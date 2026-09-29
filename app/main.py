"""FastAPI service exposing dots.tts voice cloning over HTTP.

Routes
------
* ``GET  /``                – service banner / capability document.
* ``GET  /health``          – model state + live VRAM telemetry (spec Phase 3.2).
* ``GET  /v1/models``       – descriptor of the currently loaded checkpoint.
* ``POST /v1/audio/speech`` – synthesis route (spec Phase 3.3/3.4) returning raw
  ``audio/wav`` bytes.

Inference runs in a worker thread guarded by an :class:`asyncio.Lock` so that a
single 6 GB GPU is never hit by two concurrent generations.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import LOG_NAMESPACE, __version__
from app.config import best_config, settings
from app.security import (
    SCOPE_ADMIN,
    SCOPE_SPEAKERS,
    SCOPE_TTS,
    ApiError,
    QueueGate,
    RateLimiter,
    client_principal,
    error_body,
    evaluate_exposure,
    extract_credential,
    new_request_id,
    sanitise_request_id,
    verify_key,
)
from app.speakers import SpeakerRegistry
from app.utils import (
    clear_cuda_cache,
    clear_prompt_cache,
    gpu_snapshot,
    is_out_of_memory,
    now_iso,
    reset_cuda_peak_stats,
    resolve_precision,
    safe_filename,
    temp_file_path,
    tensor_to_wav_bytes,
    wav_duration_seconds,
)

logger = logging.getLogger(f"{LOG_NAMESPACE}.api")

#: The single-page UI. Shipped inside the image by `COPY app ./app`, so it can
#: never drift from the API it talks to. Resolved once, here, so a missing file
#: is a startup-time constant rather than a 500 on every page load.
UI_INDEX = Path(__file__).resolve().parent / "static" / "index.html"

# --------------------------------------------------------------------------- #
#  Runtime bootstrap
# --------------------------------------------------------------------------- #
def _seed_everything(seed: int) -> None:
    """Reproduce the CLI's deterministic seeding (``dots.tts --seed``)."""
    try:
        from dots_tts.utils.util import seed_everything

        seed_everything(int(seed))
        return
    except Exception:  # pragma: no cover - fall back to plain torch seeding
        logger.debug("dots_tts seed_everything unavailable, using torch seeding")

    import torch

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def load_runtime() -> dict[str, Any]:
    """Blocking model load executed in a worker thread during startup."""
    from dots_tts.runtime import DotsTtsRuntime

    # The low-VRAM patches must be installed *before* the runtime builds its DiT
    # solver, because that is the point where the AdaLN fusion happens.
    patches = {"compile": False, "fused_adaln": False, "generate_audio": False}
    if settings.model.low_vram_patches:
        from app.patches import apply_patches, assert_patches_applied

        patches = apply_patches(settings.model.vocoder_chunk_patches)
        # Each patch is individually guarded so a future dots.tts rename degrades
        # instead of crashing, but on a 6 GB card a missing patch means the
        # verified configuration no longer fits. Fail here, immediately and
        # legibly, rather than OOMing minutes into the warmup.
        assert_patches_applied(patches)
    else:
        logger.warning(
            "TTS_LOW_VRAM_PATCHES=false: running the stock inference path. On a "
            "6 GB card this needs roughly 600 MB more than is available, so "
            "expect an OOM at max_generate_length=128 and lower it if it fails."
        )

    precision = resolve_precision(settings.model.precision)
    logger.info(
        "Loading checkpoint %s | precision=%s optimize=%s warmup=%s "
        "max_generate_length=%s max_sequence_length=%s patches=%s "
        "(first run downloads ~4.4 GB of weights)",
        settings.model.model_name_or_path,
        precision,
        settings.model.optimize,
        settings.model.warmup_on_optimize,
        settings.model.max_generate_length,
        settings.model.max_sequence_length,
        patches,
    )
    started = time.perf_counter()
    runtime = DotsTtsRuntime.from_pretrained(
        settings.model.model_name_or_path,
        precision=precision,
        optimize=settings.model.optimize,
        max_generate_length=settings.model.max_generate_length,
        max_sequence_length=settings.model.max_sequence_length,
        vocoder_merge_steps=settings.model.vocoder_merge_steps,
        warmup_on_optimize=settings.model.warmup_on_optimize,
    )
    load_seconds = time.perf_counter() - started

    patch_seconds: Optional[float] = None
    try:
        patch_seconds = (
            float(runtime.model.config.patch_size) * float(runtime.model.hop_size)
        ) / float(runtime.sample_rate)
    except Exception:  # pragma: no cover - purely informational
        logger.debug("could not derive audio patch duration", exc_info=True)

    logger.info(
        "Checkpoint ready in %.1fs | sample_rate=%s patch_seconds=%s",
        load_seconds,
        getattr(runtime, "sample_rate", None),
        patch_seconds,
    )
    return {
        "runtime": runtime,
        "precision": precision,
        "load_seconds": load_seconds,
        "patch_seconds": patch_seconds,
    }

# --------------------------------------------------------------------------- #
#  Application lifespan
# --------------------------------------------------------------------------- #
def _enforce_exposure() -> None:
    """Refuse to start if this configuration would publish an open voice cloner.

    Called from the lifespan, before the checkpoint loads. Raises
    ``RuntimeError`` so the container exits non-zero and ``restart:
    unless-stopped`` makes the failure *visible* rather than silently retrying
    an insecure configuration forever - a crash loop is a better outcome than a
    service quietly listening on 0.0.0.0.
    """
    sec = settings.security
    verdict = evaluate_exposure(
        sec.publish_addr,
        auth_enabled=sec.auth_enabled,
        allow_remote=sec.allow_remote,
        bind_host=settings.api.host,
        in_container=sec.in_container,
    )
    if verdict.fatal:
        raise RuntimeError(verdict.fatal)
    if verdict.warning:
        logger.warning("%s", verdict.warning)
    if sec.auth_enabled:
        logger.info(
            "Authentication enabled: %d API key(s) loaded (%s).",
            len(sec.api_keys),
            ", ".join(f"{k.label}={'+'.join(sorted(k.scopes))}" for k in sec.api_keys),
        )
    else:
        logger.info(
            "No API keys configured (TTS_API_KEYS unset) - the API is open to "
            "anything that can reach %s.",
            sec.publish_addr,
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the checkpoint once, keep it resident, release it on shutdown.

    The exposure guard runs *first*, before the 4.4 GB checkpoint is loaded and
    before the port is accepting anything. Refusing here costs a second;
    refusing after the model is resident costs a minute of the operator's time
    and leaves a brief window where the service is reachable.
    """
    settings.paths.ensure()
    _enforce_exposure()

    app.state.runtime = None
    app.state.runtime_info = None
    app.state.model_error = None
    app.state.started_at = time.time()
    app.state.generation_lock = asyncio.Lock()
    app.state.last_generation = None
    app.state.lifetime_peak_vram_mb = None
    app.state.request_count = 0
    # Bounded admission in front of the lock, and the per-key rate limiter.
    # Both are per-process state on purpose: one uvicorn worker, one GPU, and a
    # shared counter across workers would be a shared counter that lies.
    app.state.queue_gate = QueueGate(max_queue=settings.security.max_queue_size)
    app.state.rate_limiter = RateLimiter(
        limit=settings.security.rate_limit_rpm,
        window_s=settings.security.rate_limit_window_s,
    )
    app.state.speakers = SpeakerRegistry(settings.paths.speakers_dir)
    logger.info(
        "Registered speakers on disk: %d",
        len(app.state.speakers.list()),
    )

    logger.info(
        "Starting %s v%s | model=%s | max_generate_length=%s",
        settings.api.title,
        settings.api.version,
        settings.model.model_name_or_path,
        settings.model.max_generate_length,
    )
    try:
        info = await asyncio.to_thread(load_runtime)
    except Exception as exc:  # keep the API reachable so /health can explain
        app.state.model_error = f"{type(exc).__name__}: {exc}"
        logger.exception("Checkpoint load failed - /health will report unhealthy")
    else:
        app.state.runtime = info["runtime"]
        app.state.runtime_info = info

    try:
        yield
    finally:
        logger.info("Shutting down; releasing GPU memory")
        app.state.runtime = None
        app.state.runtime_info = None
        clear_cuda_cache()

app = FastAPI(
    title=settings.api.title,
    version=settings.api.version,
    description=settings.api.description,
    lifespan=lifespan,
)

# --------------------------------------------------------------------------- #
#  Request identity, authentication and structured errors
#
#  Everything below is a thin adapter: the rules live in app/security.py, which
#  imports neither torch nor FastAPI and is therefore tested on the host. This
#  layer only translates HTTP into those calls and back.
# --------------------------------------------------------------------------- #

#: Endpoints reachable without a key, whatever else is protected. ``/health``
#: is here because the Docker health check and any ``depends_on:
#: service_healthy`` gate have no way to present a credential - and a container
#: that cannot be health-checked is a container that does not restart. Its
#: detailed diagnostics are admin-only; see ``/v1/admin/health``.
PUBLIC_PATHS = frozenset({"/", "/ui", settings.api.health_path, "/openapi.json"})


def _request_id(request: Request) -> str:
    """The id for this request, taken from the state the middleware set."""
    return getattr(request.state, "request_id", "") or new_request_id()


def _error(request: Request, error: ApiError) -> JSONResponse:
    """Render an :class:`ApiError` as the documented error envelope."""
    response = JSONResponse(
        status_code=error.status, content=error_body(error, _request_id(request))
    )
    if error.retry_after is not None:
        # The header form, because that is what a generic HTTP client reads.
        # Seconds, rounded up: telling a client to retry in 0.4 s produces an
        # immediate second failure.
        response.headers["Retry-After"] = str(max(1, int(error.retry_after)))
    return response


def _refusal(request: Request, error: ApiError, auth_label: str = "") -> JSONResponse:
    """Log a refusal, then render it.

    Every 401/403/429 goes through here so a flood is attributable in the log
    to a key label or a peer address. The presented secret is never logged.
    """
    logger.warning(
        "refused %s %s -> %s %s (principal=%s)",
        request.method,
        request.url.path,
        error.status,
        error.code,
        auth_label or request.client.host if request.client else "-",
    )
    return _error(request, error)


def require_scope(scope: str):
    """Build a FastAPI dependency that demands one scope.

    Returns the dependency rather than being one, because the scope is what
    varies per route; the rest - header extraction, constant-time comparison,
    the rate-limit principal - is identical for all of them and lives here.
    """

    def _dependency(
        request: Request,
        authorization: Optional[str] = Header(None),
        x_api_key: Optional[str] = Header(None),
    ) -> Any:
        if request.url.path in PUBLIC_PATHS:
            return None
        sec = settings.security
        presented = extract_credential(authorization, x_api_key)
        auth = verify_key(presented, sec.api_keys, scope)

        if not auth.ok:
            raise _Refused(
                _refusal(
                    request,
                    ApiError(
                        401 if auth.reason == "invalid" else 403,
                        "UNAUTHORIZED"
                        if auth.reason == "invalid"
                        else "FORBIDDEN",
                        auth.message
                        or f"this endpoint requires the {scope!r} scope",
                    ),
                    auth.label or "",
                )
            )

        if sec.rate_limit_rpm > 0:
            client = request.client
            decision = request.app.state.rate_limiter.check(
                client_principal(auth, client.host if client else None)
            )
            if not decision.allowed:
                raise _Refused(
                    _refusal(
                        request,
                        ApiError(
                            429,
                            "RATE_LIMITED",
                            (
                                f"rate limit of {decision.limit} requests per "
                                f"{sec.rate_limit_window_s:g}s exceeded"
                            ),
                            retry_after=decision.retry_after,
                        ),
                        auth.label or "",
                    )
                )
            request.state.rate_remaining = decision.remaining
        request.state.auth_label = auth.label
        return auth

    return _dependency


class _Refused(Exception):
    """Carries a rendered refusal through to the exception handler.

    Raising an :class:`HTTPException` would be the obvious way to do this, but
    that produces FastAPI's ``{"detail": ...}`` body, and the documented
    envelope has to be identical for a 401, a 429 and a 503 - otherwise a
    client has to special-case every refusal. The handler below catches this and
    emits the ``{"error": {...}}`` shape, so one envelope covers all of them.
    """

    def __init__(self, response: JSONResponse) -> None:
        super().__init__("refused")
        self.response = response

@app.middleware("http")
async def _request_context(request: Request, call_next):
    """Give every request an id, and echo it back on every response.

    The id is what makes a refusal actionable: it appears in the log line, in
    the error body and in the ``X-Request-ID`` header, so a user can quote it
    and the operator can find the request without trawling the log. A
    client-supplied one is honoured when it is boring enough to be safe - see
    ``sanitise_request_id`` for why that check exists.
    """
    request.state.request_id = sanitise_request_id(request.headers.get("X-Request-ID"))
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response


@app.exception_handler(_Refused)
async def _refused_handler(request: Request, exc: _Refused) -> JSONResponse:
    """Refusals already carry their rendered envelope and ``Retry-After``."""
    response = exc.response
    response.headers["X-Request-ID"] = _request_id(request)
    return response


@app.exception_handler(StarletteHTTPException)
async def _http_error_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """Re-shape FastAPI's ``{"detail": ...}`` into the documented envelope.

    Registered against Starlette's class, not FastAPI's, and that detail is
    load-bearing: FastAPI's is a *subclass* of it, so this one handler covers
    every ``HTTPException`` the routes raise **and** the 404/405 that Starlette
    itself produces for an unknown path or method. Registering only the
    FastAPI class leaves a plain ``GET /typo`` answering in the old shape,
    which is how a client ends up parsing two different error formats.

    The status code and the message are preserved exactly; only the wrapper
    changes. ``detail`` is sometimes already a dict (the out-of-memory path in
    ``create_speech``), so it is passed through as the message rather than
    stringified into ``"{'error': 'out_of_memory'}"``.
    """
    detail = exc.detail
    message = detail if isinstance(detail, str) else str(detail)
    code = {400: "BAD_REQUEST", 401: "UNAUTHORIZED", 403: "FORBIDDEN",
            404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED",
            413: "PAYLOAD_TOO_LARGE", 429: "RATE_LIMITED",
            500: "INTERNAL_ERROR", 503: "ENGINE_BUSY"}.get(exc.status_code, "ERROR")
    return _error(request, ApiError(exc.status_code, code, message))


@app.exception_handler(RequestValidationError)
async def _validation_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """A malformed form is a 422, but still in the documented shape.

    The framework's own body lists every field error, which is useful, so its
    ``detail`` is kept verbatim under ``message`` rather than being flattened
    into prose the client cannot act on.
    """
    return _error(
        request,
        ApiError(422, "INVALID_REQUEST", str(exc.errors()[:5]), retryable=False),
    )


@app.exception_handler(Exception)
async def _unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
    """Last resort: never leak a traceback to the caller.

    The traceback goes to the log with the request id; the client gets a
    message that tells it to quote that id. Echoing ``str(exc)`` here would put
    file paths and occasionally configuration values into a response body,
    which is how internal details end up in a bug report pasted into a public
    issue tracker.
    """
    logger.exception("unhandled error (request_id=%s)", _request_id(request))
    return _error(
        request,
        ApiError(500, "INTERNAL_ERROR", "internal error; quote the request id", True),
    )

# --------------------------------------------------------------------------- #
#  Shared payload builders
# --------------------------------------------------------------------------- #
def _require_runtime(request: Request) -> Any:
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        detail = getattr(request.app.state, "model_error", None) or "runtime not ready"
        raise HTTPException(
            status_code=503,
            detail=f"TTS engine is not available yet: {detail}",
        )
    return runtime

def _model_payload(request: Request) -> dict[str, Any]:
    info = getattr(request.app.state, "runtime_info", None)
    runtime = getattr(request.app.state, "runtime", None)
    payload: dict[str, Any] = {
        "id": settings.model.model_name_or_path,
        "loaded": runtime is not None,
        "object": "model",
        "precision": (info or {}).get("precision", settings.model.precision),
        "optimize": settings.model.optimize,
        "warmup_on_optimize": settings.model.warmup_on_optimize,
        "max_generate_length": settings.model.max_generate_length,
        "default_num_steps": settings.model.default_num_steps,
        "default_guidance_scale": settings.model.default_guidance_scale,
        "default_seed": settings.model.default_seed,
        "cold_start_seconds": round((info or {}).get("load_seconds", 0.0), 3),
        "audio_patch_seconds": (info or {}).get("patch_seconds"),
    }
    if runtime is not None:
        payload["sample_rate"] = int(runtime.sample_rate)
    if getattr(request.app.state, "model_error", None):
        payload["error"] = request.app.state.model_error
    return payload


# --------------------------------------------------------------------------- #
#  Synthesis worker (blocking -> executed in a thread pool)
# --------------------------------------------------------------------------- #
def _generate_sync(
    runtime: Any,
    *,
    text: str,
    prompt_audio_path: str,
    prompt_text: str,
    num_steps: Optional[int],
    guidance_scale: Optional[float],
    seed: Optional[int],
    speaker_scale: Optional[float],
    language: Optional[str],
    normalize_text: Optional[bool],
    template_name: Optional[str],
) -> dict[str, Any]:
    """Run one dots.tts request and collect its timings / VRAM high-water mark."""
    reset_cuda_peak_stats()
    if seed is not None:
        _seed_everything(int(seed))

    started = time.perf_counter()
    try:
        result = runtime.generate(
            text=text,
            prompt_audio_path=prompt_audio_path,
            prompt_text=prompt_text,
            template_name=template_name,
            language=language,
            speaker_scale=speaker_scale,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            normalize_text=normalize_text,
        )
    except ValueError as exc:
        # Fixed-step artifacts (dots.tts-mf-1step / -2steps / -2steps-stts) reject
        # sampling overrides. Retry once, letting the artifact's own contract win.
        if num_steps is None and guidance_scale is None:
            raise
        logger.warning(
            "Sampling override rejected (%s) - retrying with artifact defaults", exc
        )
        result = runtime.generate(
            text=text,
            prompt_audio_path=prompt_audio_path,
            prompt_text=prompt_text,
            template_name=template_name,
            language=language,
            speaker_scale=speaker_scale,
            num_steps=None,
            guidance_scale=None,
            normalize_text=normalize_text,
        )
    wall_seconds = time.perf_counter() - started

    audio = result["audio"]
    sample_rate = int(result["sample_rate"])
    audio_seconds = float(audio.shape[-1]) / float(sample_rate)
    engine_seconds = float(result.get("time_used") or wall_seconds)
    peak_vram_mb = gpu_snapshot().get("vram_peak_allocated_mb")

    metrics = {
        "request_id": result.get("fid"),
        "wall_seconds": round(wall_seconds, 4),
        "engine_seconds": round(engine_seconds, 4),
        "audio_seconds": round(audio_seconds, 4),
        # RTF here follows the dots.tts convention: generation time / audio time
        # (lower is better, < 1.0 means faster than real time).
        "rtf": round(wall_seconds / audio_seconds, 4) if audio_seconds else None,
        "engine_rtf": round(engine_seconds / audio_seconds, 4) if audio_seconds else None,
        # The spec's literal formula (audio time / generation time).
        "speed_factor": round(audio_seconds / wall_seconds, 4) if wall_seconds else None,
        "peak_vram_allocated_mb": peak_vram_mb,
        "sample_rate": sample_rate,
    }
    return {"audio": audio, "sample_rate": sample_rate, "metrics": metrics}

# --------------------------------------------------------------------------- #
#  Routes
# --------------------------------------------------------------------------- #
@app.get("/", include_in_schema=False)
async def root(request: Request) -> dict[str, Any]:
    return {
        "service": settings.api.title,
        "version": __version__,
        "engine": "dots.tts",
        "model": settings.model.model_name_or_path,
        "docs": "/docs",
        "endpoints": {
            "health": f"GET {settings.api.health_path}",
            "models": "GET /v1/models",
            "speech": f"POST {settings.api.speech_path}",
        },
        "vram_limit_gb": settings.model.vram_limit_gb,
        "rtf_target": settings.model.rtf_target,
        "timestamp": now_iso(),
    }

@app.get("/ui", include_in_schema=False)
async def ui() -> FileResponse:
    """The single-page web UI.

    A self-contained HTML file served straight from the image - no build step,
    no Node, no second service. It is a thin client over the same public API
    documented at ``/docs``, so it inherits the API's exposure exactly: with no
    authentication of its own, binding the port to ``0.0.0.0`` would publish
    synthesis to the network, which is why the default is ``127.0.0.1``.

    Excluded from the schema on purpose: it is a UI, not an API surface, and
    listing it would suggest it is part of the programmatic contract.
    """
    return FileResponse(UI_INDEX, media_type="text/html")


def _health_body(request: Request) -> dict[str, Any]:
    """The full diagnostic payload, shared by both health routes."""
    state = request.app.state
    runtime = getattr(state, "runtime", None)
    gpu = gpu_snapshot()
    model = _model_payload(request)

    peak_allocated = gpu.get("vram_peak_allocated_mb")
    limit_mb = settings.model.vram_limit_gb * 1024.0
    lifetime_peak = state.lifetime_peak_vram_mb
    if peak_allocated is not None:
        lifetime_peak = max(lifetime_peak or 0.0, float(peak_allocated))
        state.lifetime_peak_vram_mb = lifetime_peak

    if runtime is not None:
        status = "ok"
    elif state.model_error:
        status = "error"
    else:
        status = "loading"

    body: dict[str, Any] = {
        "status": status,
        "ready": runtime is not None,
        "timestamp": now_iso(),
        "uptime_seconds": round(time.time() - state.started_at, 3),
        "request_count": state.request_count,
        # Queue telemetry straight from the admission gate, so it cannot report
        # a backlog the gate is not actually enforcing.
        "queue": state.queue_gate.snapshot(),
        "device": settings.model.device,
        "model": model,
        # Which resolution tier produced each setting (v3 Task 3.1). Makes it
        # obvious whether a tuned profile is actually in force.
        "config_source": {
            "best_config": best_config or None,
            "best_config_path": str(settings.paths.output_dir / "best_config.json"),
        },
        "speakers": {
            "registered": len(state.speakers.list()) if state.speakers else 0,
        },
        "vram_limit_gb": settings.model.vram_limit_gb,
        "vram_within_limit": (
            peak_allocated is None or peak_allocated <= limit_mb
        ),
        "lifetime_peak_vram_mb": lifetime_peak,
        "last_generation": state.last_generation,
        "gpu": {
            "name": gpu.get("name"),
            "cuda_available": gpu.get("cuda_available"),
            "compute_capability": gpu.get("compute_capability"),
            "torch_version": gpu.get("torch_version"),
            "cuda_version": gpu.get("cuda_version"),
            "vram_total_mb": gpu.get("vram_total_mb"),
            "vram_allocated_mb": gpu.get("vram_allocated_mb"),
            "vram_free_mb": gpu.get("vram_free_mb"),
            "vram_used_mb": gpu.get("vram_used_mb"),
            "vram_reserved_mb": gpu.get("vram_reserved_mb"),
            "vram_peak_allocated_mb": peak_allocated,
            "vram_peak_reserved_mb": gpu.get("vram_peak_reserved_mb"),
            "error": gpu.get("error"),
        },
    }
    body["security"] = {
        # Deliberately reported so an operator can confirm the state of the
        # control from outside, without reading the startup log.
        "auth_required": settings.security.auth_enabled,
        "scopes_required": {
            "speech": SCOPE_TTS,
            "speakers": SCOPE_SPEAKERS,
            "admin": SCOPE_ADMIN,
        },
        "publish_addr": settings.security.publish_addr,
        "rate_limit_rpm": settings.security.rate_limit_rpm,
        "max_queue_size": settings.security.max_queue_size,
    }
    return body


@app.get(settings.api.health_path)
async def health(request: Request) -> JSONResponse:
    """Liveness and readiness, safe to expose without a key.

    Returns 503 while the checkpoint is still loading, so the Docker health
    check (and any ``depends_on: service_healthy`` gate) waits for the engine
    rather than the bare ASGI process.

    When authentication is enabled this returns only the fields a load balancer
    needs. The full telemetry - VRAM, the loaded model, the queue, the last
    generation - moves to ``GET /v1/admin/health``, because on a published
    instance that payload tells an unauthenticated caller which model is
    loaded, how much VRAM is free and how busy the GPU is, which is reconnaissance
    for exactly the attack this P0 work exists to stop. The split is
    conditional so a local-first install sees no change at all.
    """
    body = _health_body(request)
    ready = bool(body.get("ready"))
    if settings.security.auth_enabled:
        public = {
            "status": body["status"],
            "ready": ready,
            "timestamp": body["timestamp"],
            "uptime_seconds": body["uptime_seconds"],
            "queue": {"pending": body["queue"]["pending"]},
        }
        return JSONResponse(status_code=200 if ready else 503, content=public)
    return JSONResponse(status_code=200 if ready else 503, content=body)


@app.get("/v1/admin/health", dependencies=[Depends(require_scope(SCOPE_ADMIN))])
async def admin_health(request: Request) -> JSONResponse:
    """The full diagnostic payload behind the admin scope.

    Same body ``/health`` returns when the service is open locally; available
    without a key only in that case, so the local-first workflow is unchanged.
    """
    body = _health_body(request)
    return JSONResponse(
        status_code=200 if body.get("ready") else 503, content=body
    )


@app.get("/v1/models", dependencies=[Depends(require_scope(SCOPE_TTS))])
async def list_models(request: Request) -> dict[str, Any]:
    return {"object": "list", "data": [_model_payload(request)]}

# --------------------------------------------------------------------------- #
#  Speaker registry (v3 Task 3.3)
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def _generation_slot(request: Request, state: Any):
    """Hold - or wait for - the GPU, with the queue bounded.

    One 6 GB GPU cannot run two generations at once, so the lock is what makes
    the service correct. The lock alone, though, is an *unbounded* queue: every
    request that arrives parks on it and eventually runs, so a client that
    sends 400 gets 399 connections held for the length of a synthesis each,
    and the burst is delayed rather than rejected. ``QueueGate`` refuses the
    excess with a 503 and a ``Retry-After`` instead, which the client can act on.

    The gate is also what ``/health`` reports, so the telemetry cannot drift
    from what the gate is actually doing - the two used to be separate counters
    incremented in this function.

    Both counters are decremented in a ``finally``, so a request that fails
    while waiting - or is cancelled - cannot leak a slot and wedge the queue.
    """
    admission = state.queue_gate.admit()
    if not admission.admitted:
        raise _Refused(
            _refusal(
                request,
                ApiError(
                    503,
                    "QUEUE_FULL",
                    f"the generation queue is full ({admission.pending} of "
                    f"{state.queue_gate.max_queue} slots in use). Retry after "
                    f"about {admission.retry_after:.0f}s.",
                    retry_after=admission.retry_after,
                ),
                getattr(request.state, "auth_label", "") or "",
            )
        )
    started = time.perf_counter()
    try:
        async with state.generation_lock:
            state.queue_gate.start()
            try:
                yield
            finally:
                state.queue_gate.finish()
    finally:
        # Fed the measured wall time so the next refusal quotes a real number
        # instead of the startup estimate.
        state.queue_gate.release(service_seconds=time.perf_counter() - started)


def _registry(request: Request) -> SpeakerRegistry:
    return request.app.state.speakers

@app.post(
    "/v1/speakers/register",
    status_code=201,
    dependencies=[Depends(require_scope(SCOPE_SPEAKERS))],
)
async def register_speaker(
    request: Request,
    reference_audio: UploadFile = File(..., description="Clean 5-15 s reference WAV."),
    reference_text: str = Form(
        ..., description="Exact transcript of reference_audio (continuation cloning)."
    ),
    speaker_id: Optional[str] = Form(
        None, description="Reusable name; generated when omitted."
    ),
    overwrite: bool = Form(False, description="Replace an existing speaker_id."),
    consent_acknowledged: Optional[bool] = Form(
        None,
        description=(
            "Confirm the person in this clip has agreed to being cloned. "
            "Recorded with the voice; see the 'consent' object in the response."
        ),
    ),
    consent_owner: Optional[str] = Form(
        None, description="Who owns the voice, if you record it."
    ),
    consent_basis: Optional[str] = Form(
        None, description="On what basis it may be used (contract, self, research)."
    ),
) -> dict[str, Any]:
    """Store a voice once so later requests can reference it by ``speaker_id``.

    The clip is validated for decodability and duration before being stored, so
    a broken reference is rejected at registration rather than failing on every
    subsequent chunk.
    """
    payload = await reference_audio.read()
    await reference_audio.close()

    max_bytes = settings.api.max_upload_mb * 1024 * 1024
    if len(payload) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"'reference_audio' exceeds {settings.api.max_upload_mb} MB.",
        )

    registry = _registry(request)
    _check_speaker_capacity(request, registry, speaker_id, overwrite)

    try:
        record = registry.register(
            speaker_id=speaker_id,
            audio_bytes=payload,
            audio_filename=reference_audio.filename or "reference.wav",
            reference_text=reference_text,
            overwrite=overwrite,
            consent_acknowledged=consent_acknowledged,
            consent_owner=consent_owner or "",
            consent_basis=consent_basis or "",
        )
    except ValueError as exc:
        # The registry validates the id, the audio and the transcript, and
        # signals every one of them with ValueError. Letting that escape was a
        # latent 500: a client re-registering a name got a server error for a
        # client mistake. Split on which message it is, because the two want
        # different answers - 409 says "pick another name or pass
        # overwrite=true", 400 says "this input is wrong".
        message = str(exc)
        conflict = "already exists" in message
        raise _Refused(
            _refusal(
                request,
                ApiError(
                    409 if conflict else 400,
                    "SPEAKER_EXISTS" if conflict else "INVALID_SPEAKER",
                    message,
                    retryable=False,
                ),
                getattr(request.state, "auth_label", "") or "",
            )
        ) from exc
    data = record.as_api_dict()
    data["duration_seconds"] = round(wav_duration_seconds(record.audio_path), 3)
    if record.consent_acknowledged:
        logger.info(
            "speaker %s registered with consent acknowledged (owner=%r, basis=%r)",
            record.id,
            record.consent_owner or "-",
            record.consent_basis or "-",
        )
    else:
        logger.warning(
            "speaker %s registered WITHOUT a consent acknowledgement; it will "
            "report consent.status=unknown",
            record.id,
        )
    return {"object": "speaker", **data}


def _check_speaker_capacity(
    request: Request, registry: SpeakerRegistry, speaker_id: Optional[str], overwrite: bool
) -> None:
    """Refuse a registration that would exceed ``TTS_MAX_SPEAKERS``.

    Without a cap, a loop of register calls fills the bind-mounted ``data/``
    directory, and the service stops for a reason that has nothing to do with
    the GPU. Overwrites and id collisions do not add a new voice, so they are
    not counted against the limit.
    """
    limit = settings.security.max_speakers
    if not limit:
        return
    # An overwrite of an *existing* id does not add a voice, so it is not
    # capped. Naming a new one does - otherwise the cap would be trivially
    # bypassed by passing a speaker_id, which is the obvious first thing
    # anyone would try.
    if speaker_id and registry.get(speaker_id.strip()) is not None:
        return
    current = len(registry.list())
    if current >= limit:
        raise _Refused(
            _refusal(
                request,
                ApiError(
                    409,
                    "SPEAKER_LIMIT_REACHED",
                    f"{current} voices are already registered and the limit is "
                    f"{limit}. Delete one with DELETE /v1/speakers/{{id}}, or "
                    "raise TTS_MAX_SPEAKERS.",
                    retryable=False,
                ),
                getattr(request.state, "auth_label", "") or "",
            )
        )

@app.get("/v1/speakers", dependencies=[Depends(require_scope(SCOPE_SPEAKERS))])
async def list_speakers(request: Request) -> dict[str, Any]:
    """Every registered voice, with the number of times each has been reused."""
    records = _registry(request).list()
    return {
        "object": "list",
        "data": [
            {**record.as_api_dict(), "duration_seconds": round(
                wav_duration_seconds(record.audio_path), 3
            )}
            for record in records
        ],
    }

@app.delete(
    "/v1/speakers/{speaker_id}", dependencies=[Depends(require_scope(SCOPE_SPEAKERS))]
)
async def delete_speaker(request: Request, speaker_id: str) -> dict[str, Any]:
    if not _registry(request).delete(speaker_id):
        raise HTTPException(status_code=404, detail=f"unknown speaker {speaker_id!r}.")
    return {"object": "deleted", "id": speaker_id}

@app.post("/v1/admin/clear-cache", dependencies=[Depends(require_scope(SCOPE_ADMIN))])
async def clear_cache(request: Request) -> dict[str, Any]:
    """Drop prompt latents and free cached CUDA blocks.

    ``process_txt.py`` calls this before retrying a chunk that ran out of
    memory, so a transient spike does not turn into a permanently wedged
    engine.
    """
    state = request.app.state
    released = clear_cuda_cache()
    prompt_entries = clear_prompt_cache(getattr(state, "runtime", None))
    state.request_count = 0
    logger.info(
        "Cache cleared: cuda released=%s, prompt entries=%s", released, prompt_entries
    )
    return {
        "object": "cache",
        "cuda_cache_released": released,
        "prompt_cache_entries_cleared": prompt_entries,
        "gpu": gpu_snapshot(),
    }

@app.post(
    settings.api.speech_path,
    dependencies=[Depends(require_scope(SCOPE_TTS))],
    responses={
        200: {
            "content": {"audio/wav": {}},
            "description": (
                "Raw 48 kHz WAV audio, or a JSON envelope with base64 audio when "
                "``include_metrics=true``."
            ),
        }
    },
)
async def create_speech(
    request: Request,
    text: str = Form(..., description="Text to synthesise (prosody follows . , ! ? ...)."),
    reference_audio: Optional[UploadFile] = File(
        None,
        description=(
            "Clean 5-15 s reference WAV. Omit when 'speaker_id' names a "
            "previously registered voice (v3 Task 3.3)."
        ),
    ),
    speaker_id: Optional[str] = Form(
        None, description="Reuse a voice registered via /v1/speakers/register."
    ),
    reference_text: Optional[str] = Form(
        None,
        description=(
            "Exact transcript of reference_audio (continuation cloning). "
            "Required unless 'speaker_id' is given."
        ),
    ),
    num_steps: Optional[int] = Form(None, ge=1, le=64),
    guidance_scale: Optional[float] = Form(None, gt=0.0),
    seed: Optional[int] = Form(None),
    speaker_scale: Optional[float] = Form(None, gt=0.0),
    language: Optional[str] = Form(None),
    normalize_text: Optional[bool] = Form(None),
    template_name: Optional[str] = Form(None),
    include_metrics: bool = Form(False),
) -> Response:
    """Synthesise ``text`` in the voice of ``reference_audio`` or ``speaker_id``."""
    runtime = _require_runtime(request)

    cleaned_text = (text or "").strip()
    max_chars = settings.security.max_text_chars
    if max_chars and len(cleaned_text) > max_chars:
        # Refused here rather than deeper in, where the failure is an OOM: the
        # 128-patch budget at 160 ms is ~20 s of audio, and a text long enough
        # to blow past it is a client bug worth naming precisely.
        raise _Refused(
            _refusal(
                request,
                ApiError(
                    413,
                    "TEXT_TOO_LONG",
                    f"'text' is {len(cleaned_text)} characters; the limit is "
                    f"{max_chars}. Split it into chunks, or run "
                    "scripts/process_txt.py which sizes them to the VRAM budget.",
                    retryable=False,
                ),
                getattr(request.state, "auth_label", "") or "",
            )
        )
    if not cleaned_text:
        # Defence in depth: FastAPI's form validation rejects an empty or
        # whitespace-only `text` before the route body runs, so a client sending
        # `text=` actually gets 422 {"type": "missing"} from the framework. This
        # branch catches anything that gets past it, and the 422 is what the
        # documented contract promises.
        raise HTTPException(status_code=400, detail="'text' must not be empty.")

    state = request.app.state

    # Resolve the prompt source: an inline reference_audio, or a registered
    # speaker. speaker_id wins when both are supplied (explicit reuse).
    record = None
    if speaker_id:
        record = _registry(request).get(speaker_id.strip())
        if record is None:
            raise HTTPException(
                status_code=404, detail=f"unknown speaker_id {speaker_id!r}."
            )
        prompt_text = record.reference_text
        prompt_wav = record.audio_path
        prompt_suffix = ".wav"
        if reference_audio is not None:
            await reference_audio.close()  # discarded in favour of the registry
    else:
        if reference_audio is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "provide either 'reference_audio' (inline) or 'speaker_id' "
                    "(a voice registered via /v1/speakers/register)."
                ),
            )
        cleaned_prompt_text = (reference_text or "").strip()
        if not cleaned_prompt_text:
            raise HTTPException(
                status_code=400,
                detail=(
                    "'reference_text' must not be empty: continuation voice cloning "
                    "needs the exact transcript of the reference audio."
                ),
            )
        payload = await reference_audio.read()
        await reference_audio.close()
        if not payload:
            raise HTTPException(status_code=400, detail="'reference_audio' is empty.")
        max_bytes = settings.api.max_upload_mb * 1024 * 1024
        if len(payload) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"'reference_audio' exceeds {settings.api.max_upload_mb} MB.",
            )
        prompt_suffix = Path(
            safe_filename(reference_audio.filename or "reference.wav")
        ).suffix
        prompt_wav = None  # written to a temp file below
        prompt_text = cleaned_prompt_text

    # --- parameter resolution (spec Task 3.3 defaults: 10 steps / CFG 1.2) ---
    effective_num_steps = (
        settings.model.default_num_steps if num_steps is None else int(num_steps)
    )
    effective_guidance = (
        settings.model.default_guidance_scale
        if guidance_scale is None
        else float(guidance_scale)
    )
    effective_speaker_scale = (
        settings.model.default_speaker_scale
        if speaker_scale is None
        else float(speaker_scale)
    )
    effective_language = settings.model.language if language is None else language
    effective_normalize = (
        settings.model.normalize_text if normalize_text is None else bool(normalize_text)
    )

    # A registered speaker already lives at a stable path; an inline upload has
    # to be staged in a temp file for the duration of the request only.
    @contextmanager
    def _prompt_source():
        if prompt_wav is not None:
            yield prompt_wav
        else:
            with temp_file_path(
                suffix=prompt_suffix or ".wav", prefix="ref_"
            ) as staged:
                Path(staged).write_bytes(payload)
                yield staged

    with _prompt_source() as prompt_path:
        async with _generation_slot(request, state):
            try:
                outcome = await asyncio.to_thread(
                    _generate_sync,
                    runtime,
                    text=cleaned_text,
                    prompt_audio_path=prompt_path,
                    prompt_text=prompt_text,
                    num_steps=effective_num_steps,
                    guidance_scale=effective_guidance,
                    seed=seed,
                    speaker_scale=effective_speaker_scale,
                    language=effective_language,
                    normalize_text=effective_normalize,
                    template_name=template_name,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                logger.exception("Synthesis failed")
                # Surface O distinctly: the client halves the chunk and retries,
                # which is pointless for any other failure.
                if is_out_of_memory(exc):
                    raise HTTPException(
                        status_code=503,
                        detail={
                            "error": "out_of_memory",
                            "message": f"{type(exc).__name__}: {exc}",
                            "retryable": True,
                        },
                    ) from exc
                raise HTTPException(
                    status_code=500, detail=f"{type(exc).__name__}: {exc}"
                ) from exc

            state.request_count += 1
            if record is not None:
                _registry(request).count_request(record.id)
            reference_seconds = round(wav_duration_seconds(prompt_path), 3)
            state.last_generation = {
                **outcome["metrics"],
                "timestamp": now_iso(),
                "text_chars": len(cleaned_text),
                "reference_seconds": reference_seconds,
                "speaker_id": record.id if record is not None else None,
            }

    metrics = outcome["metrics"]
    wav_bytes = tensor_to_wav_bytes(outcome["audio"], outcome["sample_rate"])

    if include_metrics:
        return JSONResponse(
            status_code=200,
            content={
                "request_id": metrics.get("request_id"),
                "media_type": "audio/wav",
                "sample_rate": outcome["sample_rate"],
                "audio_base64": base64.b64encode(wav_bytes).decode("ascii"),
                "metrics": metrics,
            },
        )

    headers = {
        "X-Request-Id": str(metrics.get("request_id") or ""),
        "X-Audio-Seconds": f"{metrics['audio_seconds']:.3f}",
        "X-Generation-Seconds": f"{metrics['wall_seconds']:.3f}",
        "X-Engine-RTF": (
            f"{metrics['engine_rtf']:.4f}" if metrics.get("engine_rtf") else ""
        ),
        "X-RTF": f"{metrics['rtf']:.4f}" if metrics.get("rtf") else "",
        "X-Speed-Factor": (
            f"{metrics['speed_factor']:.4f}" if metrics.get("speed_factor") else ""
        ),
        "X-Peak-Vram-MB": (
            f"{metrics['peak_vram_allocated_mb']:.1f}"
            if metrics.get("peak_vram_allocated_mb") is not None
            else ""
        ),
        "X-Vram-Limit-GB": f"{settings.model.vram_limit_gb:.2f}",
        "X-Sample-Rate": str(outcome["sample_rate"]),
        "Content-Disposition": 'attachment; filename="speech.wav"',
        "Cache-Control": "no-store",
    }
    return Response(content=wav_bytes, media_type="audio/wav", headers=headers)

# --------------------------------------------------------------------------- #
#  Logging
# --------------------------------------------------------------------------- #
def configure_logging() -> None:
    """Attach a stdout handler so the VoxDock loggers actually emit.

    ``uvicorn`` configures its own loggers and the ``dots_tts`` package installs
    its own, so without this the application loggers have no handler and the
    startup / patch / request records never reach the container log.

    ``voxdock.*`` is this project's namespace; ``dots_tts.*`` belongs to the
    upstream engine and keeps its own level.
    """
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        )
        root.addHandler(handler)
    root.setLevel(level)
    for module in ("api", "patches", "utils"):
        logging.getLogger(f"{LOG_NAMESPACE}.{module}").setLevel(level)

configure_logging()
