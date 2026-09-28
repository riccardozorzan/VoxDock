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

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from app import __version__
from app.config import best_config, settings
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

logger = logging.getLogger("dots_tts.api")

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
        from app.patches import apply_patches

        patches = apply_patches(settings.model.vocoder_chunk_patches)

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
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the checkpoint once, keep it resident, release it on shutdown."""
    settings.paths.ensure()
    app.state.runtime = None
    app.state.runtime_info = None
    app.state.model_error = None
    app.state.started_at = time.time()
    app.state.generation_lock = asyncio.Lock()
    app.state.last_generation = None
    app.state.lifetime_peak_vram_mb = None
    app.state.request_count = 0
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
    description=(
        "Local, GPU-accelerated text-to-speech with continuation voice cloning "
        "powered by dots.tts (48 kHz AudioVAE)."
    ),
    lifespan=lifespan,
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

def _vram_usage_mb() -> Optional[float]:
    snapshot = gpu_snapshot()
    return snapshot.get("vram_allocated_mb")

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

@app.get(settings.api.health_path)
async def health(request: Request) -> JSONResponse:
    """GPU name, total/allocated/free VRAM (MB) and model readiness.

    Returns HTTP 503 while the checkpoint is still loading so that the Docker
    health check (and any ``depends_on: service_healthy`` gate) waits for the
    engine instead of the bare ASGI process.
    """
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
        "device": settings.model.device,
        "model": model,
        # Which resolution tier produced each setting (v3 Task 3.1). Makes it
        # obvious whether a tuned profile is actually in force.
        "config_source": {
            "best_config": best_config or None,
            "best_config_path": str(settings.paths.output_dir / "best_config.json"),
        },
        "speakers": {
            "registered": len(getattr(state, "speakers").list())
            if getattr(state, "speakers", None)
            else 0,
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
    return JSONResponse(status_code=200 if runtime is not None else 503, content=body)

@app.get("/v1/models")
async def list_models(request: Request) -> dict[str, Any]:
    return {"object": "list", "data": [_model_payload(request)]}

# --------------------------------------------------------------------------- #
#  Speaker registry (v3 Task 3.3)
# --------------------------------------------------------------------------- #
def _registry(request: Request) -> SpeakerRegistry:
    return request.app.state.speakers

@app.post("/v1/speakers/register", status_code=201)
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

    record = _registry(request).register(
        speaker_id=speaker_id,
        audio_bytes=payload,
        audio_filename=reference_audio.filename or "reference.wav",
        reference_text=reference_text,
        overwrite=overwrite,
    )
    data = record.as_dict()
    data["duration_seconds"] = round(wav_duration_seconds(record.audio_path), 3)
    return {"object": "speaker", **data}

@app.get("/v1/speakers")
async def list_speakers(request: Request) -> dict[str, Any]:
    """Every registered voice, with the number of times each has been reused."""
    records = _registry(request).list()
    return {
        "object": "list",
        "data": [
            {**record.as_dict(), "duration_seconds": round(
                wav_duration_seconds(record.audio_path), 3
            )}
            for record in records
        ],
    }

@app.delete("/v1/speakers/{speaker_id}")
async def delete_speaker(request: Request, speaker_id: str) -> dict[str, Any]:
    if not _registry(request).delete(speaker_id):
        raise HTTPException(status_code=404, detail=f"unknown speaker {speaker_id!r}.")
    return {"object": "deleted", "id": speaker_id}

@app.post("/v1/admin/clear-cache")
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
    if not cleaned_text:
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
        async with state.generation_lock:
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
    """Attach a stdout handler so ``dots_tts.api`` / ``dots_tts.patches`` emit.

    ``uvicorn`` configures its own loggers and the ``dots_tts`` package installs
    its own, so without this the application loggers have no handler and the
    startup / patch / request records never reach the container log.
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
    for name in ("dots_tts.api", "dots_tts.patches"):
        logging.getLogger(name).setLevel(level)

configure_logging()
