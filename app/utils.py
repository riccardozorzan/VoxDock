"""Audio-buffer, GPU-telemetry and temporary-file helpers.

Heavy imports (``torch``, ``soundfile``, ``numpy``) are performed lazily inside
the functions so this module stays importable from a bare host interpreter.
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import tempfile
import wave
from datetime import datetime, timezone
from typing import Any, Iterator, Optional

from app.config import settings

from app import LOG_NAMESPACE

logger = logging.getLogger(f"{LOG_NAMESPACE}.utils")

__all__ = [
    "now_iso",
    "bytes_to_mb",
    "resolve_precision",
    "torch_available",
    "gpu_snapshot",
    "reset_cuda_peak_stats",
    "clear_cuda_cache",
    "clear_prompt_cache",
    "is_out_of_memory",
    "to_numpy_mono",
    "tensor_to_wav_bytes",
    "write_wav",
    "wav_duration_seconds",
    "temp_file_path",
    "safe_filename",
    "format_duration",
]

_BYTES_PER_MB = 1024 * 1024
_HALF_PRECISIONS = {"bfloat16", "bf16", "float16", "fp16", "half"}


# --------------------------------------------------------------------------- #
#  Small helpers
# --------------------------------------------------------------------------- #
def now_iso() -> str:
    """UTC timestamp in ISO-8601 form (used in every JSON report)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def bytes_to_mb(value: Optional[int]) -> Optional[float]:
    if value is None:
        return None
    return round(int(value) / _BYTES_PER_MB, 2)


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes, whole = divmod(int(seconds), 60)
    return f"{minutes:d}m{whole:02d}s" if minutes else f"{whole:d}s"


def safe_filename(name: str) -> str:
    """Strip any directory component / odd characters from a client filename."""
    base = os.path.basename(name or "reference.wav")
    cleaned = "".join(ch if ch.isalnum() or ch in "-_. " else "_" for ch in base)
    return cleaned.strip() or "reference.wav"


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:  # pragma: no cover - defensive
        return False
    return True


# --------------------------------------------------------------------------- #
#  Precision handling
# --------------------------------------------------------------------------- #
def resolve_precision(precision: str) -> str:
    """Return a precision string that is actually usable on this GPU.

    ``bfloat16`` is requested by the spec; on GPUs without native BF16 support
    (pre-Ampere) we transparently fall back to ``float16``.
    """
    requested = (precision or "bfloat16").strip().lower()
    if requested in {"bf16"}:
        requested = "bfloat16"
    if requested in {"fp16", "half"}:
        requested = "float16"
    try:
        import torch
    except Exception:  # pragma: no cover - defensive
        return requested

    if not torch.cuda.is_available():
        if requested in _HALF_PRECISIONS:
            logger.warning(
                "CUDA unavailable -> forcing float32 (requested %s)", requested
            )
            return "float32"
        return requested

    if requested == "bfloat16" and not torch.cuda.is_bf16_supported():
        logger.warning(
            "GPU %s has no native bfloat16 support -> falling back to float16",
            torch.cuda.get_device_name(0),
        )
        return "float16"
    return requested


# --------------------------------------------------------------------------- #
#  GPU telemetry (spec Phase 3, Task 3.2)
# --------------------------------------------------------------------------- #
def _empty_gpu_snapshot() -> dict[str, Any]:
    return {
        "cuda_available": False,
        "device_count": 0,
        "name": None,
        "compute_capability": None,
        "torch_version": None,
        "cuda_version": None,
        "vram_total_mb": None,
        "vram_free_mb": None,
        "vram_used_mb": None,
        "vram_allocated_mb": None,
        "vram_reserved_mb": None,
        "vram_peak_allocated_mb": None,
        "vram_peak_reserved_mb": None,
        "vram_limit_gb": settings.model.vram_limit_gb,
        "error": None,
    }


def gpu_snapshot() -> dict[str, Any]:
    """Collect a VRAM/GPU snapshot.

    ``vram_used_mb`` (``total - free`` from :func:`torch.cuda.mem_get_info`) is a
    whole-device figure; the ``*_allocated``/``*_reserved`` pair is PyTorch's own
    allocator view of *this* process, and ``vram_peak_allocated_mb`` reflects the
    high-water mark since the last :func:`reset_cuda_peak_stats` call.
    """
    snapshot = _empty_gpu_snapshot()
    try:
        import torch
    except Exception as exc:  # pragma: no cover - defensive
        snapshot["error"] = f"torch unavailable: {exc}"
        return snapshot

    snapshot["torch_version"] = getattr(torch, "__version__", None)
    snapshot["cuda_version"] = getattr(getattr(torch, "version", None), "cuda", None)
    snapshot["cuda_available"] = bool(torch.cuda.is_available())
    if not snapshot["cuda_available"]:
        snapshot["error"] = "CUDA is not available"
        return snapshot

    try:
        snapshot["device_count"] = torch.cuda.device_count()
        snapshot["name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        snapshot["compute_capability"] = f"{props.major}.{props.minor}"

        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        snapshot["vram_total_mb"] = bytes_to_mb(total_bytes)
        snapshot["vram_free_mb"] = bytes_to_mb(free_bytes)
        snapshot["vram_used_mb"] = bytes_to_mb(total_bytes - free_bytes)

        snapshot["vram_allocated_mb"] = bytes_to_mb(torch.cuda.memory_allocated(0))
        snapshot["vram_reserved_mb"] = bytes_to_mb(torch.cuda.memory_reserved(0))
        snapshot["vram_peak_allocated_mb"] = bytes_to_mb(
            torch.cuda.max_memory_allocated(0)
        )
        snapshot["vram_peak_reserved_mb"] = bytes_to_mb(
            torch.cuda.max_memory_reserved(0)
        )
    except Exception as exc:  # pragma: no cover - defensive
        snapshot["error"] = f"{type(exc).__name__}: {exc}"
    return snapshot


def reset_cuda_peak_stats() -> None:
    """Reset the allocator high-water marks (used before each request)."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(0)
    except Exception:  # pragma: no cover - defensive
        logger.debug("reset_peak_memory_stats failed", exc_info=True)


def clear_cuda_cache() -> bool:
    """Empty the CUDA caching allocator; returns whether it actually ran.

    Called between chunks and after an OOM so that freed blocks return to the
    driver instead of being held by the allocator.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        torch.cuda.empty_cache()
        return True
    except Exception:  # pragma: no cover - defensive
        logger.debug("empty_cache failed", exc_info=True)
        return False


def clear_prompt_cache(runtime: Any) -> int:
    """Drop ``dots_tts``' cached prompt latents. Returns entries removed.

    ``DotsTtsModel`` keeps an ``OrderedDict`` of prompt latents keyed by a hash
    of the reference clip. On a GPU this tight that cache is a real memory
    hold, so ``/v1/admin/clear-cache`` evicts it wholesale instead of relying on
    the library's own size limit.
    """
    model = getattr(runtime, "model", None)
    cache = getattr(model, "_prompt_feature_cache", None)
    if cache is None:
        return 0
    try:
        removed = len(cache)
        cache.clear()
        return removed
    except Exception:  # pragma: no cover - library internals may change
        logger.debug("could not clear prompt cache", exc_info=True)
        return 0


def is_out_of_memory(exc: BaseException) -> bool:
    """Whether ``exc`` (or anything in its cause chain) is a CUDA OOM.

    Classifying this server-side keeps the client's split-and-retry logic from
    having to pattern-match driver error strings.
    """
    seen: set[int] = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, MemoryError) or "out of memory" in str(current).lower():
            return True
        current = current.__cause__ or current.__context__
    return False



# --------------------------------------------------------------------------- #
#  Audio conversion helpers
# --------------------------------------------------------------------------- #
def to_numpy_mono(audio: Any) -> "Any":
    """Normalise a torch tensor / ndarray to a 1-D contiguous float32 array."""
    import numpy as np

    if hasattr(audio, "detach"):  # torch.Tensor
        audio = audio.detach().float().cpu().numpy()
    array = np.asarray(audio, dtype="float32")
    if array.ndim > 1:
        # Accept (1, samples), (channels, samples) or (samples, channels).
        array = array.reshape(-1) if array.shape[0] == 1 else array.mean(axis=0)
    return np.ascontiguousarray(array.reshape(-1), dtype="float32")


def tensor_to_wav_bytes(
    audio: Any,
    sample_rate: int,
    subtype: str = "PCM_16",
) -> bytes:
    """Serialise a generated waveform to an in-memory ``audio/wav`` payload."""
    import soundfile as sf

    buffer = io.BytesIO()
    # soundfile infers the container from the filename extension, which a
    # BytesIO does not have, so the format has to be named explicitly.
    sf.write(
        buffer,
        to_numpy_mono(audio),
        int(sample_rate),
        subtype=subtype,
        format="WAV",
    )
    return buffer.getvalue()


def write_wav(
    destination: Any,
    audio: Any,
    sample_rate: int,
    subtype: str = "PCM_16",
) -> None:
    """``soundfile.write`` wrapper accepting tensors, arrays and byte buffers."""
    import soundfile as sf

    # File-like objects (and extension-less paths) need an explicit format;
    # real paths keep the extension-based inference.
    fmt = "WAV" if isinstance(destination, (io.IOBase, io.BytesIO)) else None
    if fmt is None and not str(destination).lower().endswith(".wav"):
        fmt = "WAV"
    kwargs = {"format": fmt} if fmt else {}
    sf.write(
        destination,
        to_numpy_mono(audio),
        int(sample_rate),
        subtype=subtype,
        **kwargs,
    )



def wav_duration_seconds(path: str | os.PathLike[str]) -> float:
    """Duration of a PCM WAV file using only the standard library."""
    with contextlib.closing(wave.open(str(path), "rb")) as handle:
        frame_rate = handle.getframerate() or 1
        return handle.getnframes() / float(frame_rate)


# --------------------------------------------------------------------------- #
#  Temporary files
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def temp_file_path(
    suffix: str = ".wav",
    prefix: str = "tts_",
    directory: Optional[str | os.PathLike[str]] = None,
) -> Iterator[str]:
    """Yield a temp file path and always remove it, even on failure."""
    handle = tempfile.NamedTemporaryFile(
        suffix=suffix, prefix=prefix, dir=directory, delete=False
    )
    path = handle.name
    handle.close()
    try:
        yield path
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(path)

