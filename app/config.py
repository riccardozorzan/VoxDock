"""Hardware & model configuration for the containerised dots.tts service.

Resolution order for every setting (v3 Task 3.1)
-----------------------------------------------
1. **Environment variable** — always wins, so an operator can override a tuned
   profile for a single run without editing anything.
2. **``data/output/best_config.json``** — the profile written by
   ``scripts/autotune.py`` after measuring this host.
3. **Built-in default** — the values verified on the reference 6 GB RTX 4050.

A missing, unreadable or malformed ``best_config.json`` is never fatal: the
service falls back to the environment and defaults, and says so in ``/health``.

This module is intentionally **torch-free**: every helper here is plain Python so
that the client-side scripts can import it even on a host machine that has none
of the heavy ML dependencies installed.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "PathSettings",
    "ModelSettings",
    "APISettings",
    "ProcessingSettings",
    "Settings",
    "settings",
    "best_config",
    "load_best_config",
]

logger = logging.getLogger("dots_tts.api")


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(
    os.environ.get("TTS_PROJECT_ROOT") or Path(__file__).resolve().parent.parent
).resolve()

DATA_DIR = Path(os.environ.get("TTS_DATA_DIR") or (PROJECT_ROOT / "data")).resolve()

_TRUTHY = {"1", "true", "t", "yes", "y", "on"}
_FALSY = {"0", "false", "f", "no", "n", "off", ""}










# --------------------------------------------------------------------------- #
#  Tuned-profile loader (v3 Task 3.1)
# --------------------------------------------------------------------------- #
BEST_CONFIG_PATH = DATA_DIR / "output" / "best_config.json"


def load_best_config(path: Optional[Path] = None) -> dict[str, Any]:
    """Read ``best_config.json`` and flatten it to a ``{setting: value}`` map.

    Accepts either of the shapes ``scripts/autotune.py`` can emit::

        {"settings": {"TTS_MAX_GENERATE_LENGTH": 128, ...}}
        {"TTS_MAX_GENERATE_LENGTH": 128, ...}

    Never raises: a missing or malformed file yields an empty map, because a
    tuning artefact must never be able to stop the service from starting.
    """
    target = path or Path(os.environ.get("TTS_BEST_CONFIG") or BEST_CONFIG_PATH)
    try:
        if not target.exists():
            return {}
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("ignoring unreadable %s: %s", target, exc)
        return {}
    if not isinstance(payload, dict):
        logger.warning("ignoring %s: expected a JSON object", target)
        return {}
    values = payload.get("settings", payload)
    if not isinstance(values, dict):
        logger.warning("ignoring %s: 'settings' is not an object", target)
        return {}
    return {str(k): v for k, v in values.items()}


#: Flattened tuned profile; empty when the autotuner has never run here.
best_config: dict[str, Any] = load_best_config()


def _tuned(name: str) -> Any:
    """Value from ``best_config.json``, or ``None`` when absent.

    ``name`` is both the ``TTS_*`` environment variable and the key used in
    ``best_config.json``, which is what lets one argument drive both tiers.
    """
    return best_config.get(name)


def _str(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value is not None and value.strip() != "":
        return value.strip()
    tuned = _tuned(name)
    if tuned is not None and str(tuned).strip() != "":
        return str(tuned).strip()
    return default


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is not None:
        normalised = raw.strip().lower()
        if normalised in _TRUTHY:
            return True
        if normalised in _FALSY:
            return False
        raise ValueError(f"{name} expects a boolean, got {raw!r}")
    tuned = _tuned(name)
    if tuned is not None:
        if isinstance(tuned, bool):
            return tuned
        normalised = str(tuned).strip().lower()
        if normalised in _TRUTHY:
            return True
        if normalised in _FALSY:
            return False
        logger.warning("best_config: %s=%r is not a boolean; using %r", name, tuned, default)
    return default


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is not None and raw.strip() != "":
        return int(raw.strip())
    tuned = _tuned(name)
    if tuned is not None and str(tuned).strip() != "":
        try:
            return int(float(tuned))
        except (TypeError, ValueError):
            logger.warning("best_config: %s=%r is not an int; using %r", name, tuned, default)
    return default


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is not None and raw.strip() != "":
        return float(raw.strip())
    tuned = _tuned(name)
    if tuned is not None and str(tuned).strip() != "":
        try:
            return float(tuned)
        except (TypeError, ValueError):
            logger.warning(
                "best_config: %s=%r is not a float; using %r", name, tuned, default
            )
    return default



# --------------------------------------------------------------------------- #
#  Settings groups
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PathSettings:
    """Filesystem layout shared by the compose mounts and the scripts.

    The host project directory is mounted at ``/app`` inside the container, so
    ``data/`` is the very same directory from both sides.
    """

    project_root: Path = PROJECT_ROOT
    data_dir: Path = DATA_DIR
    references_dir: Path = field(init=False)
    input_dir: Path = field(init=False)
    temp_chunks_dir: Path = field(init=False)
    output_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "references_dir", self.data_dir / "references")
        object.__setattr__(self, "input_dir", self.data_dir / "input")
        object.__setattr__(self, "temp_chunks_dir", self.data_dir / "temp_chunks")
        object.__setattr__(self, "output_dir", self.data_dir / "output")

    # -- conventional file names -------------------------------------------
    @property
    def clean_reference_wav(self) -> Path:
        """Phase-2 output: the denoised, validated reference WAV.

        7 s / 44 patches. The 6 GB budget forces
        ``TTS_MAX_GENERATE_LENGTH=64`` (the smallest ``--optimize`` compile
        bucket), so the reference has to stay under 64 patches. An 11 s clip is
        71 patches and ``_prepare_inputs`` rejects it outright, which leaves
        ~20 patches (3.2 s) of generation room here.
        """
        return self.input_dir / "reference_clean.wav"

    @property
    def clean_reference_txt(self) -> Path:
        """Transcript that matches :attr:`clean_reference_wav` exactly."""
        return self.input_dir / "reference_clean.txt"

    @property
    def benchmark_report(self) -> Path:
        """Phase-5 output: JSON benchmark log."""
        return self.output_dir / "benchmark_report.json"

    @property
    def speakers_dir(self) -> Path:
        """Registered speaker prompts (v3 Task 3.3).

        One ``<id>.wav`` + ``<id>.txt`` per registered voice. Storing them on
        disk rather than in RAM is deliberate: ``dots_tts`` is handed a *path*,
        so an in-memory blob would have to be written to a temp file on every
        request anyway, and a registered voice would vanish on restart. On disk
        it is reloaded by id at zero per-request cost.
        """
        return self.data_dir / "speakers"

    def ensure(self) -> "PathSettings":

        for directory in (
            self.data_dir,
            self.references_dir,
            self.input_dir,
            self.temp_chunks_dir,
            self.output_dir,
            self.speakers_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return self


@dataclass(frozen=True)
class ModelSettings:
    """dots.tts runtime configuration + the VRAM/RTF acceptance targets."""

    model_name_or_path: str = "dots-studio/dots.tts-mf"
    precision: str = "bfloat16"
    optimize: bool = True
    warmup_on_optimize: bool = True

    # ``max_generate_length`` is the maximum **total** audio patch count
    # (reference prompt + generated audio). One patch ~= 160 ms.
    #
    # Measured on an RTX 4050 Laptop (6 GB / 5.64 GiB usable) with the
    # low-VRAM patches enabled and TTS_NUM_STEPS=2:
    #     max_generate_length=128 -> 5.36 GB peak, RTF 0.44   <-- configured here
    #     max_generate_length=96  -> 5.27 GB peak, RTF 1.15 (eager DiT) / OOM when
    #                                 the DiT KV cache is sized for more than
    #                                 2 sampling steps.
    # The DiT KV cache is allocated as (num_steps, layers, heads, tokens, dim),
    # so this budget is coupled to ``default_num_steps``: at 4 steps the cache
    # doubles and 128 patches no longer fits -- drop this to 96 in that case.
    max_generate_length: int = 128
    # ``max_sequence_length`` sizes the LLM KV cache. The checkpoint default is
    # 2048; halving it to 1024 saves ~170 MB, which is part of what makes the
    # 128-patch budget reachable on a 6 GB card.
    max_sequence_length: int = 1024
    vocoder_merge_steps: int = 4
    device: str = "cuda"

    # ---- low-VRAM patches (see app/patches.py) ---------------------------
    # The stock dots.tts inference path needs ~5.8 GB of transient memory on
    # top of the 4.55 GiB of bfloat16 weights, which does not fit a 6 GB card.
    # ``app.patches`` removes a 292 MiB duplicated AdaLN matrix and bounds the
    # BigVGAN vocoder's activation peak. Both are required to hold the peak
    # under ``vram_limit_gb``; disable only for debugging.
    low_vram_patches: bool = True
    vocoder_chunk_patches: int = 12

    # Sampling defaults (spec Phase 3, Task 3.3)
    # 2 is the MeanFlow checkpoint's native step count. It is also what keeps the
    # DiT KV cache (allocated as num_steps * ...) small enough for the
    # 128-patch budget above.
    default_num_steps: int = 2
    default_guidance_scale: float = 1.2
    default_speaker_scale: float = 1.5
    default_seed: int = 42

    language: str = "none"
    normalize_text: bool = False

    # Acceptance thresholds (spec "Technical Constraints")
    vram_limit_gb: float = 5.5
    rtf_target: float = 0.5

    @classmethod
    def from_env(cls) -> "ModelSettings":
        return cls(
            model_name_or_path=_str("TTS_MODEL_NAME", cls.model_name_or_path),
            precision=_str("TTS_PRECISION", cls.precision),
            optimize=_bool("TTS_OPTIMIZE", cls.optimize),
            warmup_on_optimize=_bool(
                "TTS_WARMUP_ON_OPTIMIZE", cls.warmup_on_optimize
            ),
            max_generate_length=_int(
                "TTS_MAX_GENERATE_LENGTH", cls.max_generate_length
            ),
            max_sequence_length=_int(
                "TTS_MAX_SEQUENCE_LENGTH", cls.max_sequence_length
            ),
            vocoder_merge_steps=_int(
                "TTS_VOCODER_MERGE_STEPS", cls.vocoder_merge_steps
            ),
            low_vram_patches=_bool(
                "TTS_LOW_VRAM_PATCHES", cls.low_vram_patches
            ),
            vocoder_chunk_patches=_int(
                "TTS_VOCODER_CHUNK_PATCHES", cls.vocoder_chunk_patches
            ),
            device=_str("TTS_DEVICE", cls.device),
            default_num_steps=_int("TTS_NUM_STEPS", cls.default_num_steps),
            default_guidance_scale=_float(
                "TTS_GUIDANCE_SCALE", cls.default_guidance_scale
            ),
            default_speaker_scale=_float(
                "TTS_SPEAKER_SCALE", cls.default_speaker_scale
            ),
            default_seed=_int("TTS_SEED", cls.default_seed),
            language=_str("TTS_LANGUAGE", cls.language),
            normalize_text=_bool("TTS_NORMALIZE_TEXT", cls.normalize_text),
            vram_limit_gb=_float("TTS_VRAM_LIMIT_GB", cls.vram_limit_gb),
            rtf_target=_float("TTS_RTF_TARGET", cls.rtf_target),
        )


@dataclass(frozen=True)
class APISettings:
    """FastAPI server / HTTP client settings."""

    title: str = "dots.tts Local TTS Service"
    version: str = "2.0.0"
    host: str = "0.0.0.0"
    port: int = 8000
    api_url: str = "http://localhost:8000"
    speech_path: str = "/v1/audio/speech"
    health_path: str = "/health"
    request_timeout_s: float = 900.0
    connect_timeout_s: float = 10.0
    max_upload_mb: int = 64

    @classmethod
    def from_env(cls) -> "APISettings":
        return cls(
            title=_str("TTS_API_TITLE", cls.title),
            version=_str("TTS_API_VERSION", cls.version),
            host=_str("TTS_HOST", cls.host),
            port=_int("TTS_PORT", cls.port),
            api_url=_str("TTS_API_URL", cls.api_url).rstrip("/"),
            request_timeout_s=_float("TTS_REQUEST_TIMEOUT", cls.request_timeout_s),
            connect_timeout_s=_float("TTS_CONNECT_TIMEOUT", cls.connect_timeout_s),
            max_upload_mb=_int("TTS_MAX_UPLOAD_MB", cls.max_upload_mb),
        )


@dataclass(frozen=True)
class ProcessingSettings:
    """Reference-audio cleaning and long-text chunking parameters."""

    # ---- clean_audio.py ---------------------------------------------------
    target_sample_rate: int = 48_000
    noise_prop_decrease: float = 0.85
    stationary_noise: bool = True
    reference_min_seconds: float = 5.0
    reference_max_seconds: float = 15.0
    reference_ideal_min_seconds: float = 10.0
    reference_ideal_max_seconds: float = 12.0
    peak_dbfs: float = -1.0

    # ---- process_txt.py ---------------------------------------------------
    # NOTE: the spec suggested 150-180 words per chunk, which produces ~60-70 s
    # of audio (~400 audio patches). That needs ~10 GB of VRAM and is therefore
    # impossible on a 6 GB RTX 4050. The value below is the VRAM-safe figure for
    # this hardware; `--max-words auto` (the CLI default) derives the real limit
    # from the live server configuration.
    chunk_max_words: int = 35
    chunk_min_words: int = 8
    words_per_second: float = 2.5
    max_retries: int = 3
    retry_backoff_s: float = 2.0
    concat_gap_seconds: float = 0.12

    @classmethod
    def from_env(cls) -> "ProcessingSettings":
        return cls(
            target_sample_rate=_int("TTS_SAMPLE_RATE", cls.target_sample_rate),
            noise_prop_decrease=_float(
                "TTS_NOISE_PROP_DECREASE", cls.noise_prop_decrease
            ),
            stationary_noise=_bool("TTS_STATIONARY_NOISE", cls.stationary_noise),
            reference_min_seconds=_float(
                "TTS_REF_MIN_SECONDS", cls.reference_min_seconds
            ),
            reference_max_seconds=_float(
                "TTS_REF_MAX_SECONDS", cls.reference_max_seconds
            ),
            chunk_max_words=_int("TTS_CHUNK_MAX_WORDS", cls.chunk_max_words),
            words_per_second=_float("TTS_WORDS_PER_SECOND", cls.words_per_second),
            max_retries=_int("TTS_MAX_RETRIES", cls.max_retries),
        )


@dataclass(frozen=True)
class Settings:
    """Aggregate of every settings group."""

    paths: PathSettings
    model: ModelSettings
    api: APISettings
    processing: ProcessingSettings
    log_level: str = "info"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            paths=PathSettings(),
            model=ModelSettings.from_env(),
            api=APISettings.from_env(),
            processing=ProcessingSettings.from_env(),
            log_level=_str("TTS_LOG_LEVEL", "info"),
        )

    def speech_url(self) -> str:
        return f"{self.api.api_url}{self.api.speech_path}"

    def health_url(self) -> str:
        return f"{self.api.api_url}{self.api.health_path}"


settings = Settings.from_env()
