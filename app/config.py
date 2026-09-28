"""Hardware & model configuration for the containerised dots.tts service.

All values are environment driven so that the exact same code can be reused by
the FastAPI server, the pre-processing scripts and the benchmark client.

This module is intentionally **torch-free**: every helper here is plain Python so
that the client-side scripts can import it even on a host machine that has none
of the heavy ML dependencies installed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "PathSettings",
    "ModelSettings",
    "APISettings",
    "ProcessingSettings",
    "Settings",
    "settings",
]

# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(
    os.environ.get("TTS_PROJECT_ROOT") or Path(__file__).resolve().parent.parent
).resolve()

DATA_DIR = Path(os.environ.get("TTS_DATA_DIR") or (PROJECT_ROOT / "data")).resolve()

_TRUTHY = {"1", "true", "t", "yes", "y", "on"}
_FALSY = {"0", "false", "f", "no", "n", "off", ""}


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalised = raw.strip().lower()
    if normalised in _TRUTHY:
        return True
    if normalised in _FALSY:
        return False
    raise ValueError(f"{name} expects a boolean, got {raw!r}")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw.strip())


def _env_optional_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    return int(raw.strip())


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw.strip())


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

    def ensure(self) -> "PathSettings":
        """Create every directory the pipeline needs. Idempotent."""
        for directory in (
            self.data_dir,
            self.references_dir,
            self.input_dir,
            self.temp_chunks_dir,
            self.output_dir,
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
            model_name_or_path=_env_str("TTS_MODEL_NAME", cls.model_name_or_path),
            precision=_env_str("TTS_PRECISION", cls.precision),
            optimize=_env_bool("TTS_OPTIMIZE", cls.optimize),
            warmup_on_optimize=_env_bool(
                "TTS_WARMUP_ON_OPTIMIZE", cls.warmup_on_optimize
            ),
            max_generate_length=_env_int(
                "TTS_MAX_GENERATE_LENGTH", cls.max_generate_length
            ),
            max_sequence_length=_env_int(
                "TTS_MAX_SEQUENCE_LENGTH", cls.max_sequence_length
            ),
            vocoder_merge_steps=_env_int(
                "TTS_VOCODER_MERGE_STEPS", cls.vocoder_merge_steps
            ),
            low_vram_patches=_env_bool(
                "TTS_LOW_VRAM_PATCHES", cls.low_vram_patches
            ),
            vocoder_chunk_patches=_env_int(
                "TTS_VOCODER_CHUNK_PATCHES", cls.vocoder_chunk_patches
            ),
            device=_env_str("TTS_DEVICE", cls.device),
            default_num_steps=_env_int("TTS_NUM_STEPS", cls.default_num_steps),
            default_guidance_scale=_env_float(
                "TTS_GUIDANCE_SCALE", cls.default_guidance_scale
            ),
            default_speaker_scale=_env_float(
                "TTS_SPEAKER_SCALE", cls.default_speaker_scale
            ),
            default_seed=_env_int("TTS_SEED", cls.default_seed),
            language=_env_str("TTS_LANGUAGE", cls.language),
            normalize_text=_env_bool("TTS_NORMALIZE_TEXT", cls.normalize_text),
            vram_limit_gb=_env_float("TTS_VRAM_LIMIT_GB", cls.vram_limit_gb),
            rtf_target=_env_float("TTS_RTF_TARGET", cls.rtf_target),
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
            title=_env_str("TTS_API_TITLE", cls.title),
            version=_env_str("TTS_API_VERSION", cls.version),
            host=_env_str("TTS_HOST", cls.host),
            port=_env_int("TTS_PORT", cls.port),
            api_url=_env_str("TTS_API_URL", cls.api_url).rstrip("/"),
            request_timeout_s=_env_float("TTS_REQUEST_TIMEOUT", cls.request_timeout_s),
            connect_timeout_s=_env_float("TTS_CONNECT_TIMEOUT", cls.connect_timeout_s),
            max_upload_mb=_env_int("TTS_MAX_UPLOAD_MB", cls.max_upload_mb),
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
            target_sample_rate=_env_int("TTS_SAMPLE_RATE", cls.target_sample_rate),
            noise_prop_decrease=_env_float(
                "TTS_NOISE_PROP_DECREASE", cls.noise_prop_decrease
            ),
            stationary_noise=_env_bool("TTS_STATIONARY_NOISE", cls.stationary_noise),
            reference_min_seconds=_env_float(
                "TTS_REF_MIN_SECONDS", cls.reference_min_seconds
            ),
            reference_max_seconds=_env_float(
                "TTS_REF_MAX_SECONDS", cls.reference_max_seconds
            ),
            chunk_max_words=_env_int("TTS_CHUNK_MAX_WORDS", cls.chunk_max_words),
            words_per_second=_env_float("TTS_WORDS_PER_SECOND", cls.words_per_second),
            max_retries=_env_int("TTS_MAX_RETRIES", cls.max_retries),
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
            log_level=_env_str("TTS_LOG_LEVEL", "info"),
        )

    def speech_url(self) -> str:
        return f"{self.api.api_url}{self.api.speech_path}"

    def health_url(self) -> str:
        return f"{self.api.api_url}{self.api.health_path}"


settings = Settings.from_env()
