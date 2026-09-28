"""VoxDock — Containerised high-fidelity local TTS service built on ``dots.tts``.

Package layout
--------------
* :mod:`app.config`  – env-driven settings shared by the API and the scripts.
* :mod:`app.utils`   – GPU/audio/buffer helpers (torch + soundfile live here).
* :mod:`app.main`    – FastAPI application exposing ``/health`` and
  ``/v1/audio/speech``.
"""

__title__ = "VoxDock"
__package_name__ = "voxdock-tts"
__version__ = "1.0.0"
__description__ = "Containerised high-fidelity local TTS service built on dots.tts"

# --------------------------------------------------------------------------- #
#  Deployment identity
#
#  Single source of truth for the Docker image, the container and the compose
#  project name. Docker Compose cannot read Python, so docker-compose.yml
#  repeats these as inline ${VAR:-default} fallbacks; scripts/check_identity.py
#  fails if the two ever drift apart.
# --------------------------------------------------------------------------- #
#: Docker image name, built from ``__package_name__``.
IMAGE_NAME = __package_name__
#: Fixed container name, so `docker logs voxdock-tts` and `docker exec` keep
#: working regardless of the compose project name.
CONTAINER_NAME = "voxdock-tts"
#: Compose project name (the namespace for containers, networks and volumes).
COMPOSE_PROJECT = "voxdock"
#: Persistent volume holding the ~4.4 GB of HuggingFace checkpoint weights.
#:
#: Deliberately NOT renamed from ``local_tts_hf_cache``: Docker cannot rename a
#: volume, so changing it would orphan the existing cache and force a full
#: re-download of every weight on the next build.
HF_CACHE_VOLUME = "local_tts_hf_cache"
#: Logger namespace for this project's own log lines.
#:
#: Deliberately *not* ``dots_tts.*``: that is the upstream library's namespace,
#: and reusing it makes our records indistinguishable from the engine's.
LOG_NAMESPACE = "voxdock"

__all__ = [
    "__title__",
    "__package_name__",
    "__version__",
    "__description__",
    "IMAGE_NAME",
    "CONTAINER_NAME",
    "COMPOSE_PROJECT",
    "HF_CACHE_VOLUME",
    "LOG_NAMESPACE",
]