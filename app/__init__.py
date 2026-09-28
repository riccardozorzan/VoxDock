"""Containerised high-fidelity local TTS service built on ``dots.tts``.

Package layout
--------------
* :mod:`app.config`  – env-driven settings shared by the API and the scripts.
* :mod:`app.utils`   – GPU/audio/buffer helpers (torch + soundfile live here).
* :mod:`app.main`    – FastAPI application exposing ``/health`` and
  ``/v1/audio/speech``.
"""

__all__ = ["__version__"]

__version__ = "1.0.0"
