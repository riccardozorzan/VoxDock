"""Shared fixtures for the VoxDock test suite.

The tests here are deliberately **host-side**: no torch, no GPU, no running
server. ``app.config`` is torch-free by design and the registry and chunking
helpers are plain Python, so the whole suite runs in well under a second on a
laptop without the container even being up.

One subtlety shapes the fixtures below: ``app.config`` snapshots the tuned
profile at *import* time into a module-level ``best_config`` dict, and
``PROJECT_ROOT``/``DATA_DIR`` are likewise resolved once at import. So a test
cannot just ``monkeypatch.setenv("TTS_BEST_CONFIG", ...)`` and expect tier 2 to
notice - the module has to be reloaded for the new path to be read. The
``reload_config`` fixture does exactly that and is what makes the three-tier
chain testable at all.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    """Strip this machine's TTS_* variables and point the profile at nothing.

    Without this, a developer's real ``.env`` (exported into the shell) or a
    genuine ``data/output/best_config.json`` on this machine would make the
    resolution tests pass or fail depending on who ran them.
    """
    monkeypatch.setenv("TTS_BEST_CONFIG", str(tmp_path / "absent.json"))
    for key in list(os.environ):
        if key.startswith("TTS_"):
            monkeypatch.delenv(key, raising=False)
    return tmp_path


@pytest.fixture
def reload_config(clean_env):
    """Return a callable that re-imports ``app.config`` on demand.

    ``app.config`` snapshots the tuned profile at *import* time, so writing a
    ``best_config.json`` and setting ``TTS_BEST_CONFIG`` is not enough - the
    module must be re-read. A fixture cannot do that on the test's behalf,
    because fixtures run *before* the test body writes the file, so this yields
    a function the test calls itself once its setup is done::

        monkeypatch.setenv("TTS_BEST_CONFIG", str(profile))
        config = reload_config()
        assert config.ModelSettings.from_env().default_num_steps == 4

    The module is reloaded again on teardown so the fake profile cannot leak
    into later tests.
    """
    module = importlib.import_module("app.config")

    def _reload():
        return importlib.reload(module)

    yield _reload
    importlib.reload(module)


@pytest.fixture
def registry(tmp_path):
    """A ``SpeakerRegistry`` rooted in a throwaway directory."""
    from app.speakers import SpeakerRegistry

    return SpeakerRegistry(tmp_path / "speakers")
