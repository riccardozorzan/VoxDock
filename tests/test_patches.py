"""Startup guard for the low-VRAM patches (todo 4.7).

``app/patches.py`` monkey-patches ``dots_tts`` internals, so every patch is
individually wrapped in a ``try/except`` that logs a warning and returns False
when upstream renames the symbol it needs. That is the right behaviour *when a
patch is optional* - a future ``dots.tts`` release degrades to stock instead of
crashing at import.

On a 6 GB card the patches are not optional: the stock path needs roughly
600 MB more than the GPU has, so a silently-missing patch converts a clear
error message at startup into a confusing OOM minutes into the warmup. These
tests pin the loud failure that replaced the silent one.
"""

from __future__ import annotations

import pytest

from app.patches import (
    REQUIRED_PATCHES,
    PatchFailure,
    assert_patches_applied,
)

ALL_APPLIED = {"compile": True, "fused_adaln": True, "generate_audio": True}


def test_all_applied_is_silent():
    """The happy path must not raise - this runs on every startup."""
    assert_patches_applied(ALL_APPLIED)


def test_a_missing_patch_is_fatal():
    """The whole point: losing a patch must not be a warning."""
    with pytest.raises(PatchFailure) as excinfo:
        assert_patches_applied({**ALL_APPLIED, "fused_adaln": False})

    assert "fused_adaln" in str(excinfo.value)


def test_the_error_explains_the_consequence():
    """A failure message that does not say what breaks is not worth raising."""
    with pytest.raises(PatchFailure) as excinfo:
        assert_patches_applied({**ALL_APPLIED, "generate_audio": False})

    message = str(excinfo.value)
    assert "OOM" in message
    assert "TTS_LOW_VRAM_PATCHES" in message, "must name the escape hatch"


def test_every_missing_patch_is_named():
    """Not just the first - a partial upgrade could lose several at once."""
    with pytest.raises(PatchFailure) as excinfo:
        assert_patches_applied({"compile": True, "fused_adaln": False, "generate_audio": False})

    message = str(excinfo.value)
    assert "fused_adaln" in message
    assert "generate_audio" in message


def test_an_empty_result_is_fatal():
    """Nothing applied at all is the worst case and must not pass silently."""
    with pytest.raises(PatchFailure):
        assert_patches_applied({})


def test_the_required_set_is_the_one_that_matters():
    """Guard against someone narrowing REQUIRED_PATCHES to make this pass."""
    assert set(REQUIRED_PATCHES) == set(ALL_APPLIED)
