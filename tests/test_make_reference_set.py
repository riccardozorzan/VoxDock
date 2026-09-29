"""``make_reference_set.py`` must produce a manifest the autotuner accepts.

The interesting assertion is the round trip: the script writes JSON, and
``scripts/autotune.py::load_reference_set`` - the real consumer, with its own
path resolution and duration probing - reads it back. A generator that produced
almost-right JSON would pass any test that only checked its own output.

The validation cases matter for the same reason the autotuner's own manifest is
committed: a bad entry is not a cosmetic problem, it is a profile measured
against a reference the server will later reject.
"""

from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.autotune import load_reference_set  # noqa: E402
from scripts.make_reference_set import (  # noqa: E402
    MIN_GENERATION_PATCHES,
    PATCH_SECONDS,
    build_entries,
)

SAMPLE_RATE = 48_000
TRANSCRIPT = "Questa e una frase di prova per il set di riferimento."
BUDGET = 128


def write_wav(path: Path, seconds: float) -> Path:
    """A silent 48 kHz mono PCM clip - the duration is all that matters here."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(b"\x00\x00" * int(SAMPLE_RATE * seconds))
    return path


def make_ref(tmp_path: Path, seconds: float, *, name: str = "ref", text: str | None = TRANSCRIPT):
    audio = write_wav(tmp_path / f"{name}.wav", seconds)
    transcript = tmp_path / f"{name}.txt"
    if text is not None:
        transcript.write_text(text, encoding="utf-8")
    return [(name, "a label", str(audio), str(transcript))]


def build(refs, *, min_seconds: float = 0.0, max_seconds: float = 30.0, budget: int = BUDGET):
    return build_entries(
        refs, min_seconds=min_seconds, max_seconds=max_seconds, budget_patches=budget
    )


# --------------------------------------------------------------------------- #
#  The happy path
# --------------------------------------------------------------------------- #
def test_it_reports_seconds_and_patches(tmp_path):
    entries, warnings = build(make_ref(tmp_path, 4.0))

    assert warnings == []
    assert entries[0]["seconds"] == pytest.approx(4.0, abs=0.01)
    assert entries[0]["patches"] == 25  # 4.0 / 0.16
    assert entries[0]["name"] == "ref"


def test_patches_use_the_160ms_patch(tmp_path):
    """The AudioVAE patch size is not a guess; it is the memory budget unit."""
    entries, _ = build(make_ref(tmp_path, 1.6))

    assert entries[0]["patches"] == round(1.6 / PATCH_SECONDS)


def test_the_autotuner_accepts_what_this_script_writes(tmp_path):
    """Round trip through the real consumer, not just our own shape check."""
    refs = make_ref(tmp_path, 4.0, name="ref_s") + make_ref(tmp_path, 8.0, name="ref_m")
    entries, _ = build(refs)

    manifest = tmp_path / "reference_set.json"
    manifest.write_text(
        json.dumps(
            {"references": [{k: e[k] for k in ("name", "label", "audio", "text")} for e in entries]}
        ),
        encoding="utf-8",
    )

    loaded = load_reference_set(manifest)

    assert [r.name for r in loaded] == ["ref_s", "ref_m"]
    assert loaded[0].audio == tmp_path / "ref_s.wav"
    assert loaded[0].seconds == pytest.approx(4.0, abs=0.01)


# --------------------------------------------------------------------------- #
#  Validation
# --------------------------------------------------------------------------- #
def test_a_missing_clip_is_fatal(tmp_path):
    with pytest.raises(SystemExit, match="audio not found"):
        build([("ghost", "l", str(tmp_path / "ghost.wav"), str(tmp_path / "g.txt"))])


def test_a_missing_transcript_is_fatal(tmp_path):
    audio = write_wav(tmp_path / "ref.wav", 4.0)

    with pytest.raises(SystemExit, match="transcript not found"):
        build([("ref", "l", str(audio), str(tmp_path / "absent.txt"))])


def test_an_empty_transcript_is_fatal(tmp_path):
    refs = make_ref(tmp_path, 4.0, text="   \n")

    with pytest.raises(SystemExit, match="transcript is empty"):
        build(refs)


def test_a_clip_past_the_ceiling_is_fatal(tmp_path):
    with pytest.raises(SystemExit, match="exceeds the 15s ceiling"):
        build(make_ref(tmp_path, 15.5), max_seconds=15.0)


def test_a_short_slice_warns_instead_of_failing(tmp_path):
    """The committed ref_s is 4.05 s, and it is a perfectly valid reference.

    TTS_REF_MIN_SECONDS is clean_audio's intake rule for a *raw recording*; a
    reference for the sweep is usually a slice cut out of one take, so being under
    it is worth flagging, never fatal. An earlier version of this script enforced
    the window on both sides and rejected the repository's own manifest.
    """
    entries, warnings = build(make_ref(tmp_path, 4.05), min_seconds=5.0, max_seconds=15.0)

    assert entries[0]["patches"] == 26  # exactly the committed ref_s
    assert len(warnings) == 1
    assert "under the 5s" in warnings[0]


def test_duplicate_names_are_fatal(tmp_path):
    refs = make_ref(tmp_path, 4.0) + make_ref(tmp_path, 6.0)

    with pytest.raises(SystemExit, match="duplicate name"):
        build(refs)


# --------------------------------------------------------------------------- #
#  The patch budget
#
#  15 s = 94 patches, and the shipped budget is 128, so the worst-case reference
#  leaves 34 patches (~5.4 s). The warning therefore does NOT fire on the default
#  configuration - it exists for a tighter budget. These tests pin that arithmetic
#  so the constant is not mistaken for a threshold that always applies.
# --------------------------------------------------------------------------- #
def test_the_shipped_defaults_leave_room_and_do_not_warn(tmp_path):
    entries, warnings = build(make_ref(tmp_path, 15.0))

    assert entries[0]["patches"] == 94
    assert BUDGET - entries[0]["patches"] == 34
    assert warnings == []


def test_a_tight_budget_warns(tmp_path):
    """At 100 patches a 15 s reference leaves 6 - under one usable utterance."""
    _, warnings = build(make_ref(tmp_path, 15.0), budget=100)

    assert len(warnings) == 1
    assert "leaving 6" in warnings[0]


def test_a_reference_that_fills_the_budget_is_fatal(tmp_path):
    """The server would answer every request with 400, so refuse the manifest."""
    with pytest.raises(SystemExit, match="fill the whole"):
        build(make_ref(tmp_path, 15.0), budget=94)


def test_the_floor_is_32_patches(tmp_path):
    """5.1 s: the shortest utterance worth synthesising."""
    assert MIN_GENERATION_PATCHES == 32
    assert MIN_GENERATION_PATCHES * PATCH_SECONDS == pytest.approx(5.12, abs=0.01)
