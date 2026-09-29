"""Autotuner bookkeeping: the winning profile and the variant table.

The bug these cover is subtle and shipped once: a config can be measured twice
in a single sweep (rejected as the incumbent baseline, then accepted when its
own dimension is reached), and the artefact writer used the *first* match. The
profile it emitted therefore carried a null score and a worst_rtf above the
target - numbers from a rejected measurement attached to a winner.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.autotune import _variant_comparison, write_best_config  # noqa: E402

WINNER = {"TTS_MODEL_NAME": "dots-studio/dots.tts-mf", "TTS_NUM_STEPS": "2"}
ARGS = SimpleNamespace(
    objective="balanced", vram_limit_gb=5.5, vram_margin_gb=0.02, rtf_target=0.5
)
REFERENCE = SimpleNamespace(name="ref_l", label="3 sentences", seconds=11.45, patches=72)


def make_record(config, *, feasible, worst_rtf, score, peak=5560.0):
    return SimpleNamespace(
        config=dict(config),
        feasible=feasible,
        rejected=None if feasible else "worst-profile RTF 0.52 exceeds the target 0.5",
        worst_rtf=worst_rtf,
        mean_rtf=worst_rtf,
        peak_vram_mb=peak,
        usable_audio_seconds=16.48,
        score=score,
    )


def test_feasible_record_wins_over_an_earlier_rejected_one(tmp_path):
    """The exact shape that produced the bad artefact.

    The incumbent is measured first and rejected; the same config is then
    measured again and passes. The emitted ``measured`` block must come from the
    feasible pass.
    """
    rejected = make_record(WINNER, feasible=False, worst_rtf=None, score=None)
    accepted = make_record(WINNER, feasible=True, worst_rtf=0.4804, score=0.4141)
    evaluator = SimpleNamespace(records=[rejected, accepted])

    out = tmp_path / "best_config.json"
    write_best_config(out, WINNER, evaluator, ARGS, REFERENCE)

    measured = json.loads(out.read_text(encoding="utf-8"))["measured"]
    assert measured["worst_rtf"] == pytest.approx(0.4804)
    assert measured["score"] == pytest.approx(0.4141)
    assert measured["worst_rtf"] < ARGS.rtf_target, (
        "a winner must never carry an RTF above the target"
    )


def test_variant_table_marks_the_winner_even_when_measured_twice():
    """``is_winner`` came from setdefault, so it inherited the first record."""
    rejected = make_record(WINNER, feasible=False, worst_rtf=None, score=None)
    accepted = make_record(WINNER, feasible=True, worst_rtf=0.4804, score=0.4141)
    evaluator = SimpleNamespace(records=[rejected, accepted])

    rows = _variant_comparison(evaluator, WINNER)

    assert len(rows) == 1
    assert rows[0]["is_winner"] is True
    assert rows[0]["feasible"] == 1, "the rejected pass must not count as feasible"
    assert rows[0]["attempts"] == 2


def test_a_config_measured_only_once_still_records(tmp_path):
    """The single-measurement case must keep working, not just the doubled one."""
    evaluator = SimpleNamespace(
        records=[make_record(WINNER, feasible=True, worst_rtf=0.42, score=0.30)]
    )

    out = tmp_path / "best_config.json"
    write_best_config(out, WINNER, evaluator, ARGS, REFERENCE)

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["measured"]["worst_rtf"] == 0.42


def test_values_are_casted_to_real_types(tmp_path):
    """The sweep works in strings; the profile is read by the service, so cast."""
    evaluator = SimpleNamespace(
        records=[make_record(WINNER, feasible=True, worst_rtf=0.4, score=0.3)]
    )

    out = tmp_path / "best_config.json"
    write_best_config(out, WINNER, evaluator, ARGS, REFERENCE)

    settings = json.loads(out.read_text(encoding="utf-8"))["settings"]
    assert settings["TTS_NUM_STEPS"] == 2
    assert isinstance(settings["TTS_NUM_STEPS"], int)
