"""The three-tier settings chain: env -> best_config.json -> defaults (v3 Task 3.1).

These are the tests that protect the autotuner. If tier 2 silently stopped
working, ``autotune.py`` would write a winner that the server never applied, and
nothing in the running service would complain - which is exactly the failure
mode that makes ``/health``'s ``config_source`` field worth asserting on.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from app.config import ModelSettings, ProcessingSettings, load_best_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
#  Tier 3: built-in defaults
# --------------------------------------------------------------------------- #
def test_defaults_without_env_or_profile(clean_env):
    """No env, no profile -> the values verified on the 6 GB RTX 4050."""
    model = ModelSettings.from_env()
    assert model.max_generate_length == 128
    assert model.default_num_steps == 2, "the MeanFlow default is 2 steps"
    assert model.max_sequence_length == 1024
    assert model.low_vram_patches is True, "the patches are not optional on 6 GB"
    assert model.precision == "bfloat16"
    assert model.vram_limit_gb == 5.5
    assert model.rtf_target == 0.5

    processing = ProcessingSettings.from_env()
    assert processing.chunk_max_words == 35
    assert processing.concat_gap_seconds == 0.30
    assert processing.target_sample_rate == 48_000


# --------------------------------------------------------------------------- #
#  Tier 1: environment wins
# --------------------------------------------------------------------------- #
def test_env_beats_everything(clean_env, monkeypatch, tmp_path, reload_config):
    """An operator override outranks a tuned profile."""
    profile = tmp_path / "best_config.json"
    profile.write_text(
        json.dumps({"settings": {"TTS_MAX_GENERATE_LENGTH": 64}}), encoding="utf-8"
    )
    monkeypatch.setenv("TTS_BEST_CONFIG", str(profile))
    monkeypatch.setenv("TTS_MAX_GENERATE_LENGTH", "256")

    config = reload_config()
    assert config.ModelSettings.from_env().max_generate_length == 256


def test_numeric_profile_values_are_coerced(
    clean_env, monkeypatch, tmp_path, reload_config
):
    """The autotuner writes strings; a float-y value must still land as an int."""
    profile = tmp_path / "best_config.json"
    profile.write_text(
        json.dumps({"settings": {"TTS_VOCODER_CHUNK_PATCHES": "20.0"}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("TTS_BEST_CONFIG", str(profile))

    config = reload_config()
    model = config.ModelSettings.from_env()
    assert model.vocoder_chunk_patches == 20


# --------------------------------------------------------------------------- #
#  Tier 2: the tuned profile
# --------------------------------------------------------------------------- #
def test_profile_beats_defaults(clean_env, monkeypatch, tmp_path, reload_config):
    profile = tmp_path / "best_config.json"
    profile.write_text(
        json.dumps(
            {"settings": {"TTS_NUM_STEPS": 4, "TTS_VOCODER_CHUNK_PATCHES": 20}}
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("TTS_BEST_CONFIG", str(profile))

    config = reload_config()
    model = config.ModelSettings.from_env()
    assert model.default_num_steps == 4
    assert model.vocoder_chunk_patches == 20


def test_flat_profile_shape_is_accepted(clean_env, tmp_path):
    """autotune.py can emit either ``{"settings": {...}}`` or a bare map."""
    profile = tmp_path / "flat.json"
    profile.write_text(json.dumps({"TTS_NUM_STEPS": 1}), encoding="utf-8")

    assert load_best_config(profile) == {"TTS_NUM_STEPS": 1}




# --------------------------------------------------------------------------- #
#  A malformed artefact must never stop the service
# --------------------------------------------------------------------------- #
def test_malformed_profile_falls_back_silently(
    clean_env, monkeypatch, tmp_path, reload_config
):
    """Bad JSON, a non-object payload, and a non-dict 'settings' are all non-fatal.

    A tuning artefact is an optimisation; it must not be able to prevent startup.
    """
    broken = tmp_path / "broken.json"
    broken.write_text("{not json at all", encoding="utf-8")
    monkeypatch.setenv("TTS_BEST_CONFIG", str(broken))
    assert reload_config().load_best_config() == {}
    config = reload_config()
    assert config.ModelSettings.from_env().max_generate_length == 128

    array = tmp_path / "array.json"
    array.write_text("[1, 2, 3]", encoding="utf-8")
    monkeypatch.setenv("TTS_BEST_CONFIG", str(array))
    assert reload_config().load_best_config() == {}

    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps({"settings": "nope"}), encoding="utf-8")
    monkeypatch.setenv("TTS_BEST_CONFIG", str(wrong))
    assert reload_config().load_best_config() == {}
    config = reload_config()
    assert config.ModelSettings.from_env().max_generate_length == 128


def test_missing_profile_is_not_an_error(clean_env, monkeypatch, tmp_path):
    monkeypatch.setenv("TTS_BEST_CONFIG", str(tmp_path / "never_written.json"))
    assert load_best_config() == {}


def test_wrongly_typed_profile_value_uses_default(
    clean_env, monkeypatch, tmp_path, reload_config
):
    """A non-numeric int in the profile warns and keeps the default."""
    profile = tmp_path / "best_config.json"
    profile.write_text(
        json.dumps({"settings": {"TTS_VOCODER_CHUNK_PATCHES": "twelve"}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("TTS_BEST_CONFIG", str(profile))

    config = reload_config()
    assert config.ModelSettings.from_env().vocoder_chunk_patches == 12


def test_blank_env_value_falls_through(clean_env, monkeypatch):
    """An empty env var is treated as unset, not as a malformed value."""
    monkeypatch.setenv("TTS_MAX_GENERATE_LENGTH", "   ")
    assert ModelSettings.from_env().max_generate_length == 128


# --------------------------------------------------------------------------- #
#  Boolean parsing
# --------------------------------------------------------------------------- #
def test_truthy_spellings(clean_env, monkeypatch):
    for raw in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv("TTS_OPTIMIZE", raw)
        assert ModelSettings.from_env().optimize is True, raw


def test_falsy_spellings(clean_env, monkeypatch):
    for raw in ("0", "false", "No", "off"):
        monkeypatch.setenv("TTS_OPTIMIZE", raw)
        assert ModelSettings.from_env().optimize is False, raw


def test_nonsense_boolean_is_rejected_loudly(clean_env, monkeypatch):
    """Unlike a bad profile value, a bad *env* value is an operator error.

    Silently defaulting would hide a typo in `.env`, so this raises instead.
    """
    monkeypatch.setenv("TTS_OPTIMIZE", "maybe")
    with pytest.raises(ValueError, match="expects a boolean"):
        ModelSettings.from_env()


# --------------------------------------------------------------------------- #
#  The checkpoint is pinned
#
#  "Pin dots.tts==0.3.1" was in the README for the Python package; the weights
#  needed the same treatment, because the verified configuration (2 sampling
#  steps, a 128-patch budget, 5457 MB peak) describes one specific revision of
#  the checkpoint. Track a branch and the numbers quietly stop meaning anything.
# --------------------------------------------------------------------------- #
def test_the_model_revision_is_pinned_to_a_commit():
    from app.config import ModelSettings

    revision = ModelSettings().model_revision

    assert revision == "c28105adc8228143392b4e346994ff613ee48a06", (
        "the pinned revision changed. If that was deliberate, every measured "
        "number in the README needs re-taking: scripts/benchmark_test.py, then "
        "the steps A/B in data/output/steps_ab.json."
    )


def test_the_pinned_revision_is_a_sha_and_not_a_branch():
    from app.config import ModelSettings

    revision = ModelSettings().model_revision

    assert len(revision) == 40, "a branch name is not a pin"
    assert all(c in "0123456789abcdef" for c in revision)


def test_an_operator_can_still_track_main_deliberately(clean_env, monkeypatch):
    from app.config import ModelSettings

    monkeypatch.setenv("TTS_MODEL_REVISION", "main")

    assert ModelSettings.from_env().model_revision == "main"


def test_the_revision_survives_the_environment_and_compose_defaults_agree():
    """A pin in app/config.py that compose overrides is not a pin."""
    from app.config import ModelSettings

    compose = (PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    match = re.search(
        r"TTS_MODEL_REVISION:\s*\$\{TTS_MODEL_REVISION:-([0-9a-f]{40})\}", compose
    )

    assert match is not None, "compose does not forward TTS_MODEL_REVISION"
    assert match.group(1) == ModelSettings().model_revision
