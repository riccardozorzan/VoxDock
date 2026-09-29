"""Speaker registry: CRUD, id validation, and path-traversal defence (v3 Task 3.5).

The traversal tests are the important ones. ``SpeakerRegistry`` builds on-disk
paths as ``root / f"{speaker_id}.wav"``, so an id containing ``../`` would
normally escape the speakers directory. ``register()`` validates, but ``get()``
and ``delete()`` are reachable directly from the HTTP routes with a
client-supplied id, so they are covered explicitly here.
"""

from __future__ import annotations

import json

import pytest

WAV = b"RIFF----WAVEfmt fake payload for registry tests"
TEXT = "Questa e una frase di prova per la registrazione della voce."


def register(registry, speaker_id=None, text=TEXT):
    return registry.register(
        speaker_id=speaker_id,
        audio_bytes=WAV,
        audio_filename="reference.wav",
        reference_text=text,
    )


# --------------------------------------------------------------------------- #
#  CRUD
# --------------------------------------------------------------------------- #
def test_register_persists_audio_text_and_metadata(registry):
    record = register(registry, "riccardo")

    assert record.id == "riccardo"
    assert record.audio_bytes == len(WAV)
    assert record.requests == 0
    # All three artefacts exist on disk.
    assert registry.root.joinpath("riccardo.wav").read_bytes() == WAV
    assert registry.root.joinpath("riccardo.txt").read_text(encoding="utf-8") == TEXT
    meta = json.loads(registry.root.joinpath("riccardo.json").read_text("utf-8"))
    assert meta["id"] == "riccardo"
    assert meta["audio_filename"] == "reference.wav"


def test_no_staging_file_is_left_behind(registry):
    """The register-then-rename dance must not leave a .wav.tmp on failure paths."""
    register(registry, "riccardo")
    assert not registry.root.joinpath("riccardo.wav.tmp").exists()


def test_get_returns_the_record(registry):
    register(registry, "riccardo")
    record = registry.get("riccardo")

    assert record is not None
    assert record.reference_text == TEXT
    assert record.audio_path.endswith("riccardo.wav")


def test_get_unknown_returns_none(registry):
    assert registry.get("nobody") is None


def test_get_returns_none_when_audio_vanished(registry, tmp_path):
    """A clip deleted behind our back must not resolve as usable."""
    register(registry, "riccardo")
    registry.root.joinpath("riccardo.wav").unlink()

    assert registry.get("riccardo") is None


def test_get_returns_none_on_corrupt_metadata(registry):
    """Truncated JSON must not raise out of a request path."""
    register(registry, "riccardo")
    registry.root.joinpath("riccardo.json").write_text("{truncated", encoding="utf-8")

    assert registry.get("riccardo") is None


def test_list_and_delete(registry):
    register(registry, "alpha")
    register(registry, "beta")

    assert [r.id for r in registry.list()] == ["alpha", "beta"]

    assert registry.delete("alpha") is True
    assert [r.id for r in registry.list()] == ["beta"]
    assert not registry.root.joinpath("alpha.wav").exists()
    assert not registry.root.joinpath("alpha.json").exists()


def test_delete_unknown_returns_false(registry):
    assert registry.delete("ghost") is False


def test_clear_removes_everything(registry):
    register(registry, "alpha")
    register(registry, "beta")

    assert registry.clear() == 2
    assert registry.list() == []
    # The directory is usable again afterwards.
    assert registry.root.is_dir()


def test_overwrite_is_refused_by_default(registry):
    register(registry, "riccardo", text="prima")

    with pytest.raises(ValueError):
        register(registry, "riccardo", text="dopo")


def test_overwrite_when_explicitly_allowed(registry):
    register(registry, "riccardo", text="prima")
    record = registry.register(
        speaker_id="riccardo",
        audio_bytes=WAV,
        audio_filename="reference.wav",
        reference_text="dopo",
        overwrite=True,
    )

    assert record.reference_text == "dopo"


def test_auto_id_skips_taken_names(registry):
    assert register(registry).id == "speaker-1"
    assert register(registry).id == "speaker-2"
    assert register(registry).id == "speaker-3"


def test_count_request_increments_the_reuse_counter(registry):
    """This counter is the visible evidence that prompt reuse is working."""
    register(registry, "riccardo")

    for expected in (1, 2, 3):
        registry.count_request("riccardo")
        assert registry.get("riccardo").requests == expected


def test_count_request_on_unknown_is_a_noop(registry):
    registry.count_request("ghost")  # must not raise


# --------------------------------------------------------------------------- #
#  Id validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "candidate",
    ["riccardo", "a", "A1", "my.voice", "my_voice", "my-voice", "x" * 64],
)
def test_valid_ids_are_accepted(candidate):
    from app.speakers import SpeakerRegistry

    assert SpeakerRegistry.validate_id(candidate) == candidate


@pytest.mark.parametrize(
    "candidate",
    [
        "",
        "   ",
        "../evil",
        "../../etc/passwd",
        "a/b",
        "a\\b",
        "-leading-dash",
        ".hidden",
        "x" * 65,          # over the 64-character limit
        "with space",
        "null\x00byte",
        "emoji-\N{SNOWMAN}",
    ],
)
def test_invalid_ids_are_rejected(candidate):
    from app.speakers import SpeakerRegistry

    with pytest.raises(ValueError):
        SpeakerRegistry.validate_id(candidate)


def test_register_rejects_a_traversal_id(registry, tmp_path):
    victim = tmp_path / "victim.txt"
    victim.write_text("secret", encoding="utf-8")

    with pytest.raises(ValueError):
        register(registry, "../victim")

    assert victim.read_text(encoding="utf-8") == "secret"


# --------------------------------------------------------------------------- #
#  Path traversal defence on get() / delete()
#
#  These two do not call validate_id(); they are reached straight from the HTTP
#  routes with a client-supplied id. They are currently safe only because both
#  require a `<id>.json` sidecar to exist before acting, and register() is the
#  only thing that can create one. These tests pin that behaviour so a future
#  refactor cannot quietly drop the sidecar precondition.
# --------------------------------------------------------------------------- #
TRAVERSAL_IDS = [
    "../evil",
    "../../etc/passwd",
    "..%2f..%2fetc%2fpasswd",
    "./../outside",
]


@pytest.mark.parametrize("bad_id", TRAVERSAL_IDS)
def test_get_refuses_an_audio_path_outside_the_root(registry, tmp_path, bad_id):
    """A planted sidecar must not make the registry hand back an outside path.

    ``audio_path`` from the JSON is what ``POST /v1/audio/speech`` reads as the
    synthesis prompt (app/main.py), so an unvalidated value would let a writable
    host bind mount redirect the engine at an arbitrary file.
    """
    register(registry, "riccardo")
    planted = tmp_path / f"{bad_id.replace('/', '_')}.json"
    planted.write_text(
        json.dumps({"id": bad_id, "audio_path": "/etc/passwd"}), encoding="utf-8"
    )

    record = registry.get(bad_id)
    assert record is None, f"{bad_id} resolved to {getattr(record, 'audio_path', None)}"


@pytest.mark.parametrize("bad_id", TRAVERSAL_IDS)
def test_delete_cannot_remove_a_file_outside_the_root(registry, tmp_path, bad_id):
    victim = tmp_path / "victim.txt"
    victim.write_text("secret", encoding="utf-8")
    register(registry, "riccardo")

    registry.delete(bad_id)  # the return value is False; the point is the file

    assert victim.exists(), "delete() escaped the speakers directory"


# --------------------------------------------------------------------------- #
#  Consent metadata
#
#  A voice clone is a likeness, not a file, so "may we use this recording?" has
#  to be answerable after the fact. These tests pin the distinction the field
#  exists for: unknown is not the same as refused.
# --------------------------------------------------------------------------- #
def register_with_consent(registry, speaker_id="voice", **consent):
    return registry.register(
        speaker_id=speaker_id,
        audio_bytes=WAV,
        audio_filename="reference.wav",
        reference_text=TEXT,
        **consent,
    )


def test_consent_is_acknowledged_and_timestamped(registry):
    record = register_with_consent(registry, consent_acknowledged=True)

    assert record.consent_acknowledged is True
    assert record.consent_status == "granted"
    assert record.consent_at, "the moment it was acknowledged must be recorded"


def test_no_timestamp_is_stamped_without_an_acknowledgement(registry):
    """A timestamp next to ``false`` would imply consent that was not given."""
    record = register_with_consent(registry, consent_acknowledged=False)

    assert record.consent_at == ""
    assert record.consent_status == "not_acknowledged"


def test_an_omitted_acknowledgement_is_unknown_not_false(registry):
    record = register_with_consent(registry)

    assert record.consent_acknowledged is None
    assert record.consent_status == "unknown"


def test_owner_and_basis_are_stored(registry):
    record = register_with_consent(
        registry, consent_acknowledged=True,
        consent_owner="Ada Lovelace", consent_basis="signed release",
    )

    assert record.consent_owner == "Ada Lovelace"
    assert record.consent_basis == "signed release"


def test_consent_is_trimmed(registry):
    record = register_with_consent(registry, consent_owner="  Ada  ")

    assert record.consent_owner == "Ada"


def test_consent_survives_a_reload(registry):
    register_with_consent(registry, consent_acknowledged=True, consent_owner="Grace")

    assert registry.get("voice").consent_owner == "Grace"


def test_a_sidecar_without_consent_fields_still_loads(registry):
    """Upgrading must not orphan voices registered before this existed."""
    register_with_consent(registry, consent_acknowledged=True)
    meta = registry.root / "voice.json"
    payload = json.loads(meta.read_text(encoding="utf-8"))
    for field in ("consent_acknowledged", "consent_at", "consent_owner", "consent_basis"):
        payload.pop(field, None)
    meta.write_text(json.dumps(payload), encoding="utf-8")

    record = registry.get("voice")

    assert record is not None, "a pre-consent sidecar must still resolve"
    assert record.consent_acknowledged is None
    assert record.consent_status == "unknown"


def test_a_string_boolean_in_a_sidecar_is_read(registry):
    """The sidecar is a file in a bind mount, so it is not fully trusted."""
    register_with_consent(registry)
    meta = registry.root / "voice.json"
    payload = json.loads(meta.read_text(encoding="utf-8"))
    payload["consent_acknowledged"] = "true"
    meta.write_text(json.dumps(payload), encoding="utf-8")

    assert registry.get("voice").consent_acknowledged is True


def test_a_nonsense_boolean_reads_as_unknown(registry):
    """Never coerce junk into consent."""
    register_with_consent(registry)
    meta = registry.root / "voice.json"
    payload = json.loads(meta.read_text(encoding="utf-8"))
    payload["consent_acknowledged"] = "maybe"
    meta.write_text(json.dumps(payload), encoding="utf-8")

    assert registry.get("voice").consent_acknowledged is None


def test_the_api_form_nests_consent_and_the_stored_form_stays_flat(registry):
    """Two shapes, on purpose: one for disk, one for clients."""
    register_with_consent(registry, consent_acknowledged=True, consent_owner="Ada")

    record = registry.get("voice")

    assert record.consent_status == "granted"
    api = record.as_api_dict()
    assert api["consent"] == {
        "status": "granted",
        "acknowledged": True,
        "recorded_at": api["consent_at"],
        "owner": "Ada",
        "basis": "",
    }
    # The sidecar stays the flat set of stored fields, with no derived key.
    stored = json.loads((registry.root / "voice.json").read_text(encoding="utf-8"))
    assert "consent" not in stored


def test_overwriting_refreshes_the_consent_record(registry):
    register_with_consent(registry, consent_acknowledged=True, consent_owner="first")
    second = register_with_consent(
        registry, consent_acknowledged=True, consent_owner="second", overwrite=True
    )

    assert second.consent_owner == "second"
    assert registry.get("voice").consent_owner == "second"
