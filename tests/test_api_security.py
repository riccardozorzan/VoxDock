"""The HTTP layer over ``app.security``: does the wiring actually hold?

``tests/test_security.py`` proves the rules. This file proves they are *wired
to the routes* - that ``/v1/audio/speech`` really is behind a key, that a
speakers key really cannot clear the cache, and that a refusal really comes
back in the documented envelope with a ``Retry-After``.

The app is importable on the host because ``app.utils`` imports torch lazily and
the lifespan is not started here (``TestClient`` without a ``with`` block), so
no checkpoint is loaded and no GPU is touched. ``app.state`` is populated by
hand with just the objects the request path needs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

pytest.importorskip("fastapi", reason="the HTTP layer needs fastapi (see requirements-dev.txt)")
pytest.importorskip("httpx", reason="TestClient needs httpx")

from fastapi.testclient import TestClient  # noqa: E402

import app.config as config  # noqa: E402
from app.security import QueueGate, RateLimiter, parse_api_keys  # noqa: E402

TTS_KEY = "sk-tts-9f2a7c41d0e5b8a2"
SPEAKERS_KEY = "sk-spk-4c81e6b39aaa7d11"
ADMIN_KEY = "sk-adm-77bd1e0c55aa22f3"

KEYS = (
    f"pipeline:{TTS_KEY}:tts;library:{SPEAKERS_KEY}:speakers;ops:{ADMIN_KEY}:admin"
)


def build_client(tmp_path, *, queue_size=4, rpm=1000, **state_overrides):
    """A TestClient over the real app with its request-path state populated.

    The lifespan is deliberately not entered: it would load the 4.4 GB
    checkpoint. Everything the *request* path touches is therefore set up here
    by hand, which is also what makes these tests fast enough to run on every
    commit.
    """
    import app.main as main

    test_client = TestClient(main.app, raise_server_exceptions=False)
    state = test_client.app.state
    state.runtime = None
    state.runtime_info = None
    state.model_error = None
    state.last_generation = None
    state.lifetime_peak_vram_mb = None
    state.request_count = 0
    state.started_at = 1000.0
    state.queue_gate = QueueGate(max_queue=queue_size)
    state.rate_limiter = RateLimiter(limit=rpm)
    state.speakers = main.SpeakerRegistry(tmp_path / "speakers")
    for name, value in state_overrides.items():
        setattr(state, name, value)
    return test_client


@pytest.fixture
def client(tmp_path):
    """The default case: three scopes configured, limits wide open."""
    configure_security(api_keys=parse_api_keys(KEYS), rate_limit_rpm=1000, max_queue_size=4)
    return build_client(tmp_path)


def auth(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def configure_security(**fields) -> None:
    """Override fields on the frozen ``SecuritySettings`` for one test.

    ``SecuritySettings`` is a frozen dataclass, which is right - it is read by
    every request - but it means ``monkeypatch.setattr`` cannot be used on it.
    The originals are stashed and restored by the ``restore_security`` fixture,
    so a test that raises a rate limit does not leak that limit into the next.
    """
    sec = config.settings.security
    _SAVED.setdefault(id(sec), {}).update(
        {name: getattr(sec, name) for name in fields}
    )
    for name, value in fields.items():
        object.__setattr__(sec, name, value)


_SAVED: dict[int, dict[str, object]] = {}


@pytest.fixture(autouse=True)
def restore_security():
    yield
    for fields in _SAVED.values():
        for name, value in fields.items():
            object.__setattr__(config.settings.security, name, value)
    _SAVED.clear()


# --------------------------------------------------------------------------- #
#  A valid key gets in
# --------------------------------------------------------------------------- #
def test_a_synthesis_key_reaches_the_speech_endpoint(client):
    response = client.post("/v1/audio/speech", data={"text": "hi"}, headers=auth(TTS_KEY))

    # 503 (engine not loaded), not 401: the key was accepted and the route ran.
    assert response.status_code == 503
    assert response.json()["error"]["code"] != "UNAUTHORIZED"


def test_a_speakers_key_reaches_the_speaker_list(client):
    response = client.get("/v1/speakers", headers=auth(SPEAKERS_KEY))

    assert response.status_code == 200


# --------------------------------------------------------------------------- #
#  A missing or wrong key does not
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "path,method",
    [
        ("/v1/audio/speech", "post"),
        ("/v1/speakers", "get"),
        ("/v1/speakers/x", "delete"),
        ("/v1/admin/clear-cache", "post"),
        ("/v1/models", "get"),
    ],
)
def test_every_protected_endpoint_refuses_an_anonymous_caller(client, path, method):
    call = getattr(client, method)
    response = call(path, data={"text": "hi"}) if method == "post" else call(path)

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "UNAUTHORIZED"


def test_a_wrong_key_is_refused(client):
    response = client.get("/v1/speakers", headers=auth("sk-nope-0000000000000000"))

    assert response.status_code == 401


# --------------------------------------------------------------------------- #
#  Scope separation: the P0 "admin authorization" item
# --------------------------------------------------------------------------- #
def test_a_speakers_key_cannot_clear_the_cache(client):
    response = client.post("/v1/admin/clear-cache", headers=auth(SPEAKERS_KEY))

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "FORBIDDEN"


def test_a_tts_key_cannot_list_or_delete_speakers(client):
    assert client.get("/v1/speakers", headers=auth(TTS_KEY)).status_code == 403
    assert client.delete("/v1/speakers/x", headers=auth(TTS_KEY)).status_code == 403


def test_an_admin_key_can_reach_admin_routes(client):
    """Admin is a separate scope, not a synonym for "has any key"."""
    response = client.post("/v1/admin/clear-cache", headers=auth(ADMIN_KEY))

    # It got past authentication: not a 401/403.
    assert response.status_code not in (401, 403)


def test_the_refusal_names_the_missing_scope_without_leaking_the_key(client):
    response = client.get("/v1/speakers", headers=auth(TTS_KEY))
    message = response.json()["error"]["message"]

    assert "speakers" in message
    assert TTS_KEY not in message
    assert "pipeline" in message, "the label is what makes the audit log useful"


# --------------------------------------------------------------------------- #
#  The envelope
# --------------------------------------------------------------------------- #
def test_every_refusal_uses_the_documented_error_shape(client):
    body = client.get("/v1/speakers").json()

    assert set(body) == {"error"}
    assert set(body["error"]) >= {"code", "message", "retryable", "request_id"}
    assert body["error"]["retryable"] is False


def test_a_request_id_is_returned_on_a_refusal(client):
    body = client.get("/v1/speakers").json()

    assert len(body["error"]["request_id"]) == 32


def test_the_request_id_is_also_a_header(client):
    response = client.get("/v1/speakers")

    assert response.headers["X-Request-ID"] == response.json()["error"]["request_id"]


def test_a_client_supplied_request_id_is_echoed(client):
    response = client.get(
        "/v1/speakers", headers={**auth(SPEAKERS_KEY), "X-Request-ID": "job-42"}
    )

    assert response.headers["X-Request-ID"] == "job-42"


def test_a_forged_request_id_is_replaced_not_echoed(client):
    """Otherwise a caller can forge a line in the operator's log."""
    forged = "x\nINFO | voxdock.api | admin cleared the cache"
    response = client.get(
        "/v1/speakers", headers={**auth(SPEAKERS_KEY), "X-Request-ID": forged}
    )

    assert "\n" not in response.headers["X-Request-ID"]


def test_a_framework_404_is_also_in_the_envelope(client):
    body = client.get("/v1/definitely-not-a-route", headers=auth(TTS_KEY)).json()

    assert body["error"]["code"] == "NOT_FOUND"


# --------------------------------------------------------------------------- #
#  Public routes stay public: the health check has no credential
# --------------------------------------------------------------------------- #
def test_health_is_reachable_without_a_key(client):
    """A container that cannot be health-checked does not restart."""
    response = client.get("/health")

    assert response.status_code in (200, 503)


def test_health_hides_diagnostics_when_auth_is_enabled(client):
    body = client.get("/health").json()

    assert "status" in body and "ready" in body
    assert "gpu" not in body, "VRAM and model detail must not be public"
    assert "last_generation" not in body


def test_the_admin_health_route_carries_the_full_payload(client):
    response = client.get("/v1/admin/health", headers=auth(ADMIN_KEY))

    if response.status_code == 200:
        assert "gpu" in response.json()
    else:
        assert response.status_code == 503, "admin health must not 401 with a key"


def test_admin_health_is_refused_without_the_admin_scope(client):
    assert client.get("/v1/admin/health", headers=auth(TTS_KEY)).status_code == 403


# --------------------------------------------------------------------------- #
#  Open mode: the local-first default is unchanged
# --------------------------------------------------------------------------- #
def test_with_no_keys_configured_nothing_is_refused(tmp_path):
    """The local install must behave exactly as it did before this work."""
    configure_security(api_keys=())
    test_client = build_client(tmp_path)

    assert test_client.get("/v1/speakers").status_code == 200
    assert test_client.post("/v1/admin/clear-cache").status_code != 401


# --------------------------------------------------------------------------- #
#  Rate limiting and the queue, end to end
# --------------------------------------------------------------------------- #
def test_exceeding_the_rate_limit_returns_429_with_retry_after(tmp_path):
    configure_security(api_keys=parse_api_keys(KEYS), rate_limit_rpm=3)
    test_client = build_client(tmp_path, rpm=3)

    for _ in range(3):
        assert test_client.get("/v1/speakers", headers=auth(SPEAKERS_KEY)).status_code == 200

    refused = test_client.get("/v1/speakers", headers=auth(SPEAKERS_KEY))

    assert refused.status_code == 429
    assert refused.json()["error"]["code"] == "RATE_LIMITED"
    assert refused.json()["error"]["retryable"] is True
    assert int(refused.headers["Retry-After"]) >= 1


def test_the_rate_limit_is_per_key_not_global(tmp_path):
    """One client exhausting its budget must not lock out another."""
    configure_security(api_keys=parse_api_keys(KEYS), rate_limit_rpm=2)
    test_client = build_client(tmp_path, rpm=2)

    for _ in range(3):
        test_client.get("/v1/speakers", headers=auth(SPEAKERS_KEY))

    assert test_client.get("/v1/speakers", headers=auth(ADMIN_KEY)).status_code in (200, 403)


# --------------------------------------------------------------------------- #
#  The synthesis path
#
#  ``create_speech`` checks the runtime *before* the queue and the text cap,
#  which is the right order - there is no point queueing work for an engine
#  that is not there. So these tests stub a runtime and register a real (tiny)
#  reference clip to get past the earlier checks. Neither reaches torch: the
#  queue refusal happens before generation, and so does the text cap.
# --------------------------------------------------------------------------- #
def silent_wav(seconds: float = 0.5, sample_rate: int = 48000) -> bytes:
    """A valid, silent 48 kHz mono WAV - stdlib ``wave``, so no torch."""
    import io
    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * int(seconds * sample_rate))
    return buffer.getvalue()


def register_speaker(client, name="ref", **fields):
    data = {
        "reference_text": "one two three four five",
        "speaker_id": name,
        "consent_acknowledged": "true",
        **fields,
    }
    return client.post(
        "/v1/speakers/register",
        files={"reference_audio": ("ref.wav", silent_wav(), "audio/wav")},
        data=data,
        headers=auth(SPEAKERS_KEY),
    )


def test_a_full_queue_is_refused_even_with_an_engine_ready(tmp_path):
    """The gate is the last line, and it still refuses with a usable answer."""
    configure_security(api_keys=parse_api_keys(KEYS), max_queue_size=1)
    test_client = build_client(tmp_path, queue_size=1, runtime=object())
    assert register_speaker(test_client).status_code == 201
    # Occupy the only slot, as a synthesis in progress would.
    test_client.app.state.queue_gate.admit()

    response = test_client.post(
        "/v1/audio/speech", data={"text": "hello", "speaker_id": "ref"},
        headers=auth(TTS_KEY),
    )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "QUEUE_FULL"
    assert int(response.headers["Retry-After"]) >= 1


def test_the_text_cap_is_checked_before_the_gpu_is_touched(tmp_path):
    """An oversized request is refused, not attempted and failed."""
    configure_security(api_keys=parse_api_keys(KEYS), max_text_chars=10)
    test_client = build_client(tmp_path, runtime=object())
    assert register_speaker(test_client).status_code == 201

    response = test_client.post(
        "/v1/audio/speech", data={"text": "x" * 50, "speaker_id": "ref"},
        headers=auth(TTS_KEY),
    )

    assert response.status_code == 413
    body = response.json()["error"]
    assert body["code"] == "TEXT_TOO_LONG"
    assert "process_txt" in body["message"], "the message should say what to do"
    assert body["retryable"] is False
    # And it was not queued on the way to being refused.
    assert test_client.app.state.queue_gate.pending == 0


def test_the_default_text_cap_cannot_break_the_long_form_pipeline():
    """``process_txt.py`` sends at most ``chunk_max_words`` per request.

    The cap has to sit above that or the measured 7-chunk pipeline would start
    failing at 413 - so the default is asserted against the pipeline's own
    setting rather than against a round number someone liked.
    """
    from app.config import ProcessingSettings

    words = ProcessingSettings().chunk_max_words
    # ~7 characters per word is generous for English prose; even at 10 it has
    # to clear the cap.
    assert words * 10 < config.settings.security.max_text_chars


# --------------------------------------------------------------------------- #
#  Consent metadata
# --------------------------------------------------------------------------- #
def test_consent_is_recorded_with_the_voice(tmp_path):
    test_client = build_client(tmp_path)

    body = register_speaker(
        test_client, consent_owner="Ada", consent_basis="self-recorded"
    ).json()

    assert body["consent"]["status"] == "granted"
    assert body["consent"]["acknowledged"] is True
    assert body["consent"]["owner"] == "Ada"
    assert body["consent"]["basis"] == "self-recorded"
    assert body["consent"]["recorded_at"], "the moment must be recorded"


def test_a_registration_without_consent_reports_unknown_not_false(tmp_path):
    """"We never asked" and "we were told no" are different answers."""
    test_client = build_client(tmp_path)

    body = register_speaker(test_client, consent_acknowledged="false").json()

    assert body["consent"]["status"] == "not_acknowledged"
    assert body["consent"]["acknowledged"] is False


def test_an_omitted_consent_field_is_unknown(tmp_path):
    test_client = build_client(tmp_path)

    body = test_client.post(
        "/v1/speakers/register",
        files={"reference_audio": ("ref.wav", silent_wav(), "audio/wav")},
        data={"reference_text": "one two three", "speaker_id": "legacy"},
        headers=auth(SPEAKERS_KEY),
    ).json()

    assert body["consent"]["status"] == "unknown"
    assert body["consent"]["acknowledged"] is None


def test_a_pre_consent_sidecar_still_loads(tmp_path):
    """Existing voices must survive the upgrade; a missing key is not an error."""
    test_client = build_client(tmp_path)
    assert register_speaker(test_client).status_code == 201

    # Simulate a sidecar written before consent was tracked.
    import json

    meta = tmp_path / "speakers" / "ref.json"
    payload = json.loads(meta.read_text())
    for field in ("consent_acknowledged", "consent_at", "consent_owner", "consent_basis"):
        payload.pop(field, None)
    meta.write_text(json.dumps(payload))

    listed = test_client.get("/v1/speakers", headers=auth(SPEAKERS_KEY)).json()["data"]

    assert [record["id"] for record in listed] == ["ref"]
    assert listed[0]["consent"]["status"] == "unknown"


def test_consent_survives_a_reload_from_disk(tmp_path):
    test_client = build_client(tmp_path)
    register_speaker(test_client, consent_owner="Grace", consent_basis="contract")

    fresh = build_client(tmp_path)
    listed = fresh.get("/v1/speakers", headers=auth(SPEAKERS_KEY)).json()["data"]

    assert listed[0]["consent"]["owner"] == "Grace"
    assert listed[0]["consent"]["recorded_at"]


def test_overwriting_a_voice_refreshes_the_consent_record(tmp_path):
    """The timestamp must describe the audio on disk, not the first upload."""
    test_client = build_client(tmp_path)
    first = register_speaker(test_client, consent_owner="first").json()
    second = register_speaker(
        test_client, consent_owner="second", overwrite="true"
    ).json()

    assert first["consent"]["owner"] == "first"
    assert second["consent"]["owner"] == "second"


# --------------------------------------------------------------------------- #
#  The speaker cap
# --------------------------------------------------------------------------- #
def test_the_registry_is_capped(tmp_path):
    """Otherwise a loop of register calls fills the bind-mounted data dir."""
    configure_security(api_keys=parse_api_keys(KEYS), max_speakers=2)
    test_client = build_client(tmp_path)

    assert register_speaker(test_client, name="a").status_code == 201
    assert register_speaker(test_client, name="b").status_code == 201
    refused = register_speaker(test_client, name="c")

    assert refused.status_code == 409
    assert refused.json()["error"]["code"] == "SPEAKER_LIMIT_REACHED"


def test_a_named_registration_bypasses_the_cap_for_overwrites(tmp_path):
    """Overwriting an existing id does not add a voice, so it is not capped."""
    configure_security(api_keys=parse_api_keys(KEYS), max_speakers=1)
    test_client = build_client(tmp_path)

    assert register_speaker(test_client, name="only").status_code == 201
    assert register_speaker(test_client, name="only", overwrite="true").status_code == 201


# --------------------------------------------------------------------------- #
#  Registry errors are client errors
#
#  A latent 500: register() signals every validation failure with ValueError and
#  the route did not catch it, so re-registering a name returned a server error
#  for what is plainly a client mistake. Found by running the real service.
# --------------------------------------------------------------------------- #
def test_re_registering_a_name_is_a_409_not_a_500(tmp_path):
    test_client = build_client(tmp_path)
    assert register_speaker(test_client, name="twice").status_code == 201

    clash = register_speaker(test_client, name="twice")

    assert clash.status_code == 409
    assert clash.json()["error"]["code"] == "SPEAKER_EXISTS"
    assert clash.json()["error"]["retryable"] is False


def test_the_conflict_message_says_how_to_resolve_it(tmp_path):
    test_client = build_client(tmp_path)
    register_speaker(test_client, name="twice")

    assert "overwrite" in register_speaker(test_client, name="twice").json()["error"]["message"]


def test_overwrite_resolves_the_conflict(tmp_path):
    test_client = build_client(tmp_path)
    register_speaker(test_client, name="twice")

    assert register_speaker(test_client, name="twice", overwrite="true").status_code == 201


def test_a_malformed_speaker_id_is_a_400_not_a_500(tmp_path):
    test_client = build_client(tmp_path)

    response = test_client.post(
        "/v1/speakers/register",
        files={"reference_audio": ("ref.wav", silent_wav(), "audio/wav")},
        data={"reference_text": "one two three", "speaker_id": "bad id!"},
        headers=auth(SPEAKERS_KEY),
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_SPEAKER"
