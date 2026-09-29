"""Tests for the P0 security rules in ``app.security``.

Everything here is a pure function or a plain class, so the whole P0 surface -
authentication, scope separation, the exposure guard, rate limiting and the
bounded queue - is verified on the host, with no GPU, no container and no
running server. That is the reason those rules do not live in ``app/main.py``:
rules that can only be tested by starting a 6 GB service do not get tested.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.security import (  # noqa: E402
    SCOPE_ADMIN,
    SCOPE_ALL,
    SCOPE_SPEAKERS,
    SCOPE_TTS,
    ApiError,
    ConfigError,
    QueueGate,
    RateLimiter,
    client_principal,
    error_body,
    evaluate_exposure,
    extract_credential,
    is_loopback,
    parse_api_keys,
    sanitise_request_id,
    verify_key,
)

TTS_KEY = "sk-tts-9f2a7c41d0e5b8a2"
SPEAKERS_KEY = "sk-spk-4c81e6b39aaa7d11"
ADMIN_KEY = "sk-adm-77bd1e0c55aa22f3"
STAR_KEY = "sk-all-1234abcd5678ef90"

KEYS = parse_api_keys(
    f"pipeline:{TTS_KEY}:{SCOPE_TTS};"
    f"library:{SPEAKERS_KEY}:{SCOPE_SPEAKERS};"
    f"ops:{ADMIN_KEY}:{SCOPE_ADMIN};"
    f"root:{STAR_KEY}:{SCOPE_ALL}"
)


class FakeClock:
    """A clock the test steps, so nothing here has to sleep."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


# --------------------------------------------------------------------------- #
#  Parsing TTS_API_KEYS
# --------------------------------------------------------------------------- #
def test_keys_parse_into_label_secret_and_scopes():
    assert [(k.label, sorted(k.scopes)) for k in KEYS] == [
        ("pipeline", [SCOPE_TTS]),
        ("library", [SCOPE_SPEAKERS]),
        ("ops", [SCOPE_ADMIN]),
        ("root", [SCOPE_ALL]),
    ]


def test_unset_means_authentication_disabled():
    """The local-first default. The exposure guard is what covers this case."""
    assert parse_api_keys(None) == ()
    assert parse_api_keys("   ") == ()


def test_a_short_secret_is_rejected_loudly():
    """``TTS_API_KEYS=dev`` is the likeliest way this control gets defeated."""
    with pytest.raises(ConfigError, match="guessable word"):
        parse_api_keys("dev:short:tts")


def test_a_misspelled_scope_is_rejected_not_ignored():
    """``:adminn`` must not quietly mean "no admin access"."""
    with pytest.raises(ConfigError, match="unknown scope"):
        parse_api_keys(f"ops:{ADMIN_KEY}:adminn")


def test_an_entry_without_scopes_is_rejected():
    with pytest.raises(ConfigError, match="at least one scope"):
        parse_api_keys(f"ops:{ADMIN_KEY}:")


def test_a_bare_key_without_label_or_scope_is_rejected():
    with pytest.raises(ConfigError, match="must be"):
        parse_api_keys(ADMIN_KEY)


def test_duplicate_labels_are_rejected():
    with pytest.raises(ConfigError, match="duplicate label"):
        parse_api_keys(f"ops:{ADMIN_KEY}:admin;ops:{STAR_KEY}:admin")


def test_one_secret_may_not_back_two_labels():
    """Otherwise a flood is attributable to neither of them."""
    with pytest.raises(ConfigError, match="already"):
        parse_api_keys(f"ops:{ADMIN_KEY}:admin;ops2:{ADMIN_KEY}:admin")


# --------------------------------------------------------------------------- #
#  Authentication
# --------------------------------------------------------------------------- #
def test_a_valid_key_authenticates():
    result = verify_key(TTS_KEY, KEYS, SCOPE_TTS)

    assert result.ok
    assert result.label == "pipeline"


def test_a_wrong_key_is_refused_without_saying_which_part_was_wrong():
    result = verify_key(TTS_KEY[:-1] + "X", KEYS, SCOPE_TTS)

    assert not result.ok
    assert result.reason == "invalid"
    # A message naming the prefix would turn this into an oracle.
    assert TTS_KEY[:8] not in result.message


def test_a_valid_key_without_the_scope_is_refused_as_insufficient():
    result = verify_key(SPEAKERS_KEY, KEYS, SCOPE_ADMIN)

    assert not result.ok
    assert result.reason == "insufficient_scope"
    assert result.label == "library", "the audit log needs to know who was refused"


def test_the_star_scope_grants_everything():
    for scope in (SCOPE_TTS, SCOPE_SPEAKERS, SCOPE_ADMIN):
        assert verify_key(STAR_KEY, KEYS, scope).ok


def test_speaker_access_does_not_imply_synthesis():
    """Least privilege: registering a voice is not permission to generate."""
    assert verify_key(SPEAKERS_KEY, KEYS, SCOPE_TTS).reason == "insufficient_scope"


def test_synthesis_access_does_not_imply_admin():
    assert verify_key(TTS_KEY, KEYS, SCOPE_ADMIN).reason == "insufficient_scope"


def test_no_configured_keys_admits_everything():
    """Open by default on loopback; see evaluate_exposure for the other half."""
    result = verify_key("", (), SCOPE_ADMIN)

    assert result.ok
    assert result.reason == "disabled"


def test_a_non_ascii_key_is_refused_rather_than_raising():
    """Headers are attacker controlled: a 500 here would be a trivial DoS.

    ``hmac.compare_digest`` rejects non-ASCII ``str``, so the bytes encode has
    to happen before it, not after.
    """
    result = verify_key("ключ-ñ-🔑", KEYS, SCOPE_TTS)

    assert not result.ok


def test_the_last_matching_key_is_not_preferred_by_position():
    """The comparison is over every key, not a prefix scan."""
    assert verify_key(ADMIN_KEY, KEYS).label == "ops"


# --------------------------------------------------------------------------- #
#  Credential extraction
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "headers,expected",
    [
        ({"Authorization": f"Bearer {TTS_KEY}"}, TTS_KEY),
        ({"Authorization": f"bearer   {TTS_KEY}  "}, TTS_KEY),
        ({"Authorization": f"Token {TTS_KEY}"}, TTS_KEY),
        ({"X-API-Key": TTS_KEY}, TTS_KEY),
        ({"X-API-Key": f"  {TTS_KEY} "}, TTS_KEY),
        ({"Authorization": "Basic dXNlcjpwYXNz"}, ""),
        ({"Authorization": "Bearer"}, ""),
        ({"Authorization": ""}, ""),
        ({}, ""),
    ],
)
def test_credentials_come_from_either_header(headers, expected):
    assert extract_credential(
        headers.get("Authorization"), headers.get("X-API-Key")
    ) == expected


def test_authorization_wins_over_the_api_key_header():
    """Ambiguity is resolved in favour of the standard, not merged."""
    both = extract_credential("Bearer " + TTS_KEY, SPEAKERS_KEY)

    assert both == TTS_KEY


# --------------------------------------------------------------------------- #
#  Rate limiting
# --------------------------------------------------------------------------- #
def test_requests_within_budget_are_allowed():
    limiter = RateLimiter(3, window_s=60.0, clock=FakeClock())

    assert all(limiter.check("k").allowed for _ in range(3))


def test_the_request_after_the_budget_is_refused_with_a_retry_after():
    clock = FakeClock()
    limiter = RateLimiter(3, window_s=60.0, clock=clock)
    for _ in range(3):
        limiter.check("k")
    clock.advance(10)

    decision = limiter.check("k")

    assert not decision.allowed
    # 60 s window, first hit was 10 s ago -> 50 s of it left.
    assert decision.retry_after == pytest.approx(50.0)


def test_retry_after_is_never_advertised_as_less_than_a_second():
    """A 0.4 s wait must not be rounded down to "retry immediately"."""
    clock = FakeClock()
    limiter = RateLimiter(1, window_s=60.0, clock=clock)
    limiter.check("k")
    clock.advance(59.9)

    assert limiter.check("k").retry_after == 1.0


def test_budget_returns_as_the_window_slides():
    clock = FakeClock()
    limiter = RateLimiter(2, window_s=60.0, clock=clock)
    limiter.check("k")
    limiter.check("k")
    assert not limiter.check("k").allowed

    clock.advance(61)

    assert limiter.check("k").allowed


def test_the_limit_holds_across_a_window_boundary():
    """A fixed-window counter would reset at t=60 and let these through.

    Three requests land at t=59 and a fourth at t=61. A counter keyed on
    ``floor(now / 60)`` sees "3 used" in the first window and "0 used" in the
    second and admits the fourth; a sliding window sees the t=59 hits are still
    inside 60 s and refuses. A token bucket has the same problem at a different
    edge, which is why neither is used here.
    """
    clock = FakeClock()
    limiter = RateLimiter(3, window_s=60.0, clock=clock)

    clock.advance(59)
    for _ in range(3):
        assert limiter.check("k").allowed, "the first window's budget is available"

    clock.advance(2)  # t=61: a fresh fixed window, but only 2 s since the last hit
    assert not limiter.check("k").allowed, (
        "a fixed-window counter would have reset here and admitted this"
    )


def test_refusing_a_request_does_not_extend_the_lockout():
    """A client hammering a closed door must not push its own window out."""
    clock = FakeClock()
    limiter = RateLimiter(1, window_s=10.0, clock=clock)
    limiter.check("k")
    clock.advance(5)
    for _ in range(20):
        limiter.check("k")
    clock.advance(6)

    assert limiter.check("k").allowed


def test_budgets_are_per_principal():
    limiter = RateLimiter(1, window_s=60.0, clock=FakeClock())

    assert limiter.check("key:ops").allowed
    assert not limiter.check("key:ops").allowed
    assert limiter.check("key:pipeline").allowed


def test_the_principal_table_is_bounded():
    """The limiter must not become the memory exhaustion it exists to stop."""
    limiter = RateLimiter(5, window_s=60.0, clock=FakeClock(), max_principals=10)

    for index in range(100):
        limiter.check(f"ip:10.0.0.{index}")

    assert limiter.tracked_principals() == 10


def test_a_zero_limit_is_a_configuration_error():
    with pytest.raises(ValueError):
        RateLimiter(0)


def test_principals_are_keyed_by_label_when_authenticated():
    from app.security import AuthResult

    authed = AuthResult(ok=True, label="ops")

    assert client_principal(authed, "10.0.0.5") == "key:ops"


def test_principals_fall_back_to_the_address_when_open():
    from app.security import AUTH_DISABLED

    assert client_principal(AUTH_DISABLED, "10.0.0.5") == "ip:10.0.0.5"
    assert client_principal(AUTH_DISABLED, None) == "ip:unknown"


# --------------------------------------------------------------------------- #
#  Bounded queue
# --------------------------------------------------------------------------- #
def test_the_queue_admits_up_to_its_capacity():
    gate = QueueGate(max_queue=3)

    assert [gate.admit().admitted for _ in range(3)] == [True, True, True]


def test_a_full_queue_refuses_with_a_retry_after():
    gate = QueueGate(max_queue=2)
    gate.admit()
    gate.admit()

    admission = gate.admit()

    assert not admission.admitted
    assert admission.reason == "queue_full"
    assert admission.retry_after >= 1.0


def test_a_refused_request_does_not_consume_capacity():
    """Otherwise a burst permanently poisons the queue for everyone."""
    gate = QueueGate(max_queue=1)
    gate.admit()
    for _ in range(5):
        assert not gate.admit().admitted
    assert gate.pending == 1


def test_releasing_frees_a_slot():
    gate = QueueGate(max_queue=1)
    gate.admit()
    gate.release()

    assert gate.pending == 0
    assert gate.admit().admitted


def test_pending_never_goes_negative():
    """A mismatched release must not corrupt the telemetry in /health."""
    gate = QueueGate(max_queue=2)
    gate.admit()
    gate.release()
    gate.release()

    assert gate.pending == 0


def test_a_zero_queue_refuses_everything_but_the_first():
    """``TTS_MAX_QUEUE_SIZE=0`` means "no queue", not "no service"."""
    gate = QueueGate(max_queue=0)

    assert not gate.admit().admitted


def test_waiting_counts_only_the_queue_not_the_gpu():
    gate = QueueGate(max_queue=4)
    gate.admit()
    gate.start()
    gate.admit()
    gate.admit()

    assert gate.snapshot() == {
        "pending": 3,
        "in_flight": 1,
        "waiting": 2,
        "capacity": 4,
        "service_estimate_s": 8.0,
    }


def test_retry_after_follows_the_measured_service_time():
    """An honest number beats a constant, and the first refusal has no data."""
    gate = QueueGate(max_queue=1, service_estimate_s=8.0)
    gate.admit()
    assert gate.retry_after_s() == 8.0

    for _ in range(5):
        gate.admit()
        gate.release(service_seconds=20.0)

    # EWMA over 5 observations of 20 s, from an 8 s start, converges on 20.
    assert gate.retry_after_s() > 12.0


def test_a_measured_service_time_moves_the_estimate_down_as_well():
    gate = QueueGate(max_queue=1, service_estimate_s=30.0)
    for _ in range(5):
        gate.admit()
        gate.release(service_seconds=2.0)

    assert gate.retry_after_s() < 20.0


# --------------------------------------------------------------------------- #
#  Request IDs
# --------------------------------------------------------------------------- #
def test_a_boring_request_id_is_kept():
    assert sanitise_request_id("job-42_a.b") == "job-42_a.b"


def test_a_log_injection_attempt_is_replaced():
    """The id is echoed into logs, so a newline here is a forged log line."""
    forged = "abc\nINFO | api | admin cleared the cache"

    assert sanitise_request_id(forged) != forged
    assert "\n" not in sanitise_request_id(forged)


def test_an_oversized_request_id_is_replaced():
    assert len(sanitise_request_id("x" * 500)) < 64


def test_a_missing_request_id_is_generated():
    generated = sanitise_request_id(None)

    assert len(generated) == 32


def test_two_different_requests_get_different_ids():
    assert sanitise_request_id(None) != sanitise_request_id(None)


# --------------------------------------------------------------------------- #
#  Structured errors
# --------------------------------------------------------------------------- #
def test_an_error_body_carries_the_code_and_the_request_id():
    body = error_body(
        ApiError(429, "RATE_LIMITED", "too many requests", retry_after=30), "req-1"
    )

    assert body == {
        "error": {
            "code": "RATE_LIMITED",
            "message": "too many requests",
            "retryable": True,
            "request_id": "req-1",
            "retry_after": 30,
        }
    }


def test_retryable_is_derived_from_the_code():
    assert ApiError(429, "RATE_LIMITED", "x").is_retryable
    assert ApiError(503, "QUEUE_FULL", "x").is_retryable
    assert not ApiError(400, "TEXT_TOO_LONG", "x").is_retryable
    assert not ApiError(401, "UNAUTHORIZED", "x").is_retryable


def test_an_explicit_retryable_overrides_the_code():
    """A 429 that a client must not retry is still expressible."""
    assert not ApiError(429, "RATE_LIMITED", "x", retryable=False).is_retryable


def test_retry_after_is_absent_when_there_is_nothing_to_wait_for():
    body = error_body(ApiError(400, "TEXT_TOO_LONG", "x"), "req-1")

    assert "retry_after" not in body["error"]


# --------------------------------------------------------------------------- #
#  Network exposure
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "address",
    ["127.0.0.1", "127.1.2.3", "localhost", "::1", "[::1]", "::ffff:127.0.0.1"],
)
def test_loopback_addresses_are_recognised(address):
    assert is_loopback(address)


@pytest.mark.parametrize(
    "address", ["0.0.0.0", "192.168.1.5", "10.0.0.1", "::", "example.com", "", None]
)
def test_remote_addresses_are_not_mistaken_for_loopback(address):
    assert not is_loopback(address)


def test_the_documented_default_is_allowed_but_says_so():
    """No keys, published on loopback: fine, and it must still be visible."""
    verdict = evaluate_exposure("127.0.0.1", auth_enabled=False)

    assert verdict.safe
    assert "No API keys configured" in verdict.warning


def test_remote_without_keys_refuses_to_start():
    """The control the whole P0 section exists for."""
    verdict = evaluate_exposure("0.0.0.0", auth_enabled=False)

    assert not verdict.safe
    assert "Refusing to publish" in verdict.fatal
    assert "TTS_API_KEYS" in verdict.fatal, "the refusal must say how to fix it"


def test_remote_with_keys_is_allowed_with_a_warning():
    verdict = evaluate_exposure("0.0.0.0", auth_enabled=True)

    assert verdict.safe
    assert verdict.fatal is None
    assert "authentication enabled" in verdict.warning


def test_allow_remote_is_an_explicit_escape_hatch():
    verdict = evaluate_exposure("0.0.0.0", auth_enabled=False, allow_remote=True)

    assert verdict.safe
    assert "impersonate" not in (verdict.warning or "")
    assert "Anything that can reach" in verdict.warning


def test_a_container_binding_all_interfaces_is_not_itself_an_exposure():
    """Inside Docker, 0.0.0.0 is how the published port reaches the service.

    Refusing on the *bind* host would make the documented deployment
    impossible, so the bind host is not consulted when running in a container.
    """
    verdict = evaluate_exposure(
        "127.0.0.1", auth_enabled=False, bind_host="0.0.0.0", in_container=True
    )

    assert verdict.safe


def test_a_host_run_bound_to_every_interface_is_refused():
    """Without a container, the bind address is the exposure."""
    verdict = evaluate_exposure(
        "127.0.0.1", auth_enabled=False, bind_host="0.0.0.0", in_container=False
    )

    assert not verdict.safe
    assert "TTS_BIND_HOST" in verdict.fatal


def test_an_empty_publish_address_is_treated_as_loopback():
    assert evaluate_exposure("", auth_enabled=False).safe


def test_an_empty_publish_address_with_keys_does_not_warn():
    assert evaluate_exposure(None, auth_enabled=True).warning is None


# --------------------------------------------------------------------------- #
#  Provenance
# --------------------------------------------------------------------------- #
def test_provenance_records_what_produced_the_audio():
    from app.security import build_provenance

    record = build_provenance(
        request_id="abc123",
        model="dots-studio/dots.tts-mf",
        text="Questo e un test.",
        audio_seconds=2.512,
        sample_rate=48000,
        speaker_id="riccardo",
        consent_status="granted",
    )

    assert record["generator"] == "VoxDock"
    assert record["request_id"] == "abc123"
    assert record["model"] == "dots-studio/dots.tts-mf"
    assert record["audio_seconds"] == 2.512
    assert record["sample_rate"] == 48000


def test_provenance_carries_the_consent_status():
    """The hard half of any abuse question: was this voice permitted?"""
    from app.security import build_provenance

    permitted = build_provenance(
        request_id="a", model="m", text="t", audio_seconds=1.0, sample_rate=48000,
        consent_status="granted",
    )
    inline = build_provenance(
        request_id="b", model="m", text="t", audio_seconds=1.0, sample_rate=48000
    )

    assert permitted["consent_status"] == "granted"
    assert inline["consent_status"] is None, (
        "an inline upload has no registered voice, and saying so honestly beats "
        "implying consent"
    )


def test_the_text_fingerprint_is_stable_and_one_way():
    from app.security import text_fingerprint

    first = text_fingerprint("Questo e un test.")
    assert first == text_fingerprint("Questo e un test.")
    assert first != text_fingerprint("Questo e un altro test.")
    assert len(first) == 16
    assert "Questo" not in first, "the fingerprint must not leak the text"


def test_identical_text_gives_an_identical_fingerprint():
    """That is the point: it is a lookup key for a request, not a nonce."""
    from app.security import text_fingerprint

    assert text_fingerprint("same") == text_fingerprint("same")


def test_provenance_headers_omit_what_is_unknown():
    """A client testing for a header's presence must not be misled by an empty one."""
    from app.security import build_provenance, provenance_headers

    inline = provenance_headers(
        build_provenance(
            request_id="a", model="m", text="t", audio_seconds=1.0, sample_rate=48000
        )
    )

    assert "X-Speaker-Id" not in inline
    assert "X-Consent-Status" not in inline
    assert inline["X-VoxDock-Generator"] == "VoxDock"
    assert inline["X-Text-SHA256"]


def test_provenance_headers_are_ascii_and_header_safe():
    """A model name or text hash with a newline would split the header."""
    from app.security import provenance_headers

    headers = provenance_headers(
        {
            "generator": "VoxDock",
            "version": "1.1.0",
            "model": "dots-studio/dots.tts-mf",
            "text_sha256": "abc",
            "speaker_id": "riccardo",
            "consent_status": "granted",
        }
    )

    for name, value in headers.items():
        assert "\n" not in value and "\r" not in value, name
        assert value.isascii(), name


# --------------------------------------------------------------------------- #
#  Input and budget caps
# --------------------------------------------------------------------------- #
def test_a_reference_within_the_cap_is_allowed():
    from app.security import check_reference_duration

    assert check_reference_duration(7.0, 20.0) is None


def test_an_over_long_reference_is_refused_with_the_numbers():
    from app.security import check_reference_duration

    error = check_reference_duration(45.0, 20.0)

    assert error is not None
    assert error.status == 413
    assert error.code == "REFERENCE_TOO_LONG"
    assert "45" in error.message and "20" in error.message
    assert error.is_retryable is False


def test_a_zero_limit_disables_the_reference_cap():
    from app.security import check_reference_duration

    assert check_reference_duration(600.0, 0) is None


def test_the_generation_budget_names_the_real_constraint():
    from app.security import check_generation_budget

    error = check_generation_budget(90.0, 20.0)

    assert error is not None
    assert error.code == "TEXT_TOO_LONG"
    assert "patch" in error.message, "the VRAM budget is the actual reason"


def test_the_generation_budget_allows_what_fits():
    from app.security import check_generation_budget

    assert check_generation_budget(9.0, 20.0) is None
    assert check_generation_budget(90.0, 0) is None, "0 disables"


def test_the_text_cap_allows_what_fits():
    from app.security import check_text_length

    assert check_text_length("short", 2000) is None
    assert check_text_length("x" * 2001, 2000) is not None
    assert check_text_length("x" * 99999, 0) is None, "0 disables"


def test_the_generation_estimate_scales_with_the_text():
    from app.security import estimate_generation_seconds

    short = estimate_generation_seconds(60, 2.5)
    long = estimate_generation_seconds(600, 2.5)

    assert long == pytest.approx(short * 10)
    assert estimate_generation_seconds(60, 0) == 0.0, "no division by zero"
    assert estimate_generation_seconds(-5, 2.5) == 0.0, "negative text is zero"
