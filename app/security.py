"""Authentication, rate limiting and abuse limits for the HTTP surface.

Deliberately pure
-----------------
This module imports **nothing** but the standard library - no torch, no
FastAPI, no ``app.config``. That is a hard constraint, not a style choice: the
host-side test suite installs neither torch nor FastAPI (see
``requirements-dev.txt``), so ``app/main.py`` cannot be imported on the host at
all. If the security rules lived in the route handlers they would be the one
part of the service with no tests - precisely the part that must not be taken
on faith. Here they are plain functions and classes, directly unit-tested, and
:mod:`app.main` only adapts HTTP onto them.

The split has a second benefit: the routing layer stays a routing layer. Route
bodies read as HTTP handling, not as a pile of ``if len(text) > N`` branches
whose interaction nobody can reason about.

Threat model
------------
This service clones any voice from any uploaded WAV. A caller who can reach it
can impersonate a person, so the two things that matter are (a) only the
operator can reach the endpoints at all, and (b) if they can, one caller cannot
monopolise the single GPU or fill the disk with voices. ``TODO/todo.md`` P0
lists these as authentication, network-exposure protection, rate limiting and a
bounded queue; they live here and are wired up in :mod:`app.main`.

With no keys configured the service stays **open**, which is the right default
for something bound to ``127.0.0.1``. The counterpart is that exposing it to a
network without keys is a mistake the operator should be stopped from making,
which is what :func:`evaluate_exposure` is for.
"""

from __future__ import annotations

import hmac
import re
import time
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

__all__ = [
    "SCOPE_TTS",
    "SCOPE_SPEAKERS",
    "SCOPE_ADMIN",
    "SCOPE_ALL",
    "VALID_SCOPES",
    "APIKey",
    "ConfigError",
    "parse_api_keys",
    "AuthResult",
    "AUTH_DISABLED",
    "verify_key",
    "extract_credential",
    "RateLimiter",
    "RateDecision",
    "MAX_TRACKED_PRINCIPALS",
    "QueueGate",
    "Admission",
    "DEFAULT_SERVICE_ESTIMATE_S",
    "ApiError",
    "error_body",
    "RETRYABLE_CODES",
    "new_request_id",
    "sanitise_request_id",
    "client_principal",
    "is_loopback",
    "ExposureVerdict",
    "evaluate_exposure",
    "MIN_KEY_LENGTH",
]

# --------------------------------------------------------------------------- #
#  Scopes
# --------------------------------------------------------------------------- #

#: Synthesis - the expensive, GPU-saturating operation.
SCOPE_TTS = "tts"
#: Registering, listing and deleting stored voices.
SCOPE_SPEAKERS = "speakers"
#: Cache clearing and the detailed diagnostics in /health.
SCOPE_ADMIN = "admin"
#: Every scope. Must be spelled out; see :func:`parse_api_keys`.
SCOPE_ALL = "*"

VALID_SCOPES = frozenset({SCOPE_TTS, SCOPE_SPEAKERS, SCOPE_ADMIN, SCOPE_ALL})

#: A key shorter than this is not a secret, it is a guessable word. Refused at
#: parse time with a message that says so, because ``TTS_API_KEYS=dev`` looks
#: like it works and is the likeliest way this control gets defeated.
MIN_KEY_LENGTH = 16


class ConfigError(ValueError):
    """A malformed ``TTS_API_KEYS``. Raised at startup, never per request."""


@dataclass(frozen=True)
class APIKey:
    """One credential: the secret, who it owns, and what it may do.

    The label is not decoration - it is what appears in the audit log when a
    request is refused, so a flood can be attributed to a key rather than to an
    anonymous address. Never log ``key`` itself.
    """

    label: str
    key: str
    scopes: frozenset[str]

    def allows(self, scope: str) -> bool:
        return SCOPE_ALL in self.scopes or scope in self.scopes


_LABEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,31}$")


def parse_api_keys(raw: Optional[str]) -> tuple[APIKey, ...]:
    """Parse ``TTS_API_KEYS`` into credentials.

    Format, one entry per ``;``::

        <label>:<secret>:<scope>[|<scope>...]

    for example::

        TTS_API_KEYS="ci:sk-live-9f2a7c41d0e5b8a2:tts|speakers;ops:sk-ops-4c81e6b3:admin"

    Every decision here fails loudly, which is the point:

    * **A bare key is rejected.** Each entry must name a label and at least one
      scope. Defaulting is wrong in both directions: guessing ``admin`` would be
      a privilege escalation, and guessing ``tts`` would lock the operator out
      of the endpoint they were trying to reach.
    * **Scopes are enumerated, not free text.** ``:adminn`` is rejected rather
      than quietly meaning "no admin access", which is how a security control
      ends up believed-in and absent.
    * **Keys come from the environment only** (see
      ``SecuritySettings.from_env``), never from a tuned profile, so a profile
      that gets committed cannot carry a credential.

    Returns an empty tuple when unset, meaning "authentication disabled".
    """
    if raw is None or not raw.strip():
        return ()

    keys: list[APIKey] = []
    seen_labels: set[str] = set()
    seen_secrets: set[str] = set()

    for index, entry in enumerate(raw.split(";"), start=1):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split(":")
        if len(parts) != 3:
            raise ConfigError(
                f"TTS_API_KEYS entry {index} must be "
                f"'<label>:<secret>:<scope>[|<scope>...]', got {entry!r}"
            )
        label, secret, scope_text = (part.strip() for part in parts)

        if not _LABEL_PATTERN.match(label):
            raise ConfigError(
                f"TTS_API_KEYS entry {index}: label {label!r} must be 1-32 "
                "characters of letters, digits, '.', '_' or '-'"
            )
        if len(secret) < MIN_KEY_LENGTH:
            raise ConfigError(
                f"TTS_API_KEYS entry {index} ({label}): the secret must be at "
                f"least {MIN_KEY_LENGTH} characters, got {len(secret)}. A short "
                "key is a guessable word, not a credential."
            )
        scopes = frozenset(s for s in scope_text.split("|") if s)
        if not scopes:
            raise ConfigError(
                f"TTS_API_KEYS entry {index} ({label}): at least one scope is "
                f"required, one of {', '.join(sorted(VALID_SCOPES))} or {SCOPE_ALL!r}"
            )
        unknown = scopes - VALID_SCOPES
        if unknown:
            raise ConfigError(
                f"TTS_API_KEYS entry {index} ({label}): unknown scope(s) "
                f"{sorted(unknown)}; valid scopes are "
                f"{', '.join(sorted(VALID_SCOPES))}"
            )
        if label in seen_labels:
            raise ConfigError(
                f"TTS_API_KEYS entry {index}: duplicate label {label!r}; labels "
                "must be unique or the audit log becomes ambiguous"
            )
        if secret in seen_secrets:
            raise ConfigError(
                f"TTS_API_KEYS entry {index} ({label}): that secret is already "
                "used by another label"
            )
        seen_labels.add(label)
        seen_secrets.add(secret)
        keys.append(APIKey(label=label, key=secret, scopes=scopes))

    return tuple(keys)


# --------------------------------------------------------------------------- #
#  Verification
# --------------------------------------------------------------------------- #

_SCHEME_PATTERN = re.compile(r"^\s*(\w+)\s+(.+?)\s*$", re.IGNORECASE)


def extract_credential(authorization: Optional[str], api_key: Optional[str]) -> str:
    """Pull the presented secret out of the request headers.

    Accepts ``Authorization: Bearer <key>`` and ``X-API-Key: <key>``. Parsing
    happens here so the rest of the module deals in plain strings and stays
    testable without an ASGI request object.
    """
    if authorization:
        match = _SCHEME_PATTERN.match(authorization)
        if not match:
            return ""
        scheme, value = match.group(1).lower(), match.group(2)
        if scheme not in {"bearer", "token"}:
            return ""
        return value
    return (api_key or "").strip()


@dataclass(frozen=True)
class AuthResult:
    """Outcome of authenticating one request."""

    ok: bool
    label: Optional[str] = None
    scopes: frozenset[str] = field(default_factory=frozenset)
    #: ``None`` when ``ok``; otherwise one of ``invalid``,
    #: ``insufficient_scope`` or ``disabled``.
    reason: Optional[str] = None
    message: str = ""
    required_scope: Optional[str] = None

    def allows(self, scope: str) -> bool:
        return self.ok and (SCOPE_ALL in self.scopes or scope in self.scopes)


#: The result every route sees when no keys are configured. ``ok`` is True so
#: route bodies need no special case, but the scopes are empty: nothing is
#: granted, everything is merely not refused. ``main.py`` checks whether keys
#: exist separately, where it needs to tell the operator the service is open.
AUTH_DISABLED = AuthResult(ok=True, reason="disabled", message="authentication disabled")


def verify_key(
    presented: str,
    keys: Sequence[APIKey],
    required_scope: Optional[str] = None,
) -> AuthResult:
    """Authenticate ``presented`` against ``keys``, then check one scope.

    Constant time. Every candidate is compared with :func:`hmac.compare_digest`
    and the loop does not return early, so the response time carries no
    information about which prefix of a guess was right, and the total time
    does not reveal how many keys are configured. An attacker who can time
    responses should learn nothing beyond "that key is not valid".

    Non-ASCII header values are encoded to bytes first: ``compare_digest``
    rejects non-ASCII ``str`` inputs, and headers are attacker controlled, so
    raising here would be a trivially reachable 500.
    """
    if not keys:
        return AUTH_DISABLED

    candidate = presented.encode("utf-8")
    matched: Optional[APIKey] = None
    for entry in keys:
        # No `if matched: break` - see the docstring.
        if hmac.compare_digest(entry.key.encode("utf-8"), candidate):
            matched = entry

    if matched is None:
        return AuthResult(
            ok=False, reason="invalid", message="invalid or unknown API key"
        )

    if required_scope and not matched.allows(required_scope):
        return AuthResult(
            ok=False,
            label=matched.label,
            reason="insufficient_scope",
            required_scope=required_scope,
            message=(
                f"API key {matched.label!r} is not authorised for "
                f"{required_scope!r}. Granted: {', '.join(sorted(matched.scopes))}."
            ),
        )

    return AuthResult(ok=True, label=matched.label, scopes=matched.scopes)


# --------------------------------------------------------------------------- #
#  Rate limiting
# --------------------------------------------------------------------------- #

#: Cap on tracked principals. With authentication off a caller is identified by
#: address, so the table is otherwise bounded only by how many can reach the
#: port; oldest entries are evicted rather than allowed to grow, because the
#: limiter must not become the memory-exhaustion vector it exists to prevent.
MAX_TRACKED_PRINCIPALS = 4096


@dataclass(frozen=True)
class RateDecision:
    allowed: bool
    #: Seconds until a refused request would be admitted. ``None`` when allowed.
    retry_after: Optional[float] = None
    limit: int = 0
    remaining: int = 0


class RateLimiter:
    """Sliding-window rate limiter, one budget per principal.

    A sliding window over a fixed window, rather than a token bucket: a bucket
    sized "N per minute" lets its whole budget be spent in one burst at the
    start of every window, and a burst at the start of a window is exactly what
    an abuse attempt looks like. This keeps the timestamps of recent requests
    and counts what is still inside the window, so the limit holds across
    boundaries.

    The clock is injected so the tests step time instead of sleeping.
    """

    def __init__(
        self,
        limit: int,
        window_s: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        max_principals: int = MAX_TRACKED_PRINCIPALS,
    ) -> None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        self.limit = int(limit)
        self.window_s = float(window_s)
        self._clock = clock
        self._max_principals = max_principals
        self._hits: "OrderedDict[str, deque[float]]" = OrderedDict()

    def _prune(self, principal: str) -> deque[float]:
        """Drop timestamps that have left the window; return the live ones."""
        cutoff = self._clock() - self.window_s
        hits = self._hits.get(principal)
        if hits is None:
            hits = deque()
            self._hits[principal] = hits
        while hits and hits[0] <= cutoff:
            hits.popleft()
        return hits

    def check(self, principal: str) -> RateDecision:
        """Record an attempt and say whether it is within budget.

        A refused request is not recorded, so a client hammering a closed door
        does not extend its own lockout.
        """
        now = self._clock()
        hits = self._prune(principal)

        if len(hits) >= self.limit:
            return RateDecision(
                allowed=False,
                # Round *up*: a 0.4 s wait must not be advertised as "retry now".
                retry_after=max(1.0, hits[0] + self.window_s - now),
                limit=self.limit,
                remaining=0,
            )

        hits.append(now)
        # Re-insert to mark it most-recently-used for the eviction order.
        self._hits.move_to_end(principal)
        while len(self._hits) > self._max_principals:
            self._hits.popitem(last=False)
        return RateDecision(
            allowed=True, limit=self.limit, remaining=self.limit - len(hits)
        )

    def reset(self, principal: Optional[str] = None) -> None:
        """Forget one principal's history, or all of it."""
        if principal is None:
            self._hits.clear()
        else:
            self._hits.pop(principal, None)

    def tracked_principals(self) -> int:
        return len(self._hits)


# --------------------------------------------------------------------------- #
#  Bounded queue
# --------------------------------------------------------------------------- #

#: Assumed service time before anything has been measured, used only so the
#: first callers get a plausible ``Retry-After`` instead of a fabricated one.
DEFAULT_SERVICE_ESTIMATE_S = 8.0


@dataclass(frozen=True)
class Admission:
    admitted: bool
    #: Requests holding the GPU plus those queued behind them.
    pending: int = 0
    in_flight: int = 0
    waiting: int = 0
    retry_after: Optional[float] = None
    reason: Optional[str] = None


class QueueGate:
    """Bounded admission control in front of the single generation lock.

    Why not a semaphore
    -------------------
    An unbounded queue does not queue, it accumulates: a client that sends 400
    requests to a 1-GPU service gets 399 parked for the length of a synthesis
    each, holding memory and a request context, and every one of them
    eventually runs - so the burst is not rejected, merely delayed past its own
    timeout. Bounding admission means the excess is *refused*, with a
    ``Retry-After`` the client can act on.

    This is also the single source of truth for the queue counters in
    ``/health``, replacing the pair of counters that used to be incremented
    inside the route body and could drift from what was actually happening.

    The counters are plain attributes rather than an :class:`asyncio.Semaphore`
    because admission has to be *rejected* with a number the client should be
    told, and a semaphore can only block - it has no "full" state to report.
    The exclusive lock stays where it belongs, on the generation lock.
    """

    def __init__(
        self,
        max_queue: int = 8,
        service_estimate_s: float = DEFAULT_SERVICE_ESTIMATE_S,
    ) -> None:
        if max_queue < 0:
            raise ValueError("max_queue must not be negative")
        self.max_queue = int(max_queue)
        #: Exponential moving average of observed generation time, used for
        #: ``Retry-After``. Starts as an estimate and is replaced by measurement
        #: as soon as there is one; an honest number beats a constant.
        self.service_estimate_s = float(service_estimate_s)
        self._pending = 0
        self._in_flight = 0
        self._observed = 0

    @property
    def pending(self) -> int:
        return self._pending

    @property
    def in_flight(self) -> int:
        return self._in_flight

    @property
    def waiting(self) -> int:
        return max(0, self._pending - self._in_flight)

    def retry_after_s(self) -> float:
        """How long a refused client should wait: about one service time.

        The GPU serves requests serially and the refused request is first in
        line once a slot frees, so one service time is the honest lower bound.
        """
        return max(1.0, round(self.service_estimate_s))

    def admit(self) -> Admission:
        """Reserve a slot, or refuse with a reason and a ``Retry-After``."""
        if self._pending >= self.max_queue:
            return Admission(
                admitted=False,
                pending=self._pending,
                in_flight=self._in_flight,
                waiting=self.waiting,
                retry_after=self.retry_after_s(),
                reason="queue_full",
            )
        self._pending += 1
        return Admission(
            admitted=True,
            pending=self._pending,
            in_flight=self._in_flight,
            waiting=self.waiting,
        )

    def release(self, service_seconds: Optional[float] = None) -> None:
        """Give the slot back, optionally feeding in the observed service time."""
        self._pending = max(0, self._pending - 1)
        if service_seconds and service_seconds > 0:
            self._observed += 1
            # EWMA weighted 1/n: cheap, convergent, and it never lets a single
            # slow request move the estimate much.
            self.service_estimate_s += (float(service_seconds) - self.service_estimate_s) / (
                1 + self._observed
            )

    def start(self) -> None:
        self._in_flight += 1

    def finish(self) -> None:
        self._in_flight = max(0, self._in_flight - 1)

    def snapshot(self) -> dict[str, Any]:
        return {
            "pending": self._pending,
            "in_flight": self._in_flight,
            "waiting": self.waiting,
            "capacity": self.max_queue,
            "service_estimate_s": round(self.service_estimate_s, 2),
        }


# --------------------------------------------------------------------------- #
#  Request IDs and structured errors
# --------------------------------------------------------------------------- #

_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def new_request_id() -> str:
    return uuid.uuid4().hex


def sanitise_request_id(candidate: Optional[str]) -> str:
    """Keep a client-supplied request id only if it is boring.

    The value is echoed into logs and error bodies, so an unvalidated header
    would let a caller inject newlines into the log stream, or megabytes of text
    into every response. Anything unexpected is *replaced* rather than repaired:
    truncating would make two different requests look alike, which defeats the
    purpose of having an id.
    """
    if candidate and _REQUEST_ID_PATTERN.match(candidate):
        return candidate
    return new_request_id()


def client_principal(auth: AuthResult, client_host: Optional[str]) -> str:
    """What to count a request against: the key, or the peer address.

    With authentication off there is no label to count against, so the address
    is used - which still bounds one misbehaving client, and is the only
    identifier available. It is *not* a security boundary: behind a reverse
    proxy every request looks like the proxy. That is a documented consequence
    of running open, not a bug.
    """
    if auth.label:
        return f"key:{auth.label}"
    return f"ip:{client_host or 'unknown'}"


#: Why a request was refused, in the shape a client branches on. ``retryable``
#: is the load-bearing part: a client that blindly retries a 400 has a bug, and
#: one that never retries stalls its own queue.
RETRYABLE_CODES = frozenset(
    {
        "RATE_LIMITED",
        "QUEUE_FULL",
        "ENGINE_BUSY",
        "OUT_OF_MEMORY",
        "INTERNAL_ERROR",
    }
)


@dataclass(frozen=True)
class ApiError:
    """A refusal expressed as data, so it is testable without HTTP."""

    status: int
    code: str
    message: str
    retryable: Optional[bool] = None
    retry_after: Optional[float] = None

    @property
    def is_retryable(self) -> bool:
        if self.retryable is not None:
            return self.retryable
        return self.code in RETRYABLE_CODES

    def as_body(self, request_id: str) -> dict[str, Any]:
        error: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "retryable": self.is_retryable,
            "request_id": request_id,
        }
        if self.retry_after is not None:
            error["retry_after"] = int(self.retry_after)
        return {"error": error}


def error_body(error: ApiError, request_id: str) -> dict[str, Any]:
    return error.as_body(request_id)


# --------------------------------------------------------------------------- #
#  Network exposure
# --------------------------------------------------------------------------- #

_LOOPBACK_NAMES = frozenset({"localhost", "ip6-localhost", "ip6-loopback"})


def is_loopback(address: Optional[str]) -> bool:
    """True for addresses only this machine can reach.

    Accepts bare IPv6 (``::1``) and the IPv4-mapped form (``::ffff:127.0.0.1``),
    both genuinely loopback and both of which a naive string comparison would
    call "remote" - a false positive that would block a safe configuration.
    """
    if not address:
        return False
    candidate = address.strip().strip("[]").lower()
    if not candidate:
        return False
    if candidate in _LOOPBACK_NAMES:
        return True
    if candidate in {"::1", "0:0:0:0:0:0:0:1", "::ffff:127.0.0.1"}:
        return True
    # Drop a zone index ("fe80::1%eth0") before testing the mapped form.
    candidate = candidate.split("%", 1)[0]
    if candidate.startswith("::ffff:"):
        candidate = candidate[len("::ffff:") :]
    return candidate.startswith("127.")


@dataclass(frozen=True)
class ExposureVerdict:
    """What to do about the address this service is published on."""

    #: True when the configuration is safe as it stands.
    safe: bool
    #: Set when the service should refuse to start. A message, because the whole
    #: point is to be legible in the container log at 3am.
    fatal: Optional[str] = None
    #: Set when it should start but say something loudly on the way up.
    warning: Optional[str] = None


def evaluate_exposure(
    publish_addr: Optional[str],
    *,
    auth_enabled: bool,
    allow_remote: bool = False,
    bind_host: Optional[str] = None,
    in_container: bool = True,
) -> ExposureVerdict:
    """Decide whether the configured exposure is acceptable.

    The subtlety is Docker
    ----------------------
    Inside a container uvicorn *must* bind ``0.0.0.0`` - that is how a
    published port reaches it - so refusing to start on a non-loopback
    ``bind_host`` would make the documented deployment impossible. The address
    that decides exposure is the one on the **host**, which is why compose
    passes its ``TTS_BIND_HOST`` value in as ``TTS_PUBLISH_ADDR``: one variable
    drives both the ``ports:`` mapping and this check, so publishing on
    ``0.0.0.0`` cannot happen without also tripping the guard.

    The rules, in order:

    1. keys configured, or the address is loopback, or ``TTS_ALLOW_REMOTE`` -
       safe;
    2. remote, unauthenticated, ``TTS_ALLOW_REMOTE=true`` - a warning naming the
       consequence, because the operator asked for this explicitly and a
       refusal would only be a speed bump;
    3. remote, unauthenticated, not allowed - refuse to start.
    """
    published = (publish_addr or "127.0.0.1").strip() or "127.0.0.1"

    if auth_enabled:
        warning = None
        if not is_loopback(published):
            warning = (
                f"Published on {published} with API authentication enabled. "
                "Anyone holding a key can synthesise from any uploaded voice - "
                "keep the keys out of shell history and out of the repository."
            )
        return ExposureVerdict(safe=True, warning=warning)

    if is_loopback(published):
        if not in_container and bind_host and not is_loopback(bind_host):
            # Running on the host rather than behind a container: the bind
            # address *is* the exposure, whatever the publish address says.
            return ExposureVerdict(
                safe=False,
                fatal=(
                    f"TTS_BIND_HOST={bind_host} publishes this unauthenticated "
                    "voice-cloning service to the network. Set TTS_API_KEYS, or "
                    "bind 127.0.0.1, or set TTS_ALLOW_REMOTE=true if another "
                    "layer (a reverse proxy or a VPN) is in front of it."
                ),
            )
        return ExposureVerdict(
            safe=True,
            warning=(
                "No API keys configured: /v1/audio/speech and the speaker "
                "endpoints are reachable by anything that can reach this port, "
                "and any of them can clone any voice it is sent. That is fine on "
                "127.0.0.1. Set TTS_API_KEYS before changing the published "
                "address."
            ),
        )

    if allow_remote:
        return ExposureVerdict(
            safe=True,
            warning=(
                f"TTS_ALLOW_REMOTE=true: publishing an unauthenticated "
                f"voice-cloning service on {published}. Anything that can reach "
                "that address can synthesise from any uploaded voice. This is "
                "the operator's call; the startup log will keep saying so."
            ),
        )

    return ExposureVerdict(
        safe=False,
        fatal=(
            f"Refusing to publish on {published} with authentication disabled. "
            "This service clones arbitrary voices, so an unauthenticated "
            "instance is an impersonation tool. Either set TTS_API_KEYS, or "
            "publish on 127.0.0.1, or set TTS_ALLOW_REMOTE=true to accept the "
            "risk explicitly (documented in README section 6)."
        ),
    )
