"""Compose must forward every ``TTS_*`` the app can actually read.

``docker-compose.yml`` is the only place that decides what reaches the container:
Compose injects the variables listed under ``environment:`` and nothing else. A
variable that ``app/config.py`` reads but that is missing from that block is
therefore *silently* ignored inside the container - the container simply runs with
the built-in default while the operator wonders why ``.env`` did nothing.

That was a real bug, not a hypothetical: ``.env.example`` advertises
``TTS_API_TITLE`` and ``TTS_API_VERSION`` ("override only when running a private
fork") and both were dropped on the floor, because compose never listed them.

These tests pin the invariant in both directions:

* every ``TTS_*`` ``app/config.py`` reads is forwarded, or is on an explicit
  host-only allow-list that says *why* the container must not see it;
* nothing is forwarded that the app does not read (a dead env var);
* a forwarded default matches the code default, because compose's default always
  wins over ``app/config.py``'s once the container starts.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import APISettings, ModelSettings  # noqa: E402

COMPOSE_FILE = PROJECT_ROOT / "docker-compose.yml"
CONFIG_FILE = PROJECT_ROOT / "app" / "config.py"
ENV_EXAMPLE = PROJECT_ROOT / ".env.example"

#: Variables that ``app/config.py`` reads but that must NOT be forwarded into the
#: container, with the reason. Every entry is checked to be absent from the
#: compose block, so this list cannot quietly excuse a genuine gap.
HOST_ONLY = {
    "TTS_HOST": "compose publishes the port; uvicorn binds 0.0.0.0:8000 itself",
    "TTS_PORT": "consumed by compose interpolation for the port mapping",
    "TTS_BIND_HOST": "consumed by compose for the published port, like TTS_PORT",
    "TTS_DATA_DIR": "/app/data is correct by construction",
    "TTS_PROJECT_ROOT": "/app is correct by construction",
    "TTS_BEST_CONFIG": (
        "a host path does not exist in the container, so load_best_config() would "
        "warn and fall back to the built-in defaults instead of the bind-mounted "
        "/app/data/output/best_config.json"
    ),
}

#: ``VAR: ${VAR:-default}`` inside the ``environment:`` block.
_ENTRY = re.compile(r'^\s+(TTS_[A-Z_]+):\s*"?\$\{\1:-([^}"]*)\}', re.M)
_ENVIRONMENT_BLOCK = re.compile(r"\n    environment:\n(.*?)\n    volumes:", re.S)
#: Variables ``app/config.py`` reads, i.e. every quoted ``TTS_*`` it hands to a
#: ``_str``/``_int``/``_float``/``_bool`` helper.
_READ = re.compile(r'"(TTS_[A-Z_]+)"')


def compose_text() -> str:
    return COMPOSE_FILE.read_text(encoding="utf-8")


def forwarded_vars(text: str) -> dict[str, str]:
    """``{VAR: default}`` for every forwarded ``TTS_*`` in the environment block."""
    block = _ENVIRONMENT_BLOCK.search(text)
    assert block is not None, "docker-compose.yml: no environment block found"
    return dict(_ENTRY.findall(block.group(1)))


def app_read_vars() -> set[str]:
    return set(_READ.findall(CONFIG_FILE.read_text(encoding="utf-8")))


def advertised_vars() -> set[str]:
    """``TTS_*`` the operator is told about, comments included."""
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    return set(re.findall(r"^#?\s*(TTS_[A-Z_]+)=", text, re.M))


def unaccounted_vars(text: str, read_vars: set[str]) -> list[str]:
    """Vars the app reads that compose neither forwards nor HOST_ONLY excuses."""
    return sorted(read_vars - set(forwarded_vars(text)) - set(HOST_ONLY))


def _same(compose_default: str, code_default: object) -> bool:
    """``"900.0"`` and ``900`` are the same default; ``"VoxDock"`` is not ``"VoxDock "``."""
    try:
        return float(compose_default) == float(code_default)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(compose_default) == str(code_default)


# --------------------------------------------------------------------------- #
#  The invariant
# --------------------------------------------------------------------------- #
def test_no_read_variable_is_silently_ignored():
    """The bug this file exists for: a documented setting that never arrives."""
    missing = unaccounted_vars(compose_text(), app_read_vars())

    assert missing == [], (
        "app/config.py reads these but docker-compose.yml does not forward them, "
        f"so setting them in .env does nothing inside the container: {missing}. "
        "Add them to the environment block, or document them in HOST_ONLY."
    )


def test_host_only_exceptions_really_are_not_forwarded():
    """Otherwise the allow-list could hide a regression instead of explaining one."""
    forwarded = set(forwarded_vars(compose_text()))
    forwarded_host_only = sorted(forwarded & set(HOST_ONLY))

    assert forwarded_host_only == [], (
        f"{forwarded_host_only} are listed as host-only but compose forwards them"
    )


def test_nothing_forwarded_is_unknown_to_the_app():
    """A forwarded variable nothing reads is dead config that invites typos."""
    unknown = sorted(set(forwarded_vars(compose_text())) - app_read_vars())

    assert unknown == [], f"docker-compose.yml forwards unread variables: {unknown}"


def test_every_advertised_variable_actually_reaches_the_container():
    """`.env.example` is a promise; anything it names must survive the trip.

    TTS_API_TITLE and TTS_API_VERSION were both advertised and both ignored -
    this is the assertion that would have caught it.
    """
    ignored = sorted(
        advertised_vars() - set(forwarded_vars(compose_text())) - set(HOST_ONLY)
    )

    assert ignored == [], (
        f".env.example advertises {ignored}, which docker-compose.yml never forwards"
    )


# --------------------------------------------------------------------------- #
#  Defaults
# --------------------------------------------------------------------------- #
def test_forwarded_defaults_match_the_code_defaults(clean_env):
    """Compose's default always wins over ``app/config.py`` once the container runs.

    So a drifted default here would override the code without a single warning.
    """
    api = APISettings.from_env()
    model = ModelSettings.from_env()
    forwarded = forwarded_vars(compose_text())

    expected = {
        "TTS_API_TITLE": api.title,
        "TTS_API_VERSION": api.version,
        "TTS_API_DESCRIPTION": api.description,
        "TTS_DEVICE": model.device,
        "TTS_MAX_UPLOAD_MB": api.max_upload_mb,
        "TTS_REQUEST_TIMEOUT": api.request_timeout_s,
        "TTS_CONNECT_TIMEOUT": api.connect_timeout_s,
    }
    drifted = {
        var: (forwarded[var], value)
        for var, value in expected.items()
        if not _same(forwarded[var], value)
    }

    assert drifted == {}, (
        f"compose default != app/config.py default (compose, code): {drifted}"
    )


# --------------------------------------------------------------------------- #
#  The checker itself
# --------------------------------------------------------------------------- #
def test_a_planted_gap_is_detected():
    """A guard that cannot fail is worse than none - feed it a known gap."""
    doctored = re.sub(
        r"^\s+TTS_DEVICE: \$\{TTS_DEVICE:-cuda\}\n", "", compose_text(), flags=re.M
    )

    assert "TTS_DEVICE" in unaccounted_vars(doctored, app_read_vars()), (
        "the gap detector failed to notice a removed line"
    )


@pytest.mark.parametrize("var", ["TTS_HOST", "TTS_PORT", "TTS_BEST_CONFIG"])
def test_host_only_list_documents_the_awkward_ones(var):
    """Removing something from the block is only OK when HOST_ONLY says why."""
    assert var in HOST_ONLY
