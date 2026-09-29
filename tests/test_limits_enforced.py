"""Every knob on ``SecuritySettings`` must either do something or say so.

This file exists because of a real defect, not a hypothetical one.
``TTS_MAX_REFERENCE_SECONDS`` and ``TTS_MAX_AUDIO_SECONDS`` were documented in
the README and ``.env.example``, read by ``app/config.py``, forwarded by
compose - and enforced nowhere. An operator setting either of them was told they
were protected and were not.

``tests/test_compose_env.py`` already guards the *config -> container* half of
that problem. This guards the *config -> behaviour* half, which nothing covered.
"""

from __future__ import annotations

import sys
from dataclasses import fields
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import SecuritySettings  # noqa: E402
from app import main  # noqa: E402
from app import security  # noqa: E402

#: Knobs that are read somewhere other than ``app/main.py``'s request path.
#: Each entry says where, so this list cannot quietly excuse a genuine gap.
ENFORCED_ELSEWHERE = {
    "api_keys": "app.security.verify_key, via the require_scope dependency",
    "allow_bearer": "app.security.extract_credential, via the require_scope dependency",
    "publish_addr": "_enforce_exposure, at startup",
    "allow_remote": "_enforce_exposure, at startup",
    "in_container": "_enforce_exposure, at startup - decides what counts as exposed",
    "rate_limit_window_s": "the RateLimiter built in the lifespan",
}

SOURCE_FILES = (Path(main.__file__), Path(security.__file__))

#: Variables the app reads but that compose *derives* from another setting, so
#: there is nothing for an operator to type. Each one must still be explained in
#: .env.example next to the variable it comes from - ``test_derived_knobs_are_
#: explained`` enforces that, because "documented nowhere" is how
#: TTS_PUBLISH_ADDR could have stayed a mystery.
DERIVED_IN_COMPOSE = {
    "TTS_PUBLISH_ADDR": "compose forwards TTS_BIND_HOST under this name (ALIASED "
    "in tests/test_compose_env.py)",
}


def test_derived_knobs_are_explained_beside_their_source():
    """A derived variable is only fair if the operator can find out about it."""
    env_example = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    compose = (PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    for derived, reason in DERIVED_IN_COMPOSE.items():
        source = reason.split("composes forwards ")[-1].split(" ")[0] if "forwards" in reason else ""
        assert derived in compose, (
            f"{derived} is declared derived, but docker-compose.yml never sets "
            f"it - so it would never reach the container at all"
        )
        assert derived in env_example, (
            f"{derived} reaches the container but .env.example never mentions "
            f"it; an operator who trips the exposure guard would have no way to "
            f"find out which setting caused it"
        )
        if source:
            assert source in env_example, (
                f"{derived} is derived from {source}, but {source} is not "
                f"documented in .env.example either"
            )


def knob_names() -> list[str]:
    return [f.name for f in fields(SecuritySettings) if not f.name.startswith("_")]


def test_the_knob_list_is_not_empty():
    """If the dataclass were ever emptied, this file would pass vacuously."""
    assert len(knob_names()) >= 12


@pytest.mark.parametrize("name", knob_names())
def test_every_knob_is_read_somewhere(name):
    """The invariant: a declared setting is a used setting."""
    if name in ENFORCED_ELSEWHERE:
        return
    # Deliberately a substring search across both source files rather than an
    # import-and-inspect: the point is that the *name* appears in the code that
    # acts on it, and a source scan cannot be fooled by a refactor that moves
    # the enforcement behind an indirection.
    hits = [
        path.name
        for path in SOURCE_FILES
        if f".{name}" in path.read_text(encoding="utf-8")
    ]

    assert hits, (
        f"SecuritySettings.{name} is declared and documented but never read by "
        f"app/main.py or app/security.py. Either enforce it, or add it to "
        f"ENFORCED_ELSEWHERE with a reason - an operator who sets a knob that "
        f"does nothing is worse off than one who never saw it."
    )


def test_the_documented_exemptions_are_real_knobs():
    """Stops ENFORCED_ELSEWHERE from accumulating names that do not exist."""
    declared = set(knob_names())

    assert set(ENFORCED_ELSEWHERE) <= declared, (
        f"ENFORCED_ELSEWHERE names settings that no longer exist: "
        f"{sorted(set(ENFORCED_ELSEWHERE) - declared)}"
    )


def test_the_documenting_knobs_are_actually_documented():
    """Every TTS_* the operator can set must appear in .env.example.

    The gap this file guards started as a knob that was documented and did
    nothing; this is the other direction - a knob that does something and that
    nobody was told about.
    """
    import re

    env_example = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    advertised = set(re.findall(r"^#?\s*(TTS_[A-Z_]+)=", env_example, re.M))
    from_env = SecuritySettings.from_env

    import inspect

    source = inspect.getsource(from_env)
    read = set(re.findall(r'"(TTS_[A-Z_]+)"', source))

    undocumented = sorted(read - advertised - set(DERIVED_IN_COMPOSE))
    assert undocumented == [], (
        f"app/config.py reads {undocumented} but .env.example never mentions "
        f"them; an operator cannot configure what they were never told exists. "
        f"(Variables compose derives from another one belong in DERIVED_IN_COMPOSE "
        f"and must be explained in .env.example.)"
    )
