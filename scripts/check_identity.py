"""Fail loudly if the project's name drifts between its places.

The identity constants in :mod:`app` (``VoxDock`` / ``voxdock-tts``) are the
source of truth. Docker Compose cannot read Python, so
``docker-compose.yml`` repeats them as inline ``${VAR:-default}`` fallbacks,
and this script is what keeps those copies honest.

Run it in CI, or after renaming anything::

    python scripts/check_identity.py

Exits 0 when consistent, 1 on any drift, printing what disagrees.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import (  # noqa: E402
    COMPOSE_PROJECT,
    CONTAINER_NAME,
    HF_CACHE_VOLUME,
    IMAGE_NAME,
    INDUCTOR_CACHE_VOLUME,
    __title__,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMPOSE_FILE = PROJECT_ROOT / "docker-compose.yml"

#: ``${VAR:-default}`` in docker-compose.yml must default to these.
EXPECTED_COMPOSE_DEFAULTS = {
    "VOXDOCK_COMPOSE_PROJECT": COMPOSE_PROJECT,
    "VOXDOCK_IMAGE": IMAGE_NAME,
    "VOXDOCK_CONTAINER": CONTAINER_NAME,
    "VOXDOCK_HF_CACHE_VOLUME": HF_CACHE_VOLUME,
    "VOXDOCK_INDUCTOR_CACHE_VOLUME": INDUCTOR_CACHE_VOLUME,
}

#: Strings that must no longer appear in the deployment files, left over from
#: the pre-rename `dots-tts` / `local-tts-docker` naming.
STALE_NAMES = ("dots-tts", "local-tts-docker")


def check_compose_defaults(text: str) -> list[str]:
    """Every ``${VAR:-default}`` must carry the value from :mod:`app`."""
    problems: list[str] = []
    found = dict(
        re.findall(r"\$\{(VOXDOCK_[A-Z_]+):-([^}]+)\}", text)
    )
    for var, expected in EXPECTED_COMPOSE_DEFAULTS.items():
        if var not in found:
            problems.append(f"docker-compose.yml: {var} is missing entirely")
        elif found[var] != expected:
            problems.append(
                f"docker-compose.yml: {var} defaults to {found[var]!r} "
                f"but app/__init__.py says {expected!r}"
            )
    for var in set(found) - set(EXPECTED_COMPOSE_DEFAULTS):
        problems.append(f"docker-compose.yml: unknown identity var {var}")
    return problems


def check_no_stale_names(text: str) -> list[str]:
    """No pre-rename identity may survive in the deployment files.

    ``dots_tts`` (underscore) is the upstream Python package and is fine; only
    the hyphenated Docker/project names are stale.
    """
    return [
        f"docker-compose.yml: still references the old name {name!r}"
        for name in STALE_NAMES
        if name in text
    ]


def main() -> int:
    if not COMPOSE_FILE.exists():
        print(f"check_identity: {COMPOSE_FILE} not found", file=sys.stderr)
        return 1
    text = COMPOSE_FILE.read_text(encoding="utf-8")
    problems = check_compose_defaults(text) + check_no_stale_names(text)

    print(f"check_identity: {__title__} ({IMAGE_NAME})")
    print(f"  image      : {IMAGE_NAME}:latest")
    print(f"  container  : {CONTAINER_NAME}")
    print(f"  compose    : {COMPOSE_PROJECT}")
    print(f"  hf cache   : {HF_CACHE_VOLUME}")
    print(f"  inductor   : {INDUCTOR_CACHE_VOLUME}")
    if problems:
        print("\nFAILED:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print("  status     : consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
