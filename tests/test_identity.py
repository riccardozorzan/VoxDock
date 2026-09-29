"""The identity drift guard (``scripts/check_identity.py``).

``docker-compose.yml`` cannot read Python, so it repeats the identity constants
from ``app/__init__.py`` as inline ``${VAR:-default}`` fallbacks - and, since the
container started running as the host user, it also repeats the Dockerfile's
uid/gid. The guard is what keeps those copies honest, but nothing asserted that
the guard itself still passes on this checkout.

That gap is not hypothetical: adding ``${VOXDOCK_UID:-1000}`` /
``${VOXDOCK_GID:-1000}`` made ``check_identity.py`` exit 1 for every run - its
"unknown identity var" rule only knew ``app/__init__.py``'s constants - and no
test noticed. These tests pin both halves: this checkout must pass, and a wrong
value must still be reported.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.check_identity import (  # noqa: E402
    HOST_USER_DEFAULTS,
    check_compose_defaults,
    check_no_stale_names,
)

COMPOSE_FILE = PROJECT_ROOT / "docker-compose.yml"
DOCKERFILE = PROJECT_ROOT / "Dockerfile"


@pytest.fixture(scope="module")
def compose_text() -> str:
    return COMPOSE_FILE.read_text(encoding="utf-8")


def test_the_real_compose_file_is_consistent(compose_text):
    """The guard must pass on this checkout - that is its entire contract."""
    assert check_compose_defaults(compose_text) == []


def test_no_stale_pre_rename_names_survive(compose_text):
    assert check_no_stale_names(compose_text) == []


@pytest.mark.parametrize("var", sorted(HOST_USER_DEFAULTS))
def test_the_host_user_defaults_match_the_dockerfile(var):
    """uid/gid are baked into ``/etc/passwd`` by the Dockerfile, so they must agree.

    This is not cosmetic drift: the container runs as that uid, and a container
    whose uid has no passwd entry dies at startup with
    ``getpwuid(): uid not found: 1000``.
    """
    declared = re.search(rf"^ARG {var}=(\d+)", DOCKERFILE.read_text("utf-8"), re.M)
    assert declared, f"Dockerfile declares no ARG {var}"
    assert declared.group(1) == HOST_USER_DEFAULTS[var]


def test_an_unknown_identity_var_is_still_rejected():
    """The rule that caught the uid/gid pair in the first place keeps working."""
    problems = check_compose_defaults("image: ${VOXDOCK_MYSTERY:-nope}\n")

    assert any("unknown identity var VOXDOCK_MYSTERY" in p for p in problems)


def test_a_drifted_image_name_is_reported():
    problems = check_compose_defaults("image: ${VOXDOCK_IMAGE:-other}:latest\n")

    assert any("VOXDOCK_IMAGE" in p for p in problems)


def test_a_drifted_uid_default_is_reported():
    problems = check_compose_defaults('user: "${VOXDOCK_UID:-1001}"\n')

    assert any("VOXDOCK_UID" in p for p in problems)


def test_a_missing_uid_default_is_reported():
    problems = check_compose_defaults("image: ${VOXDOCK_IMAGE:-voxdock-tts}:latest\n")

    assert any("VOXDOCK_UID is missing entirely" in p for p in problems)
