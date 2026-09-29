"""The ``test_api.py --assert`` smoke test (todo 4.6).

A smoke test that cannot fail is worse than none, because it looks like
coverage. Every test here feeds ``_assert_response`` a deliberately broken
response and checks that the specific defect is reported, so passing them means
the assertions have teeth rather than always returning 0.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.test_api import _assert_response  # noqa: E402

GOOD_BODY = b"RIFF\x00\x00\x00\x00WAVEfmt "
GOOD_HEADERS = {"X-Sample-Rate": "48000", "X-Peak-Vram-MB": "5451.9"}


class FakeResponse:
    def __init__(self, content=GOOD_BODY, headers=None):
        self.content = content
        self.headers = GOOD_HEADERS if headers is None else headers


def run_assert(*, content=GOOD_BODY, headers=None, duration=3.0) -> int:
    return _assert_response(
        FakeResponse(content, headers), Path("/tmp/unused.wav"), duration, 1.0
    )


# --------------------------------------------------------------------------- #
#  The happy path
# --------------------------------------------------------------------------- #
def test_a_good_response_passes():
    assert run_assert() == 0


# --------------------------------------------------------------------------- #
#  Each defect must be detected
# --------------------------------------------------------------------------- #
def test_an_empty_body_fails():
    assert run_assert(content=b"") == 1


def test_a_non_wav_body_fails():
    """A 200 carrying JSON or an error page must not pass as audio."""
    assert run_assert(content=b'{"error": "oops"}') == 1


def test_a_truncated_render_fails():
    """The failure a status code cannot see: HTTP 200, almost no audio."""
    assert run_assert(duration=0.01) == 1


def test_a_wrong_sample_rate_fails():
    assert run_assert(headers={**GOOD_HEADERS, "X-Sample-Rate": "16000"}) == 1


def test_a_missing_sample_rate_fails():
    assert run_assert(headers={"X-Peak-Vram-MB": "5451.9"}) == 1


def test_peak_vram_over_the_limit_fails():
    """The memory patches exist to hold this line, so it is worth gating on."""
    assert run_assert(headers={**GOOD_HEADERS, "X-Peak-Vram-MB": "5900.0"}) == 1


def test_a_missing_vram_header_fails():
    assert run_assert(headers={"X-Sample-Rate": "48000"}) == 1


def test_a_non_numeric_vram_header_fails():
    assert run_assert(headers={**GOOD_HEADERS, "X-Peak-Vram-MB": "lots"}) == 1


def test_peak_vram_just_under_the_limit_still_passes():
    """Guard against an off-by-one that would fail a legitimate run."""
    assert run_assert(headers={**GOOD_HEADERS, "X-Peak-Vram-MB": "5600.0"}) == 0


# --------------------------------------------------------------------------- #
#  Reporting behaviour
# --------------------------------------------------------------------------- #
def test_every_failure_is_reported_not_just_the_first(capsys):
    """One run should describe the whole picture, not need five runs."""
    run_assert(content=b"", headers={}, duration=0.0)
    err = capsys.readouterr().err
    assert "5 assertion(s) failed" in err
