"""The web UI asset (``app/static/index.html``).

The page is not exercised by a browser here - that needs a GPU and a real
browser, neither of which the host-side suite has. What *is* worth pinning is
the coupling that silently breaks: the UI hard-codes endpoint paths, so
renaming a route in ``app/main.py`` would leave the page calling a 404 and
nothing would say so. These tests fail on that drift instead.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

UI = Path(__file__).resolve().parent.parent / "app" / "static" / "index.html"


@pytest.fixture(scope="module")
def page() -> str:
    assert UI.exists(), f"UI asset missing: {UI}"
    return UI.read_text(encoding="utf-8")


def test_page_is_a_complete_document(page):
    assert page.lstrip().startswith("<!DOCTYPE html>")
    assert page.rstrip().endswith("</html>")
    assert page.count("<script>") == page.count("</script>") == 1


def test_no_build_placeholders_survived(page):
    """An unreplaced __JS__ marker would ship a page that silently does nothing."""
    leftovers = re.findall(r"__[A-Z_]+__|<!--BODY-->", page)
    assert not leftovers, f"unreplaced placeholders: {leftovers}"


def test_it_calls_endpoints_that_exist(page):
    """The drift guard: every path the page fetches must be a real route."""
    routes = {
        "health", "v1/speakers", "v1/speakers/register", "v1/audio/speech",
        # DELETE /v1/speakers/{id} - the id is appended at runtime, so the
        # literal the page fetches is this prefix.
        "v1/speakers/",
    }
    called = set(re.findall(r'fetch\("([^"?]+)"', page))
    assert called, "no fetch calls found - did the script get dropped?"
    for path in called:
        assert path in routes, f"UI calls {path!r}, which is not a known route"


def test_it_uses_the_json_envelope_rather_than_raw_wav(page):
    """include_metrics gives audio and metrics in one JSON body.

    Without it the response is raw WAV bytes, and the page's
    ``response.json()`` would throw on the first synthesis.
    """
    assert 'body.append("include_metrics", "true")' in page
    assert "audio_base64" in page


def test_every_dom_id_it_references_exists(page):
    """A typo'd getElementById returns null and the handler dies silently."""
    declared = set(re.findall(r'id="([\w-]+)"', page))
    used = set(re.findall(r'\$\("([\w-]+)"\)', page))
    assert used, "no element lookups found"
    assert not (used - declared), f"referenced but never defined: {used - declared}"


def test_it_escapes_dynamic_text_before_touching_inner_html(page):
    """Speaker ids come from disk; unescaped, one could inject markup."""
    assert "function escapeHtml" in page
    assert "escapeHtml(s.id)" in page
