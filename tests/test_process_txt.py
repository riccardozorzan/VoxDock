"""Long-form client logic: chunking, OOM auto-split, and failure classification.

These are the paths that decide what happens to a user's text file, so they are
worth pinning even though the server is mocked. The behaviour that matters most:
a 503 ``out_of_memory`` halves the chunk, but an ordinary 500 does **not** -
splitting a genuine server bug would corrupt the output ordering.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.process_txt import (  # noqa: E402
    HttpFailure,
    count_words,
    resolve_max_words,
    split_sentences,
    split_text_into_chunks,
)


# --------------------------------------------------------------------------- #
#  Sentence / chunk splitting (spec Task 4.1)
# --------------------------------------------------------------------------- #
def test_clean_text_collapses_whitespace():
    from scripts.process_txt import clean_text

    assert clean_text("Ciao   mondo.\r\n\r\nSeconda  riga.") == (
        "Ciao mondo. Seconda riga."
    )


def test_split_sentences_keeps_punctuation():
    assert split_sentences("Uno! Due? Tre. Quattro") == [
        "Uno!",
        "Due?",
        "Tre.",
        "Quattro",
    ]


def test_split_sentences_ignores_extra_whitespace():
    assert split_sentences("Uno.   \n Due.") == ["Uno.", "Due."]


def test_chunks_respect_the_word_budget(tmp_path):
    source = tmp_path / "in.txt"
    source.write_text(" ".join(f"parola{i}." for i in range(50)), encoding="utf-8")

    chunks = split_text_into_chunks(source, max_words=10)

    assert len(chunks) > 1
    for chunk in chunks:
        assert count_words(chunk) <= 10, chunk


def test_no_content_is_lost(tmp_path):
    """Splitting must never drop or duplicate a sentence."""
    sentences = [f"Frase numero {i} con parole varie." for i in range(20)]
    source = tmp_path / "in.txt"
    source.write_text(" ".join(sentences), encoding="utf-8")

    rejoined = " ".join(split_text_into_chunks(source, max_words=8))

    for sentence in sentences:
        assert sentence in rejoined


def test_an_oversized_sentence_is_never_broken(tmp_path):
    """One long sentence stays whole, even when it exceeds the budget.

    Breaking mid-sentence would wreck the prosody, so the budget is allowed to
    be exceeded instead.
    """
    long_sentence = " ".join(f"parola{i}" for i in range(50)) + "."
    source = tmp_path / "in.txt"
    source.write_text(long_sentence, encoding="utf-8")

    assert split_text_into_chunks(source, max_words=10) == [long_sentence]


def test_empty_input_yields_no_chunks(tmp_path):
    source = tmp_path / "empty.txt"
    source.write_text("   \n\n  ", encoding="utf-8")
    assert split_text_into_chunks(source, max_words=10) == []


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        split_text_into_chunks(tmp_path / "nope.txt", max_words=10)


def test_unicode_is_preserved(tmp_path):
    source = tmp_path / "it.txt"
    source.write_text(
        "Ciao, perché no? Perché sì! Città, università.", encoding="utf-8"
    )
    assert "perché" in " ".join(split_text_into_chunks(source, max_words=2))


# --------------------------------------------------------------------------- #
#  OOM classification
# --------------------------------------------------------------------------- #
def test_503_out_of_memory_is_recognised():
    assert HttpFailure(503, '{"error": "out_of_memory"}').is_out_of_memory is True


def test_oom_is_recognised_from_the_body_text():
    """A proxy that rewrites the body must not defeat the check."""
    assert HttpFailure(500, "CUDA out of memory").is_out_of_memory is True


def test_a_plain_500_is_not_an_oom():
    """A genuine server bug must NOT trigger a split.

    Splitting on a 500 would reorder the output while papering over the bug.
    """
    assert HttpFailure(500, "Internal Server Error").is_out_of_memory is False


def test_a_validation_error_is_not_an_oom():
    assert HttpFailure(422, "reference too long").is_out_of_memory is False


def test_failure_message_keeps_the_status():
    failure = HttpFailure(503, "boom")
    assert "503" in str(failure) and "boom" in str(failure)


# --------------------------------------------------------------------------- #
#  VRAM-aware chunk sizing (--max-words auto)
# --------------------------------------------------------------------------- #
HEALTHY = {
    "model": {"max_generate_length": 128, "audio_patch_seconds": 0.16}
}


def test_explicit_max_words_wins():
    words, reason = resolve_max_words("25", HEALTHY, 7.0)
    assert words == 25
    assert "explicit" in reason


def test_auto_derives_from_patch_arithmetic():
    """128 total - 44 prompt = 84 patches * 0.16 s * 2.5 w/s = 33 words."""
    words, reason = resolve_max_words("auto", HEALTHY, 7.0)
    assert words == 33
    assert "auto" in reason


def test_auto_never_exceeds_the_configured_ceiling():
    """A short reference leaves lots of room, but the ceiling still applies."""
    words, _ = resolve_max_words("auto", HEALTHY, 0.5)
    assert words <= 35


def test_auto_rejects_a_reference_that_leaves_no_room():
    """A reference that fills the whole patch budget must be refused.

    40 s / 0.16 s = 250 prompt patches against a 128-patch budget, so there is
    nothing left to generate into. (At 20 s it is 125 patches, which still
    leaves 3, so the guard correctly does not fire there.)
    """
    with pytest.raises(SystemExit, match="no room left to generate"):
        resolve_max_words("auto", HEALTHY, 40.0)


def test_auto_falls_back_when_the_server_reports_no_patch_size():
    words, reason = resolve_max_words("auto", {"model": {}}, 7.0)
    assert words == 35
    assert "no audio_patch_seconds" in reason


def test_a_tuned_profile_beats_the_estimate(monkeypatch):
    """A measured TTS_CHUNK_MAX_WORDS from best_config.json wins over arithmetic.

    It was measured end to end on this host with this reference, so it is more
    trustworthy than patch arithmetic.
    """
    import scripts.process_txt as process_txt

    monkeypatch.setattr(
        process_txt, "load_best_config", lambda: {"TTS_CHUNK_MAX_WORDS": 28}
    )

    words, reason = process_txt.resolve_max_words("auto", HEALTHY, 7.0)

    assert words == 28
    assert "best_config" in reason


def test_a_bogus_tuned_value_is_ignored(monkeypatch):
    """A non-numeric or non-positive tuned chunk size falls back to arithmetic."""
    import scripts.process_txt as process_txt

    monkeypatch.setattr(
        process_txt, "load_best_config", lambda: {"TTS_CHUNK_MAX_WORDS": "nonsense"}
    )
    words, _ = process_txt.resolve_max_words("auto", HEALTHY, 7.0)
    assert words == 33


# --------------------------------------------------------------------------- #
#  OOM auto-split ordering (v3 Task 5.2)
# --------------------------------------------------------------------------- #
def test_oom_halves_the_chunk_and_preserves_order(monkeypatch, tmp_path):
    """A 10-word chunk that OOMs must be retried as 5 + 5, in that order.

    Order is the thing worth pinning: the parts are concatenated with `-c copy`,
    so a reversed split would produce audio whose sentences are out of sequence
    while every individual request still reports success.
    """
    import scripts.process_txt as process_txt

    attempted: list[str] = []

    def fake_post_chunk(**kwargs):
        text = kwargs["text"]
        attempted.append(text)
        if text == "uno due tre quattro cinque sei sette otto nove dieci":
            raise HttpFailure(503, '{"error": "out_of_memory"}')
        return (
            {
                "audio_seconds": 1.0,
                "generation_seconds": 0.5,
                "peak_vram_mb": 100.0,
            },
            b"RIFF",
        )

    monkeypatch.setattr(process_txt, "post_chunk", fake_post_chunk)
    monkeypatch.setattr(process_txt, "clear_server_cache", lambda *a, **k: True)

    out = tmp_path / "chunk.wav"
    agg, parts, notes = process_txt.synthesize_with_split(
        api_url="http://x",
        reference_audio=None,
        reference_text="r",
        speaker_id="s",
        text="uno due tre quattro cinque sei sette otto nove dieci",
        max_splits=3,
        out_path=out,
        post_kwargs={"timeout": 1.0},
    )

    assert attempted[0].endswith("dieci"), "the full chunk is tried first"
    assert attempted[1:] == [
        "uno due tre quattro cinque",
        "sei sette otto nove dieci",
    ], f"split lost its order: {attempted}"
    assert agg["splits"] == 1
    assert agg["parts"] == 2
    assert notes and "split into 5 + 5" in notes[0]


def test_a_500_is_never_split(monkeypatch, tmp_path):
    """A non-OOM failure must propagate instead of being retried as halves."""
    import scripts.process_txt as process_txt

    calls = {"n": 0}

    def fake_post_chunk(**kwargs):
        calls["n"] += 1
        raise HttpFailure(500, "Internal Server Error")

    monkeypatch.setattr(process_txt, "post_chunk", fake_post_chunk)
    monkeypatch.setattr(process_txt, "clear_server_cache", lambda *a, **k: True)

    with pytest.raises(HttpFailure):
        process_txt.synthesize_with_split(
            api_url="http://x",
            reference_audio=None,
            reference_text="r",
            speaker_id="s",
            text="uno due tre quattro cinque sei",
            max_splits=3,
            out_path=tmp_path / "c.wav",
            post_kwargs={"timeout": 1.0},
        )

    assert calls["n"] == 1, "a 500 must not trigger a split retry"


def test_peak_ceiling_does_not_reduce_the_gain():
    """The gain is never capped to fit the peak - a limiter handles that instead.

    Capping the gain here would silently give up over a dB of loudness on
    speech, whose crest factor is far too high for -16 LUFS and -1.5 dBTP to
    coexist.
    """
    from scripts.process_txt import _linear_offset_db

    measured = {"input_i": -29.10, "input_tp": -13.40}
    # Only 11.9 dB of headroom remains, but the required gain is 13.1 dB.
    assert _linear_offset_db(
        target_lufs=-16.0, measured=measured, tp_ceiling=-1.5
    ) > (-1.5 - (-13.40))


def test_measured_loudness_comes_from_the_summary_block():
    """Only the final ``Summary:`` counts, and the last Peak is the true peak."""
    from scripts.process_txt import _parse_ebur128

    stderr = """
[Parsed_ebur128 @ 0x1] t: 1.0  M: -20.0  S: 100.0
  I:  -20.5 LUFS    LRA: 5.2 LU
  Peak:  -6.10 dBFS
Summary:

  Integrated loudness:
    I:         -18.20 LUFS
    Threshold: -28.90 LUFS

  True peak:
    Peak:       -1.42 dBFS
"""
    parsed = _parse_ebur128(stderr)

    assert parsed == {"input_i": -18.20, "input_tp": -1.42}


def test_silence_is_reported_as_unmeasurable():
    """``-inf`` has no loudness to target, so it must not become a gain."""
    from scripts.process_txt import _parse_ebur128

    silence = "Summary:\n  I:  -inf LUFS\n  True peak:\n    Peak: -inf dBFS\n"
    assert _parse_ebur128(silence) is None
    assert _parse_ebur128("no summary at all") is None
    assert _parse_ebur128("") is None


def test_the_limiter_is_backed_off_below_the_ceiling():
    """`alimiter` caps sample peak, not oversampled true peak.

    Asking it for the -1.5 dBTP ceiling directly still measured a true peak of
    -0.8 dBFS on real output, so the limit is set a margin below the ceiling. If
    this ever changes, the ceiling silently stops being enforced.
    """
    from scripts.process_txt import TRUE_PEAK_MARGIN_DB

    assert TRUE_PEAK_MARGIN_DB > 0, "a zero margin lets true peak exceed the ceiling"


def test_splitting_stops_at_max_depth(monkeypatch, tmp_path):
    """max_splits is a hard bound, not a suggestion."""
    import scripts.process_txt as process_txt

    def always_oom(**kwargs):
        raise HttpFailure(503, '{"error": "out_of_memory"}')

    monkeypatch.setattr(process_txt, "post_chunk", always_oom)
    monkeypatch.setattr(process_txt, "clear_server_cache", lambda *a, **k: True)

    with pytest.raises(HttpFailure):
        process_txt.synthesize_with_split(
            api_url="http://x",
            reference_audio=None,
            reference_text="r",
            speaker_id="s",
            text="uno due tre quattro cinque sei sette otto",
            max_splits=1,
            out_path=tmp_path / "c.wav",
            post_kwargs={"timeout": 1.0},
        )
