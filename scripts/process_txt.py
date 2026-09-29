#!/usr/bin/env python3
"""Long-form batch synthesis client (spec Phase 4).

Workflow
--------
1. ``split_text_into_chunks()`` – split a UTF-8 ``.txt`` file on sentence
   punctuation (``(?<=[.!?])\\s+``) and pack whole sentences into word-bounded
   chunks (spec Task 4.1).
2. POST every chunk to ``/v1/audio/speech`` with the reference WAV + its exact
   transcript, storing ``data/temp_chunks/chunk_001.wav`` … (spec Task 4.2),
   with retries, progress output and a resumable cache.
3. Concatenate losslessly with FFmpeg's concat demuxer and delete the chunk
   files (spec Task 4.3).

Chunk sizing
------------
The model caps **total** audio patches (reference prompt + generated) per
request (``TTS_MAX_GENERATE_LENGTH``), which is what keeps peak VRAM under the
5.5 GB budget. ``--max-words auto`` (the default) therefore asks the live server
how much generation room is left after the reference prompt and derives the word
budget from it. Pass an explicit integer to override.

Example
-------
    docker compose exec tts python scripts/process_txt.py \
        --input data/input/test_audio_sample.txt \
        --reference-audio data/input/reference_clean.wav \
        --reference-text data/input/reference_clean.txt
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import load_best_config, settings  # noqa: E402
from app.utils import now_iso, wav_duration_seconds  # noqa: E402

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
WHITESPACE_RE = re.compile(r"\s+")


# --------------------------------------------------------------------------- #
#  Text chunking (spec Task 4.1)
# --------------------------------------------------------------------------- #
def clean_text(text: str) -> str:
    """Collapse whitespace while keeping paragraph content intact."""
    return WHITESPACE_RE.sub(" ", (text or "").replace("\r\n", "\n")).strip()


def split_sentences(text: str) -> list[str]:
    """Split on ``.``, ``!``, ``?`` while keeping the punctuation attached."""
    return [part.strip() for part in SENTENCE_SPLIT_RE.split(text) if part.strip()]


def split_text_into_chunks(file_path: str | Path, max_words: int = 180) -> list[str]:
    """Group whole sentences into chunks of at most ``max_words`` words.

    ``max_words`` follows the spec signature (180). A single sentence longer than
    the budget is emitted on its own so that no sentence is ever broken.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"input text file not found: {path}")

    text = clean_text(path.read_text(encoding="utf-8"))
    if not text:
        return []

    budget = max(1, int(max_words))
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for sentence in split_sentences(text):
        sentence_words = len(sentence.split())
        if current and current_words + sentence_words > budget:
            chunks.append(" ".join(current))
            current = []
            current_words = 0
        current.append(sentence)
        current_words += sentence_words

    if current:
        chunks.append(" ".join(current))
    return chunks


def count_words(text: str) -> int:
    return len(text.split())


# CHUNK-PROCESS-CONTINUE
def server_health(api_url: str, timeout: float) -> dict:
    """``GET /health``; tolerates the 503 returned while the model loads."""
    import requests

    url = f"{api_url.rstrip('/')}{settings.api.health_path}"
    try:
        response = requests.get(url, timeout=timeout)
    except Exception as exc:
        raise SystemExit(
            f"[process_txt] cannot reach the TTS server at {url}: {exc}"
        ) from exc
    try:
        return response.json()
    except Exception as exc:
        raise SystemExit(
            f"[process_txt] {url} returned a non-JSON payload "
            f"(HTTP {response.status_code}): {exc}"
        ) from exc


def wait_until_ready(api_url: str, timeout: float, wait_seconds: float) -> dict:
    """Optionally block until ``/health`` reports ``ready``."""
    deadline = time.time() + max(0.0, wait_seconds)
    health = server_health(api_url, timeout)
    while not health.get("ready") and time.time() < deadline:
        print(
            f"  waiting for engine ({health.get('status')}) ... "
            f"{int(deadline - time.time())}s left"
        )
        time.sleep(10)
        health = server_health(api_url, timeout)
    return health


def resolve_max_words(
    requested: Optional[str], health: dict, reference_seconds: float
) -> tuple[int, str]:
    """Turn ``auto`` into a VRAM-aware word budget for one chunk."""
    cfg = settings.processing
    if requested is not None and str(requested).lower() != "auto":
        return int(requested), f"explicit --max-words={requested}"

    # A tuned profile may carry a measured chunk size (v3 Task 5.1). It wins
    # over the VRAM-derived estimate because it was measured end to end on this
    # host with this reference, not derived from patch arithmetic.
    tuned = load_best_config().get("TTS_CHUNK_MAX_WORDS")
    if tuned is not None:
        try:
            tuned_words = int(float(tuned))
        except (TypeError, ValueError):
            tuned_words = 0
        if tuned_words > 0:
            return tuned_words, (
                f"best_config.json TTS_CHUNK_MAX_WORDS={tuned_words} "
                "(measured by scripts/autotune.py)"
            )

    model = (health or {}).get("model") or {}
    total_patches = int(
        model.get("max_generate_length") or settings.model.max_generate_length
    )
    patch_seconds = model.get("audio_patch_seconds")
    if not patch_seconds:
        return (
            cfg.chunk_max_words,
            "server reported no audio_patch_seconds -> falling back to "
            f"{cfg.chunk_max_words} words",
        )

    patch_seconds = float(patch_seconds)
    prompt_patches = int(math.ceil(reference_seconds / patch_seconds))
    available_patches = total_patches - prompt_patches
    if available_patches <= 1:
        raise SystemExit(
            f"[process_txt] the reference prompt needs {prompt_patches} of the "
            f"{total_patches} available audio patches: no room left to generate.\n"
            "             Use a shorter reference clip or raise TTS_MAX_GENERATE_LENGTH."
        )

    available_seconds = available_patches * patch_seconds
    words = int(available_seconds * cfg.words_per_second)
    words = max(cfg.chunk_min_words, min(cfg.chunk_max_words, words))
    reason = (
        f"auto: {total_patches} total patches - {prompt_patches} prompt patches = "
        f"{available_patches} patches ({available_seconds:.1f}s generation) -> "
        f"{words} words/chunk"
    )
    return words, reason


def build_silence(path: Path, seconds: float, sample_rate: int) -> Path:
    """PCM_16 mono silence so that FFmpeg's ``-c copy`` stays format-consistent."""
    frames = max(1, int(round(seconds * sample_rate)))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * frames)
    return path


def _header_float(headers, name: str) -> Optional[float]:
    raw = headers.get(name)
    try:
        return float(raw) if raw not in (None, "") else None
    except (TypeError, ValueError):
        return None


class HttpFailure(RuntimeError):
    """A non-200 response, tagged so OOM can be told apart from other errors."""

    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f"HTTP {status_code}: {body}")
        self.status_code = status_code
        self.body = body

    @property
    def is_out_of_memory(self) -> bool:
        """True when the server flagged an OOM worth splitting a chunk for.

        The service answers 503 with ``error=out_of_memory``; the substring
        check is a fallback for a proxy that rewrites the body.
        """
        if self.status_code == 503 and "out_of_memory" in self.body:
            return True
        return "out of memory" in self.body.lower()


def post_chunk(
    *,
    api_url: str,
    reference_audio: Optional[Path],
    reference_text: Optional[str],
    speaker_id: Optional[str],
    text: str,
    num_steps: Optional[int],
    guidance_scale: Optional[float],
    seed: Optional[int],
    language: Optional[str],
    normalize_text: Optional[bool],
    timeout: float,
    retries: int,
    backoff: float,
) -> tuple[dict, bytes]:
    """POST a single chunk with retries; returns ``(metrics, wav_bytes)``.

    Sends ``speaker_id`` when the voice was registered and only uploads the
    reference WAV otherwise (v3 Task 5.2), so a 40-chunk job uploads 656 KB
    once instead of 40 times.
    """
    import requests

    url = f"{api_url.rstrip('/')}{settings.api.speech_path}"
    form: dict[str, str] = {"text": text}
    files = None
    if speaker_id:
        form["speaker_id"] = speaker_id
    else:
        form["reference_text"] = reference_text or ""
        files = {
            "reference_audio": (
                reference_audio.name,
                reference_audio.read_bytes(),
                "audio/wav",
            )
        }
    if num_steps is not None:
        form["num_steps"] = str(num_steps)
    if guidance_scale is not None:
        form["guidance_scale"] = str(guidance_scale)
    if seed is not None:
        form["seed"] = str(seed)
    if language is not None:
        form["language"] = str(language)
    if normalize_text is not None:
        form["normalize_text"] = "true" if normalize_text else "false"

    last_error: Optional[Exception] = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            response = requests.post(
                url,
                files=files,
                data=form,
                timeout=timeout,
            )
            if response.status_code != 200:
                raise HttpFailure(response.status_code, response.text[:300])
            if not response.content:
                raise RuntimeError("server returned an empty audio body")

            headers = response.headers
            metrics = {
                "request_id": headers.get("X-Request-Id"),
                "audio_seconds": _header_float(headers, "X-Audio-Seconds"),
                "generation_seconds": _header_float(headers, "X-Generation-Seconds"),
                "rtf": _header_float(headers, "X-RTF"),
                "engine_rtf": _header_float(headers, "X-Engine-RTF"),
                "speed_factor": _header_float(headers, "X-Speed-Factor"),
                "peak_vram_mb": _header_float(headers, "X-Peak-Vram-MB"),
                "http_status": response.status_code,
                "attempts": attempt,
            }
            return metrics, response.content
        except Exception as exc:  # noqa: BLE001 - retry on any transient failure
            last_error = exc
            # An OOM will not fix itself on a retry; the caller halves the chunk.
            if isinstance(exc, HttpFailure) and exc.is_out_of_memory:
                raise
            if attempt < max(1, retries):
                wait = backoff * attempt
                print(
                    f"      ! attempt {attempt}/{retries} failed ({exc}); "
                    f"retrying in {wait:.1f}s",
                    file=sys.stderr,
                )
                time.sleep(wait)

    raise RuntimeError(f"chunk failed after {retries} attempt(s): {last_error}")



def synthesize_with_split(
    *,
    api_url: str,
    reference_audio: Optional[Path],
    reference_text: Optional[str],
    speaker_id: Optional[str],
    text: str,
    max_splits: int,
    out_path: Path,
    post_kwargs: dict,
) -> tuple[dict, list[Path], list[str]]:
    """Synthesise ``text``, halving it on OOM until it fits (v3 Task 5.2).

    Returns ``(aggregate_metrics, part_paths, notes)``. Each generated part goes
    to its own WAV so the final stitch is unchanged. Splitting stops at four
    words, below which there is nothing left to halve.
    """
    notes: list[str] = []
    parts: list[Path] = []
    agg: dict = {
        "audio_seconds": 0.0,
        "generation_seconds": 0.0,
        "rtf": None,
        "peak_vram_mb": 0.0,
        "parts": 0,
        "splits": 0,
    }
    queue: list[str] = [text]
    depth = 0
    timeout = float(post_kwargs.get("timeout", 120.0))

    while queue:
        piece = queue.pop(0)
        # The first part of an unsplit chunk keeps the caller's filename; any
        # part after a split is numbered so the order is obvious on disk.
        target = out_path if not parts and not queue else out_path.with_name(
            f"{out_path.stem}_part{len(parts):03d}{out_path.suffix}"
        )
        try:
            metrics, payload = post_chunk(
                api_url=api_url,
                reference_audio=reference_audio,
                reference_text=reference_text,
                speaker_id=speaker_id,
                text=piece,
                **post_kwargs,
            )
        except HttpFailure as exc:
            words = piece.split()
            if not exc.is_out_of_memory or depth >= max_splits or len(words) < 4:
                raise
            depth += 1
            middle = len(words) // 2
            head, tail = " ".join(words[:middle]), " ".join(words[middle:])
            notes.append(
                f"OOM at {len(words)} words -> split into {middle} + "
                f"{len(words) - middle} words (depth {depth})"
            )
            print(
                f"      ! out of memory on {len(words)} words; splitting into "
                f"{middle} + {len(words) - middle} and retrying "
                f"(depth {depth}/{max_splits})",
                file=sys.stderr,
            )
            if not clear_server_cache(api_url, timeout):
                notes.append("cache clear request failed")
            # Keep order: head then tail, both ahead of anything still queued.
            queue.insert(0, tail)
            queue.insert(0, head)
            continue

        target.write_bytes(payload)
        parts.append(target)
        agg["parts"] += 1
        agg["audio_seconds"] += float(metrics.get("audio_seconds") or 0.0)
        agg["generation_seconds"] += float(metrics.get("generation_seconds") or 0.0)
        agg["peak_vram_mb"] = max(
            float(agg["peak_vram_mb"]), float(metrics.get("peak_vram_mb") or 0.0)
        )

    if agg["audio_seconds"] > 0 and agg["generation_seconds"] > 0:
        agg["rtf"] = round(agg["generation_seconds"] / agg["audio_seconds"], 4)
    agg["splits"] = depth
    return agg, parts, notes


def register_speaker(
    *,
    api_url: str,
    reference_audio: Path,
    reference_text: str,
    speaker_id: str,
    timeout: float,
) -> str:
    """Register the voice once, return its id, or ``""`` to fall back (Task 5.2).

    A registration failure is never fatal: the caller then uploads the reference
    with every chunk exactly as before.
    """
    import requests

    url = f"{api_url.rstrip('/')}/v1/speakers/register"
    try:
        response = requests.post(
            url,
            files={
                "reference_audio": (
                    reference_audio.name,
                    reference_audio.read_bytes(),
                    "audio/wav",
                )
            },
            data={
                "reference_text": reference_text,
                "speaker_id": speaker_id,
                "overwrite": "true",
            },
            timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001
        print(
            f"[process_txt] speaker registration failed ({exc}); falling back "
            "to uploading the reference with every chunk",
            file=sys.stderr,
        )
        return ""
    if response.status_code not in (200, 201):
        print(
            f"[process_txt] registration returned HTTP {response.status_code}: "
            f"{response.text[:200]}; falling back to per-chunk upload",
            file=sys.stderr,
        )
        return ""
    try:
        payload = response.json()
    except ValueError:
        return ""
    registered = str(payload.get("id") or "")
    if registered:
        print(
            f"[process_txt] speaker registered as {registered!r} "
            f"({payload.get('audio_bytes', 0) / 1024:.0f} KB, "
            f"{payload.get('duration_seconds', 0):.1f}s) - reused for every chunk"
        )
    return registered


def clear_server_cache(api_url: str, timeout: float) -> bool:
    """Ask the service to drop prompt latents and free cached CUDA blocks."""
    import requests

    try:
        response = requests.post(
            f"{api_url.rstrip('/')}/v1/admin/clear-cache", timeout=timeout
        )
    except Exception:  # noqa: BLE001 - best effort
        return False
    return response.status_code == 200


def normalize_loudness(
    source: Path,
    destination: Path,
    *,
    lufs: float = -16.0,
    tp: float = -1.5,
    lra: float = 11.0,
) -> Optional[list[str]]:
    """EBU R128 normalisation (v3 Task 5.3); ``None`` when ffmpeg is missing.

    A second pass over the stitched file, so every chunk of a long-form job ends
    up at one consistent level.

    **Two-pass, and peak-aware.** ffmpeg's ``loudnorm`` in single-pass mode has
    to guess the integrated loudness before it has analysed anything, so it
    normalises against an estimate. The fix is to measure first (``-f null -``)
    and then apply a computed linear gain in a second real pass.

    Two subtleties, both found by measuring the real output rather than trusting
    the filter's own numbers:

    * ffmpeg's ``target_offset`` must **not** be applied directly. It is derived
      from a *predicted* output, so using it double-counts the gain and lands on
      roughly half the needed correction. The offset that closes the loop is
      simply ``target - measured``.
    * the gain is capped so the true peak never exceeds ``tp``. On a file that
      already sits near the ceiling, the loudness target and the peak ceiling
      can be mutually exclusive; staying under the ceiling wins, and the
      residual loudness error is reported rather than hidden.

    Digital silence measures as ``-inf`` and is rejected by
    :func:`_parse_ebur128`, so the single-pass fallback covers it.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print(
            "[process_txt] ffmpeg not found - skipping EBU R128 normalisation",
            file=sys.stderr,
        )
        return None

    # ---- pass 1: measure ---------------------------------------------------- #
    # Measured with `ebur128`, NOT with `loudnorm ... print_format=json`.
    # The loudnorm probe reports a value derived from its own internal gating
    # pass, which on this material (7 short chunks separated by 300 ms of
    # silence) reads ~1.1 dB low - enough that a "correct" arithmetic offset
    # still leaves the output a full dB under target. ebur128 is a plain meter
    # with no gain applied, so target - measured closes the loop exactly.
    probe = subprocess.run(
        [
            ffmpeg, "-hide_banner", "-nostats",
            "-i", str(source),
            "-af", "ebur128=peak=true",
            "-f", "null", "-",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    measured = _parse_ebur128(probe.stderr)
    if measured is None:
        # Loudness is already right, or ffmpeg could not measure it; a plain
        # single-pass is still better than shipping the file untouched.
        print(
            "[process_txt] could not read the measured loudness; falling back "
            "to single-pass normalisation",
            file=sys.stderr,
        )
        command = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(source),
            "-af", f"loudnorm=I={lufs}:LRA={lra}:TP={tp}",
            "-ar", "48000",
            "-ac", "1",
            str(destination),
        ]
        subprocess.run(command, check=True)
        return command

    offset = _linear_offset_db(target_lufs=lufs, measured=measured, tp_ceiling=tp)
    if abs(offset) < 0.05:
        # Already within a tenth of a dB; a copy is bit-exact and avoids a
        # pointless re-encode.
        print(
            f"[process_txt] loudness already on target "
            f"(I={measured['input_i']} LUFS) - copying without a gain pass"
        )
        shutil.copyfile(source, destination)
        return [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(source), "-ar", "48000", "-ac", "1", str(destination),
        ]

    headroom = tp - measured["input_tp"]
    if offset > headroom:
        # The two targets genuinely conflict for this material. Rather than
        # give up on the loudness target, apply the full linear gain and let a
        # real limiter catch the handful of peaks that would otherwise exceed
        # the ceiling. `alimiter` is the right tool: it only touches samples
        # above the threshold, so the integrated loudness is essentially
        # unaffected while the peak is held. Capping the *gain* instead (an
        # earlier attempt) silently gave up ~1.2 LU of loudness to satisfy the
        # peak, which is a much worse trade for speech.
        print(
            f"[process_txt] true-peak limited: reaching {lufs} LUFS needs "
            f"{offset:+.2f} dB but only {headroom:+.2f} dB is available before "
            f"true peak passes {tp} dBTP (crest factor "
            f"{measured['input_tp'] - measured['input_i']:.1f} dB). Applying the "
            f"full gain and limiting peaks at {tp} dBTP.",
            file=sys.stderr,
        )
        filter_chain = (
            f"volume={offset:.2f}dB,"
            f"alimiter=limit={tp - TRUE_PEAK_MARGIN_DB:.2f}dB:level=disabled"
        )
    else:
        filter_chain = f"volume={offset:.2f}dB"

    print(
        f"[process_txt] loudness measured I={measured['input_i']} LUFS "
        f"(true peak {measured['input_tp']} dBTP) -> "
        f"applying {offset:+.2f} dB to reach {lufs} LUFS"
    )
    command = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(source),
        "-af", filter_chain,
        "-ar", "48000",
        "-ac", "1",
        str(destination),
    ]
    subprocess.run(command, check=True)
    return command


def _linear_offset_db(
    *, target_lufs: float, measured: dict[str, float], tp_ceiling: float
) -> Optional[float]:
    """Linear gain in dB that moves the measured loudness onto the target.

    Deliberately arithmetic (``target - measured``) rather than ffmpeg's
    ``target_offset``: that field is computed against a *predicted* render, so
    applying it directly undershoots by roughly half the needed correction.

    The peak ceiling is **not** applied here. When the two targets conflict -
    and for speech with a ~15 dB crest factor at -16 LUFS they always do - the
    caller gains first and limits afterwards, which keeps the loudness target
    instead of trading it away for the peak.
    """
    return target_lufs - measured["input_i"]


#: EBU R128 integrated-loudness target and true-peak ceiling for the stitch.
DEFAULT_LUFS = -16.0
DEFAULT_TRUE_PEAK = -1.5

#: `alimiter` limits *sample* peak, not oversampled true peak. Inter-sample
#: overshoot on this material runs ~1 dB above the sample peak, so asking the
#: limiter for the ceiling directly still produced a true peak of -0.8 dBFS
#: (measured). Backing the limit off by this margin is what actually holds the
#: ceiling: at -2.5 dB the same file measures a true peak of -1.80 dBFS.
TRUE_PEAK_MARGIN_DB = 1.0


def _parse_ebur128(stderr: str) -> Optional[dict[str, float]]:
    """Extract the final integrated loudness and true peak from ``ebur128``.

    ``ebur128`` prints a running per-frame block and then a ``Summary:``; only
    the summary is wanted, and of the repeated ``Peak:`` lines the last is the
    true peak. Returns ``None`` when the summary is missing or the file is
    digital silence (``-inf``), which the caller treats as "cannot measure".
    """
    marker = "Summary:"
    start = stderr.rfind(marker)
    if start == -1:
        return None
    summary = stderr[start:]

    def _number(pattern: str, text: str) -> Optional[float]:
        match = re.search(pattern, text)
        if not match:
            return None
        try:
            value = float(match.group(1))
        except ValueError:
            return None
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return value

    integrated = _number(r"I:\s*(-?[\d.]+|-inf)\s*LUFS", summary)
    peaks = re.findall(r"Peak:\s*(-?[\d.]+)\s*dBFS", summary)
    if integrated is None or not peaks:
        return None
    return {"input_i": integrated, "input_tp": float(peaks[-1])}


def concat_chunks(
    chunk_paths: list[Path],
    output_path: Path,
    file_list_path: Path,
    *,
    gap_seconds: float,
    sample_rate: int,
) -> list[str]:
    """Lossless FFmpeg concat-demuxer stitch (spec Task 4.3)."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("[process_txt] ffmpeg was not found on PATH")

    entries: list[Path] = []
    silence: Optional[Path] = None
    if gap_seconds > 0 and len(chunk_paths) > 1:
        silence = build_silence(
            file_list_path.parent / "_gap_silence.wav", gap_seconds, sample_rate
        )
    for index, chunk in enumerate(chunk_paths):
        entries.append(chunk)
        if silence is not None and index < len(chunk_paths) - 1:
            entries.append(silence)

    file_list_path.parent.mkdir(parents=True, exist_ok=True)
    file_list_path.write_text(
        "\n".join(f"file '{path.as_posix()}'" for path in entries) + "\n",
        encoding="utf-8",
    )

    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(file_list_path),
        "-c",
        "copy",
        str(output_path),
    ]
    subprocess.run(command, check=True)
    if silence is not None:
        silence.unlink(missing_ok=True)
    return command


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Chunk a .txt file, synthesise every chunk through the local dots.tts "
            "API and stitch the WAVs with FFmpeg."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", required=True, help="Input UTF-8 .txt file.")
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Final WAV path (default: data/output/<input-stem>.wav).",
    )
    parser.add_argument(
        "--reference-audio",
        default=str(settings.paths.clean_reference_wav),
        help="Cleaned reference WAV produced by scripts/clean_audio.py.",
    )
    parser.add_argument(
        "--reference-text",
        default=None,
        help=(
            "Transcript of the reference audio. Defaults to a sibling .txt file, "
            "then to data/input/reference_clean.txt."
        ),
    )
    parser.add_argument(
        "--max-words",
        default="auto",
        help=(
            "'auto' (VRAM-aware, derived from /health) or a fixed integer word "
            "budget per chunk."
        ),
    )
    parser.add_argument("--api-url", default=settings.api.api_url, help="Server base URL.")
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Skip the EBU R128 (-16 LUFS) pass over the stitched file.",
    )
    parser.add_argument(
        "--lufs",
        type=float,
        default=-16.0,
        help="Integrated loudness target for the normalisation pass.",
    )
    parser.add_argument(
        "--max-splits",
        type=int,
        default=3,
        help="How many times a chunk may be halved after an OOM (Task 5.2).",
    )
    parser.add_argument(
        "--speaker-id",
        default=None,
        help="Name to register the reference voice under (default: input stem).",
    )
    parser.add_argument(
        "--no-speaker-cache",
        action="store_true",
        help=(
            "Upload the reference WAV with every chunk instead of registering "
            "it once and reusing the id."
        ),
    )
    parser.add_argument("--num-steps", type=int, default=None, help="Omit to use the artifact default.")
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--language", default=None, help="e.g. IT, EN or 'auto_detect'.")
    parser.add_argument("--normalize-text", action="store_true", default=False)
    parser.add_argument("--timeout", type=float, default=settings.api.request_timeout_s)
    parser.add_argument("--retries", type=int, default=settings.processing.max_retries)
    parser.add_argument("--backoff", type=float, default=settings.processing.retry_backoff_s)
    parser.add_argument(
        "--gap",
        type=float,
        default=settings.processing.concat_gap_seconds,
        help="Silence inserted between chunks during concatenation.",
    )
    parser.add_argument("--chunks-dir", default=str(settings.paths.temp_chunks_dir))
    parser.add_argument("--file-list", default=None, help="FFmpeg concat list path.")
    parser.add_argument("--keep-chunks", action="store_true", help="Do not delete chunk WAVs.")
    parser.add_argument("--resume", action="store_true", help="Reuse chunk WAVs already on disk.")
    parser.add_argument("--no-concat", action="store_true", help="Synthesise only; skip stitching.")
    parser.add_argument(
        "--wait-ready",
        type=float,
        default=0.0,
        help="Seconds to wait for the engine to finish loading before starting.",
    )
    parser.add_argument("--report", default=None, help="JSON report path.")
    return parser


def resolve_reference_text(args: argparse.Namespace, reference_audio: Path) -> Path:
    """Locate the transcript that matches the reference audio."""
    candidates = []
    if args.reference_text:
        candidates.append(Path(args.reference_text).expanduser())
    candidates.append(reference_audio.with_suffix(".txt"))
    candidates.append(settings.paths.clean_reference_txt)
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    raise SystemExit(
        "[process_txt] no reference transcript found. Tried:\n"
        + "\n".join(f"  - {c}" for c in candidates)
        + "\nThe reference text must match the audio exactly (continuation cloning)."
    )



def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = settings.paths.ensure()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"[process_txt] input file not found: {input_path}")

    reference_audio = Path(args.reference_audio).expanduser().resolve()
    if not reference_audio.exists():
        raise SystemExit(
            f"[process_txt] reference audio not found: {reference_audio}\n"
            "             Run scripts/clean_audio.py first, for example:\n"
            "             python scripts/clean_audio.py "
            "-i data/references/my_voice_v2.wav --trim-to 7.0"
        )

    reference_text_path = resolve_reference_text(args, reference_audio)
    reference_text = reference_text_path.read_text(encoding="utf-8").strip()
    if not reference_text:
        raise SystemExit(
            f"[process_txt] reference transcript is empty: {reference_text_path}"
        )

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else paths.output_dir / f"{input_path.stem}.wav"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    chunks_dir = Path(args.chunks_dir).expanduser().resolve()
    chunks_dir.mkdir(parents=True, exist_ok=True)
    file_list_path = (
        Path(args.file_list).expanduser().resolve()
        if args.file_list
        else chunks_dir / "file_list.txt"
    )

    reference_seconds = wav_duration_seconds(reference_audio)
    health = wait_until_ready(args.api_url, min(args.timeout, 30.0), args.wait_ready)
    ready = bool(health.get("ready"))
    max_words, sizing_reason = resolve_max_words(
        args.max_words, health, reference_seconds
    )

    model_info = health.get("model") or {}
    sample_rate = int(model_info.get("sample_rate") or 48_000)

    chunks = split_text_into_chunks(input_path, max_words)
    if not chunks:
        raise SystemExit(f"[process_txt] no usable text found in {input_path}")
    total_words = sum(count_words(chunk) for chunk in chunks)

    print("=" * 74)
    print("  dots.tts long-form batch synthesis")
    print("=" * 74)
    print(f"  input            : {input_path}")
    print(f"  output           : {output_path}")
    print(f"  reference audio  : {reference_audio} ({reference_seconds:.2f}s)")
    print(
        f"  reference text   : {reference_text_path} "
        f"({count_words(reference_text)} words)"
    )
    print(f"  engine ready     : {ready} (status={health.get('status')})")
    print(f"  chunk sizing     : {max_words} words/chunk")
    print(f"                     {sizing_reason}")
    print(f"  chunks / words   : {len(chunks)} / {total_words}")
    print(f"  retries / gap    : {args.retries} / {args.gap}s")
    print("=" * 74)
    if not ready:
        print(
            "  ! the engine is still loading; requests will fail with HTTP 503",
            file=sys.stderr,
        )

    results: list[dict] = []
    chunk_paths: list[Path] = []
    failed = 0
    generated_audio_seconds = 0.0
    generated_words = 0
    job_started = time.time()

    # Register the voice once so each chunk sends a ~20 byte id instead of a
    # 656 KB WAV (v3 Task 5.2). Falls back silently to per-chunk upload.
    speaker_id = ""
    if not args.no_speaker_cache:
        speaker_id = register_speaker(
            api_url=args.api_url,
            reference_audio=reference_audio,
            reference_text=reference_text,
            speaker_id=args.speaker_id or input_path.stem,
            timeout=args.timeout,
        )

    post_kwargs = {
        "num_steps": args.num_steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "language": args.language,
        "normalize_text": args.normalize_text or None,
        "timeout": args.timeout,
        "retries": args.retries,
        "backoff": args.backoff,
    }

    for index, chunk_text in enumerate(chunks, start=1):
        chunk_path = chunks_dir / f"chunk_{index:03d}.wav"
        chunk_paths.append(chunk_path)
        words = count_words(chunk_text)

        if args.resume and chunk_path.exists() and chunk_path.stat().st_size > 0:
            duration = wav_duration_seconds(chunk_path)
            generated_audio_seconds += duration
            generated_words += words
            results.append(
                {
                    "index": index,
                    "words": words,
                    "characters": len(chunk_text),
                    "audio_seconds": round(duration, 3),
                    "cached": True,
                    "path": str(chunk_path),
                }
            )
            print(
                f"  [{index:03d}/{len(chunks):03d}] cached  {words:3d} words "
                f"-> {duration:6.2f}s audio"
            )
            continue

        chunk_started = time.time()
        try:
            agg, part_paths, notes = synthesize_with_split(
                api_url=args.api_url,
                reference_audio=reference_audio,
                reference_text=reference_text,
                speaker_id=speaker_id or None,
                text=chunk_text,
                max_splits=args.max_splits,
                out_path=chunk_path,
                post_kwargs=post_kwargs,
            )
        except Exception as exc:  # noqa: BLE001 - report and continue with the rest
            failed += 1
            print(f"  [{index:03d}/{len(chunks):03d}] FAILED  {exc}", file=sys.stderr)
            results.append({"index": index, "words": words, "error": str(exc)})
            continue

        # A chunk that was split produced several parts; the final stitch must
        # include all of them, in order.
        if len(part_paths) > 1:
            chunk_paths[-1:] = list(part_paths)
        wall_seconds = time.time() - chunk_started
        audio_seconds = float(
            agg.get("audio_seconds") or wav_duration_seconds(chunk_path)
        )
        generated_audio_seconds += audio_seconds
        generated_words += words
        metrics = {
            "audio_seconds": round(audio_seconds, 3),
            "generation_seconds": round(float(agg["generation_seconds"]), 3),
            "rtf": agg.get("rtf"),
            "peak_vram_mb": round(float(agg["peak_vram_mb"]), 1)
            if agg["peak_vram_mb"]
            else None,
            "parts": agg["parts"],
            "splits": agg["splits"],
        }

        rtf = metrics.get("rtf")
        eta_text = ""
        if rtf and generated_words:
            seconds_per_word = generated_audio_seconds / generated_words
            remaining_words = max(0, total_words - generated_words)
            eta_text = (
                f"  eta {remaining_words * seconds_per_word * float(rtf):6.1f}s"
            )

        print(
            f"  [{index:03d}/{len(chunks):03d}] {words:3d} words -> "
            f"{audio_seconds:6.2f}s audio in {wall_seconds:6.2f}s"
            + (f" (rtf {float(rtf):.3f}" if rtf else " (rtf n/a")
            + (
                f", peak {float(metrics['peak_vram_mb']):.0f} MB){eta_text}"
                if metrics.get("peak_vram_mb")
                else "){}".format(eta_text)
            )
            + (
                f"  [split into {metrics['parts']} parts after OOM]"
                if metrics.get("splits")
                else ""
            )
        )
        results.append(
            {
                "index": index,
                "words": words,
                "characters": len(chunk_text),
                "path": str(chunk_path),
                "audio_seconds": round(audio_seconds, 3),
                "wall_seconds": round(wall_seconds, 3),
                "oom_notes": notes,
                **{k: v for k, v in metrics.items() if k != "http_status"},
            }
        )

    total_wall_seconds = time.time() - job_started
    gap_total = (
        args.gap * max(0, len(chunk_paths) - 1) if len(chunk_paths) > 1 else 0.0
    )

    final_seconds: float | None = None
    concat_command: list[str] | None = None

    if failed:
        print(
            f"\n  {failed} chunk(s) failed - skipping the FFmpeg concatenation.",
            file=sys.stderr,
        )
    elif args.no_concat:
        print(f"\n  --no-concat given: chunk WAVs kept in {chunks_dir}")
    else:
        print("\n  Concatenating with FFmpeg (concat demuxer, -c copy) ...")
        concat_command = concat_chunks(
            chunk_paths,
            output_path,
            file_list_path,
            gap_seconds=args.gap,
            sample_rate=sample_rate,
        )
        # EBU R128 as a second pass, so chunks stay bit-identical through the
        # -c copy stitch and only the finished file is re-encoded (v3 Task 5.3).
        if not args.no_normalize:
            raw_path = output_path.with_name(f"{output_path.stem}_raw.wav")
            output_path.replace(raw_path)
            try:
                loudnorm_command = normalize_loudness(
                    raw_path, output_path, lufs=args.lufs
                )
            finally:
                raw_path.unlink(missing_ok=True)
            if loudnorm_command:
                concat_command = loudnorm_command
                print(
                    f"  loudness         : EBU R128 {args.lufs} LUFS "
                    "(TP -1.5, LRA 11)"
                )
        final_seconds = wav_duration_seconds(output_path)
        expected_seconds = generated_audio_seconds + gap_total
        print(f"  output           : {output_path}")
        print(
            f"  duration         : {final_seconds:.2f}s  "
            f"(chunks {generated_audio_seconds:.2f}s + gaps {gap_total:.2f}s "
            f"= {expected_seconds:.2f}s expected)"
        )
        if abs(final_seconds - expected_seconds) > 1.0:
            print(
                f"  ! duration mismatch: expected ~{expected_seconds:.2f}s, "
                f"got {final_seconds:.2f}s",
                file=sys.stderr,
            )
        if not args.keep_chunks:
            for path in chunk_paths:
                path.unlink(missing_ok=True)
            file_list_path.unlink(missing_ok=True)
            print(f"  temp chunks      : removed {len(chunk_paths)} file(s)")

    overall_rtf = (
        total_wall_seconds / generated_audio_seconds
        if generated_audio_seconds > 0
        else None
    )
    peak_vram = max(
        (float(r.get("peak_vram_mb") or 0) for r in results), default=0.0
    )

    report = {
        "timestamp": now_iso(),
        "input": str(input_path),
        "output": (
            str(output_path) if (final_seconds is not None) else None
        ),
        "api_url": args.api_url,
        "model": model_info,
        "reference": {
            "audio": str(reference_audio),
            "text_file": str(reference_text_path),
            "seconds": round(reference_seconds, 3),
        },
        "chunking": {
            "max_words": max_words,
            "reason": sizing_reason,
            "chunk_count": len(chunks),
            "total_words": total_words,
            "gap_seconds": args.gap,
        },
        "parameters": {
            "num_steps": args.num_steps,
            "guidance_scale": args.guidance_scale,
            "seed": args.seed,
            "language": args.language,
            "normalize_text": bool(args.normalize_text),
        },
        "totals": {
            "wall_seconds": round(total_wall_seconds, 3),
            "generated_audio_seconds": round(generated_audio_seconds, 3),
            "final_audio_seconds": (
                round(final_seconds, 3) if final_seconds is not None else None
            ),
            "overall_rtf": round(overall_rtf, 4) if overall_rtf else None,
            "overall_speed_factor": (
                round(1.0 / overall_rtf, 4) if overall_rtf else None
            ),
            "failed_chunks": failed,
            "peak_vram_mb": round(peak_vram, 1) if peak_vram else None,
        },
        "chunks": results,
        "ffmpeg_command": concat_command,
        "speaker_id": speaker_id or None,
        "speaker_reused": bool(speaker_id),
        "loudnorm_lufs": None if args.no_normalize else args.lufs,
        "oom_splits": sum(int(r.get("splits") or 0) for r in results),
    }

    report_path = (
        Path(args.report).expanduser().resolve()
        if args.report
        else paths.output_dir / f"{input_path.stem}.report.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("=" * 74)
    print("  Summary")
    print("=" * 74)
    print(f"  chunks ok/failed : {len(chunks) - failed}/{failed}")
    print(f"  generated audio  : {generated_audio_seconds:.2f}s")
    if final_seconds is not None:
        print(f"  final audio      : {final_seconds:.2f}s -> {output_path}")
    print(f"  total wall time  : {total_wall_seconds:.2f}s")
    if overall_rtf:
        print(
            f"  overall RTF      : {overall_rtf:.4f} "
            f"({1.0 / overall_rtf:.2f}x faster than real time)"
        )
    if peak_vram:
        print(f"  peak VRAM        : {peak_vram:.0f} MB")
    print(f"  report           : {report_path}")
    print("=" * 74)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())


