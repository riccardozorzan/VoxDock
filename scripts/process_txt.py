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

from app.config import settings  # noqa: E402
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
        raise SystemExit(f"[process_txt] cannot reach the TTS server at {url}: {exc}")
    try:
        return response.json()
    except Exception as exc:
        raise SystemExit(
            f"[process_txt] {url} returned a non-JSON payload "
            f"(HTTP {response.status_code}): {exc}"
        )


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


def post_chunk(
    *,
    api_url: str,
    reference_audio: Path,
    reference_text: str,
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
    """POST a single chunk with retries; returns ``(metrics, wav_bytes)``."""
    import requests

    url = f"{api_url.rstrip('/')}{settings.api.speech_path}"
    form: dict[str, str] = {"text": text, "reference_text": reference_text}
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
            with open(reference_audio, "rb") as handle:
                response = requests.post(
                    url,
                    files={
                        "reference_audio": (
                            reference_audio.name,
                            handle,
                            "audio/wav",
                        )
                    },
                    data=form,
                    timeout=timeout,
                )
            if response.status_code != 200:
                raise RuntimeError(
                    f"HTTP {response.status_code}: {response.text[:300]}"
                )
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
            if attempt < max(1, retries):
                wait = backoff * attempt
                print(
                    f"      ! attempt {attempt}/{retries} failed ({exc}); "
                    f"retrying in {wait:.1f}s",
                    file=sys.stderr,
                )
                time.sleep(wait)

    raise RuntimeError(f"chunk failed after {retries} attempt(s): {last_error}")


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
            metrics, audio_bytes = post_chunk(
                api_url=args.api_url,
                reference_audio=reference_audio,
                reference_text=reference_text,
                text=chunk_text,
                num_steps=args.num_steps,
                guidance_scale=args.guidance_scale,
                seed=args.seed,
                language=args.language,
                normalize_text=args.normalize_text or None,
                timeout=args.timeout,
                retries=args.retries,
                backoff=args.backoff,
            )
        except Exception as exc:  # noqa: BLE001 - report and continue with the rest
            failed += 1
            print(f"  [{index:03d}/{len(chunks):03d}] FAILED  {exc}", file=sys.stderr)
            results.append({"index": index, "words": words, "error": str(exc)})
            continue

        chunk_path.write_bytes(audio_bytes)
        wall_seconds = time.time() - chunk_started
        audio_seconds = float(
            metrics.get("audio_seconds") or wav_duration_seconds(chunk_path)
        )
        generated_audio_seconds += audio_seconds
        generated_words += words

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
        )
        results.append(
            {
                "index": index,
                "words": words,
                "characters": len(chunk_text),
                "path": str(chunk_path),
                "audio_seconds": round(audio_seconds, 3),
                "wall_seconds": round(wall_seconds, 3),
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


