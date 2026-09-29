#!/usr/bin/env python3
"""Sanity test for the running dots.tts API (spec Phase 1 / 6).

Checks ``/health`` and performs a single short synthesis request, saving the
result to ``data/output/test_api.wav``.

    docker compose exec tts python scripts/test_api.py

With ``--assert`` it additionally *verifies* the response rather than just
printing it, and exits non-zero on the first failure - so it can be wired into
CI or a post-deploy check. The default stays a human-readable smoke test,
because that is what it was written as.

What is asserted, and why each one:

* HTTP 200 and a non-empty body - catches a route or exception-handling change.
* the payload is a real RIFF/WAVE - a 200 carrying an error page in the body
  would otherwise pass silently.
* the duration is plausible for the text - catches a silent near-empty render,
  which is the failure mode a status code cannot see.
* ``X-Sample-Rate`` is the configured rate - catches a resampling regression.
* ``X-Peak-Vram-MB`` is present and within the configured limit - the memory
  patches exist to hold this line, so it is the one number worth gating on.

The thresholds are deliberately loose. This is a smoke test, not a benchmark:
``benchmark_test.py`` owns performance acceptance. Its job is to fail loudly
when the service is broken, not to notice that a run was 5 % slower.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings  # noqa: E402
from app.utils import wav_duration_seconds  # noqa: E402

SHORT_TEXT = "Ciao! Questo è un rapido test di sintesi vocale in italiano."

#: A correct render of SHORT_TEXT is a few seconds. Anything under this is a
#: truncated or empty generation that still returns HTTP 200.
MIN_PLAUSIBLE_SECONDS = 0.5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sanity-check the local dots.tts API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--api-url", default=settings.api.api_url)
    parser.add_argument("--text", default=SHORT_TEXT)
    parser.add_argument("--reference-audio", default=str(settings.paths.clean_reference_wav))
    parser.add_argument("--reference-text", default=None)
    parser.add_argument("--output", default=str(settings.paths.output_dir / "test_api.wav"))
    parser.add_argument("--timeout", type=float, default=settings.api.request_timeout_s)
    parser.add_argument("--wait-ready", type=float, default=600.0)
    parser.add_argument(
        "--assert",
        dest="assert_",
        action="store_true",
        help=(
            "Verify the response (WAV magic, plausible duration, sample rate, "
            "peak VRAM within budget) and exit non-zero on failure. For CI."
        ),
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    import requests

    args = build_parser().parse_args(argv)
    settings.paths.ensure()

    health_url = f"{args.api_url.rstrip('/')}{settings.api.health_path}"
    print(f"[test_api] GET {health_url}")
    deadline = time.time() + max(0.0, args.wait_ready)
    health: dict = {}
    while True:
        try:
            response = requests.get(health_url, timeout=30)
            health = response.json()
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"[test_api] cannot reach the API: {exc}")
        if health.get("ready") or time.time() >= deadline:
            break
        print(f"[test_api]   status={health.get('status')} - waiting ...")
        time.sleep(10)

    print(json.dumps(health, indent=2, ensure_ascii=False))
    if not health.get("ready"):
        print("[test_api] FAIL: the engine is not ready", file=sys.stderr)
        return 1

    reference_audio = Path(args.reference_audio).expanduser()
    if not reference_audio.exists():
        print(
            f"[test_api] FAIL: reference audio not found: {reference_audio}\n"
            "           Run scripts/clean_audio.py first.",
            file=sys.stderr,
        )
        return 1

    reference_text_path = (
        Path(args.reference_text).expanduser()
        if args.reference_text
        else reference_audio.with_suffix(".txt")
    )
    if not reference_text_path.exists():
        print(
            f"[test_api] FAIL: reference transcript not found: {reference_text_path}",
            file=sys.stderr,
        )
        return 1
    reference_text = reference_text_path.read_text(encoding="utf-8").strip()

    speech_url = f"{args.api_url.rstrip('/')}{settings.api.speech_path}"
    print(f"[test_api] POST {speech_url}")
    started = time.perf_counter()
    with open(reference_audio, "rb") as handle:
        response = requests.post(
            speech_url,
            files={"reference_audio": (reference_audio.name, handle, "audio/wav")},
            data={"text": args.text, "reference_text": reference_text},
            timeout=args.timeout,
        )
    elapsed = time.perf_counter() - started

    if response.status_code != 200:
        print(
            f"[test_api] FAIL: HTTP {response.status_code}: {response.text[:400]}",
            file=sys.stderr,
        )
        return 1

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(response.content)
    duration = wav_duration_seconds(output_path)

    print("[test_api] OK")
    print(f"  http status      : {response.status_code}")
    print(f"  output           : {output_path} ({output_path.stat().st_size / 1024:.0f} KiB)")
    print(f"  audio duration   : {duration:.2f}s")
    print(f"  request time     : {elapsed:.2f}s")
    print(f"  rtf (gen/audio)  : {elapsed / duration:.4f}" if duration else "")
    print(f"  x-rtf header     : {response.headers.get('X-RTF')}")
    print(f"  x-peak-vram-mb   : {response.headers.get('X-Peak-Vram-MB')}")
    print(f"  x-sample-rate    : {response.headers.get('X-Sample-Rate')}")

    if not args.assert_:
        return 0
    return _assert_response(response, output_path, duration, elapsed)


def _assert_response(response, output_path: Path, duration: float, elapsed: float) -> int:
    """Verify a successful response, printing every failure rather than the first.

    All checks run so one invocation reports the whole picture; only the exit
    code reflects whether any of them failed.
    """
    failures: list[str] = []

    if not response.content:
        failures.append("empty response body")

    # A 200 carrying something that is not a WAV would otherwise sail through:
    # an error page or a JSON blob both parse fine as "bytes were returned".
    if response.content[:4] != b"RIFF" or response.content[8:12] != b"WAVE":
        failures.append(
            f"body is not a RIFF/WAVE payload (starts {response.content[:12]!r})"
        )

    if duration < MIN_PLAUSIBLE_SECONDS:
        failures.append(
            f"audio is {duration:.2f}s, too short to be a real render "
            f"(expected at least {MIN_PLAUSIBLE_SECONDS}s)"
        )

    expected_rate = settings.processing.target_sample_rate
    reported_rate = response.headers.get("X-Sample-Rate")
    if reported_rate is not None and reported_rate.isdigit():
        if int(reported_rate) != expected_rate:
            failures.append(
                f"X-Sample-Rate is {reported_rate}, expected {expected_rate}"
            )
    else:
        failures.append("X-Sample-Rate header missing or not numeric")

    peak = response.headers.get("X-Peak-Vram-MB")
    limit_mb = settings.model.vram_limit_gb * 1024
    if peak is None:
        failures.append("X-Peak-Vram-MB header missing")
    else:
        try:
            peak_mb = float(peak)
        except ValueError:
            failures.append(f"X-Peak-Vram-MB is not numeric: {peak!r}")
        else:
            if peak_mb > limit_mb:
                failures.append(
                    f"peak VRAM {peak_mb:.0f} MB exceeds the "
                    f"{limit_mb:.0f} MB limit"
                )

    if failures:
        print(f"\n[test_api] {len(failures)} assertion(s) failed:", file=sys.stderr)
        for problem in failures:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    print("\n[test_api] all assertions passed")
    print(f"  rtf               : {elapsed / duration:.4f}" if duration else "")
    print(f"  peak vram         : {peak} MB / {limit_mb:.0f} MB limit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
