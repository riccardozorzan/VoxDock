#!/usr/bin/env python3
"""Quick sanity test for the running dots.tts API (spec Phase 1 / 6).

Checks ``/health`` and performs a single short synthesis request, saving the
result to ``data/output/test_api.wav``.

    docker compose exec tts python scripts/test_api.py
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
