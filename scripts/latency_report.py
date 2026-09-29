#!/usr/bin/env python3
"""Measure the three latencies that matter, in one run.

Why
---
"Faster" is ambiguous in this project, and the three costs respond to
completely different levers:

  * **cold start** - container start to `/health` 200. Dominated by
    torch.compile; already cut 219 s -> 69 s by caching the kernels.
  * **first request** - the one request right after a restart. Measured at
    RTF 3.3-4.5 against a steady-state 0.45, because the warmup does not
    cover every shape a real request hits.
  * **steady state** - RTF once warm.

Optimising one and reporting it as "faster" would hide the other two, so this
measures all three together.

Usage
-----
    docker compose exec tts python scripts/latency_report.py
    docker compose exec tts python scripts/latency_report.py --report data/output/latency.json
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
from app.utils import now_iso, wav_duration_seconds  # noqa: E402

#: Long enough that the fixed per-request cost is amortised the way real
#: long-form work is, and short enough to keep the measurement quick.
LONG_TEXT = (
    "Il sole tramonta lentamente dietro le colline mentre la campana del paese "
    "suona lontano, e attraverso i campi l'aria porta il profumo del fieno "
    "appena mietuto. Mia nonna racconta sempre che le sere d'estate duravano il "
    "doppio, quando non c'era la televisione e si aspettava che le stelle "
    "cadessero una dopo l'altra."
)
SHORT_TEXT = "Ciao, questo è un test."


def _synth(text: str, reference: Path, output: Path, api_url: str, timeout: float):
    """One synthesis, returning the metrics the service reports."""
    import requests

    url = f"{api_url.rstrip('/')}{settings.api.speech_path}"
    reference_text = reference.with_suffix(".txt").read_text(encoding="utf-8").strip()
    started = time.perf_counter()
    with open(reference, "rb") as handle:
        response = requests.post(
            url,
            files={"reference_audio": (reference.name, handle, "audio/wav")},
            data={"text": text, "reference_text": reference_text},
            timeout=timeout,
        )
    elapsed = time.perf_counter() - started
    if response.status_code != 200:
        raise SystemExit(f"HTTP {response.status_code}: {response.text[:300]}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(response.content)
    audio = wav_duration_seconds(output)
    return {
        "wall_seconds": round(elapsed, 3),
        "audio_seconds": round(audio, 3),
        "engine_rtf": _header(response, "X-Engine-RTF"),
        "rtf": _header(response, "X-RTF"),
        "peak_vram_mb": _header(response, "X-Peak-Vram-MB"),
    }


def _header(response, name: str) -> Optional[float]:
    raw = response.headers.get(name)
    try:
        return float(raw) if raw not in (None, "") else None
    except ValueError:
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure cold-start, first-request and steady-state latency.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--api-url", default=settings.api.api_url)
    parser.add_argument(
        "--reference-audio", default=str(settings.paths.clean_reference_wav)
    )
    parser.add_argument("--repeat", type=int, default=3, help="Steady-state samples.")
    parser.add_argument("--report", default=None, help="Optional JSON report path.")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    reference = Path(args.reference_audio)
    if not reference.exists():
        print(f"[latency] reference not found: {reference}", file=sys.stderr)
        return 1

    import requests

    out_dir = settings.paths.output_dir / "_latency"
    health_url = f"{args.api_url.rstrip('/')}{settings.api.health_path}"
    health = requests.get(health_url, timeout=30).json()

    print("=" * 70)
    print("  Latency report")
    print("=" * 70)
    print(f"  service uptime     : {health.get('uptime_seconds')} s")
    print(f"  cold start         : {health['model'].get('cold_start_seconds')} s")
    print(f"  num_steps          : {health['model'].get('default_num_steps')}")
    print(f"  vocoder chunk      : "
          f"{(health.get('config_source', {}).get('best_config') or {}).get('TTS_VOCODER_CHUNK_PATCHES')}")

    first = _synth(
        SHORT_TEXT, reference, out_dir / "first.wav", args.api_url,
        settings.api.request_timeout_s,
    )
    print(f"  first request      : {first['wall_seconds']:.2f} s "
          f"({first['audio_seconds']:.2f} s audio, rtf {first['rtf']})")

    long_rtf, long_audio, short_rtfs = [], None, []
    for index in range(args.repeat):
        sample = _synth(
            LONG_TEXT, reference, out_dir / f"steady{index}.wav", args.api_url,
            settings.api.request_timeout_s,
        )
        long_rtf.append(sample["rtf"] or 0.0)
        long_audio = sample["audio_seconds"]
        short_rtfs.append(sample["rtf"])
        print(f"  steady {index + 1}/{args.repeat}       : "
              f"{sample['wall_seconds']:.2f} s ({sample['audio_seconds']:.2f} s audio, "
              f"rtf {sample['rtf']})")

    steady = sum(long_rtf) / len(long_rtf)
    report = {
        "timestamp": now_iso(),
        "cold_start_seconds": health["model"].get("cold_start_seconds"),
        "first_request": first,
        "steady_state": {
            "samples": len(long_rtf),
            "audio_seconds": long_audio,
            "rtf_mean": round(steady, 4),
            "rtf_min": round(min(long_rtf), 4),
            "rtf_max": round(max(long_rtf), 4),
        },
    }
    print("-" * 70)
    print(f"  steady-state RTF   : {steady:.4f} "
          f"({1 / steady:.2f}x faster than real time)")
    print(f"  first-request RTF  : {first['rtf']} "
          f"({(first['rtf'] or 0) / steady:.1f}x the steady cost)")
    if args.report:
        path = Path(args.report).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"  report             : {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
