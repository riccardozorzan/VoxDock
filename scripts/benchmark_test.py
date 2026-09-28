#!/usr/bin/env python3
"""Automated integration test, VRAM + RTF benchmark (spec Phase 5).

Measures
--------
* **Cold start** – how long the API took to become ``ready`` (and the server's own
  checkpoint-load time), i.e. container startup + model load into VRAM.
* **Single-chunk inference** – a standardised 50-word Italian test sentence.
* **Peak VRAM** – sampled through ``/health`` before, during (background poller)
  and immediately after every synthesis call, plus the per-request high-water
  mark reported by the API. Must stay **< 5.5 GB**.
* **RTF** – reported in both conventions: ``generation_time / audio_duration``
  (dots.tts convention, lower is better, target < 0.5) and the spec's literal
  ``audio_duration / generation_time`` (speed factor).
* **Report** – ``data/output/benchmark_report.json`` with timestamps, RTF, peak
  VRAM, HTTP status and the pass/fail verdict.

Example
-------
    docker compose exec tts python scripts/benchmark_test.py
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings  # noqa: E402
from app.utils import now_iso  # noqa: E402

# Standardised 50-word benchmark sentence (varied punctuation -> prosody test).
STANDARD_TEST_TEXT = (
    "Questa è una prova di sintesi vocale pensata per misurare con precisione "
    "la velocità di generazione, l'uso della memoria video e la stabilità del "
    "modello. Il sole tramonta lentamente dietro le colline mentre una brezza "
    "leggera attraversa il prato: che cosa succederà domani? Forse arriverà "
    "davvero la pioggia, forse no."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark the local dots.tts API (cold start, RTF, peak VRAM).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--api-url", default=settings.api.api_url)
    parser.add_argument(
        "--reference-audio", default=str(settings.paths.clean_reference_wav)
    )
    parser.add_argument(
        "--reference-text",
        default=None,
        help="Defaults to the sibling .txt of the reference WAV.",
    )
    parser.add_argument("--text", default=STANDARD_TEST_TEXT, help="Text to synthesise.")
    parser.add_argument(
        "--wait-ready",
        type=float,
        default=3600.0,
        help="Seconds to wait for the engine to become ready (cold start).",
    )
    parser.add_argument("--warmup", type=int, default=1, help="Unmeasured warmup calls.")
    parser.add_argument("--samples", type=int, default=3, help="Measured calls.")
    parser.add_argument("--poll-interval", type=float, default=0.25)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=settings.api.request_timeout_s)
    parser.add_argument("--retries", type=int, default=settings.processing.max_retries)
    parser.add_argument("--vram-limit-gb", type=float, default=settings.model.vram_limit_gb)
    parser.add_argument("--rtf-target", type=float, default=settings.model.rtf_target)
    parser.add_argument("--output", default=None, help="Where to store the last WAV.")
    parser.add_argument("--report", default=None, help="JSON report path.")
    parser.add_argument(
        "--no-fail",
        action="store_true",
        help="Always exit 0, even when a threshold is violated.",
    )
    return parser


# --------------------------------------------------------------------------- #
#  HTTP helpers
# --------------------------------------------------------------------------- #
def _header_float(headers, name: str) -> Optional[float]:
    raw = headers.get(name)
    try:
        return float(raw) if raw not in (None, "") else None
    except (TypeError, ValueError):
        return None


def get_health(api_url: str, timeout: float) -> dict[str, Any]:
    """``GET /health`` - tolerates the 503 while the model is loading."""
    import requests

    url = f"{api_url.rstrip('/')}{settings.api.health_path}"
    started = time.perf_counter()
    try:
        response = requests.get(url, timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"[benchmark] cannot reach the TTS server at {url}: {exc}")
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    try:
        payload = response.json()
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"[benchmark] {url} returned a non-JSON payload "
            f"(HTTP {response.status_code}): {exc}"
        )
    payload["_http_status"] = response.status_code
    payload["_latency_ms"] = latency_ms
    return payload


def wait_until_ready(
    api_url: str, wait_limit: float, request_timeout: float
) -> tuple[dict[str, Any], float]:
    """Poll ``/health`` until the engine is ready; returns (health, seconds)."""
    started = time.perf_counter()
    deadline = started + max(0.0, wait_limit)
    while True:
        health = get_health(api_url, min(request_timeout, 30.0))
        if health.get("ready"):
            return health, time.perf_counter() - started
        if health.get("status") == "error":
            raise SystemExit(
                "[benchmark] the engine reported an error: "
                + json.dumps(health.get("model"), ensure_ascii=False)
            )
        if time.perf_counter() >= deadline:
            raise SystemExit(
                f"[benchmark] engine still '{health.get('status')}' after "
                f"{wait_limit:.0f}s (uptime {health.get('uptime_seconds')}s)"
            )
        print(
            f"  waiting for the engine (status={health.get('status')}, "
            f"uptime={health.get('uptime_seconds')}s, "
            f"waited={time.perf_counter() - started:.0f}s)"
        )
        time.sleep(5.0)


class VramPoller(threading.Thread):
    """Background ``/health`` sampler used to catch the VRAM high-water mark."""

    def __init__(self, api_url: str, interval: float, request_timeout: float):
        super().__init__(daemon=True, name="vram-poller")
        self.api_url = api_url
        self.interval = max(0.05, float(interval))
        self.request_timeout = request_timeout
        self.samples: list[dict[str, Any]] = []
        # NB: do NOT name this `_stop` - `threading.Thread` has a private
        # `_stop()` method that `Thread.join()` calls internally, so shadowing it
        # with an Event makes every `join()` fail with
        # "TypeError: 'Event' object is not callable".
        self._stop_event = threading.Event()

    def run(self) -> None:  # pragma: no cover - thread body
        while not self._stop_event.is_set():
            try:
                health = get_health(self.api_url, min(self.request_timeout, 15.0))
            except SystemExit:
                break
            gpu = health.get("gpu") or {}
            self.samples.append(
                {
                    "vram_allocated_mb": gpu.get("vram_allocated_mb"),
                    "vram_reserved_mb": gpu.get("vram_reserved_mb"),
                    "vram_peak_allocated_mb": gpu.get("vram_peak_allocated_mb"),
                    "vram_used_mb": gpu.get("vram_used_mb"),
                    "vram_free_mb": gpu.get("vram_free_mb"),
                    "lifetime_peak_vram_mb": health.get("lifetime_peak_vram_mb"),
                }
            )
            self._stop_event.wait(self.interval)

    def stop(self) -> None:
        self._stop_event.set()
        self.join(timeout=5.0)

    def _max_of(self, key: str) -> Optional[float]:
        values = [
            float(sample[key])
            for sample in self.samples
            if sample.get(key) is not None
        ]
        return max(values) if values else None

    @property
    def peak_allocated_mb(self) -> Optional[float]:
        return self._max_of("vram_peak_allocated_mb")

    @property
    def peak_reserved_mb(self) -> Optional[float]:
        return self._max_of("vram_reserved_mb")

    @property
    def peak_lifetime_mb(self) -> Optional[float]:
        return self._max_of("lifetime_peak_vram_mb")


def synthesize(
    args: argparse.Namespace,
    reference_audio: Path,
    reference_text: str,
    text: str,
) -> tuple[dict[str, Any], bytes, int]:
    """POST one synthesis request; returns ``(metrics, wav_bytes, http_status)``."""
    import requests

    url = f"{args.api_url.rstrip('/')}{settings.api.speech_path}"
    form: dict[str, str] = {"text": text, "reference_text": reference_text}
    if args.num_steps is not None:
        form["num_steps"] = str(args.num_steps)
    if args.guidance_scale is not None:
        form["guidance_scale"] = str(args.guidance_scale)
    if args.seed is not None:
        form["seed"] = str(args.seed)

    with open(reference_audio, "rb") as handle:
        response = requests.post(
            url,
            files={"reference_audio": (reference_audio.name, handle, "audio/wav")},
            data=form,
            timeout=args.timeout,
        )

    if response.status_code != 200:
        return {"error": response.text[:500]}, b"", response.status_code

    headers = response.headers
    metrics = {
        "request_id": headers.get("X-Request-Id"),
        "audio_seconds": _header_float(headers, "X-Audio-Seconds"),
        "generation_seconds": _header_float(headers, "X-Generation-Seconds"),
        "rtf": _header_float(headers, "X-RTF"),
        "engine_rtf": _header_float(headers, "X-Engine-RTF"),
        "speed_factor": _header_float(headers, "X-Speed-Factor"),
        "peak_vram_mb": _header_float(headers, "X-Peak-Vram-MB"),
        "sample_rate": _header_float(headers, "X-Sample-Rate"),
    }
    return metrics, response.content, response.status_code


def run_once(
    args: argparse.Namespace,
    reference_audio: Path,
    reference_text: str,
    *,
    measure: bool,
) -> dict[str, Any]:
    """Run one synthesis call, optionally polling VRAM around it."""
    poller: Optional[VramPoller] = None
    before = get_health(args.api_url, min(args.timeout, 30.0))
    if measure:
        poller = VramPoller(args.api_url, args.poll_interval, args.timeout)
        poller.start()

    started = time.perf_counter()
    try:
        metrics, wav_bytes, status = synthesize(
            args, reference_audio, reference_text, args.text
        )
    except Exception as exc:  # noqa: BLE001
        if poller is not None:
            poller.stop()
        return {"http_status": 0, "error": f"{type(exc).__name__}: {exc}"}
    wall_seconds = time.perf_counter() - started

    if poller is not None:
        poller.stop()
    after = get_health(args.api_url, min(args.timeout, 30.0))

    record: dict[str, Any] = {
        "http_status": status,
        "client_wall_seconds": round(wall_seconds, 4),
        **metrics,
    }
    if poller is not None:
        record["poller"] = {
            "samples": len(poller.samples),
            "peak_allocated_mb": poller.peak_allocated_mb,
            "peak_reserved_mb": poller.peak_reserved_mb,
            "peak_lifetime_mb": poller.peak_lifetime_mb,
        }
    record["health_before"] = {
        "vram_allocated_mb": (before.get("gpu") or {}).get("vram_allocated_mb"),
        "vram_free_mb": (before.get("gpu") or {}).get("vram_free_mb"),
        "lifetime_peak_vram_mb": before.get("lifetime_peak_vram_mb"),
    }
    record["health_after"] = {
        "vram_allocated_mb": (after.get("gpu") or {}).get("vram_allocated_mb"),
        "vram_free_mb": (after.get("gpu") or {}).get("vram_free_mb"),
        "vram_peak_allocated_mb": (after.get("gpu") or {}).get("vram_peak_allocated_mb"),
        "lifetime_peak_vram_mb": after.get("lifetime_peak_vram_mb"),
    }
    record["_wav_bytes"] = wav_bytes
    return record



# --------------------------------------------------------------------------- #
#  Aggregation helpers
# --------------------------------------------------------------------------- #
def percentile(values: list[float], quantile: float) -> Optional[float]:
    """Linear-interpolated percentile (no numpy dependency)."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def collect_peak_vram(samples: list[dict[str, Any]]) -> Optional[float]:
    """Highest VRAM figure observed anywhere around the measured calls (MB)."""
    peaks: list[float] = []
    for sample in samples:
        if sample.get("peak_vram_mb") is not None:
            peaks.append(float(sample["peak_vram_mb"]))
        poller = sample.get("poller") or {}
        for key in ("peak_allocated_mb", "peak_reserved_mb", "peak_lifetime_mb"):
            if poller.get(key) is not None:
                peaks.append(float(poller[key]))
        after = sample.get("health_after") or {}
        for key in ("vram_peak_allocated_mb", "lifetime_peak_vram_mb"):
            if after.get(key) is not None:
                peaks.append(float(after[key]))
    return max(peaks) if peaks else None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = settings.paths.ensure()

    reference_audio = Path(args.reference_audio).expanduser().resolve()
    if not reference_audio.exists():
        raise SystemExit(
            f"[benchmark] reference audio not found: {reference_audio}\n"
            "            Run scripts/clean_audio.py first."
        )
    reference_text_path = (
        Path(args.reference_text).expanduser().resolve()
        if args.reference_text
        else reference_audio.with_suffix(".txt")
    )
    if not reference_text_path.exists():
        raise SystemExit(
            f"[benchmark] reference transcript not found: {reference_text_path}"
        )
    reference_text = reference_text_path.read_text(encoding="utf-8").strip()

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else paths.output_dir / "benchmark_output.wav"
    )
    report_path = (
        Path(args.report).expanduser().resolve()
        if args.report
        else paths.benchmark_report
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    word_count = len(args.text.split())
    print("=" * 74)
    print("  dots.tts benchmark: cold start / RTF / peak VRAM")
    print("=" * 74)
    print(f"  api url          : {args.api_url}")
    print(f"  reference audio  : {reference_audio}")
    print(f"  reference text   : {reference_text_path}")
    print(f"  test text        : {word_count} words / {len(args.text)} chars")
    print(
        f"  thresholds       : peak VRAM < {args.vram_limit_gb:.2f} GB, "
        f"RTF(gen/audio) <= {args.rtf_target}"
    )
    print(f"  calls            : {args.warmup} warmup + {args.samples} measured")
    print("=" * 74)

    health, cold_start = wait_until_ready(args.api_url, args.wait_ready, args.timeout)
    model = health.get("model") or {}
    gpu = health.get("gpu") or {}
    server_cold_start = model.get("cold_start_seconds")
    print(
        f"  engine ready in  : {cold_start:.1f}s "
        f"(server checkpoint load: {server_cold_start}s)"
    )
    print(
        f"  gpu              : {gpu.get('name')} | "
        f"{gpu.get('vram_total_mb')} MB total | "
        f"{gpu.get('vram_used_mb')} MB used"
    )
    print(
        f"  model            : {model.get('id')} | precision={model.get('precision')} "
        f"| optimize={model.get('optimize')} "
        f"| max_generate_length={model.get('max_generate_length')} "
        f"| patch={model.get('audio_patch_seconds')}s"
    )
    print(f"  sample rate      : {model.get('sample_rate')}")
    print("-" * 74)

    for index in range(max(0, args.warmup)):
        print(f"  warmup {index + 1}/{args.warmup} (not measured) ...")
        warmup = run_once(args, reference_audio, reference_text, measure=False)
        warmup.pop("_wav_bytes", None)
        if warmup.get("http_status") != 200:
            raise SystemExit(f"[benchmark] warmup call failed: {warmup.get('error')}")

    samples: list[dict[str, Any]] = []
    for index in range(max(1, args.samples)):
        record = run_once(args, reference_audio, reference_text, measure=True)
        wav_bytes = record.pop("_wav_bytes", b"")
        if wav_bytes:
            output_path.write_bytes(wav_bytes)
        samples.append(record)

        if record.get("http_status") == 200:
            print(
                f"  sample {index + 1}/{args.samples} : "
                f"audio {record.get('audio_seconds')}s in "
                f"{record.get('generation_seconds')}s | "
                f"rtf {record.get('rtf')} | "
                f"speed {record.get('speed_factor')}x | "
                f"peak {record.get('peak_vram_mb')} MB | "
                f"poller {(record.get('poller') or {}).get('samples', 0)} snapshots"
            )
        else:
            print(
                f"  sample {index + 1}/{args.samples} : FAILED "
                f"HTTP {record.get('http_status')} {record.get('error')}",
                file=sys.stderr,
            )

    # ---- aggregate ---------------------------------------------------------
    ok_samples = [s for s in samples if s.get("http_status") == 200]
    audio_values = [
        float(s["audio_seconds"]) for s in ok_samples if s.get("audio_seconds")
    ]
    rtf_values = [float(s["rtf"]) for s in ok_samples if s.get("rtf") is not None]
    speed_values = [
        float(s["speed_factor"]) for s in ok_samples if s.get("speed_factor") is not None
    ]
    gen_values = [
        float(s["generation_seconds"])
        for s in ok_samples
        if s.get("generation_seconds") is not None
    ]

    peak_vram_mb = collect_peak_vram(samples)
    limit_mb = args.vram_limit_gb * 1024.0
    vram_ok = peak_vram_mb is not None and peak_vram_mb <= limit_mb
    http_ok = bool(samples) and len(ok_samples) == len(samples)

    mean_rtf = statistics.fmean(rtf_values) if rtf_values else None
    rtf_p50 = percentile(rtf_values, 0.50)
    rtf_p90 = percentile(rtf_values, 0.90)
    mean_speed = statistics.fmean(speed_values) if speed_values else None
    rtf_ok = mean_rtf is not None and mean_rtf <= args.rtf_target
    passed = bool(vram_ok and http_ok and rtf_ok)

    after = get_health(args.api_url, min(args.timeout, 30.0))

    report: dict[str, Any] = {
        "timestamp": now_iso(),
        "ok": passed,
        "thresholds": {
            "peak_vram_limit_gb": args.vram_limit_gb,
            "peak_vram_limit_mb": round(limit_mb, 1),
            "rtf_target_gen_over_audio": args.rtf_target,
        },
        "verdict": {
            "peak_vram_ok": vram_ok,
            "rtf_ok": rtf_ok,
            "http_ok": http_ok,
        },
        "cold_start": {
            "script_wait_seconds": round(cold_start, 3),
            "server_checkpoint_load_seconds": server_cold_start,
            "model_load_seconds": model.get("load_seconds"),
        },
        "environment": {
            "api_url": args.api_url,
            "model": model,
            "gpu": gpu,
        },
        "request": {
            "test_text": args.text,
            "test_text_words": word_count,
            "reference_audio": str(reference_audio),
            "reference_text_file": str(reference_text_path),
            "reference_text_words": len(reference_text.split()),
            "num_steps": args.num_steps,
            "guidance_scale": args.guidance_scale,
            "seed": args.seed,
            "warmup_calls": args.warmup,
            "measured_calls": args.samples,
            "poll_interval_seconds": args.poll_interval,
        },
        "results": {
            "http_success": len(ok_samples),
            "http_failures": len(samples) - len(ok_samples),
            "audio_seconds_mean": (
                round(statistics.fmean(audio_values), 3) if audio_values else None
            ),
            "generation_seconds_mean": (
                round(statistics.fmean(gen_values), 3) if gen_values else None
            ),
            "rtf_mean": round(mean_rtf, 4) if mean_rtf else None,
            "rtf_p50": round(rtf_p50, 4) if rtf_p50 else None,
            "rtf_p90": round(rtf_p90, 4) if rtf_p90 else None,
            "speed_factor_mean": round(mean_speed, 4) if mean_speed else None,
            "peak_vram_mb": round(peak_vram_mb, 1) if peak_vram_mb else None,
            "peak_vram_gb": round(peak_vram_mb / 1024.0, 3) if peak_vram_mb else None,
            "vram_free_mb_after": (after.get("gpu") or {}).get("vram_free_mb"),
            "lifetime_peak_vram_mb": after.get("lifetime_peak_vram_mb"),
        },
        "samples": samples,
        "output_wav": str(output_path) if output_path.exists() else None,
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ---- verdict -----------------------------------------------------------
    results = report["results"]
    print("-" * 74)
    print("  Results")
    print("-" * 74)
    print(f"  http ok / failed : {len(ok_samples)} / {len(samples) - len(ok_samples)}")
    print(f"  audio mean       : {results['audio_seconds_mean']}s")
    print(f"  generation mean  : {results['generation_seconds_mean']}s")
    print(
        f"  RTF mean/p50/p90 : {results['rtf_mean']} / {results['rtf_p50']} / "
        f"{results['rtf_p90']}  (generation/audio, lower is better)"
    )
    print(
        f"  speed factor     : {results['speed_factor_mean']}x real time "
        f"(audio/generation)"
    )
    print(
        f"  peak VRAM        : {results['peak_vram_gb']} GB "
        f"({results['peak_vram_mb']} MB) | limit {args.vram_limit_gb} GB"
    )
    print(f"  vram free after  : {results['vram_free_mb_after']} MB")
    print(f"  output wav       : {output_path}")
    print(f"  report           : {report_path}")
    print("=" * 74)
    print(f"  VERDICT peak VRAM < {args.vram_limit_gb} GB : {'PASS' if vram_ok else 'FAIL'}")
    print(f"          RTF <= {args.rtf_target}          : {'PASS' if rtf_ok else 'FAIL'}")
    print(f"          HTTP 200 x{len(samples)}         : {'PASS' if http_ok else 'FAIL'}")
    print(f"          OVERALL              : {'PASS' if passed else 'FAIL'}")
    print("=" * 74)

    if args.no_fail:
        return 0
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())


