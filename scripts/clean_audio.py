#!/usr/bin/env python3
"""Reference-audio pre-processing: denoise, level and validate (spec Phase 2).

Pipeline
--------
1. Force-load the reference at **48 000 Hz mono** with ``librosa``.
2. Trim leading/trailing silence (optional).
3. Optional sentence-safe ``--trim-to`` cut so that a long reference can be made
   VRAM-friendly (a warning is emitted because the transcript must match!).
4. Stationary noise reduction via :func:`noisereduce.reduce_noise`
   (``prop_decrease=0.85`` by default).
5. Peak-normalise to -1 dBFS and write a 48 kHz / mono / PCM_16 WAV.
6. Validate the duration (must be 5-15 s, ideally 10-12 s) and report whether
   the clip leaves room for generation under the configured
   ``max_generate_length`` (total prompt + generated audio patches).

Example
-------
    docker compose exec tts python scripts/clean_audio.py \
        --input data/references/my_voice_v2.wav --trim-to 7.0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:  # allow `python scripts/clean_audio.py`
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings  # noqa: E402
from app.utils import now_iso, write_wav  # noqa: E402

# One dots.tts audio latent patch ~= 160 ms (patch_size * hop_size / 48 kHz).
DEFAULT_PATCH_SECONDS = 0.16


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def load_audio(path: Path, sample_rate: int):
    """Force-load at ``sample_rate`` Hz mono (spec Task 2.1)."""
    import librosa

    return librosa.load(str(path), sr=sample_rate, mono=True)


def trim_edges(data, top_db: float = 40.0):
    """Remove leading/trailing silence."""
    import librosa

    trimmed, _ = librosa.effects.trim(data, top_db=top_db)
    return trimmed if trimmed.size else data


def denoise(data, rate: int, prop_decrease: float, stationary: bool):
    """Stationary noise reduction (spec Task 2.1)."""
    import noisereduce as nr

    return nr.reduce_noise(
        y=data,
        sr=rate,
        prop_decrease=prop_decrease,
        stationary=stationary,
    )


def peak_dbfs(data) -> float | None:
    """Peak level of a signal in dBFS (``None`` for silence)."""
    import numpy as np

    if not data.size:
        return None
    peak = float(np.max(np.abs(data)))
    return 20.0 * float(np.log10(peak)) if peak > 1e-9 else None


def normalise_peak(data, peak_dbfs_target: float):
    """Peak-normalise and return ``(samples, peak_before_dbfs)``."""
    import numpy as np

    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak <= 1e-9:
        return data.astype("float32"), None
    before_dbfs = 20.0 * float(np.log10(peak))
    target = 10.0 ** (peak_dbfs_target / 20.0)
    return (data * (target / peak)).astype("float32"), before_dbfs


def validate_duration(seconds: float) -> tuple[bool, list[str]]:
    """Apply the spec Task 2.2 duration rules."""
    cfg = settings.processing
    notes: list[str] = []
    ok = cfg.reference_min_seconds <= seconds <= cfg.reference_max_seconds
    if not ok:
        notes.append(
            f"duration {seconds:.2f}s is outside the required "
            f"{cfg.reference_min_seconds:.0f}-{cfg.reference_max_seconds:.0f}s window"
        )
    elif not (
        cfg.reference_ideal_min_seconds
        <= seconds
        <= cfg.reference_ideal_max_seconds
    ):
        notes.append(
            f"duration {seconds:.2f}s is valid but outside the ideal "
            f"{cfg.reference_ideal_min_seconds:.0f}-"
            f"{cfg.reference_ideal_max_seconds:.0f}s window"
        )
    return ok, notes


def patch_budget(seconds: float, patch_seconds: float) -> dict:
    """How much generation room is left after this prompt on the current GPU."""
    import math

    prompt_patches = int(math.ceil(seconds / patch_seconds))
    total = int(settings.model.max_generate_length)
    remaining = total - prompt_patches
    return {
        "audio_patch_seconds": round(patch_seconds, 4),
        "prompt_patches": prompt_patches,
        "max_generate_length": total,
        "generation_patches_available": remaining,
        "generation_seconds_available": round(remaining * patch_seconds, 2),
    }


def process(args: argparse.Namespace) -> tuple[dict, bool]:
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"[clean_audio] input file not found: {input_path}")

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rate = settings.processing.target_sample_rate
    data, rate = load_audio(input_path, rate)
    source_seconds = len(data) / float(rate)

    if args.trim_silence:
        data = trim_edges(data)
    if args.trim_to is not None:
        data = data[: int(round(args.trim_to * rate))]

    peak_before_dbfs = peak_dbfs(data)
    data = denoise(data, rate, args.prop_decrease, args.stationary)
    data, _ = normalise_peak(data, args.peak_dbfs)

    cleaned_seconds = len(data) / float(rate)
    valid, notes = validate_duration(cleaned_seconds)

    if args.trim_to is not None:
        notes.append(
            "--trim-to was used: update the reference transcript so that it "
            "matches the shortened audio exactly (mismatches degrade cloning)."
        )
    if not valid and not args.force:
        for note in notes:
            print(f"[clean_audio] FAIL: {note}", file=sys.stderr)
        raise SystemExit(
            "Duration validation failed - re-record (5-15 s, ideally 10-12 s) or "
            "pass --force to write the file anyway."
        )

    write_wav(output_path, data, rate, subtype="PCM_16")

    transcript_out = None
    if args.transcript:
        source_txt = Path(args.transcript).expanduser().resolve()
        if not source_txt.exists():
            notes.append(f"transcript not found, skipped: {source_txt}")
        else:
            transcript_out = output_path.with_suffix(".txt")
            transcript_out.write_text(
                source_txt.read_text(encoding="utf-8").strip() + "\n", encoding="utf-8"
            )

    report = {
        "timestamp": now_iso(),
        "input": str(input_path),
        "output": str(output_path),
        "sample_rate": rate,
        "channels": 1,
        "subtype": "PCM_16",
        "source_seconds": round(source_seconds, 3),
        "cleaned_seconds": round(cleaned_seconds, 3),
        "peak_dbfs_before": (
            round(peak_before_dbfs, 2) if peak_before_dbfs is not None else None
        ),
        "peak_dbfs_target": args.peak_dbfs,
        "parameters": {
            "prop_decrease": args.prop_decrease,
            "stationary": bool(args.stationary),
            "trim_silence": bool(args.trim_silence),
            "trim_to_seconds": args.trim_to,
        },
        "duration_valid": valid,
        "notes": notes,
        "transcript_output": str(transcript_out) if transcript_out else None,
        "patch_budget": patch_budget(cleaned_seconds, DEFAULT_PATCH_SECONDS),
    }
    return report, valid


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Denoise + validate a dots.tts reference audio clip.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", required=True, help="Source audio file.")
    parser.add_argument(
        "--output",
        "-o",
        default=str(settings.paths.clean_reference_wav),
        help="Destination WAV (48 kHz mono PCM_16).",
    )
    parser.add_argument(
        "--prop-decrease",
        type=float,
        default=settings.processing.noise_prop_decrease,
        help="noisereduce strength (0-1).",
    )
    parser.add_argument(
        "--stationary",
        dest="stationary",
        action="store_true",
        default=settings.processing.stationary_noise,
        help="Treat the noise floor as stationary (recommended for studio hiss).",
    )
    parser.add_argument(
        "--non-stationary",
        dest="stationary",
        action="store_false",
        help="Use the non-stationary noise estimator instead.",
    )
    parser.add_argument(
        "--no-trim-silence",
        dest="trim_silence",
        action="store_false",
        default=True,
        help="Keep leading/trailing silence.",
    )
    parser.add_argument(
        "--trim-to",
        type=positive_float,
        default=None,
        help=(
            "Keep the first N seconds (sentence-safe VRAM reduction). The "
            "matching transcript must be shortened accordingly!"
        ),
    )
    parser.add_argument(
        "--peak-dbfs",
        type=float,
        default=settings.processing.peak_dbfs,
        help="Target peak level after normalisation.",
    )
    parser.add_argument(
        "--transcript",
        default=None,
        help=(
            "Optional transcript side-car file to copy next to the cleaned WAV "
            "(written to <output>.txt)."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Write the output even when the duration validation fails.",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Optional path for a JSON report of the operation.",
    )
    parser.add_argument("--quiet", action="store_true", help="Print less.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report, valid = process(args)

    if args.report:
        report_path = Path(args.report).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    if not args.quiet:
        budget = report["patch_budget"]
        print("=" * 68)
        print("  Reference audio cleaned")
        print("=" * 68)
        print(f"  input           : {report['input']}")
        print(f"  output          : {report['output']}")
        print(
            f"  duration        : {report['source_seconds']}s -> "
            f"{report['cleaned_seconds']}s"
        )
        print(
            f"  format          : {report['sample_rate']} Hz / mono / "
            f"{report['subtype']}"
        )
        print(
            f"  peak            : {report['peak_dbfs_before']} -> "
            f"{report['peak_dbfs_target']} dBFS"
        )
        print(f"  duration valid  : {report['duration_valid']}")
        print(
            f"  prompt patches  : {budget['prompt_patches']} / "
            f"{budget['max_generate_length']} (generation room: "
            f"{budget['generation_seconds_available']}s)"
        )
        if report["transcript_output"]:
            print(f"  transcript      : {report['transcript_output']}")
        for note in report["notes"]:
            print(f"  NOTE            : {note}")
        print("=" * 68)

    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())

