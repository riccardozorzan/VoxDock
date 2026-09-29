"""Write the ``reference_set.json`` the autotuner sweeps, from your own clips.

Why this exists
---------------
``data/input/reference_set.json`` is committed, but every ``.wav`` it points at is
gitignored. On a fresh clone the autotuner therefore finds a manifest full of
missing files and quietly falls back to a single ``--reference-audio`` clip
(``autotune.py`` handles the failure and says so), so the "sweep the reference
length" feature is unavailable to anyone but the machine that recorded the voice.

Editing that JSON by hand is also easy to get wrong in a way nothing catches: the
reference-prompt prefill is a *fixed* cost on every request, so clip length is the
dominant lever for short-utterance RTF, and the patch arithmetic (1 patch = 160 ms,
prompt patches count against ``TTS_MAX_GENERATE_LENGTH``) is exactly the part you
should not be doing in a text editor.

This script takes the clips and transcripts you already have, validates them, and
writes the manifest with the arithmetic visible. It cannot invent a transcript:
continuation cloning needs the text to match the audio word for word, so every
clip must come with a ``.txt`` you wrote.

Usage
-----
    python scripts/make_reference_set.py \\
        --ref ref_s "1 sentence"  data/input/ref_s.wav  data/input/ref_s.txt \\
        --ref ref_m "2 sentences" data/input/ref_m.wav data/input/ref_m.txt

Then point the autotuner at it:

    docker compose exec tts python scripts/autotune.py \\
        --reference-set data/input/reference_set.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings  # noqa: E402
from scripts.autotune import load_reference_seconds, repo_relative  # noqa: E402

#: One latent patch = 160 ms of audio, per the dots.tts AudioVAE config.
PATCH_SECONDS = 0.16

#: Generation needs room left over. 32 patches is 5.1 s of audio - the shortest
#: utterance worth synthesising - so below that the reference is eating the
#: request budget. At the shipped settings (5-15 s clips, 128 patches) the worst
#: case uses 94 patches and leaves 34, so this does not fire by default: it
#: exists for a tighter budget (TTS_MAX_GENERATE_LENGTH=96) or a longer
#: TTS_REF_MAX_SECONDS.
MIN_GENERATION_PATCHES = 32


def _resolve(value: str) -> Path:
    """Absolute path, relative to the project root when not already absolute."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def build_entries(
    refs: list[tuple[str, str, str, str]],
    *,
    min_seconds: float,
    max_seconds: float,
    budget_patches: int,
    min_generation_patches: int = MIN_GENERATION_PATCHES,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Validate every reference; return ``(entries, warnings)``.

    Raises ``SystemExit`` on anything unusable: a manifest that silently carries a
    bad entry is worse than no manifest, because the autotuner would measure
    against it and report a number that means nothing.
    """
    entries: list[dict[str, Any]] = []
    warnings: list[str] = []
    seen: set[str] = set()

    for name, label, audio_arg, text_arg in refs:
        if name in seen:
            raise SystemExit(f"[refset] duplicate name: {name}")
        seen.add(name)

        audio = _resolve(audio_arg)
        text = _resolve(text_arg)
        if not audio.exists():
            raise SystemExit(f"[refset] {name}: audio not found: {audio}")
        if not text.exists():
            raise SystemExit(
                f"[refset] {name}: transcript not found: {text}\n"
                "        The transcript must match the audio word for word - "
                "continuation cloning has no other alignment cue."
            )
        if not text.read_text(encoding="utf-8").strip():
            raise SystemExit(f"[refset] {name}: transcript is empty: {text}")

        seconds = load_reference_seconds(audio)
        if seconds > max_seconds:
            raise SystemExit(
                f"[refset] {name}: {seconds:.2f}s exceeds the {max_seconds:g}s ceiling "
                "that clean_audio.py enforces.\n"
                "        Re-cut it with: python scripts/clean_audio.py --trim-to <seconds>"
            )

        patches = int(seconds / PATCH_SECONDS + 0.999)
        remaining = budget_patches - patches
        if remaining <= 0:
            raise SystemExit(
                f"[refset] {name}: {patches} prompt patches fill the whole "
                f"{budget_patches}-patch budget ({seconds:.2f}s), so the server would "
                "reject every request made with this reference.\n"
                "        Shorten it with clean_audio.py --trim-to, or raise "
                "TTS_MAX_GENERATE_LENGTH if the GPU has the headroom."
            )
        entries.append(
            {
                "name": name,
                "label": label,
                "audio": repo_relative(audio),
                "text": repo_relative(text),
                "seconds": round(seconds, 2),
                "patches": patches,
            }
        )
        if remaining < min_generation_patches:
            warnings.append(
                f"{name}: {patches} of {budget_patches} patches are prompt ({seconds:.2f}s), "
                f"leaving {remaining} ({remaining * PATCH_SECONDS:.1f}s) to generate - "
                "shorter than a usable utterance"
            )
        if seconds < min_seconds:
            # Not fatal, and deliberately so: TTS_REF_MIN_SECONDS is clean_audio's
            # intake rule for a *raw recording*, whereas a reference for the sweep is
            # usually a slice cut out of one take. The committed ref_s is 4.05 s and
            # perfectly valid. A raw clip that short is simply worth flagging.
            warnings.append(
                f"{name}: {seconds:.2f}s is under the {min_seconds:g}s clean_audio.py asks "
                "of a recording - fine if it was cut from a longer take, thin if it is the "
                "whole recording"
            )

    return entries, warnings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write a reference_set.json for scripts/autotune.py from your clips.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ref",
        action="append",
        nargs=4,
        metavar=("NAME", "LABEL", "AUDIO", "TEXT"),
        required=True,
        help="One reference: name, label, WAV, transcript. Repeatable.",
    )
    parser.add_argument(
        "--out",
        default=str(settings.paths.input_dir / "reference_set.json"),
        help="Manifest to write.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite an existing manifest.")
    args = parser.parse_args()

    out_path = _resolve(args.out)
    if out_path.exists() and not args.force:
        raise SystemExit(
            f"[refset] {out_path} already exists; pass --force to overwrite it.\n"
            "        It is committed to git, so the change shows up in git diff."
        )

    processing = settings.processing
    entries, warnings = build_entries(
        [tuple(item) for item in args.ref],
        min_seconds=processing.reference_min_seconds,
        max_seconds=processing.reference_max_seconds,
        budget_patches=settings.model.max_generate_length,
    )

    payload = {
        "_comment": [
            "Reference clips the autotuner may choose between.",
            "The reference-prompt prefill is a FIXED cost paid on every request, so its",
            "length is usually the dominant lever for short-utterance RTF -- more",
            "important than any env setting. Continuation cloning needs the transcript to",
            "match the audio exactly, hence one .txt per .wav.",
            f"Generated by scripts/make_reference_set.py; 1 patch = {PATCH_SECONDS}s of audio.",
        ],
        "references": [
            {key: entry[key] for key in ("name", "label", "audio", "text")}
            for entry in entries
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    budget = settings.model.max_generate_length
    print(f"[refset] wrote {out_path}")
    print(f"  {'name':<10} {'seconds':>8} {'patches':>8}  label")
    for entry in entries:
        print(
            f"  {entry['name']:<10} {entry['seconds']:>8.2f} {entry['patches']:>8}  "
            f"{entry['label']}"
        )
    for warning in warnings:
        print(f"[refset] warning: {warning}", file=sys.stderr)
    print(
        f"\n[refset] budget: {budget} patches total "
        f"({budget * PATCH_SECONDS:.1f}s of audio, prompt included)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
