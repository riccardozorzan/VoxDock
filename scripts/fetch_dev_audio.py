"""Fetch the non-personal assets a fresh clone needs.

Why this exists
---------------
Everything under ``data/`` that matters is gitignored on purpose - a voice is
personal and large - so a clean checkout has no audio at all. Two consequences:

* ``scripts/speaker_similarity.py`` cannot report its *impostor floor*, which is
  the whole reason its scores are interpretable. A cosine similarity without a
  different-speaker control is a number with no scale.
* the autotuner's three-reference sweep has nothing to sweep.

This script closes the first gap. The two control clips are the example files
published alongside the ``speechbrain/spkrec-ecapa-voxceleb`` model (VoxCeleb
speakers id10003 and id10004) - public benchmark audio, about 140 KB together,
and nothing here is personal. The control files are not shipped in the repository
because the whole ``data/references/controls/`` directory is ignored, not because
they are private.

What this cannot do
-------------------
Your own reference voice. No script can produce that: continuation cloning needs
the transcript to match the audio word for word, and no tool here can transcribe
your voice for you. Record 5-15 s yourself and run ``scripts/clean_audio.py``;
``scripts/make_reference_set.py`` then writes the ``reference_set.json`` the
autotuner expects.

Usage
-----
    python scripts/fetch_dev_audio.py                 # fetch what is missing
    python scripts/fetch_dev_audio.py --check         # report, download nothing
    python scripts/fetch_dev_audio.py --force         # re-download

    python scripts/speaker_similarity.py \\
        --reference data/references/my_voice_v2.wav \\
        --generated data/output/test_audio_sample.wav \\
        --impostor data/references/controls/example1.wav
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings  # noqa: E402

#: ``(filename, source)`` for the control clips shipped with the ECAPA model.
BASE_URL = "https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/resolve/main/"
ASSETS = (
    ("example1.wav", f"{BASE_URL}example1.wav"),
    ("example2.flac", f"{BASE_URL}example2.flac"),
)

#: Anything smaller than this is an error page or a redirect stub, not audio.
MIN_BYTES = 1024


def download(url: str, destination: Path, timeout: float) -> int:
    """Fetch ``url`` into ``destination`` atomically. Returns the byte count."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    scratch = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = response.read()
        if len(payload) < MIN_BYTES:
            raise ValueError(
                f"{url} returned {len(payload)} bytes - that is an error page, not audio"
            )
        scratch.write_bytes(payload)
        scratch.replace(destination)
        return len(payload)
    except urllib.error.URLError as exc:
        raise SystemExit(f"[fetch] could not download {url}: {exc}") from exc
    finally:
        scratch.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch the public ECAPA control clips a fresh clone needs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dest",
        default=str(settings.paths.references_dir / "controls"),
        help="Directory the control clips are written to.",
    )
    parser.add_argument("--force", action="store_true", help="Re-download existing files.")
    parser.add_argument("--check", action="store_true", help="Report status, download nothing.")
    parser.add_argument("--timeout", type=float, default=60.0, help="Per-request timeout (s).")
    args = parser.parse_args()

    destination_dir = Path(args.dest).expanduser()
    if not destination_dir.is_absolute():
        destination_dir = PROJECT_ROOT / destination_dir

    if destination_dir.exists() and not destination_dir.is_dir():
        raise SystemExit(f"[fetch] {destination_dir} exists and is not a directory")

    print(f"[fetch] destination: {destination_dir}")
    missing: list[str] = []
    for name, url in ASSETS:
        target = destination_dir / name
        if target.exists() and target.stat().st_size >= MIN_BYTES and not args.force:
            print(f"  [ok]    {name:<14} {target.stat().st_size:>7} bytes  (already present)")
            continue
        if args.check:
            print(f"  [missing] {name}")
            missing.append(name)
            continue
        size = download(url, target, args.timeout)
        print(f"  [saved] {name:<14} {size:>7} bytes")

    if missing:
        print(
            "\n[fetch] missing: " + ", ".join(missing) + "\n"
            "        re-run without --check to download them.",
            file=sys.stderr,
        )
        return 1

    if not args.check:
        first = destination_dir / ASSETS[0][0]
        print(
            "\n[fetch] the impostor floor for speaker_similarity.py:\n"
            f"        --impostor {first.relative_to(PROJECT_ROOT) if first.is_relative_to(PROJECT_ROOT) else first}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
