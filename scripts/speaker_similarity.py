#!/usr/bin/env python3
"""Measure speaker similarity between a reference clip and synthesised audio.

Why this exists
---------------
Every quality claim in this project so far has been inferred from *performance*
numbers - RTF and peak VRAM - which say nothing about whether the output
actually sounds like the speaker. The checkpoint choice (``dots.tts-mf`` over
``soar``) and the sampling configuration (2 steps) were both decided on VRAM
grounds alone, with no measurement of the thing they actually trade against.

This scores that trade. It embeds a reference clip and one or more generated
files with a speaker-verification model (ECAPA-TDNN via ``speechbrain``) and
reports cosine similarity, which is the standard metric for this task.

Interpreting the numbers
------------------------
Cosine similarity is a *relative* measure. What matters is the ordering and the
gap to controls, not the absolute value, so this script always reports two
controls alongside the real score:

* **self** - the reference against itself. This is the ceiling, and anything
  below ~0.99 means the metric or the resampling is lossy rather than the clone
  being bad.
* **impostor** - the reference against a *different* speaker's clip. This is
  the floor, and a clone that scores near it is not cloning.

Without both, a single number cannot be interpreted: 0.75 might be excellent or
hopeless depending on the model's scale and the recording conditions.

Usage
-----
    # score generated audio against a reference
    docker compose exec tts python scripts/speaker_similarity.py \
        --reference data/input/reference_clean.wav \
        --generated data/output/test_audio_sample.wav

    # add an impostor control (a different speaker)
    docker compose exec tts python scripts/speaker_similarity.py \
        --reference data/references/my_voice_v2.wav \
        --generated data/output/test_audio_sample.wav \
        --impostor data/speakers/riccardo.wav

    # JSON report for tracking over time
    docker compose exec tts python scripts/speaker_similarity.py ... \
        --report data/output/similarity_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils import now_iso, wav_duration_seconds  # noqa: E402

#: ECAPA-TDNN is the standard lightweight speaker-verification model (~22 MB on
#: CPU), which keeps this usable on a 6 GB card that is already saturated by
#: the TTS engine.
DEFAULT_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
TARGET_SAMPLE_RATE = 16_000


def load_audio(path: Path, sample_rate: int = TARGET_SAMPLE_RATE):
    """Load a WAV as mono float32 at the model's expected rate."""
    import librosa

    audio, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    return audio


class Embedder:
    """Lazily-loaded speaker embedding model.

    The model is only instantiated on first use so ``--help`` and argument
    validation do not pay for a ~22 MB download, and so an import error in
    ``speechbrain`` surfaces at a useful point rather than at import time.
    """

    def __init__(self, model_id: str = DEFAULT_MODEL, device: str = "cpu") -> None:
        self.model_id = model_id
        self.device = device
        self._model = None

    def _ensure(self):
        if self._model is not None:
            return self._model
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise SystemExit(
                "speechbrain is not installed. Add it to requirements.txt and "
                "rebuild the image:\n"
                "    docker compose up -d --build\n"
                f"(import failed: {exc})"
            ) from exc
        # CPU by default: the GPU is already saturated by the TTS engine, and
        # this 22 MB model costs well under a second per clip on CPU.
        self._model = EncoderClassifier.from_hparams(
            source=self.model_id,
            savedir=os.environ.get("SPEECHBRAIN_CACHE", "/app/data/.speechbrain"),
            run_opts={"device": self.device},
        )
        return self._model

    def embed(self, audio) -> "Any":
        import torch

        model = self._ensure()
        tensor = torch.from_numpy(audio).unsqueeze(0)
        with torch.no_grad():
            embedding = model.encode_batch(tensor)
        # encode_batch returns (batch, 1, dim) for some speechbrain versions
        # and (batch, dim) for others, so flatten to a single 1-D vector rather
        # than assuming a rank.
        embedding = embedding.reshape(-1)
        # L2-normalise so the dot product below is a true cosine similarity
        # and cannot be inflated by a longer or louder clip.
        return torch.nn.functional.normalize(embedding, dim=0)


def cosine(a: "Any", b: "Any") -> float:
    """Cosine similarity of two already-normalised embeddings, in [-1, 1]."""
    import torch

    return float(torch.dot(a, b).item())


def relative_position(score: float, low: float, high: float) -> Optional[float]:
    """Where ``score`` sits between two control scores, as a 0..1 fraction.

    Returns ``None`` when the controls are indistinguishable, which would make
    the ratio meaningless rather than merely uninformative.
    """
    span = high - low
    if span <= 1e-6:
        return None
    return (score - low) / span


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score how closely synthesised audio matches a reference voice.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--reference", required=True, help="Reference WAV the voice was cloned from."
    )
    parser.add_argument(
        "--generated",
        nargs="+",
        required=True,
        help="One or more generated WAVs to score against the reference.",
    )
    parser.add_argument(
        "--impostor",
        default=None,
        help="A different speaker's WAV, reported as the floor. A clone scoring "
        "near it is not cloning.",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="Any speechbrain speaker-encoder id."
    )
    parser.add_argument(
        "--device", default="cpu", help="cpu or cuda. CPU is ample for this model."
    )
    parser.add_argument("--report", default=None, help="Optional JSON report path.")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    reference = Path(args.reference)
    if not reference.exists():
        print(f"[sim] reference not found: {reference}", file=sys.stderr)
        return 1
    missing = [p for p in args.generated if not Path(p).exists()]
    for path in missing:
        print(f"[sim] generated file not found: {path}", file=sys.stderr)
    if missing:
        return 1
    if args.impostor and not Path(args.impostor).exists():
        print(f"[sim] impostor not found: {args.impostor}", file=sys.stderr)
        return 1

    embedder = Embedder(args.model, args.device)
    reference_embedding = embedder.embed(load_audio(reference))

    # ---- controls ---------------------------------------------------------- #
    self_score = cosine(reference_embedding, reference_embedding)
    print(f"  model           : {args.model}")
    print(f"  reference       : {reference} ({wav_duration_seconds(reference):.2f}s)")
    print(f"  control self    : {self_score:.4f}   (ceiling - below 0.99 means "
          f"the metric is lossy, not the clone)")

    impostor_score: Optional[float] = None
    if args.impostor:
        impostor_embedding = embedder.embed(load_audio(Path(args.impostor)))
        impostor_score = cosine(reference_embedding, impostor_embedding)
        print(f"  control impostor: {impostor_score:.4f}   (floor - a different speaker)")

    low = impostor_score if impostor_score is not None else 0.0

    # ---- score ------------------------------------------------------------- #
    results = []
    for path in args.generated:
        candidate = Path(path)
        score = cosine(reference_embedding, embedder.embed(load_audio(candidate)))
        relative = relative_position(score, low, self_score)
        results.append(
            {
                "file": str(candidate),
                "seconds": round(wav_duration_seconds(candidate), 3),
                "similarity": round(score, 4),
                "relative_to_controls": round(relative, 4) if relative is not None else None,
            }
        )
        suffix = (
            f"   ({relative * 100:.0f}% of the way from impostor to self)"
            if relative is not None
            else ""
        )
        print(f"  {candidate.name:<30} {score:.4f}{suffix}")

    report = {
        "timestamp": now_iso(),
        "model": args.model,
        "reference": str(reference),
        "reference_seconds": round(wav_duration_seconds(reference), 3),
        "controls": {
            "self": round(self_score, 4),
            "impostor": round(impostor_score, 4) if impostor_score is not None else None,
        },
        "generated": results,
        "note": (
            "Cosine similarity is relative. Judge the generated scores against "
            "the two controls, not against an absolute threshold."
        ),
    }
    if args.report:
        report_path = Path(args.report).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"  report          : {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

