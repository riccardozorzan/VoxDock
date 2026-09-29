#!/usr/bin/env python3
"""Attribute the per-request fixed generation cost (todo 4.3).

Why
---
The autotuner fits a cost model across the sweep:

    generation = fixed + rate x audio_seconds
    RTF        = 0.413 + 0.79 / audio_seconds

That says the ~0.79 s of *fixed* cost is why short speech can never reach
RTF 0.5, and that it is independent of reference length. What it does not say is
**where that 0.79 s goes**. The split decides whether it is worth attacking: if
it is dominated by prompt prefill then a shorter reference clip is the lever; if
it is a fixed per-call CUDA-graph replay, no configuration change will move it.

Method
------
``torch.profiler`` with CUDA activities, grouping the operator table by the
``dots_tts`` module that owns each op. Grouping by module rather than reading
raw op names is what makes the output readable, and prefill vs decode vs
vocoder is the distinction that actually matters.

This loads its own copy of the checkpoint, so **the service must be stopped
first** - two copies of a 4.5 GiB model will not fit in 6 GB.

    docker compose stop tts
    docker compose run --rm tts python scripts/profile_request.py
    docker compose up -d
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import LOG_NAMESPACE  # noqa: E402
from app.config import settings  # noqa: E402
from app.utils import now_iso  # noqa: E402

logger = logging.getLogger(f"{LOG_NAMESPACE}.profile")

#: Classification of a torch profiler event by op kind.
#:
#: The first attempt matched on module path (`vocoder`, `llm_inference`, ...),
#: which produced 100% "other": torch reports the *kernel*, not the Python
#: module that launched it, so the dots_tts module names never appear. What the
#: trace does contain is the op kind, which is enough to attribute the time -
#: BigVGAN is a convolutional decoder, so convolution is the vocoder; the
#: transformer stack is matmul and attention.
#:
#: Order matters: the first match wins.
OP_GROUPS: list[tuple[str, str]] = [
    # A full CUDA command queue with nothing to run means the GPU was starved
    # waiting for the host to launch work. That is a *host-bound* signal, not
    # GPU compute, and it is the most actionable number in the whole profile.
    ("command buffer", "GPU idle (host launch gap)"),
    # Convolution, its transpose, and the cudnn dgrad kernels are BigVGAN.
    ("cudnn_convolution_transpose", "vocoder (BigVGAN conv)"),
    ("cudnn_convolution", "vocoder (BigVGAN conv)"),
    ("dgrad", "vocoder (BigVGAN conv)"),
    ("convolution", "vocoder (BigVGAN conv)"),
    # Attention before generic matmul, so it is not swallowed by "linear".
    ("attention", "attention (LLM + DiT)"),
    ("scaled_dot_product", "attention (LLM + DiT)"),
    # Tensor-core matmuls: the transformer linear layers.
    ("cutlass", "linear layers (LLM + DiT)"),
    ("gemm", "linear layers (LLM + DiT)"),
    ("aten::mm", "linear layers (LLM + DiT)"),
    ("aten::addmm", "linear layers (LLM + DiT)"),
    ("aten::bmm", "linear layers (LLM + DiT)"),
    ("aten::linear", "linear layers (LLM + DiT)"),
    # Host/device transfers and normalisation.
    ("memcpy", "host<->device transfer"),
    ("memset", "host<->device transfer"),
    ("layer_norm", "normalisation"),
    ("softmax", "normalisation"),
]


def classify(name: str) -> str:
    """Map a torch op or module name onto the phase that owns it."""
    lowered = name.lower()
    for needle, label in OP_GROUPS:
        if needle in lowered:
            return label
    return "other"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Attribute the per-request fixed generation cost by phase.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--reference-audio",
        default=str(settings.paths.clean_reference_wav),
        help="Reference clip whose prompt prefill is being profiled.",
    )
    parser.add_argument(
        "--text",
        default=(
            "Il sole tramonta lentamente dietro le colline mentre la campana "
            "del paese suona lontano."
        ),
        help="Short text, so the fixed cost dominates the measurement.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Unprofiled runs first, so compile is not inside the profile.",
    )
    parser.add_argument("--rows", type=int, default=20, help="Top operators to print.")
    parser.add_argument("--report", default=None, help="Optional JSON report path.")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    reference = Path(args.reference_audio)
    if not reference.exists():
        print(f"[profile] reference not found: {reference}", file=sys.stderr)
        print("[profile] Run scripts/clean_audio.py first.", file=sys.stderr)
        return 1

    import torch

    from app.main import load_runtime
    from app.patches import apply_patches, assert_patches_applied

    apply_patches(settings.model.vocoder_chunk_patches)
    assert_patches_applied(
        {"compile": True, "fused_adaln": True, "generate_audio": True}
    )
    # load_runtime() returns a dict ({"runtime": ..., "precision": ...}); the
    # object exposing .generate() is the "runtime" entry.
    runtime = load_runtime()["runtime"]

    prompt_path = str(reference)
    prompt_text = reference.with_suffix(".txt").read_text(encoding="utf-8").strip()

    def generate() -> Any:
        # Keyword set mirrors the service's own call in app/main.py, so the
        # profile measures the same code path a real request takes.
        return runtime.generate(
            text=args.text,
            prompt_audio_path=prompt_path,
            prompt_text=prompt_text,
            num_steps=settings.model.default_num_steps,
            guidance_scale=settings.model.default_guidance_scale,
        )

    for index in range(args.warmup):
        logger.info("unprofiled warmup %d/%d", index + 1, args.warmup)
        generate()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    ) as prof:
        generate()
        torch.cuda.synchronize()

    events = prof.key_averages()
    print("\n" + "=" * 78)
    print("  Top operators by self CUDA time")
    print("=" * 78)
    # torch 2.8 has no key_averages_table() - it was removed - so the table is
    # formatted from the event objects directly. Same data, fewer columns.
    ranked = sorted(
        (e for e in events if (getattr(e, "self_device_time_total", 0) or 0) > 0),
        key=lambda e: e.self_device_time_total,
        reverse=True,
    )[: args.rows]
    print(f"  {'operator':<44} {'self ms':>9} {'calls':>7}")
    print("  " + "-" * 62)
    for event in ranked:
        print(
            f"  {event.key[:44]:<44} {event.self_device_time_total / 1000:9.2f} "
            f"{event.count:>7}"
        )

    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for event in events:
        spent = getattr(event, "self_device_time_total", 0) or 0
        if spent <= 0:
            continue
        label = classify(event.key)
        totals[label] += spent
        counts[label] += 1

    total_ms = sum(totals.values())
    print("=" * 78)
    print("  Attributed by phase (self CUDA time)")
    print("=" * 78)
    for label, spent in sorted(totals.items(), key=lambda kv: -kv[1]):
        share = (spent / total_ms * 100) if total_ms else 0.0
        print(
            f"  {label:<34} {spent / 1000:8.2f} ms  {share:5.1f}%  ({counts[label]} ops)"
        )

    report = {
        "timestamp": now_iso(),
        "reference": str(reference),
        "text": args.text,
        "num_steps": settings.model.default_num_steps,
        "total_self_cuda_ms": round(total_ms / 1000, 3),
        "by_phase": {
            label: {
                "self_cuda_ms": round(spent / 1000, 3),
                "share_percent": round(spent / total_ms * 100, 1) if total_ms else 0.0,
                "op_count": counts[label],
            }
            for label, spent in sorted(totals.items(), key=lambda kv: -kv[1])
        },
        "note": (
            "Self CUDA time only, so overlapping work is not double counted. The "
            "fitted model puts the per-request fixed cost at ~0.79 s; this says "
            "which phase that time belongs to."
        ),
    }
    if args.report:
        path = Path(args.report).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\n  report : {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
