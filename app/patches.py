"""Low-VRAM patches for ``dots_tts`` on a 6 GB RTX 4050 Laptop GPU.

Why this module exists
----------------------
``dots.tts`` (the ``dots-studio/dots.tts-mf`` MeanFlow checkpoint) needs
~4.55 GiB of bfloat16 weights on the GPU. On the 5.64 GiB usable RTX 4050 that
leaves roughly 1.1 GiB for activations, KV caches and the BigVGAN vocoder, and
the stock inference path spends that headroom very inefficiently:

1. **DiT AdaLN fusion doubles the modulation weights.**
   :class:`~dots_tts.modules.backbone.dit_inference.FusedAdaLNDiT.__init__` builds
   a single ``fused_adaln`` projection of shape
   ``(model_dim * (6 * num_layers + 2), model_dim)`` - 149 504 x 1024, i.e. 292 MiB
   in bfloat16 - then copies the per-block ``adaLN_modulation`` linears into it
   with one ``torch.cat``. Two problems: the ``cat`` needs *another* 292 MiB
   scratch buffer (which is what pushed eager ``max_generate_length=128`` ~50 MiB
   over the physical limit), and the original per-block linears (another 292 MiB)
   are never released.

2. **The BigVGAN decoder is decoded in one shot.**
   ``generate_audio`` concatenates every latent patch and runs the full decoder,
   whose activations scale linearly with generated duration. On this GPU that is
   the single largest transient allocation and OOMs even when weights fit.

3. **The static generate workspace is retained during vocoder decode.**
   ``_static_generate_workspaces`` pins the FM sequence/cfg buffers for the whole
   process lifetime, so the vocoder has to fit around them.

What this module changes
------------------------
* :func:`patch_fused_adaln_inplace` - chunked, in-place AdaLN fusion: copies each
  linear into its slice of ``fused_adaln`` and immediately drops the source
  weight (``lin.weight = None``), so the peak transient is one block's weights
  rather than a full duplicate, and the un-fused copies are freed.
* :func:`patch_generate_audio` - frees the generate workspaces, runs
  ``gc.collect()`` + ``torch.cuda.empty_cache()`` and then decodes the vocoder in
  fixed-size patch chunks, so vocoder activations stay bounded no matter how
  long the utterance is.
* :func:`apply_patches` - idempotent entry point installing the above.

All patches are no-ops when the upstream internals do not match expectations, so
a future ``dots.tts`` release degrades to stock behaviour instead of crashing.
"""

from __future__ import annotations

import gc
import logging
from typing import Any

from app import LOG_NAMESPACE

logger = logging.getLogger(f"{LOG_NAMESPACE}.patches")

__all__ = [
    "VOCODER_CHUNK_PATCHES",
    "REQUIRED_PATCHES",
    "PatchFailure",
    "apply_patches",
    "assert_patches_applied",
    "disable_cudagraphs",
    "patch_fused_adaln_inplace",
    "patch_generate_audio",
]

# Number of latent patches decoded per BigVGAN call. 12 patches ~= 1.92 s of
# audio, which keeps the decoder's transient activations around 220 MiB on a
# 6 GB card while still amortising per-call overhead.
VOCODER_CHUNK_PATCHES = 12

# Guard so repeated calls (e.g. a reload or a second import path) stay cheap.
_APPLIED = False

#: The patches that MUST have applied on a 6 GB card, keyed by the name
#: :func:`apply_patches` reports. Losing any of these is not a graceful
#: degradation - the process starts and then OOMs at ``max_generate_length=128``
#: some minutes later, with nothing in the log but a single WARNING line pointing
#: at an upstream rename. :func:`assert_patches_applied` turns that into a loud,
#: immediate failure instead.
REQUIRED_PATCHES = ("compile", "fused_adaln", "generate_audio")


class PatchFailure(RuntimeError):
    """A low-VRAM patch that is required on this GPU did not apply.

    Raised by :func:`assert_patches_applied`. This is deliberately fatal rather
    than a warning: on a 6 GB card these patches are what make the verified
    configuration fit at all, and silently running without them trades a clear
    error message for a confusing OOM minutes later.
    """


def assert_patches_applied(
    result: dict[str, bool], *, required: tuple[str, ...] = REQUIRED_PATCHES
) -> None:
    """Raise :class:`PatchFailure` if any required patch did not apply.

    The patches are best-effort by nature - each one is guarded so a future
    ``dots.tts`` release degrades to stock behaviour instead of crashing at
    import. That is the right default *when the patches are optional*, but on
    this hardware they are not: the stock path needs roughly 600 MB more than a
    6 GB card has. So the outcome is reported once, loudly, at startup.
    """
    missing = [name for name in required if not result.get(name)]
    if missing:
        raise PatchFailure(
            "low-VRAM patches did not apply: "
            + ", ".join(missing)
            + f" (got {result}). This GPU needs them to fit "
            "max_generate_length=128; without them the service will almost "
            "certainly OOM during warmup or the first request. Either pin "
            "dots.tts to a compatible version, or set "
            "TTS_LOW_VRAM_PATCHES=false to run unpatched and accept the lower "
            "TTS_MAX_GENERATE_LENGTH. See app/patches.py and the startup log for "
            "which upstream symbol each patch could not find."
        )


# --------------------------------------------------------------------------- #
#  1. Chunked, in-place AdaLN fusion
# --------------------------------------------------------------------------- #
def patch_fused_adaln_inplace() -> bool:
    """Replace ``FusedAdaLNDiT.__init__`` with a memory-frugal variant.

    Upstream allocates the fused projection and then ``torch.cat``s the sources
    into it, so the fused weights, a full-size scratch copy *and* the originals
    are all live at once. We copy block-by-block into the destination slices and
    null out each source tensor as soon as it is consumed, which removes the
    scratch buffer entirely and releases the un-fused weights.
    """
    try:
        import torch
        import torch.nn as nn

        from dots_tts.modules.backbone import dit_inference as dit_inf
    except Exception:  # pragma: no cover - upstream layout changed
        logger.warning(
            "dots_tts DiT inference module unavailable; skipping AdaLN fusion patch"
        )
        return False

    if getattr(dit_inf.FusedAdaLNDiT.__init__, "_low_vram_patch", False):
        return True

    def __init__(self: Any, old_dit: nn.Module) -> None:  # noqa: N807
        nn.Module.__init__(self)
        if any(not block.modulation for block in old_dit.blocks):
            raise ValueError(
                "FusedAdaLNDiT requires modulation=True for every block."
            )
        for block in old_dit.blocks:
            block.attn = dit_inf.FusedQKVMultiHeadAttention(block.attn)

        self.input_layer = old_dit.input_layer
        self.time_embedder = old_dit.time_embedder
        self.duration_embedder = getattr(old_dit, "duration_embedder", None)
        self.blocks = old_dit.blocks
        self.output_layer = old_dit.output_layer

        model_dim = int(self.input_layer.out_features)
        adaln_linears = [block.adaLN_modulation[-1] for block in self.blocks]
        adaln_linears.append(self.output_layer.adaLN_modulation[-1])
        first_linear = adaln_linears[0]

        out_dim = model_dim * (6 * len(self.blocks) + 2)
        self.fused_adaln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_dim, out_dim, bias=first_linear.bias is not None),
        )
        self.fused_adaln.to(
            device=first_linear.weight.device,
            dtype=first_linear.weight.dtype,
        )
        with torch.no_grad():
            offset = 0
            for lin in adaln_linears:
                rows = int(lin.weight.shape[0])
                # Copy straight into the destination slice: no torch.cat scratch.
                self.fused_adaln[-1].weight[offset : offset + rows].copy_(lin.weight)
                # Release the un-fused weight immediately.
                lin.weight = None
                if first_linear.bias is not None and lin.bias is not None:
                    self.fused_adaln[-1].bias[offset : offset + rows].copy_(lin.bias)
                    lin.bias = None
                offset += rows

        for block in self.blocks:
            block.adaLN_modulation = nn.Identity()
        self.output_layer.adaLN_modulation = nn.Identity()

    __init__._low_vram_patch = True  # type: ignore[attr-defined]
    dit_inf.FusedAdaLNDiT.__init__ = __init__
    logger.info("Applied low-VRAM AdaLN fusion patch (chunked, frees source weights)")
    return True


# --------------------------------------------------------------------------- #
#  2. Chunked vocoder decode + workspace release
# --------------------------------------------------------------------------- #
def patch_generate_audio(chunk_patches: int = VOCODER_CHUNK_PATCHES) -> bool:
    """Decode the BigVGAN vocoder in bounded chunks.

    Upstream concatenates all latent patches and decodes once, so the decoder's
    activation peak grows with utterance length. We release the generate
    workspaces first, then walk the patches in fixed-size chunks and concatenate
    the resulting waveforms. The AudioVAE decoder is causal with strided
    upsampling, so each chunk decodes independently of the others.
    """
    try:
        from dots_tts.models.dots_tts.model import DotsTtsModel
    except Exception:  # pragma: no cover - upstream layout changed
        logger.warning("DotsTtsModel unavailable; skipping vocoder chunking patch")
        return False

    if getattr(DotsTtsModel.generate_audio, "_low_vram_patch", False):
        return True

    import torch

    @torch.no_grad()
    def generate_audio(  # noqa: PLR0913 - mirrors the upstream signature
        self: Any,
        data: dict[str, Any],
        *,
        precision: str,
        ode_method: str,
        num_steps: int,
        guidance_scale: float,
        speaker_scale: float = 1.5,
    ) -> torch.Tensor:
        latent_patches = list(
            self._generate_latents_stream(
                data,
                precision=precision,
                ode_method=ode_method,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                speaker_scale=speaker_scale,
            )
        )
        logger.debug(
            "Vocoder decode starting: latent_patch_count=%s chunk_patches=%s",
            len(latent_patches),
            chunk_patches,
        )

        # The DiT/LLM scratch buffers are dead once the latents exist; release
        # them so the vocoder can use that memory instead of fragmenting it.
        try:
            self._static_generate_workspaces.clear()
        except Exception:  # pragma: no cover - attribute renamed upstream
            logger.debug("no static generate workspaces to clear", exc_info=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        adapter = self._get_vocoder_inference()
        total = len(latent_patches)
        if total <= chunk_patches:
            return adapter.decode_latents(torch.cat(latent_patches, dim=1))

        waveform_parts = []
        for start in range(0, total, chunk_patches):
            chunk = torch.cat(latent_patches[start : start + chunk_patches], dim=1)
            waveform_parts.append(adapter.decode_latents(chunk))
            del chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return torch.cat(waveform_parts, dim=-1)

    generate_audio._low_vram_patch = True  # type: ignore[attr-defined]
    DotsTtsModel.generate_audio = generate_audio
    logger.info("Applied chunked-vocoder patch (chunk_patches=%s)", chunk_patches)
    return True


# --------------------------------------------------------------------------- #
#  0. Drop CUDA graphs from the compiled inference steps
# --------------------------------------------------------------------------- #
def disable_cudagraphs() -> bool:
    """Strip CUDA graphs from every ``torch.compile`` the runtime performs.

    ``dots_tts.modules.backbone.inference_utils._COMPILE_OPTIONS`` compiles the
    LLM step, the DiT decode/prefill steps and the vocoder stream step with
    ``mode="reduce-overhead"``, which is a request for Inductor's cudagraph
    trees. Those trees (a) reserve a private memory pool -- 146-170 MiB on this
    checkpoint, per the OOM reports -- and (b) checkpoint the caching allocator
    around replays, which on a card with ~1 GiB of headroom aborts with::

        RuntimeError: Expected curr_block->next == nullptr to be true, but got false

    Rewriting the options to ``mode="default"`` removes both problems. The cost
    is a little extra kernel-launch overhead per decode step; the benefit is
    that the optimized path actually fits next to the vocoder on a 6 GB card.
    """
    try:
        from dots_tts.modules.backbone import inference_utils
    except Exception:  # pragma: no cover - upstream layout changed
        logger.warning("dots_tts inference_utils unavailable; skipping compile patch")
        return False

    options = getattr(inference_utils, "_COMPILE_OPTIONS", None)
    if not isinstance(options, dict):  # pragma: no cover - renamed upstream
        logger.warning("_COMPILE_OPTIONS not a dict (%r); skipping compile patch", options)
        return False

    original = dict(options)
    if options.get("mode") == "default" and "options" in options:
        return True

    # NOTE: torch.compile rejects `mode` and `options` together, so
    # `mode="default"` alone is the switch; `triton.cudagraphs` is then set
    # globally below in case anything else compiles.
    options.pop("options", None)
    options["mode"] = "default"
    logger.info("Patched dots_tts compile options: %s -> %s", original, dict(options))

    # The block-mask helpers are compiled at import time with their own options.
    try:
        import torch._inductor.config as inductor_config

        inductor_config.triton.cudagraphs = False
    except Exception:  # pragma: no cover - torch build without inductor
        logger.debug("could not set inductor_config.triton.cudagraphs", exc_info=True)
    return True


# --------------------------------------------------------------------------- #
#  Entry point
# --------------------------------------------------------------------------- #
def apply_patches(chunk_patches: int = VOCODER_CHUNK_PATCHES) -> dict[str, bool]:
    """Install every low-VRAM patch exactly once. Safe to call repeatedly."""
    global _APPLIED
    if _APPLIED:
        return {
            "compile": True,
            "fused_adaln": True,
            "generate_audio": True,
        }

    result = {
        # Must run first: it changes how every module is compiled, so it has to
        # be in effect before the runtime builds its solvers.
        "compile": disable_cudagraphs(),
        "fused_adaln": patch_fused_adaln_inplace(),
        "generate_audio": patch_generate_audio(chunk_patches),
    }
    _APPLIED = True
    logger.info("Low-VRAM patches applied: %s", result)
    return result
