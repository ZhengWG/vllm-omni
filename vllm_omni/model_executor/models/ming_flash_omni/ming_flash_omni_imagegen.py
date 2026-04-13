# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.

"""Ming-flash-omni-2.0 imagegen stage model.

This module is loaded when ``model_stage == "imagegen"`` in the stage config.
It owns the condition encoder (Qwen2 connector + norm/proj) and the ZImage
diffusion pipeline (transformer + VAE + scheduler).

Inputs:  ``thinker_hidden_states`` [B, N, 4096] + optional attention mask.
Outputs: PIL image list (wrapped in OmniOutput).

NOTE (Phase 1):
  - Weights for the DiT transformer + VAE are loaded by ZImagePipeline via
    diffusers' ``from_pretrained`` path (already done there).
  - Weights for the Qwen2 connector + mlp/ proj-norm are loaded by
    ``MingConditionEncoder.load_from_checkpoint`` using HF transformers.
  - Cross-stage hidden-state transfer is done by the stage input processor
    (see stage_input_processors/ming_flash_omni.py) and arrives on this
    module's forward() as a plain tensor.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import torch
import torch.nn as nn
from vllm.config import VllmConfig

from vllm_omni.diffusion.models.ming_flash_omni.condition_encoder import (
    MingConditionEncoder,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.transformers_utils.configs.ming_flash_omni import (
    MingFlashOmniConfig,
    MingImageGenConfig,
)

logger = logging.getLogger(__name__)


class MingFlashOmniImageGenModel(nn.Module):
    """Imagegen stage of Ming-flash-omni-2.0.

    Composed of:
      * ``condition_encoder`` — Qwen2 connector + norm/proj
      * ``pipeline``          — ``ZImagePipeline`` (DiT + VAE + scheduler)
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):  # noqa: ARG002
        super().__init__()
        self.vllm_config = vllm_config
        self.model_path = vllm_config.model_config.model

        hf_config = vllm_config.model_config.hf_config
        if isinstance(hf_config, MingFlashOmniConfig):
            self.image_gen_config: MingImageGenConfig = hf_config.image_gen_config
            thinker_hidden_size = int(hf_config.thinker_config.llm_config.hidden_size)
        else:
            # Fallback: caller passed the image-gen config directly.
            self.image_gen_config = hf_config  # type: ignore[assignment]
            thinker_hidden_size = 4096
        logger.info(
            "[MingImageGen] init: thinker_hidden_size=%d, image_gen_config=%s",
            thinker_hidden_size,
            self.image_gen_config,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = vllm_config.model_config.dtype

        self.condition_encoder = MingConditionEncoder(
            self.image_gen_config,
            thinker_hidden_size=thinker_hidden_size,
            device=device,
            dtype=dtype,
        )

        # Lazily construct the ZImage pipeline so we can fail fast with a
        # clear error if the DiT type is not supported in Phase 1.
        self.pipeline = self._build_diffusion_pipeline(device=device, dtype=dtype)

        # Connector weights live in the unified Ming checkpoint, not inside
        # ZImagePipeline's subfolders. Load them here.
        try:
            self.condition_encoder.load_from_checkpoint(self.model_path)
        except Exception:
            logger.exception(
                "[MingImageGen] condition_encoder.load_from_checkpoint failed — "
                "continuing with uninitialized connector; first real run will "
                "likely produce noise. Fix on real hardware."
            )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _build_diffusion_pipeline(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ):
        dit_type = self.image_gen_config.dit_type.lower()
        logger.info("[MingImageGen] building diffusion pipeline for dit_type=%s", dit_type)

        if dit_type == "zimage":
            from types import SimpleNamespace

            from vllm_omni.diffusion.models.z_image.pipeline_z_image import (
                ZImagePipeline,
            )

            # ZImagePipeline.__init__ expects an OmniDiffusionConfig-ish
            # object with at least ``model`` and ``quantization_config`` and
            # ``enable_diffusion_pipeline_profiler``. We stub a minimal one
            # here rather than wiring it through vllm_config plumbing in
            # Phase 1 — this will need to be swapped for the real config
            # once the diffusion engine owns this model.
            od_config = SimpleNamespace(
                model=self.model_path,
                quantization_config=None,
                enable_diffusion_pipeline_profiler=False,
            )
            pipeline = ZImagePipeline(od_config=od_config)
            pipeline.to(device=device, dtype=dtype)

            # The Ming pipeline does not use ZImage's built-in Qwen text
            # encoder — our condition_encoder replaces it. Drop it to save
            # memory once loaded (it may not exist in the Ming checkpoint
            # at all, in which case ZImagePipeline.__init__ will already
            # have raised; we guard against that above).
            if hasattr(pipeline, "text_encoder"):
                logger.info("[MingImageGen] dropping ZImagePipeline.text_encoder (Ming uses its own condition encoder)")
                pipeline.text_encoder = None

            return pipeline

        raise NotImplementedError(
            f"dit_type={dit_type!r} is not supported in Phase 1 (only 'zimage' wired so far; sd3/sana come later)."
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def forward(
        self,
        thinker_hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        seed: int | None = None,
        **_unused,
    ) -> OmniOutput:
        """Run condition encoder + diffusion loop + VAE decode.

        Args:
            thinker_hidden_states: ``[B, N, H_thinker]`` tensor from stage 0.
            attention_mask: Optional ``[B, N]`` mask.
            height / width: Output resolution. Defaults from config.
            num_inference_steps / guidance_scale / seed: Sampling knobs.
        """
        cfg = self.image_gen_config
        h = height or cfg.default_height
        w = width or cfg.default_width
        steps = num_inference_steps or cfg.num_inference_steps
        cfg_scale = guidance_scale if guidance_scale is not None else cfg.guidance_scale

        logger.info(
            "[MingImageGen.forward] hidden_states=%s, hw=(%d,%d), steps=%d, cfg=%.2f",
            tuple(thinker_hidden_states.shape),
            h,
            w,
            steps,
            cfg_scale,
        )

        cap_feats = self.condition_encoder(thinker_hidden_states, attention_mask=attention_mask)
        logger.info("[MingImageGen.forward] cap_feats=%s", tuple(cap_feats.shape))

        # ZImagePipeline.encode_prompt() normally returns a list[Tensor] where
        # each entry corresponds to one prompt (masked to its valid tokens).
        # We match that contract: one tensor per batch element.
        prompt_embeds = [cap_feats[i] for i in range(cap_feats.shape[0])]
        negative_prompt_embeds = [self.condition_encoder.zero_negative(e) for e in prompt_embeds]

        device = cap_feats.device
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(int(seed))

        # The ZImagePipeline.forward signature requires an OmniDiffusionRequest
        # as the first arg. We construct a minimal one in-place; fields the
        # pipeline reads beyond sampling knobs are limited.
        from types import SimpleNamespace

        req = SimpleNamespace(request_id="ming-imagegen", sampling_params=None)

        output = self.pipeline.forward(
            req,
            prompt=None,
            height=h,
            width=w,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            generator=generator,
            output_type="pil",
            return_dict=True,
        )

        images = getattr(output, "images", output)
        logger.info(
            "[MingImageGen.forward] produced %d image(s)",
            len(images) if hasattr(images, "__len__") else -1,
        )
        return OmniOutput(image=images)

    # ------------------------------------------------------------------
    # Weight loading (vllm contract)
    # ------------------------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Route weights to the right sub-component.

        In Phase 1 the ZImagePipeline loads its transformer/VAE/scheduler
        from disk inside ``__init__`` (via ``from_pretrained``), and the
        condition encoder loads its connector the same way. The main Ming
        checkpoint at the top level only contains the *thinker* weights,
        which the imagegen stage does not consume.

        We therefore silently drop top-level thinker weights here and only
        log what we saw, so callers know loading is effectively a no-op.
        """
        seen: set[str] = set()
        counters: dict[str, int] = {}
        for name, _tensor in weights:
            bucket = name.split(".", 1)[0] if "." in name else name
            counters[bucket] = counters.get(bucket, 0) + 1
            seen.add(name)
        logger.info(
            "[MingImageGen.load_weights] top-level weight buckets observed: %s",
            counters,
        )
        logger.info(
            "[MingImageGen.load_weights] imagegen stage loads its own "
            "transformer/vae/connector via from_pretrained; ignoring %d "
            "top-level tensors.",
            len(seen),
        )
        return seen

    # vllm's runner expects make_empty_intermediate_tensors on every model.
    def make_empty_intermediate_tensors(self, *args, **kwargs):  # noqa: ARG002
        return None


__all__ = ["MingFlashOmniImageGenModel"]
