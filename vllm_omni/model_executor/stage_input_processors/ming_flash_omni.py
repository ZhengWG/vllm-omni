# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.

"""Stage input processor for Ming-flash-omni-2.0 text-to-image.

Wired into ``ming_flash_omni_dual.yaml`` as the stage 1 (imagegen diffusion)
``custom_process_input_func``. For every thinker-stage output that arrives,
this module slices the final-layer hidden states at the learnable
``<imagePatch>`` positions and packages them into a diffusion-stage prompt
dict under ``extra[thinker_hidden_states]``. The ``OmniMsgpackEncoder``
handles the torch.Tensor payload natively via a uint8-view path, so no
fp32 cast is needed (unlike the old llm-stage msgspec path).

The receiving pipeline (``vllm_omni/diffusion/models/ming_flash_omni/
pipeline_ming.py::MingImagePipeline``) reads
``req.prompts[0]["extra"]["thinker_hidden_states"]`` and runs the Ming
condition encoder + ZImage diffusion loop.

Pattern adapted from ``glm_image/stage_input_processors/glm_image.py
::ar2diffusion``.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


# BailingMoeV2 ``image_patch_token`` id (see upstream config.json for
# Ming-flash-omni-2.0). The thinker reuses this token both for vision-input
# placeholders AND for image-gen query slots — the two paths are disambiguated
# inside the thinker's ``embed_input_ids`` by whether the request carries
# vision pixel values. We mask at this id on the stage-0 output to recover
# the query positions.
IMAGE_PATCH_TOKEN_ID = 157157


def _validate_stage_inputs(stage_list, engine_input_source):
    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")
    stage_id = engine_input_source[0]
    if stage_id >= len(stage_list):
        raise IndexError(f"Invalid stage_id: {stage_id}")
    stage = stage_list[stage_id]
    if stage.engine_outputs is None:
        raise RuntimeError(f"Stage {stage_id} has no outputs yet")
    return stage.engine_outputs


def _ensure_list(x) -> list[int]:
    """Convert ConstantList / tensor-like to plain list."""
    if hasattr(x, "_x"):
        return list(x._x)
    if isinstance(x, list):
        return x
    if hasattr(x, "tolist"):
        return x.tolist()
    return list(x)


def thinker2imagegen(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: Any | None = None,  # noqa: ARG001
    requires_multimodal_data: bool = False,  # noqa: ARG001
) -> list[dict[str, Any]]:
    """Bridge thinker (stage 0 llm) outputs to MingImagePipeline (stage 1 diffusion).

    For each thinker output we:
      1. Read ``output.multimodal_output["final_hidden_states"]`` — the full
         prompt-sequence hidden states ``[L, H]``.
      2. Build a mask ``prompt_token_ids == IMAGE_PATCH_TOKEN_ID`` and slice
         at those positions → ``[256, H]`` (for ``img_gen_scales=[16]``).
      3. Wrap in ``{"prompt": "", "extra": {"thinker_hidden_states": tensor}}``.

    The ``extra`` dict survives cross-stage serialization because
    ``OmniMsgpackEncoder`` in ``vllm_omni/distributed/omni_connectors/utils/
    serialization.py`` has a ``_encode_tensor`` hook (uint8 view + bytes).
    """
    from vllm_omni.diffusion.models.ming_flash_omni.pipeline_ming import (
        THINKER_HIDDEN_STATES_KEY,
    )

    thinker_outputs = _validate_stage_inputs(stage_list, engine_input_source)
    imagegen_inputs: list[dict[str, Any]] = []

    for i, thinker_output in enumerate(thinker_outputs):
        output = thinker_output.outputs[0]
        mm_out = getattr(output, "multimodal_output", None) or {}

        full_hidden = mm_out.get("final_hidden_states")
        if full_hidden is None:
            logger.warning(
                "[thinker2imagegen] req %d: missing 'final_hidden_states' in multimodal_output (keys=%s); skipping",
                i,
                list(mm_out.keys()),
            )
            continue

        prompt_ids = _ensure_list(thinker_output.prompt_token_ids)
        prompt_ids_t = torch.tensor(prompt_ids, dtype=torch.long, device=full_hidden.device)
        patch_mask = prompt_ids_t == IMAGE_PATCH_TOKEN_ID
        num_patches = int(patch_mask.sum().item())
        if num_patches == 0:
            logger.warning(
                "[thinker2imagegen] req %d: no <imagePatch> tokens in "
                "prompt_ids (length=%d); cannot build imagegen conditioning",
                i,
                len(prompt_ids),
            )
            continue

        if full_hidden.dim() == 3:
            assert full_hidden.shape[0] == 1, f"expected batch=1, got {full_hidden.shape}"
            full_hidden = full_hidden[0]
        if full_hidden.dim() != 2:
            logger.warning(
                "[thinker2imagegen] req %d: unexpected final_hidden_states shape %s; skipping",
                i,
                tuple(full_hidden.shape),
            )
            continue
        if full_hidden.shape[0] != patch_mask.shape[0]:
            logger.warning(
                "[thinker2imagegen] req %d: hidden length %d != prompt length %d; skipping",
                i,
                full_hidden.shape[0],
                patch_mask.shape[0],
            )
            continue

        hidden = full_hidden[patch_mask].detach().contiguous()  # [N, H]
        f = hidden.float()
        logger.info(
            "[thinker2imagegen] req %d: sliced %s dtype=%s "
            "mean=%+.4f std=%.4f |x|/tok=%.3f (found %d <imagePatch> positions)",
            i,
            tuple(hidden.shape),
            hidden.dtype,
            f.mean().item(),
            f.std().item(),
            f.norm(dim=-1).mean().item(),
            num_patches,
        )

        imagegen_inputs.append(
            {
                "prompt": "",
                "extra": {
                    THINKER_HIDDEN_STATES_KEY: hidden,
                },
            }
        )

    return imagegen_inputs


__all__ = [
    "IMAGE_PATCH_TOKEN_ID",
    "thinker2imagegen",
]
