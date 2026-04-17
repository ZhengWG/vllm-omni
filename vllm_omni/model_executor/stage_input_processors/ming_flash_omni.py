# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.

"""Stage input processor for Ming-flash-omni-2.0 text-to-image.

Acts as the bridge between the Ming thinker stage (an AR LLM that has
produced a full sequence of hidden states) and the Ming image-generation
diffusion pipeline (a DiT that wants those hidden states as conditioning).

For every thinker output this module:

  1. Reads ``output.multimodal_output["final_hidden_states"]`` — the
     last-layer hidden states over the full prompt ``[L, H]``.
  2. Masks positions where the prompt token id equals the thinker's
     learnable ``<imagePatch>`` slot (resolved from the source stage's
     HF config at first call, cached thereafter).
  3. Slices those positions out to form the conditioning tensor
     ``[N, H]`` and packages it in a diffusion-stage prompt dict under
     ``extra[thinker_hidden_states]``.

The payload survives cross-stage transport because the omni msgpack
encoder has a native ``torch.Tensor`` codec — no fp32 cast is required.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


# Documented fallback used when the source stage's config cannot be
# introspected (older checkpoints, unusual deployments, tests with
# mocked clients). This matches ``llm_config.image_patch_token`` in
# the released Ming-flash-omni-2.0 checkpoint's ``config.json``.
_DEFAULT_IMAGE_PATCH_TOKEN_ID = 157157


def _resolve_image_patch_token_id(stage: Any) -> int:
    """Return the ``<imagePatch>`` token id from *stage*'s HF config.

    Tries ``stage.vllm_config.model_config.hf_config.llm_config.
    image_patch_token`` and falls back to the documented default. The
    result is cached on the stage object (``_image_patch_token_id``) so
    subsequent calls are O(1).
    """
    cached = getattr(stage, "_image_patch_token_id", None)
    if isinstance(cached, int):
        return cached

    token_id = _DEFAULT_IMAGE_PATCH_TOKEN_ID
    try:
        hf_config = stage.vllm_config.model_config.hf_config
        llm_config = getattr(hf_config, "llm_config", None)
        resolved = getattr(llm_config, "image_patch_token", None)
        if isinstance(resolved, int):
            token_id = resolved
    except AttributeError:
        # Stage client predates the current surface; fall back silently.
        pass

    try:
        stage._image_patch_token_id = token_id
    except AttributeError:
        pass
    return token_id


def _validate_stage_inputs(stage_list, engine_input_source):
    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")
    stage_id = engine_input_source[0]
    if stage_id >= len(stage_list):
        raise IndexError(f"Invalid stage_id: {stage_id}")
    stage = stage_list[stage_id]
    if stage.engine_outputs is None:
        raise RuntimeError(f"Stage {stage_id} has no outputs yet")
    return stage, stage.engine_outputs


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
    """Bridge thinker AR stage outputs into image-generation DiT inputs.

    For each thinker output we:
      1. Read ``output.multimodal_output["final_hidden_states"]`` — the
         full prompt-sequence hidden states ``[L, H]``.
      2. Build a mask ``prompt_token_ids == image_patch_token_id`` and
         slice the hidden states at those positions → ``[N, H]`` (e.g.
         ``N = 256`` for ``img_gen_scales=[16]``).
      3. Wrap in ``{"prompt": "", "extra": {"thinker_hidden_states": tensor}}``.

    The ``"thinker_hidden_states"`` key is a plain string contract shared with
    ``MingImagePipeline.forward`` on the diffusion side (no shared constant,
    matching the ``prior_token_ids`` convention used by ``glm_image``).
    """
    source_stage, thinker_outputs = _validate_stage_inputs(stage_list, engine_input_source)
    image_patch_token_id = _resolve_image_patch_token_id(source_stage)
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
        patch_mask = prompt_ids_t == image_patch_token_id
        num_patches = int(patch_mask.sum().item())
        if num_patches == 0:
            logger.warning(
                "[thinker2imagegen] req %d: no <imagePatch> (id=%d) tokens in "
                "prompt_ids (length=%d); cannot build imagegen conditioning",
                i,
                image_patch_token_id,
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
        if logger.isEnabledFor(logging.DEBUG):
            f = hidden.float()
            logger.debug(
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
                    "thinker_hidden_states": hidden,
                },
            }
        )

    return imagegen_inputs


__all__ = [
    "thinker2imagegen",
]
