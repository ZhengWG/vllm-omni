# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.

"""Stage input processors for Ming-flash-omni-2.0 text-to-image.

Two functions are wired from ``ming_flash_omni_dual.yaml``:

* ``expand_cfg_prompts`` (stage 0 ``prompt_expand_func``) — when the user
  provides a ``negative_prompt``, expand into one CFG companion that runs
  through the thinker in parallel.
* ``thinker2imagegen`` (stage 1 ``custom_process_input_func``) — for each
  thinker output, slice ``final_hidden_states`` at ``<imagePatch>``
  positions and pack them into the diffusion prompt. When a CFG companion
  output is present, its sliced hidden is packed under
  ``extra[negative_thinker_hidden_states]`` alongside the parent's
  ``extra[thinker_hidden_states]``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch

logger = logging.getLogger(__name__)


CFG_TEXT_SUFFIX = "__cfg_text"


# Fallback when stage config introspection fails; matches
# llm_config.image_patch_token on the released Ming-flash-omni-2.0 checkpoint.
_DEFAULT_IMAGE_PATCH_TOKEN_ID = 157157


# ---------------------------------------------------------------------------
# CFG prompt expansion (stage 0: prompt_expand_func)
# ---------------------------------------------------------------------------


@dataclass
class _CfgExpandedPrompt:
    """Minimal structural object consumed by ``AsyncOmniEngine._enqueue_cfg_companions``."""

    prompt: dict[str, Any]
    role: str
    request_id_suffix: str

    def apply_overrides(self, base_params: Any, base_spl: list[Any]) -> tuple[Any, list[Any]]:
        return base_params, base_spl


def expand_cfg_prompts(
    prompt: dict[str, Any] | str,
    sampling_params: Any,
) -> list[_CfgExpandedPrompt]:
    """Expand a text-to-image request into one CFG-text companion (opt-in).

    Triggers only when a non-empty
    ``sampling_params.extra_args["image_gen"]["negative_prompt"]`` is set on
    the stage-0 params; otherwise returns ``[]`` and the pipeline falls back
    to zero negative (Ming's default behavior).
    """
    if not isinstance(prompt, dict):
        return []
    if prompt.get("modalities") != ["image"]:
        return []

    extra_args = getattr(sampling_params, "extra_args", None) or {}
    image_gen_args = extra_args.get("image_gen") or {}
    negative = image_gen_args.get("negative_prompt")
    if not isinstance(negative, str) or not negative.strip():
        return []

    neg_prompt_dict: dict[str, Any] = {
        "prompt": negative,
        "modalities": prompt.get("modalities"),
    }
    mm_kwargs = prompt.get("mm_processor_kwargs")
    if mm_kwargs:
        neg_prompt_dict["mm_processor_kwargs"] = dict(mm_kwargs)

    return [_CfgExpandedPrompt(prompt=neg_prompt_dict, role="cfg_text", request_id_suffix=CFG_TEXT_SUFFIX)]


# ---------------------------------------------------------------------------
# Thinker → imagegen bridge (stage 1: custom_process_input_func)
# ---------------------------------------------------------------------------


def _resolve_image_patch_token_id(stage: Any) -> int:
    """Return the ``<imagePatch>`` token id from *stage*'s HF config, cached on first call."""
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


def _slice_patch_hidden(
    thinker_output: Any,
    image_patch_token_id: int,
    tag: str,
) -> torch.Tensor | None:
    """Return ``[N, H]`` hidden at ``<imagePatch>`` positions, or ``None`` if unrecoverable."""
    output = thinker_output.outputs[0]
    mm_out = getattr(output, "multimodal_output", None) or {}
    full_hidden = mm_out.get("final_hidden_states")
    if full_hidden is None:
        logger.warning("[thinker2imagegen] %s: missing final_hidden_states (keys=%s)", tag, list(mm_out.keys()))
        return None

    prompt_ids = _ensure_list(thinker_output.prompt_token_ids)
    prompt_ids_t = torch.tensor(prompt_ids, dtype=torch.long, device=full_hidden.device)
    patch_mask = prompt_ids_t == image_patch_token_id
    num_patches = int(patch_mask.sum().item())
    if num_patches == 0:
        logger.warning("[thinker2imagegen] %s: no <imagePatch> tokens in prompt (len=%d)", tag, len(prompt_ids))
        return None

    if full_hidden.dim() == 3:
        assert full_hidden.shape[0] == 1, f"expected batch=1, got {full_hidden.shape}"
        full_hidden = full_hidden[0]
    if full_hidden.dim() != 2 or full_hidden.shape[0] != patch_mask.shape[0]:
        logger.warning(
            "[thinker2imagegen] %s: hidden shape %s inconsistent with prompt len %d",
            tag,
            tuple(full_hidden.shape),
            patch_mask.shape[0],
        )
        return None

    hidden = full_hidden[patch_mask].detach().contiguous()
    if logger.isEnabledFor(logging.DEBUG):
        f = hidden.float()
        logger.debug(
            "[thinker2imagegen] %s sliced=%s mean=%+.4f std=%.4f |x|/tok=%.3f (%d patches)",
            tag,
            tuple(hidden.shape),
            f.mean().item(),
            f.std().item(),
            f.norm(dim=-1).mean().item(),
            num_patches,
        )
    return hidden


def thinker2imagegen(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: Any | None = None,  # noqa: ARG001
    requires_multimodal_data: bool = False,  # noqa: ARG001
) -> list[dict[str, Any]]:
    """Bridge thinker AR outputs into image-generation DiT inputs.

    ``stage.engine_outputs`` holds ``[parent_output, *companion_outputs]``
    (bundled by the orchestrator). Parent outputs feed
    ``extra[thinker_hidden_states]``; the cfg_text companion feeds
    ``extra[negative_thinker_hidden_states]`` used by MingImagePipeline as real
    CFG negative conditioning. Unknown-suffix outputs are skipped.
    """
    source_stage, thinker_outputs = _validate_stage_inputs(stage_list, engine_input_source)
    image_patch_token_id = _resolve_image_patch_token_id(source_stage)

    parent_output = None
    negative_output = None
    for o in thinker_outputs:
        rid = getattr(o, "request_id", "")
        if rid.endswith(CFG_TEXT_SUFFIX):
            negative_output = o
        elif parent_output is None:
            parent_output = o

    if parent_output is None:
        logger.warning("[thinker2imagegen] no parent output in engine_outputs; skipping")
        return []

    parent_hidden = _slice_patch_hidden(parent_output, image_patch_token_id, tag="parent")
    if parent_hidden is None:
        return []

    extra: dict[str, Any] = {"thinker_hidden_states": parent_hidden}
    if negative_output is not None:
        neg_hidden = _slice_patch_hidden(negative_output, image_patch_token_id, tag="cfg_text")
        if neg_hidden is not None:
            extra["negative_thinker_hidden_states"] = neg_hidden

    return [{"prompt": "", "extra": extra}]


__all__ = [
    "CFG_TEXT_SUFFIX",
    "expand_cfg_prompts",
    "thinker2imagegen",
]
