# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.

"""Stage input processor for Ming-flash-omni-2.0 text-to-image.

Cross-stage protocol (Phase 1 — transfer mode **B**):

    Stage 0 (thinker)                   Stage 1 (imagegen)
    ─────────────────                   ───────────────────
    1. Append ``num_query_tokens``      4. Receive
       learnable image-gen tokens          {"ming_imagegen_hidden_states":
       to the end of the user prompt.       tensor[B, N, 4096]}
    2. Run thinker forward with         5. Call MingFlashOmniImageGenModel
       output_hidden_states=True.          .forward(hidden_states=...)
    3. Slice the final layer at the     6. Return PIL image.
       query-token positions
       (shape [B, N, 4096]) and ship
       it to stage 1 via the stage
       connector as a named tensor
       payload.

Phase-1 scope:
  * Only text->image requests (``modalities == ["image"]``) are routed here.
    Other requests short-circuit and behave exactly like today's thinker.
  * The transport under the hood still uses ``mooncake_connector`` (re-used
    from bagel_multiconnector.yaml); we attach our hidden-state tensor as a
    side-channel payload rather than a full KV cache. The concrete transfer
    mechanics live in the OmniKVTransferManager; this file only defines the
    *semantics* of what's transferred and how it's consumed.

TODO:
  * Wire this module into the actual ``omni_kv_config`` side-channel. For the
    first real run on hardware we may need to extend OmniKVTransferManager to
    carry a raw tensor under a string key — or, as a simpler first pass,
    co-locate stage 0 and stage 1 on the same process and hand the tensor
    across via a shared dict keyed by request_id.
  * Validate the query-token positions: the exact ID(s) of the image-gen
    learnable tokens must match what the thinker was trained with. We log
    the prompt and the slice indices so the first run tells us if they line
    up.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch

logger = logging.getLogger(__name__)

# Name used for the side-channel payload dict key. Kept as a module constant
# so both producer (stage 0) and consumer (stage 1) import from one place.
HIDDEN_STATES_PAYLOAD_KEY = "ming_imagegen_hidden_states"
ATTENTION_MASK_PAYLOAD_KEY = "ming_imagegen_attention_mask"

# Special token marker inserted by the prompt preprocessor to indicate where
# the learnable query tokens should be appended. The actual numeric token id
# must come from the Ming tokenizer; for Phase 1 we store the STRING marker
# and resolve it to an id at runtime via the tokenizer attached to the
# thinker's engine.
QUERY_TOKEN_PLACEHOLDER = "<|image_gen_query|>"


@dataclass
class MingImageGenRequestMeta:
    """Per-request metadata carried across the two stages.

    Attached to the request object under a dedicated attribute so both the
    thinker stage (producer) and the imagegen stage (consumer) can find it
    without polluting the sampling params.
    """

    num_query_tokens: int
    height: int
    width: int
    num_inference_steps: int
    guidance_scale: float
    seed: int | None = None


# ----------------------------------------------------------------------
# Stage 0: thinker side — prompt expansion + hidden-state extraction
# ----------------------------------------------------------------------


def expand_prompt_for_image_gen(
    prompt: dict[str, Any] | str,
    sampling_params: Any,  # noqa: ARG001
    *,
    num_query_tokens: int,
) -> dict[str, Any] | str:
    """Append ``num_query_tokens`` learnable query tokens to the user prompt.

    Called by the stage-0 prompt preprocessor when ``modalities`` includes
    ``"image"``. Returns a new prompt dict (the caller is responsible for
    passing it into the thinker engine).
    """
    if not isinstance(prompt, dict):
        prompt = {"prompt": prompt, "modalities": ["image"]}

    base_text = prompt.get("prompt", "")
    suffix = QUERY_TOKEN_PLACEHOLDER * num_query_tokens
    new_prompt = dict(prompt)
    new_prompt["prompt"] = f"{base_text}{suffix}"
    logger.info(
        "[MingImageGen.expand_prompt] appended %d query tokens (original len=%d, new len=%d)",
        num_query_tokens,
        len(base_text),
        len(new_prompt["prompt"]),
    )
    return new_prompt


def extract_query_token_hidden_states(
    hidden_states: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    query_token_id: int,
    num_query_tokens: int,
) -> torch.Tensor:
    """Slice the thinker's final-layer hidden states at query-token positions.

    Args:
        hidden_states: ``[B, L, H]`` — thinker's last-layer hidden states
            for the full prompt.
        token_ids: ``[B, L]`` — the prompt token ids that were fed in.
        query_token_id: Numeric id of the learnable query token, resolved by
            the caller from the tokenizer's vocab.
        num_query_tokens: Expected number of query-token positions per batch
            element. Used for shape validation and an early warning when the
            tokenizer didn't actually expand into as many tokens as expected
            (e.g. if ``QUERY_TOKEN_PLACEHOLDER`` was normalized away).

    Returns:
        ``[B, num_query_tokens, H]``.
    """
    if hidden_states.dim() != 3:
        raise ValueError(f"expected [B, L, H], got {tuple(hidden_states.shape)}")
    if token_ids.shape != hidden_states.shape[:2]:
        raise ValueError(
            f"token_ids shape {tuple(token_ids.shape)} does not match "
            f"hidden_states shape {tuple(hidden_states.shape[:2])}"
        )

    b, _l, h = hidden_states.shape
    mask = token_ids == query_token_id  # [B, L]
    counts = mask.sum(dim=1)  # [B]
    if (counts != num_query_tokens).any():
        logger.warning(
            "[MingImageGen.extract] expected %d query tokens per batch, got %s; "
            "ID=%d. Falling back to taking the LAST %d positions.",
            num_query_tokens,
            counts.tolist(),
            query_token_id,
            num_query_tokens,
        )
        return hidden_states[:, -num_query_tokens:, :].contiguous()

    out = torch.empty(
        (b, num_query_tokens, h),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    for i in range(b):
        out[i] = hidden_states[i][mask[i]]
    logger.info(
        "[MingImageGen.extract] sliced query-token hidden states: %s",
        tuple(out.shape),
    )
    return out


# ----------------------------------------------------------------------
# Stage 1: imagegen side — payload ingestion
# ----------------------------------------------------------------------


def load_payload_from_cross_stage(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Pull the Ming-specific tensors out of the cross-stage payload.

    The payload is whatever the stage-0 connector delivered to stage 1.
    For Phase 1 we expect a plain dict with two keys.
    """
    hidden = payload.get(HIDDEN_STATES_PAYLOAD_KEY)
    attn = payload.get(ATTENTION_MASK_PAYLOAD_KEY)
    if hidden is None:
        raise KeyError(
            f"cross-stage payload missing required key {HIDDEN_STATES_PAYLOAD_KEY!r}; got keys={list(payload.keys())}"
        )
    logger.info(
        "[MingImageGen.consume] received hidden_states=%s, attn_mask=%s",
        tuple(hidden.shape) if isinstance(hidden, torch.Tensor) else type(hidden),
        tuple(attn.shape) if isinstance(attn, torch.Tensor) else None,
    )
    result = {"thinker_hidden_states": hidden}
    if attn is not None:
        result["attention_mask"] = attn
    return result


__all__ = [
    "HIDDEN_STATES_PAYLOAD_KEY",
    "ATTENTION_MASK_PAYLOAD_KEY",
    "QUERY_TOKEN_PLACEHOLDER",
    "MingImageGenRequestMeta",
    "expand_prompt_for_image_gen",
    "extract_query_token_hidden_states",
    "load_payload_from_cross_stage",
]
