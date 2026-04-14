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

# Ming reuses the ``<imagePatch>`` token (id=157157 in BailingMoeV2 vocab) as
# the insertion slot for BOTH image comprehension and image generation; the
# distinction is made inside ``MingFlashOmniThinker.embed_input_ids`` by
# whether vision features are present. For image generation we wrap the patch
# run with ``<image>...</image>`` to mirror the comprehension prompt shape.
IMAGE_PATCH_TOKEN_STR = "<imagePatch>"
IMAGE_START_TOKEN_STR = "<image>"
IMAGE_END_TOKEN_STR = "</image>"
IMAGE_PATCH_TOKEN_ID = 157157  # BailingMoeV2 image_patch_token


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
    """Append ``<image><imagePatch>*N</image>`` to the user prompt.

    Mirrors the upstream Ming convention (reuses ``image_patch_token`` for
    image generation, distinguished from comprehension by the absence of
    vision features — see ``MingFlashOmniThinker.embed_input_ids``).

    Called by the stage-0 prompt preprocessor when the request is a
    text-to-image request. Returns a new prompt dict — the caller passes it
    into the thinker engine.
    """
    if not isinstance(prompt, dict):
        prompt = {"prompt": prompt, "modalities": ["image"]}

    base_text = prompt.get("prompt", "")
    patches = IMAGE_PATCH_TOKEN_STR * num_query_tokens
    suffix = f"{IMAGE_START_TOKEN_STR}{patches}{IMAGE_END_TOKEN_STR}"
    new_prompt = dict(prompt)
    new_prompt["prompt"] = f"{base_text}{suffix}"
    logger.info(
        "[MingImageGen.expand_prompt] appended <image>%s*%d</image> (original len=%d, new len=%d)",
        IMAGE_PATCH_TOKEN_STR,
        num_query_tokens,
        len(base_text),
        len(new_prompt["prompt"]),
    )
    return new_prompt


def extract_query_token_hidden_states(
    hidden_states: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    query_token_id: int = IMAGE_PATCH_TOKEN_ID,
    num_query_tokens: int = 256,
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
# Dual-stage transport: thinker -> imagegen
# ----------------------------------------------------------------------


def _validate_stage_inputs(stage_list, engine_input_source):
    """Mirror of qwen3_omni._validate_stage_inputs."""
    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")
    stage_id = engine_input_source[0]
    if stage_id >= len(stage_list):
        raise IndexError(f"Invalid stage_id: {stage_id}")
    stage = stage_list[stage_id]
    if stage.engine_outputs is None:
        raise RuntimeError(f"Stage {stage_id} has no outputs yet")
    return stage.engine_outputs


def thinker2imagegen(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: Any | None = None,  # noqa: ARG001
    requires_multimodal_data: bool = False,  # noqa: ARG001
) -> list[Any]:
    """Bridge thinker stage outputs to the imagegen stage.

    Reads each thinker output's ``multimodal_output["final_hidden_states"]``
    (the full prompt sequence hidden states, shape ``[L, H]``), masks it to
    the positions where ``prompt_token_ids == image_patch_token (157157)``,
    and packages the resulting ``[256, H]`` tensor as
    ``additional_information`` on a dummy ``OmniTokensPrompt`` for the
    imagegen stage.

    NOTE: an earlier attempt added a ``ming_imagegen_hidden_states`` key on
    the thinker side, but vllm-omni's stage output serialization only
    surfaces a fixed set of keys (``final_hidden_states``, ``latent``, …),
    so that custom key never made it across. Slicing happens on the
    receiver (this function) instead — zero infra change, zero new keys.

    Args:
        stage_list: Upstream stage list injected by the runtime.
        engine_input_source: Source stage IDs (``[0]`` for thinker).
        prompt: Original user prompt (unused — imagegen needs only the
            hidden-state tensor).
        requires_multimodal_data: Unused.

    Returns:
        ``list[OmniTokensPrompt]`` — one per thinker output, each with
        ``additional_information[HIDDEN_STATES_PAYLOAD_KEY]`` set.
    """
    from vllm_omni.inputs.data import OmniTokensPrompt

    thinker_outputs = _validate_stage_inputs(stage_list, engine_input_source)
    imagegen_inputs: list[Any] = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i, thinker_output in enumerate(thinker_outputs):
        output = thinker_output.outputs[0]
        mm_out = getattr(output, "multimodal_output", None) or {}

        # The thinker ships the full prompt sequence hidden states under
        # ``final_hidden_states`` ([L, H] or [B, L, H]). We select only the
        # positions that correspond to ``<imagePatch>`` tokens — that is
        # where the learnable query embeddings were injected and where the
        # thinker produced the conditioning hidden states we need.
        full_hidden = mm_out.get("final_hidden_states")
        if full_hidden is None:
            logger.warning(
                "[thinker2imagegen] request %d: missing 'final_hidden_states' in multimodal_output (keys=%s); skipping",
                i,
                list(mm_out.keys()),
            )
            continue

        prompt_ids = _ensure_list(thinker_output.prompt_token_ids)
        prompt_ids_t = torch.tensor(prompt_ids, dtype=torch.long, device=full_hidden.device)
        patch_mask = prompt_ids_t == IMAGE_PATCH_TOKEN_ID  # [L]
        num_patches = int(patch_mask.sum().item())
        if num_patches == 0:
            logger.warning(
                "[thinker2imagegen] request %d: no <imagePatch> tokens in "
                "prompt_ids (length=%d); cannot build imagegen conditioning",
                i,
                len(prompt_ids),
            )
            continue

        # Normalize full_hidden shape to [L, H] before masking.
        if full_hidden.dim() == 3:
            # [B, L, H] — assume batch size 1 (vllm prefills one request at a time)
            assert full_hidden.shape[0] == 1, f"unexpected batch dim: {full_hidden.shape}"
            full_hidden = full_hidden[0]
        if full_hidden.dim() != 2:
            logger.warning(
                "[thinker2imagegen] request %d: unexpected final_hidden_states shape %s; skipping",
                i,
                tuple(full_hidden.shape),
            )
            continue
        if full_hidden.shape[0] != patch_mask.shape[0]:
            logger.warning(
                "[thinker2imagegen] request %d: hidden length %d != prompt length %d; skipping",
                i,
                full_hidden.shape[0],
                patch_mask.shape[0],
            )
            continue

        hidden = full_hidden[patch_mask].detach().to(device=device)  # [N, H]
        # vllm-omni's cross-stage ``serialize_additional_information`` uses
        # ``tensor.cpu().numpy().tobytes()`` which numpy does not support for
        # bfloat16 / float16. Cast to float32 here; the condition encoder
        # recasts back to its own dtype on the receiver side.
        if hidden.dtype in (torch.bfloat16, torch.float16):
            hidden = hidden.to(torch.float32)
        f = hidden.float()
        logger.info(
            "[thinker2imagegen] request %d: sliced %s dtype=%s "
            "mean=%+.4f std=%.4f |x|/tok=%.3f (found %d <imagePatch> positions)",
            i,
            tuple(hidden.shape),
            hidden.dtype,
            f.mean().item(),
            f.std().item(),
            f.norm(dim=-1).mean().item(),
            num_patches,
        )

        info: dict[str, Any] = {HIDDEN_STATES_PAYLOAD_KEY: hidden}
        imagegen_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0],  # dummy — imagegen only runs prefill once
                additional_information=info,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return imagegen_inputs


def _ensure_list(x) -> list[int]:
    """Convert ConstantList / tensor-like to plain list (qwen3_omni pattern)."""
    if hasattr(x, "_x"):
        return list(x._x)
    if isinstance(x, list):
        return x
    if hasattr(x, "tolist"):
        return x.tolist()
    return list(x)


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
    "IMAGE_PATCH_TOKEN_STR",
    "IMAGE_START_TOKEN_STR",
    "IMAGE_END_TOKEN_STR",
    "IMAGE_PATCH_TOKEN_ID",
    "MingImageGenRequestMeta",
    "expand_prompt_for_image_gen",
    "extract_query_token_hidden_states",
    "load_payload_from_cross_stage",
    "thinker2imagegen",
]
