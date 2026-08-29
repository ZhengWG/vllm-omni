# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Utilities for handling multimodal outputs / building multimodal output
payloads, most of which are shared by the prefix cache / no prefix cache path.
"""

from collections.abc import Mapping

import torch
from vllm.logger import init_logger

from vllm_omni.distributed.omni_connectors.connectors.gpu_placement import gpu_key_matches

logger = init_logger(__name__)

# Flat payload keys partitioned at worker output into inter-stage connector
# payloads vs client-facing multimodal outputs.  Only final output roots are
# listed here; everything else remains available for stage-to-stage transport.
_CLIENT_MM_ROOT_KEYS: frozenset[str] = frozenset(
    {
        "model_outputs",
        "sr",
        "audio",
        "image",
        "images",
        "video",
        "videos",
        "trajectory_latents",
        "latents",
    }
)

_CLIENT_MM_META_KEYS: frozenset[str] = frozenset(
    {
        "audio_text_total_chars",
        "duplex_epoch",
        "duplex_turn_id",
        "llm_output_text_utf8",
        "segment_end",
        "tts_is_last_chunk",
        "turn_end",
    }
)


def partition_flat_payload(
    payload: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    """Split a flattened per-request payload into inter-stage vs client mm dicts."""
    if not payload:
        return {}, {}
    inter_stage: dict[str, object] = {}
    client_mm: dict[str, object] = {}
    for key, value in payload.items():
        root = key.split(".", 1)[0]
        if root in _CLIENT_MM_ROOT_KEYS:
            client_mm[key] = value
        elif root == "meta" and "." in key and key.split(".", 1)[1] in _CLIENT_MM_META_KEYS:
            # Small final-output metadata needed by serving (for example
            # transcript text attached to audio) must ride with client MM
            # output. Keep it in inter-stage too so downstream stages that read
            # metadata from the full payload are not starved.
            inter_stage[key] = value
            client_mm[key] = value
        else:
            inter_stage[key] = value
    return inter_stage, client_mm


def partition_payload_list(
    payloads: list[dict[str, object]],
) -> tuple[list[dict[str, object] | None] | None, list[dict[str, object] | None] | None]:
    inter_stage_list: list[dict[str, object] | None] = []
    client_mm_list: list[dict[str, object] | None] = []
    for payload in payloads:
        inter_stage, client_mm = partition_flat_payload(payload)
        inter_stage_list.append(inter_stage or None)
        client_mm_list.append(client_mm or None)
    return (
        None if all(item is None for item in inter_stage_list) else inter_stage_list,
        None if all(item is None for item in client_mm_list) else client_mm_list,
    )


def build_mm_cpu(
    multimodal_outputs: dict,
    gpu_keys: "frozenset[str] | None" = None,
    skip_clone: bool = False,
) -> dict[str, object]:
    """Pre-copies multimodal tensor to CPU once (not per-request) to avoid
    redundant D2H transfers when gpu_resident_buffer_keys keeps them on GPU.

    In the case of prefix caching, the multimodal outputs provided will
    only contain the passthrough data.

    Args:
        multimodal_outputs: Multimodal dict mapping strings to objects.
        gpu_keys: Stable per-edge placement set: tensors under a listed key
            stay on GPU for a GPU-direct connector edge instead of dropping
            to CPU.  Client-facing output roots never stay on GPU — they
            ride the msgpack wire to the API server.
        skip_clone: If True, GPU-kept tensors are already independent
            snapshot clones, so the defensive ``.clone()`` is skipped.
    """
    if not multimodal_outputs:
        return {}

    # Pre-copy multimodal tensors to CPU once (not per-request) to avoid
    # redundant D2H transfers when gpu_resident_buffer_keys keeps them on GPU.
    mm_cpu: dict[str, object] = {}
    # Currently there are some cases where this is true at the
    # moment, which should be fixed.
    if not isinstance(multimodal_outputs, Mapping):
        logger.warning("Multimodal outputs are not a dict and will not be passed")

    for k, v in multimodal_outputs.items():
        key_on_gpu = False
        if gpu_keys is not None and isinstance(k, str) and k.split(".", 1)[0] not in _CLIENT_MM_ROOT_KEYS:
            key_on_gpu = gpu_key_matches(k, gpu_keys)
        converted = _detach_tensor(v, key_on_gpu, skip_clone)
        if converted is not None:
            mm_cpu[k] = converted
    return mm_cpu


def snapshot_mm_payload(multimodal_outputs: dict) -> dict[str, object]:
    """Snapshot a multimodal payload without moving it off its device.

    Request-end full-payload producers retain these tensors across decode
    steps. CUDA graph outputs and runner input buffers may be reused on the
    next step, so retaining views is not sufficient: CUDA tensors must be
    copied first. Compatible per-request tensor lists are packed with one
    ``torch.cat`` and restored as views, avoiding one tiny copy kernel per
    active request.
    """
    if not multimodal_outputs:
        return {}
    return {key: _snapshot_payload_value(value) for key, value in multimodal_outputs.items()}


def _snapshot_payload_value(value):
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
        return tensor.clone() if tensor.device.type != "cpu" else tensor
    if isinstance(value, dict):
        return {key: _snapshot_payload_value(item) for key, item in value.items()}
    if isinstance(value, list):
        if not value:
            return value
        first = value[0]
        if (
            len(value) > 1
            and isinstance(first, torch.Tensor)
            and first.ndim > 0
            and first.layout == torch.strided
            and all(
                isinstance(item, torch.Tensor)
                and item.ndim == first.ndim
                and item.layout == first.layout
                and item.device == first.device
                and item.dtype == first.dtype
                and tuple(item.shape[1:]) == tuple(first.shape[1:])
                for item in value
            )
        ):
            packed = torch.cat([item.detach() for item in value], dim=0)
            output = []
            offset = 0
            for item in value:
                length = item.shape[0]
                output.append(packed.narrow(0, offset, length))
                offset += length
            return output
        return [_snapshot_payload_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_snapshot_payload_value(item) for item in value)
    return value


def _detach_tensor(value, keep_on_gpu: bool = False, skip_clone: bool = False):
    """Recursively detach tensors; move to CPU unless ``keep_on_gpu``."""
    if isinstance(value, torch.Tensor):
        if keep_on_gpu and value.is_cuda:
            # GPU-direct edge: stay on device for D2D transport.  Clone
            # defensively unless the caller already snapshot-cloned, so step
            # buffer reuse can never corrupt the payload.
            detached = value.detach()
            return detached if skip_clone else detached.clone()
        return value.detach().to("cpu").contiguous()
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            converted = _detach_tensor(v, keep_on_gpu, skip_clone)
            if converted is not None:
                out[k] = converted
        return out or None
    if isinstance(value, list):
        if not value:
            return value
        return [_detach_tensor(v, keep_on_gpu, skip_clone) for v in value]
    return value


def to_payload_element(
    element: object,
    idx: int,
    start: int,
    end: int,
    pass_lists_through: bool = False,
    seq_len: int | None = None,
    scheduled_seq_len: int | None = None,
):
    """Build an mm payload element corresponding to one request index
    from an element containing 0 or more CPU tensors.

    Args:
        element: The object to be added to the payload.
        idx: The index of the request.
        start: The start index corresponding to the request idx.
        end: The end index corresponding to the request idx.
        pass_lists_through: bool Whether or not lists should be treated as
            passthrough data; this should be False in normal cases, but True
            if we need to avoid splitting nonempty lists prior to calling
            postprocess, which is the case for prefix cache.
        seq_len: Optional hidden-aligned batch length (``hidden_states.shape[0]``).
            When a tensor's first dimension equals this value, it is sliced
            per request using ``start:end``.
        scheduled_seq_len: Optional scheduler-aligned batch length
            (``scheduler_output.total_num_scheduled_tokens``). Some full-payload
            mm tensors (e.g. batched ``codes.audio`` with tail-only hidden states)
            are laid out by scheduled tokens instead of the hidden tail shape.
            When omitted, ``seq_len`` is reused for backward compatibility.
    """
    if scheduled_seq_len is None:
        scheduled_seq_len = seq_len

    # Cached per-token tensors are merged elsewhere; here a first dim equal to
    # either the hidden-aligned or scheduler-aligned batch length means a
    # per-request slice is required.
    if isinstance(element, torch.Tensor) and (
        (seq_len is not None and element.shape[0] == seq_len)
        or (scheduled_seq_len is not None and element.shape[0] == scheduled_seq_len)
    ):
        return element[start:end].contiguous()
    # Every other case is shared between prefix cache (passthrough data)
    # and running a model without prefix caching.
    elif isinstance(element, dict):
        return {
            sk: to_payload_element(
                sv,
                idx,
                start,
                end,
                pass_lists_through=pass_lists_through,
                seq_len=seq_len,
                scheduled_seq_len=scheduled_seq_len,
            )
            for sk, sv in element.items()
        }
    elif isinstance(element, list):
        # For lists, clone tensors to avoid cross-request aliasing
        if pass_lists_through:
            return [elem.clone() if isinstance(elem, torch.Tensor) else elem for elem in element]
        element = element[idx] if idx < len(element) else element[0]
        if isinstance(element, torch.Tensor):
            element = element.clone()
        return element
    elif isinstance(element, torch.Tensor):
        # List-derived tensor payloads are request-invariant; clone to
        # avoid accidental cross-request aliasing on downstream mutation.
        return element.clone()
    return element
