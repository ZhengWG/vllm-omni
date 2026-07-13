# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Payload placement policy for GPU-tensor (CUDA-IPC) send edges.

Single home for every "does this tensor stay on GPU for the send edge?"
decision: the worker's snapshot predicate, the payload builders' per-tensor
placement, and the engine-core wire strip. Keyed on the generic
``supports_gpu_tensor`` capability; ``CudaIPCConnector`` is currently its
only producer. Framework call sites read one function each.
"""

from collections.abc import Mapping
from typing import Any

import torch

# Marker on a dict value indicating "this slot held a GPU tensor that was
# stripped before crossing the engine-core -> API msgspec boundary; the real
# payload travelled via the connector pool". Downstream consumers of
# ``OmniEngineCoreOutput.multimodal_output`` for non-terminal stages should
# treat such values as opaque metadata and never call tensor methods on them.
STRIPPED_GPU_TENSOR_MARKER = "_omni_stripped_gpu_tensor"


def connector_keeps_gpu(connector: Any) -> bool:
    """Send-edge capability read: True when payload tensors may stay on GPU.

    A routed connector answers for its output edge; ``None`` (no connector)
    means the legacy CPU pipeline.
    """
    return bool(getattr(connector, "supports_gpu_tensor", False))


def place_payload_tensor(tensor: torch.Tensor | None, keep_on_gpu: bool) -> torch.Tensor | None:
    """Detach a payload tensor and place it for the send edge.

    GPU-direct edges keep CUDA tensors on device (D2D via the connector
    pool); everything else drops to CPU for the classic SHM/msgpack path.
    """
    if tensor is None:
        return None
    return tensor.detach() if (keep_on_gpu and tensor.is_cuda) else tensor.detach().cpu()


def payload_wants_gpu(payload: Any, gpu_keys: "frozenset | None") -> bool:
    """Snapshot predicate: does this payload contain tensors the placement
    policy will keep on GPU?

    With ``gpu_keys`` (stable per-key placement) only listed key roots count —
    per-instance size decisions flap devices across chunks and break the
    receiver's stream cat. Without a filter, any CUDA tensor counts.
    """
    if gpu_keys is not None:
        return _has_gpu_tensor_under_keys(payload, gpu_keys)
    return _has_any_gpu_tensor(payload)


def _has_gpu_tensor_under_keys(value: Any, keys: frozenset) -> bool:
    if isinstance(value, dict):
        for k, v in value.items():
            if isinstance(k, str) and k.split(".", 1)[0] in keys:
                if _has_any_gpu_tensor(v):
                    return True
            elif _has_gpu_tensor_under_keys(v, keys):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(_has_gpu_tensor_under_keys(v, keys) for v in value)
    return False


def _has_any_gpu_tensor(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return value.is_cuda
    if isinstance(value, dict):
        return any(_has_any_gpu_tensor(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_any_gpu_tensor(v) for v in value)
    return False


def strip_gpu_tensors_for_engine_output(value: Any) -> Any:
    """Replace CUDA tensors with small shape/dtype descriptor dicts (recursive).

    Avoids a redundant D2H during msgpack encoding for payloads whose real
    data travels via the connector. Never mutates the input; callers gate on
    the send edge."""
    if isinstance(value, torch.Tensor):
        if value.device.type == "cuda":
            return {
                STRIPPED_GPU_TENSOR_MARKER: True,
                "shape": list(value.shape),
                "dtype": str(value.dtype).removeprefix("torch."),
            }
        return value
    if isinstance(value, Mapping):
        return {k: strip_gpu_tensors_for_engine_output(v) for k, v in value.items()}
    if isinstance(value, list):
        return [strip_gpu_tensors_for_engine_output(v) for v in value]
    if isinstance(value, tuple):
        return tuple(strip_gpu_tensors_for_engine_output(v) for v in value)
    return value
