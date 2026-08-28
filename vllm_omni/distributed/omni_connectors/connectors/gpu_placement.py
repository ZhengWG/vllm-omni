# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Payload placement policy for GPU-direct (device-to-device) send edges.

Single home for every "does this payload tensor stay on GPU for the send
edge?" decision.  Placement is keyed on two connector attributes:

* ``supports_gpu_tensor`` — the connector can move CUDA tensors between
  same-host processes without a host bounce (e.g. ``TorchIpcConnector``).
* ``gpu_tensor_keys`` — an explicit, per-edge-stable set of payload key
  roots (or full dotted keys) eligible to stay on GPU.

A tensor stays on GPU only when its key is listed **and** it meets the size
floor (``gpu_tensor_min_bytes``, default 256 KiB).  The rationale:

1. **Large tensors** (prefill handoffs, per-segment hidden states) are the
   only payloads where a device-to-device plane beats the existing
   async-materialization CPU pipeline; they are also the ones sitting on
   the first-packet latency path.
2. **Small tensors** (per-token decode embeddings, control flags) would pay
   a *synchronous* device-to-host serialization on the connector send
   thread — which can queue behind other requests' prefill kernels — for no
   transport gain.  The size floor keeps them on the CPU pipeline, whose
   device-to-host copy runs on a dedicated stream off the critical path.

Because the floor makes placement size-dependent, any producer-side
``torch.cat`` across accumulated payload pieces must normalize devices at
the cat site (see the Qwen3-Omni thinker chunk-0 accumulation).  Consumers
already normalize at use (``.to(device)`` before model consumption).
"""

from typing import Any

import torch

__all__ = [
    "GPU_PLACEMENT_MIN_BYTES",
    "connector_gpu_keys",
    "connector_gpu_min_bytes",
    "gpu_key_matches",
    "keep_tensor_on_gpu",
    "place_payload_tensor",
]

# Default size floor for GPU placement.  Below this, the host round-trip is
# cheaper than the per-tensor IPC export/import bookkeeping and never worth
# a synchronous D2H on the send thread.
GPU_PLACEMENT_MIN_BYTES = 256 * 1024


def connector_gpu_keys(connector: Any) -> frozenset[str] | None:
    """Return the edge's stable GPU key set, or ``None`` when GPU placement
    is disabled (no connector, capability absent, or no keys configured)."""
    if connector is None or not getattr(connector, "supports_gpu_tensor", False):
        return None
    keys = getattr(connector, "gpu_tensor_keys", None)
    if not keys:
        return None
    return frozenset(str(k) for k in keys)


def connector_gpu_min_bytes(connector: Any) -> int:
    """Return the edge's GPU placement size floor in bytes."""
    value = getattr(connector, "gpu_tensor_min_bytes", None)
    if value is None:
        return GPU_PLACEMENT_MIN_BYTES
    return int(value)


def gpu_key_matches(key: str, gpu_keys: frozenset[str] | None) -> bool:
    """Match a flat payload key against the configured set.

    Both full dotted keys (``"embed.prefill"``) and key roots
    (``"hidden_states"`` matching ``"hidden_states.output"``) are accepted.
    """
    if not gpu_keys or not isinstance(key, str):
        return False
    return key in gpu_keys or key.split(".", 1)[0] in gpu_keys


def keep_tensor_on_gpu(
    tensor: torch.Tensor,
    key: str | None,
    gpu_keys: frozenset[str] | None,
    min_bytes: int = GPU_PLACEMENT_MIN_BYTES,
) -> bool:
    """Full placement predicate: listed key, CUDA tensor, above the floor."""
    if gpu_keys is None or key is None or not tensor.is_cuda:
        return False
    if not gpu_key_matches(key, gpu_keys):
        return False
    return tensor.numel() * tensor.element_size() >= min_bytes


def place_payload_tensor(tensor: torch.Tensor | None, keep_on_gpu: bool) -> torch.Tensor | None:
    """Detach a payload tensor and place it for the send edge.

    ``keep_on_gpu`` is the caller's per-key placement decision; CPU tensors
    always pass through unchanged.
    """
    if tensor is None:
        return None
    if keep_on_gpu and tensor.is_cuda:
        return tensor.detach()
    return tensor.detach().cpu()
