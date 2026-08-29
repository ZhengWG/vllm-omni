# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Payload placement policy for GPU-direct (device-to-device) send edges.

Single home for every "does this payload tensor stay on GPU for the send
edge?" decision.  Placement is keyed on two connector attributes:

* ``supports_gpu_tensor`` — the connector can move CUDA tensors between
  same-host processes without a host bounce (e.g. ``TorchIpcConnector``).
* ``gpu_tensor_keys`` — an explicit, per-edge-stable set of payload key
  roots (or full dotted keys) eligible to stay on GPU.

The key set is a **consumer-locality contract**: list exactly the payload
keys the downstream stage consumes on GPU (model embeddings, hidden
states).  Data consumed on CPU (token ids, codec codes fed into prompt
construction, control flags) must stay on the CPU pipeline — moving it
device-to-device only to bounce it back with a synchronous ``.cpu()`` on
the receive thread would invert the win.

Size is *not* part of the default policy: the GPU data plane shares
handles instead of serializing tensor bytes and runs on dedicated streams
on both ends, so small per-token packets pay no host synchronization and
no compute-stream coupling.  ``gpu_tensor_min_bytes`` remains available as
a tuning floor (default 0 = disabled) should a deployment measure a
crossover.

With a nonzero floor, placement becomes size-dependent, so any
producer-side ``torch.cat`` across accumulated payload pieces must
normalize devices at the cat site (see the Qwen3-Omni thinker chunk-0
accumulation).  Consumers already normalize at use (``.to(device)``).
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

# Default size floor for GPU placement: disabled.  The data plane is
# handle-based and stream-decoupled, so listed keys ride it at any size;
# set ``gpu_tensor_min_bytes`` per edge to reintroduce a floor if a
# deployment measures a crossover.
GPU_PLACEMENT_MIN_BYTES = 0


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
