# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for TorchIpcConnector and its GPU placement policy.

Two layers, gated independently:

1. Placement policy (``gpu_placement``) and connector behavior on CPU
   payloads (including SHM wire compatibility): CPU-only.
2. Functional GPU put/get: requires a real GPU and two processes (a CUDA
   IPC handle cannot be opened in the process that exported it).
"""

from __future__ import annotations

import multiprocessing as mp
import uuid
from typing import Any

import pytest
import torch

from vllm_omni.distributed.omni_connectors.connectors.gpu_placement import (
    connector_gpu_keys,
    gpu_key_matches,
    keep_tensor_on_gpu,
    place_payload_tensor,
)
from vllm_omni.distributed.omni_connectors.connectors.shm_connector import SharedMemoryConnector
from vllm_omni.distributed.omni_connectors.connectors.torch_ipc_connector import (
    _TORCH_IPC_MARKER,
    TorchIpcConnector,
    _compact_for_share,
    _payload_has_cuda,
    _payload_has_marker,
)

pytestmark = [pytest.mark.core_model]

CPU = pytest.mark.cpu


def _unique_key(prefix: str = "torch_ipc_test") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# ════════════════════════════════════════════════════════════════════
# Layer 1a — placement policy (CPU)
# ════════════════════════════════════════════════════════════════════


@CPU
def test_gpu_key_matches_root_and_full_key():
    keys = frozenset({"embed.prefill", "hidden_states"})
    assert gpu_key_matches("embed.prefill", keys)
    assert gpu_key_matches("hidden_states", keys)
    assert gpu_key_matches("hidden_states.output", keys)  # root match
    assert gpu_key_matches("hidden_states.layers.0", keys)
    assert not gpu_key_matches("embed.decode", keys)
    assert not gpu_key_matches("meta.finished", keys)
    assert not gpu_key_matches("embed.prefill", None)
    assert not gpu_key_matches("embed.prefill", frozenset())


@CPU
def test_connector_gpu_keys_requires_capability_and_keys():
    class _NoCap:
        supports_gpu_tensor = False
        gpu_tensor_keys = frozenset({"hidden_states"})

    class _NoKeys:
        supports_gpu_tensor = True
        gpu_tensor_keys = None

    class _Ok:
        supports_gpu_tensor = True
        gpu_tensor_keys = frozenset({"hidden_states"})

    assert connector_gpu_keys(None) is None
    assert connector_gpu_keys(_NoCap()) is None
    assert connector_gpu_keys(_NoKeys()) is None
    assert connector_gpu_keys(_Ok()) == frozenset({"hidden_states"})


@CPU
def test_keep_tensor_on_gpu_rejects_cpu_tensors_and_unlisted_keys():
    keys = frozenset({"hidden_states"})
    cpu_tensor = torch.zeros(4, 4)
    # CPU tensor never kept, regardless of key.
    assert not keep_tensor_on_gpu(cpu_tensor, "hidden_states", keys)
    # Missing key / missing key set never kept.
    assert not keep_tensor_on_gpu(cpu_tensor, None, keys)
    assert not keep_tensor_on_gpu(cpu_tensor, "hidden_states", None)


@CPU
def test_place_payload_tensor_cpu_pass_through():
    t = torch.arange(8, dtype=torch.float32)
    placed = place_payload_tensor(t, keep_on_gpu=True)
    assert placed.device.type == "cpu"
    assert torch.equal(placed, t)
    assert place_payload_tensor(None, keep_on_gpu=True) is None


@CPU
def test_payload_scan_helpers():
    cpu_payload = {"a": torch.zeros(2), "b": [1, {"c": torch.ones(1)}]}
    assert not _payload_has_cuda(cpu_payload)
    assert not _payload_has_marker(cpu_payload)

    marker = {_TORCH_IPC_MARKER: True, "buf": b""}
    assert _payload_has_marker({"x": [marker]})
    assert _payload_has_marker(("y", {"z": marker}))


@CPU
def test_compact_for_share_views_and_contiguity():
    base = torch.arange(64, dtype=torch.float32)
    full = base.clone()
    # Full-storage contiguous tensor passes through untouched.
    assert _compact_for_share(full).data_ptr() == full.data_ptr()

    # A slice view gets its own compact storage.
    view = base[8:24]
    compact = _compact_for_share(view)
    assert compact.data_ptr() != base.data_ptr()
    assert compact.untyped_storage().nbytes() == view.numel() * view.element_size()
    assert torch.equal(compact, view)

    # Non-contiguous input is materialized contiguously.
    strided = torch.arange(40, dtype=torch.float32).reshape(4, 10).transpose(0, 1)
    compact = _compact_for_share(strided)
    assert compact.is_contiguous()
    assert torch.equal(compact, strided)


# ════════════════════════════════════════════════════════════════════
# Layer 1b — connector behavior on CPU payloads (CPU-only)
# ════════════════════════════════════════════════════════════════════


@CPU
def test_torch_ipc_connector_cpu_round_trip():
    """Without CUDA the connector must behave exactly like SHM."""
    connector = TorchIpcConnector({"stage_id": 0, "gpu_tensor_keys": ["hidden_states"]})
    try:
        payload = {"hidden_states": {"output": torch.randn(3, 4)}, "ids": [1, 2, 3]}
        key = _unique_key()
        ok, size, _meta = connector.put("0", "1", key, payload)
        assert ok and size > 0
        result = connector.get("0", "1", key)
        assert result is not None
        obj, _ = result
        assert torch.equal(obj["hidden_states"]["output"], payload["hidden_states"]["output"])
        assert obj["ids"] == [1, 2, 3]
    finally:
        connector.close()


@CPU
def test_torch_ipc_is_wire_compatible_with_shm():
    """CPU payloads written by TorchIpc are readable by SharedMemoryConnector
    and vice versa — the GPU plane is a per-edge opt-in, not a wire format."""
    ipc = TorchIpcConnector({"stage_id": 1})
    shm = SharedMemoryConnector({"stage_id": 2})
    try:
        payload = {"codes": {"audio": torch.arange(6, dtype=torch.long)}, "meta": {"finished": torch.tensor(False)}}
        key = _unique_key()
        ok, _size, _meta = ipc.put("1", "2", key, payload)
        assert ok
        result = shm.get("1", "2", key)
        assert result is not None
        obj, _ = result
        assert torch.equal(obj["codes"]["audio"], payload["codes"]["audio"])

        key2 = _unique_key()
        ok, _size, _meta = shm.put("0", "1", key2, payload)
        assert ok
        result = ipc.get("0", "1", key2)
        assert result is not None
        obj, _ = result
        assert torch.equal(obj["codes"]["audio"], payload["codes"]["audio"])
    finally:
        ipc.close()
        shm.close()


@CPU
def test_torch_ipc_gpu_keys_property_requires_cuda():
    connector = TorchIpcConnector({"stage_id": 0, "gpu_tensor_keys": ["hidden_states"]})
    try:
        if torch.cuda.is_available():
            assert connector.supports_gpu_tensor
            assert connector.gpu_tensor_keys == frozenset({"hidden_states"})
        else:
            assert not connector.supports_gpu_tensor
            assert connector.gpu_tensor_keys is None
    finally:
        connector.close()


@CPU
def test_import_gpu_tensors_rebuilds_markers_in_nested_payloads(mocker):
    connector = TorchIpcConnector({"stage_id": 1})
    try:
        sentinel = torch.full((2, 2), 7.0)
        mocker.patch.object(connector, "_import_one", side_effect=lambda marker, dsts, views: sentinel)
        marker = {_TORCH_IPC_MARKER: True, "buf": b""}
        payload = {
            "embed": {"prefill": marker, "decode": torch.zeros(1)},
            "list": [marker, "keep"],
            "tuple": (marker,),
        }
        out = connector._import_gpu_tensors(payload, [], [])
        assert torch.equal(out["embed"]["prefill"], sentinel)
        assert torch.equal(out["list"][0], sentinel)
        assert torch.equal(out["tuple"][0], sentinel)
        assert out["list"][1] == "keep"
        assert torch.equal(out["embed"]["decode"], torch.zeros(1))
    finally:
        connector.close()


@CPU
def test_get_skips_import_walk_for_marker_free_payloads(mocker):
    """CPU-only payloads never enter the GPU import path at all."""
    connector = TorchIpcConnector({"stage_id": 1})
    try:
        spy = mocker.spy(connector, "_import_payload")
        payload = {"codes": {"audio": torch.zeros(4, dtype=torch.long)}, "ids": [1]}
        key = _unique_key()
        ok, _size, _meta = connector.put("0", "1", key, payload)
        assert ok
        result = connector.get("0", "1", key)
        assert result is not None
        spy.assert_not_called()
    finally:
        connector.close()


@CPU
def test_put_falls_back_to_host_copy_when_export_fails(mocker):
    connector = TorchIpcConnector({"stage_id": 0, "gpu_tensor_keys": ["hidden_states"]})
    try:
        # Force the capability on and make the payload look CUDA so the
        # export path is attempted, then fail it.
        connector.supports_gpu_tensor = True
        mocker.patch(
            "vllm_omni.distributed.omni_connectors.connectors.torch_ipc_connector._payload_has_cuda",
            return_value=True,
        )
        mocker.patch.object(connector, "_export_payload", side_effect=RuntimeError("boom"))
        payload = {"hidden_states": {"output": torch.randn(2, 2)}}
        key = _unique_key()
        ok, _size, _meta = connector.put("0", "1", key, payload)
        assert ok
        result = connector.get("0", "1", key)
        assert result is not None
        obj, _ = result
        assert torch.equal(obj["hidden_states"]["output"], payload["hidden_states"]["output"])
    finally:
        connector.close()


# ════════════════════════════════════════════════════════════════════
# Layer 2 — functional GPU put/get (two processes; CUDA IPC handles
# cannot be opened in the exporting process)
# ════════════════════════════════════════════════════════════════════


def _sender_proc(key: str, put_done: Any, consumed: Any, ok: Any) -> None:
    connector = TorchIpcConnector({"stage_id": 0, "gpu_tensor_keys": ["hidden_states"]})
    try:
        payload = {
            "hidden_states": {"output": torch.arange(4096, dtype=torch.float32, device="cuda").reshape(4, 1024)},
            "meta": {"finished": torch.tensor(False)},
            "ids": [1, 2, 3],
        }
        success, _size, _meta = connector.put("0", "1", key, payload)
        ok.value = 1 if success else 0
        put_done.set()
        # Keep the exporting process (and its CUDA allocation) alive until
        # the receiver has finished its device-to-device copy.
        consumed.wait(timeout=60)
    finally:
        connector.close()


@pytest.mark.gpu
@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_torch_ipc_gpu_round_trip_across_processes():
    ctx = mp.get_context("spawn")
    put_done = ctx.Event()
    consumed = ctx.Event()
    ok = ctx.Value("i", 0)
    key = _unique_key("torch_ipc_gpu")
    sender = ctx.Process(target=_sender_proc, args=(key, put_done, consumed, ok))
    sender.start()
    try:
        assert put_done.wait(timeout=120), "sender did not finish put()"
        assert ok.value == 1, "sender put() failed"

        receiver = TorchIpcConnector({"stage_id": 1, "gpu_tensor_keys": ["hidden_states"]})
        try:
            result = receiver.get("0", "1", key)
            assert result is not None
            obj, _size = result
            out = obj["hidden_states"]["output"]
            assert out.is_cuda
            expected = torch.arange(4096, dtype=torch.float32, device="cuda").reshape(4, 1024)
            torch.accelerator.synchronize()
            assert torch.equal(out, expected)
            assert obj["ids"] == [1, 2, 3]
        finally:
            consumed.set()
            receiver.close()
    finally:
        consumed.set()
        sender.join(timeout=30)
        if sender.is_alive():
            sender.terminate()
