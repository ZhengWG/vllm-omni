# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for TorchIpcConnector, its placement policy, and edge routing.

Three layers, gated independently:

1. Placement policy (``gpu_placement``) and the restricted reduce-spec codec:
   pure CPU logic.
2. Connector behavior on CPU payloads and stage-connector resolution
   (dual input/output configs, ``EdgeRoutedConnector`` routing): CPU-only.
3. Functional GPU put/get: requires a real GPU and two processes (a CUDA IPC
   handle cannot be opened in the process that exported it).
"""

from __future__ import annotations

import multiprocessing as mp
import uuid
from typing import Any

import pytest
import torch

from vllm_omni.distributed.omni_connectors.connectors.edge_routed_connector import EdgeRoutedConnector
from vllm_omni.distributed.omni_connectors.connectors.gpu_placement import (
    GPU_PLACEMENT_MIN_BYTES,
    connector_gpu_keys,
    connector_gpu_min_bytes,
    gpu_key_matches,
    keep_tensor_on_gpu,
    place_payload_tensor,
)
from vllm_omni.distributed.omni_connectors.connectors.shm_connector import SharedMemoryConnector
from vllm_omni.distributed.omni_connectors.connectors.torch_ipc_connector import (
    _TORCH_IPC_MARKER,
    TorchIpcConnector,
    _compact_for_share,
    _decode_reduce_atom,
    _encode_reduce_atom,
)
from vllm_omni.distributed.omni_connectors.factory import OmniConnectorFactory
from vllm_omni.distributed.omni_connectors.utils.config import (
    build_stage_connector_config,
    get_stage_connector_role,
    stage_connector_extra,
)
from vllm_omni.distributed.omni_connectors.utils.initialization import resolve_stage_connector_spec

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
def test_connector_gpu_min_bytes_default_and_override():
    class _Default:
        pass

    class _Override:
        gpu_tensor_min_bytes = 1024

    assert connector_gpu_min_bytes(None) == GPU_PLACEMENT_MIN_BYTES
    assert connector_gpu_min_bytes(_Default()) == GPU_PLACEMENT_MIN_BYTES
    assert connector_gpu_min_bytes(_Override()) == 1024


@CPU
def test_keep_tensor_on_gpu_rejects_cpu_tensors_and_small_tensors():
    keys = frozenset({"hidden_states"})
    cpu_tensor = torch.zeros(1024, 1024)
    # CPU tensor never kept, regardless of key or size.
    assert not keep_tensor_on_gpu(cpu_tensor, "hidden_states", keys, min_bytes=1)
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


# ════════════════════════════════════════════════════════════════════
# Layer 1b — restricted reduce-spec codec (CPU)
# ════════════════════════════════════════════════════════════════════


@CPU
def test_reduce_atom_round_trip_primitives_and_dtype():
    for value in (None, True, 7, 2.5, "x", b"\x00\x01"):
        assert _decode_reduce_atom(_encode_reduce_atom(value)) == value

    dtype = torch.bfloat16
    assert _decode_reduce_atom(_encode_reduce_atom(dtype)) is dtype

    seq = torch.Size([4, 2048])
    decoded = _decode_reduce_atom(_encode_reduce_atom(seq))
    assert tuple(decoded) == (4, 2048)

    assert _decode_reduce_atom(_encode_reduce_atom(torch.Tensor)) is torch.Tensor
    assert _decode_reduce_atom(_encode_reduce_atom(torch.UntypedStorage)) is torch.UntypedStorage


@CPU
def test_reduce_atom_rejects_unknown_class_and_object():
    class _Evil:
        pass

    with pytest.raises(TypeError):
        _encode_reduce_atom(_Evil)
    with pytest.raises(TypeError):
        _encode_reduce_atom(_Evil())
    with pytest.raises(TypeError):
        _decode_reduce_atom({"__k": "cls", "v": "os.system"})
    with pytest.raises(TypeError):
        _decode_reduce_atom({"__k": "wat", "v": 1})


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
# Layer 2 — connector behavior on CPU + stage connector resolution
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
        mocker.patch.object(connector, "_import_one", return_value=sentinel)
        marker = {_TORCH_IPC_MARKER: True, "spec": []}
        payload = {
            "embed": {"prefill": marker, "decode": torch.zeros(1)},
            "list": [marker, "keep"],
            "tuple": (marker,),
        }
        out = connector._import_gpu_tensors(payload)
        assert torch.equal(out["embed"]["prefill"], sentinel)
        assert torch.equal(out["list"][0], sentinel)
        assert torch.equal(out["tuple"][0], sentinel)
        assert out["list"][1] == "keep"
        assert torch.equal(out["embed"]["decode"], torch.zeros(1))
    finally:
        connector.close()


@CPU
def test_put_falls_back_to_host_copy_when_export_fails(mocker):
    connector = TorchIpcConnector({"stage_id": 0, "gpu_tensor_keys": ["hidden_states"]})
    try:
        # Force the capability on so the export path is attempted, then fail it.
        connector.supports_gpu_tensor = True
        mocker.patch.object(connector, "_export_gpu_tensors", side_effect=RuntimeError("boom"))
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


@CPU
def test_factory_creates_edge_routed_connector_for_dual_config():
    cfg = {
        "input": {"name": "TorchIpcConnector", "extra": {"stage_id": 1, "gpu_tensor_keys": ["hidden_states"]}},
        "output": {"name": "SharedMemoryConnector", "extra": {"stage_id": 1}},
    }
    connector = OmniConnectorFactory.create_stage_connector(cfg)
    try:
        assert isinstance(connector, EdgeRoutedConnector)
        assert isinstance(connector._input, TorchIpcConnector)
        assert isinstance(connector._output, SharedMemoryConnector)
        # Send-side capability answers for the output edge (SHM: no GPU plane).
        assert not connector.supports_gpu_tensor
        assert connector.gpu_tensor_keys is None
        assert connector.stage_id == 1
    finally:
        connector.close()


@CPU
def test_factory_single_and_none_configs():
    assert OmniConnectorFactory.create_stage_connector(None) is None
    connector = OmniConnectorFactory.create_stage_connector({"name": "SharedMemoryConnector", "extra": {}})
    try:
        assert isinstance(connector, SharedMemoryConnector)
    finally:
        connector.close()


@CPU
def test_edge_routed_connector_routes_by_direction(mocker):
    inp = mocker.Mock(spec_set=["get", "close", "config", "stage_id", "request_scoped_cleanup", "cleanup", "health"])
    out = mocker.Mock(spec_set=["put", "close", "config", "stage_id", "request_scoped_cleanup", "cleanup", "health"])
    inp.config, out.config = {}, {}
    inp.stage_id = out.stage_id = 1
    inp.request_scoped_cleanup = True
    out.request_scoped_cleanup = False
    inp.get.return_value = ("payload", 3)
    out.put.return_value = (True, 3, None)

    routed = EdgeRoutedConnector(inp, out)
    assert routed.put("1", "2", "k", {"a": 1}) == (True, 3, None)
    out.put.assert_called_once()
    assert routed.get("0", "1", "k") == ("payload", 3)
    inp.get.assert_called_once()

    # Request-scoped cleanup fans out only to opted-in backends.
    assert routed.request_scoped_cleanup
    routed.cleanup("req")
    inp.cleanup.assert_called_once_with("req")
    out.cleanup.assert_not_called()

    routed.close()
    inp.close.assert_called_once()
    out.close.assert_called_once()


@CPU
def test_edge_routed_connector_missing_sides():
    routed = EdgeRoutedConnector(None, None)
    assert routed.put("0", "1", "k", {}) == (False, 0, None)
    assert routed.get("0", "1", "k") is None
    routed.close()


@CPU
def test_resolve_stage_connector_spec_picks_lowest_edges():
    cfg = {
        "from_stage_2": {"spec": {"name": "B"}},
        "from_stage_0": {"spec": {"name": "A"}},
        "to_stage_3": {"spec": {"name": "D"}},
        "to_stage_1": {"spec": {"name": "C"}},
    }
    resolved = resolve_stage_connector_spec(cfg)
    assert resolved["input"]["name"] == "A"
    assert resolved["output"]["name"] == "C"
    assert resolve_stage_connector_spec({}) == {}
    assert resolve_stage_connector_spec({"from_stage_0": {"spec": {}}}) == {}


@CPU
def test_build_stage_connector_config_legacy_and_dual():
    legacy = build_stage_connector_config({"name": "SharedMemoryConnector", "extra": {"x": 1}}, stage_id=2)
    assert legacy == {"name": "SharedMemoryConnector", "extra": {"x": 1, "stage_id": 2}}

    dual = build_stage_connector_config(
        {
            "input": {"name": "TorchIpcConnector", "extra": {"role": "receiver", "a": 1}},
            "output": {"name": "SharedMemoryConnector", "extra": {"role": "sender", "b": 2}},
        },
        stage_id=1,
    )
    assert dual["input"]["extra"]["stage_id"] == 1
    assert dual["output"]["extra"]["stage_id"] == 1
    # Merged legacy view: name from output, extras merged, role stripped.
    assert dual["name"] == "SharedMemoryConnector"
    assert dual["extra"]["a"] == 1 and dual["extra"]["b"] == 2
    assert "role" not in dual["extra"]

    empty = build_stage_connector_config({}, stage_id=0)
    assert empty["name"] == "SharedMemoryConnector"


@CPU
def test_stage_connector_extra_merges_dual_shape():
    dual = {
        "input": {"name": "A", "extra": {"x": 1, "shared": "in"}},
        "output": {"name": "B", "extra": {"y": 2, "shared": "out"}},
    }
    merged = stage_connector_extra(dual)
    assert merged == {"x": 1, "shared": "out", "y": 2}
    assert stage_connector_extra(None) == {}
    assert stage_connector_extra({"name": "A", "extra": {"z": 3}}) == {"z": 3}


@CPU
def test_get_stage_connector_role_dual_input_takes_precedence():
    class _Cfg:
        def __init__(self, stage_connector_config):
            self.stage_connector_config = stage_connector_config

    both = _Cfg(
        {
            "input": {"name": "A", "extra": {"role": "receiver"}},
            "output": {"name": "B", "extra": {"role": "sender"}},
        }
    )
    # A stage that receives chunks keeps reporting "receiver" even when it
    # also owns an output edge (legacy from_stage_* precedence).
    assert get_stage_connector_role(both) == "receiver"

    out_only = _Cfg({"output": {"name": "B", "extra": {}}})
    assert get_stage_connector_role(out_only) == "sender"

    legacy = _Cfg({"name": "A", "extra": {"role": "sender"}})
    assert get_stage_connector_role(legacy) == "sender"

    assert get_stage_connector_role(_Cfg(None)) is None


# ════════════════════════════════════════════════════════════════════
# Layer 3 — functional GPU put/get (two processes; CUDA IPC handles
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
            torch.cuda.synchronize()
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
