# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the CudaIPC connector and its CudaIpcControlRing control plane.

Two layers, gated independently:

1. ``CudaIpcControlRing`` (the lock-free SPSC keyed-mailbox control plane in ``cuda_ipc_control_ring.py``):
   pure-Python (struct + POSIX shm), CPU-only CI.

2. ``CudaIPCConnector`` functional put/get: requires a real GPU. CUDA IPC handles cannot be
   opened in the same process that created them, so these spawn sender + receiver processes.
   The GPU gate does not skip the CPU ring tests above.
"""

from __future__ import annotations

import multiprocessing as mp
import uuid
from typing import Any

import pytest
import torch

# ════════════════════════════════════════════════════════════════════
# Layer 1 — CudaIpcControlRing control plane (CPU-only, runs in CI without a GPU)
# ════════════════════════════════════════════════════════════════════
#
# Single-mapping publish/poll protocol tests.
from vllm_omni.distributed.omni_connectors.connectors.cuda_ipc_control_ring import (
    RING_PCLASS_INLINE,
    RING_PCLASS_POOL,
    CudaIpcControlRing,
    RingFullError,
    key_hash16,
    ring_shm_name,
)


@pytest.fixture()
def ring():
    """A small sender-owned ring; unlinks on close."""
    name = f"test_ipc_ring_{uuid.uuid4().hex[:12]}"
    r = CudaIpcControlRing.create(name, n_slots=8, body_max=64, header_bytes=32)
    yield r
    r.close()


def test_ring_header_round_trip(ring):
    blob = b"edge-constant-handles\x00\x01\x02"
    ring.write_header(blob)
    assert ring.read_header(len(blob)) == blob


def test_ring_create_translates_shm_oserror_to_actionable_message(monkeypatch):
    """When /dev/shm is too small (the most common deploy gotcha), CudaIpcControlRing.create
    must raise an OSError whose message names the requested size and tells the operator how
    to fix it. The default 64 MB tmpfs limit in many containers would otherwise surface as a
    bare ENOSPC out of multiprocessing.shared_memory with no actionable context."""
    from vllm_omni.distributed.omni_connectors.connectors import cuda_ipc_control_ring as ring_mod

    def _fake_shm_init(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(ring_mod.shm_pkg, "SharedMemory", _fake_shm_init)
    name = f"test_ipc_ring_oserr_{uuid.uuid4().hex[:12]}"
    with pytest.raises(OSError) as excinfo:
        ring_mod.CudaIpcControlRing.create(name, n_slots=2048, body_max=524288, header_bytes=32)
    msg = str(excinfo.value)
    assert "shared memory" in msg
    assert "shm-size" in msg or "ring_entries" in msg, "operator-actionable hint must be present"
    # Size should be reported in MB — round-trip the math here so a future refactor that
    # silently changes the layout fails this assertion explicitly.
    expected_size_mb_lo = (8 + 32 + 2048 * 524288) // (1024 * 1024) - 1
    assert any(
        token in msg
        for token in (f"{expected_size_mb_lo} MB", f"{expected_size_mb_lo + 1} MB", f"{expected_size_mb_lo + 2} MB")
    ), f"expected size ~{expected_size_mb_lo} MB to appear in: {msg}"


def test_ring_header_overflow_rejected(ring):
    with pytest.raises(ValueError):
        ring.write_header(b"x" * 33)  # header_bytes=32


def test_ring_publish_then_poll(ring):
    kh = key_hash16("req-A_0_1")
    ring.publish(kh, pclass=0, body=b"hello")
    got = ring.poll(kh)
    assert got is not None
    pclass, body = got
    assert pclass == 0 and body == b"hello"


def test_ring_poll_miss_returns_none(ring):
    assert ring.poll(key_hash16("never-published")) is None


def test_ring_pclass_is_carried(ring):
    ring.publish(key_hash16("k-inline"), pclass=RING_PCLASS_INLINE, body=b"a")
    ring.publish(key_hash16("k-pool"), pclass=RING_PCLASS_POOL, body=b"bb")
    assert ring.poll(key_hash16("k-inline"))[0] == RING_PCLASS_INLINE
    assert ring.poll(key_hash16("k-pool"))[0] == RING_PCLASS_POOL


def test_ring_poll_marks_consumed_once(ring):
    kh = key_hash16("once")
    ring.publish(kh, 0, b"x")
    assert ring.poll(kh) is not None  # first poll consumes
    assert ring.poll(kh) is None  # second poll: already consumed


def test_ring_consumed_slot_is_reused(ring):
    """Producer must reuse a slot the consumer has taken — else the ring wedges after
    n_slots publishes. Round-trips far more entries than slots."""
    for i in range(8 * 20):  # 160 entries through 8 slots
        kh = key_hash16(f"seq-{i}")
        ring.publish(kh, 0, b"%d" % i)
        got = ring.poll(kh)
        assert got is not None and got[1] == b"%d" % i


def test_ring_open_addressed_collision(ring):
    """Distinct keys that may land on the same home slot must each be retrievable."""
    keys = [key_hash16(f"collide-{i}") for i in range(6)]  # < n_slots, all live at once
    for i, k in enumerate(keys):
        ring.publish(k, 0, b"v%d" % i)
    for i, k in enumerate(keys):
        got = ring.poll(k)
        assert got is not None and got[1] == b"v%d" % i


def test_ring_full_raises(ring):
    for i in range(8):  # fill all 8 slots without consuming
        ring.publish(key_hash16(f"fill-{i}"), 0, b"z")
    with pytest.raises(RingFullError):
        ring.publish(key_hash16("one-too-many"), 0, b"z")


def test_ring_body_too_big_raises(ring):
    with pytest.raises(ValueError):
        ring.publish(key_hash16("big"), 0, b"x" * 65)  # body_max=64


def test_ring_ttl_reclaims_stale_entry(ring):
    """An occupied-but-unconsumed slot older than ttl_sec is reclaimed in place so an
    aborted/never-polled request cannot wedge the ring. Fresh entries are NOT reclaimed."""
    for i in range(8):  # fill at t=100, never consumed
        ring.publish(key_hash16(f"stale-{i}"), 0, b"old", ts=100, ttl_sec=30)
    with pytest.raises(RingFullError):  # t=110 within ttl -> still full
        ring.publish(key_hash16("fresh"), 0, b"new", ts=110, ttl_sec=30)
    # t=200: the t=100 entries are stale (>30s) -> reclaimed in place, publish succeeds
    ring.publish(key_hash16("after-ttl"), 0, b"new", ts=200, ttl_sec=30)


def test_ring_ttl_zero_never_reclaims(ring):
    """ttl_sec=0 disables reclaim — full ring stays full regardless of ts."""
    for i in range(8):
        ring.publish(key_hash16(f"x-{i}"), 0, b"o", ts=100, ttl_sec=0)
    with pytest.raises(RingFullError):
        ring.publish(key_hash16("y"), 0, b"n", ts=999999, ttl_sec=0)


def test_ring_name_isolates_edge_and_replica():
    """Ring shm name is deterministic, unique per (edge, replica_id), and a valid POSIX shm name."""

    def name(rid, a, b):
        return ring_shm_name(a, b, rid)

    base = name(0, "0", "1")
    assert name(0, 0, 1) == base  # int (sender) / str (receiver) stage agree
    assert name(3, "0", "1") != base  # replica-unique
    assert name(0, "1", "2") != base  # edge-unique
    assert "/" not in base and " " not in base and len(base) < 40


# ════════════════════════════════════════════════════════════════════
# Layer 2 — CudaIPCConnector functional put/get (requires a GPU)
# ════════════════════════════════════════════════════════════════════


def _sender_proc(cmd_q: mp.Queue, res_q: mp.Queue, cfg: dict):
    import torch

    torch.cuda.set_device(cfg["device"])
    from vllm_omni.distributed.omni_connectors.connectors.cuda_ipc_connector import CudaIPCConnector

    sender = CudaIPCConnector(cfg)
    try:
        res_q.put(("ready",))
        while True:
            msg = cmd_q.get()
            if msg[0] == "put":
                _, fs, ts, key, spec = msg
                data = _materialize(spec, "cuda")
                ok, size, meta = sender.put(fs, ts, key, data)
                res_q.put(("put_done", ok, size, meta))
            elif msg[0] == "health":
                res_q.put(("health", sender.health()))
            elif msg[0] == "metrics":
                res_q.put(("metrics", dict(sender._metrics)))
            elif msg[0] == "quit":
                break
    finally:
        sender.close()


def _receiver_proc(cmd_q: mp.Queue, res_q: mp.Queue, cfg: dict):
    import torch

    torch.cuda.set_device(cfg["device"])
    from vllm_omni.distributed.omni_connectors.connectors.cuda_ipc_connector import CudaIPCConnector

    receiver = CudaIPCConnector(cfg)
    try:
        res_q.put(("ready",))
        while True:
            msg = cmd_q.get()
            if msg[0] == "get":
                _, fs, ts, key, meta = msg
                result = receiver.get(fs, ts, key, metadata=meta)
                if result is None:
                    res_q.put(("get_done", None))
                else:
                    obj, rsize = result
                    res_q.put(("get_done", _summarize(obj), rsize))
            elif msg[0] == "quit":
                break
    finally:
        receiver.close()


# ════════════════════════════════════════════════════════════════════
# Layer 1.5 — register_producer_stream wiring (CPU-only, no real GPU work)
# ════════════════════════════════════════════════════════════════════
#
# These tests exercise the producer-stream registration plumbing on a
# bare CudaIPCConnector instance built via ``object.__new__`` so we can
# call _init_runtime_state without booting CUDA. They lock in the
# contract that:
#
# 1. ``register_producer_stream`` stashes the stream on the connector and
#    resets the warn-once latch.
# 2. ``_maybe_warn_ambient_fallback`` warns exactly once per registration
#    cycle, so a future PTDS rollout can't silently corrupt put() data
#    without surfacing in the logs.


def _bare_connector():
    """Skeleton CudaIPCConnector with only the runtime-state slots populated.

    Avoids ``__init__`` (which would try to load cudart and allocate a pool).
    """
    from vllm_omni.distributed.omni_connectors.connectors.cuda_ipc_connector import CudaIPCConnector

    conn = object.__new__(CudaIPCConnector)
    CudaIPCConnector._init_runtime_state(conn)
    return conn


def test_register_producer_stream_sets_field_and_resets_warn_latch():
    conn = _bare_connector()
    assert conn._producer_stream is None
    assert conn._producer_fallback_warned is False

    sentinel = object()  # opaque placeholder; the connector never inspects it
    conn.register_producer_stream(sentinel)
    assert conn._producer_stream is sentinel
    assert conn._producer_fallback_warned is False

    conn._producer_fallback_warned = True
    conn.register_producer_stream(None)
    assert conn._producer_stream is None
    assert conn._producer_fallback_warned is False, (
        "Re-registration must reset the warn-once latch so the next ambient "
        "fallback (e.g. after clearing the producer stream) is observable."
    )


def test_get_refuses_on_non_transfer_rank_and_warns_once(caplog):
    """Defence-in-depth for the SPSC ring: a non-transfer rank that ends up
    calling ``get()`` (e.g. a future caller bypassing the mixin's
    ``is_data_transfer_rank`` gate) must be refused returning ``None``, and
    must surface a single warning per process so the bug is observable."""
    import logging

    from vllm_omni.distributed.omni_connectors.connectors import cuda_ipc_connector as conn_mod
    from vllm_omni.distributed.omni_connectors.connectors.cuda_ipc_connector import CudaIPCConnector

    conn = _bare_connector()
    conn._closed = False
    conn._is_transfer_rank = False
    conn.role = "receiver"
    conn.stage_id = 1
    conn._replica_id = 0

    with caplog.at_level(logging.WARNING, logger=conn_mod.logger.name):
        for _ in range(5):
            result = CudaIPCConnector.get(conn, "0", "1", "any-key", metadata=None)
            assert result is None, "non-transfer rank must not consume from the SPSC ring"

    refusal_warnings = [r for r in caplog.records if "non-transfer rank refused" in r.getMessage()]
    assert len(refusal_warnings) == 1, (
        f"Expected exactly one SPSC refusal warning, got {len(refusal_warnings)}; "
        "must be one-shot or steady-state recv loops would spam the log."
    )


def test_get_passes_through_on_transfer_rank():
    """Sanity counterpart to the gate test: when ``_is_transfer_rank`` is set,
    ``get()`` must dispatch to ``_get_control_plane`` (we capture the call
    rather than exercise the real ring/pool, which need CUDA + a peer process)."""
    from vllm_omni.distributed.omni_connectors.connectors.cuda_ipc_connector import CudaIPCConnector

    conn = _bare_connector()
    conn._closed = False
    conn._is_transfer_rank = True
    captured: dict = {}

    def fake_get_control_plane(from_stage, to_stage, get_key, composite_key):
        captured.update(
            from_stage=from_stage,
            to_stage=to_stage,
            get_key=get_key,
            composite_key=composite_key,
        )
        return ({"hello": "world"}, 16)

    conn._get_control_plane = fake_get_control_plane

    result = CudaIPCConnector.get(conn, "0", "1", "k1", metadata=None)
    assert result == ({"hello": "world"}, 16)
    assert captured["from_stage"] == "0"
    assert captured["to_stage"] == "1"
    assert captured["get_key"] == "k1"
    assert "k1" in captured["composite_key"]


def test_maybe_warn_ambient_fallback_is_one_shot(caplog):
    import logging

    from vllm_omni.distributed.omni_connectors.connectors import cuda_ipc_connector as conn_mod

    conn = _bare_connector()

    with caplog.at_level(logging.WARNING, logger=conn_mod.logger.name):
        conn._maybe_warn_ambient_fallback()
        conn._maybe_warn_ambient_fallback()
        conn._maybe_warn_ambient_fallback()

    fallback_warnings = [r for r in caplog.records if "ambient-stream fallback" in r.getMessage()]
    assert len(fallback_warnings) == 1, (
        f"Expected exactly one ambient-fallback warning, got {len(fallback_warnings)}; "
        "the warning must be one-shot or the log will spam under steady-state put() traffic."
    )


def _materialize(spec: dict, device: str) -> dict:
    import torch

    out: dict[str, Any] = {}
    for k, v in spec.items():
        if isinstance(v, dict) and v.get("__t"):
            out[k] = torch.randn(*v["shape"], device=device, dtype=getattr(torch, v["dtype"]))
        else:
            out[k] = v
    return out


def _tspec(shape: tuple, dtype: str = "bfloat16") -> dict:
    return {"__t": True, "shape": list(shape), "dtype": dtype}


def _summarize(obj: dict) -> dict:
    summary: dict[str, Any] = {}
    for k, v in obj.items():
        if isinstance(v, torch.Tensor):
            summary[k] = {"shape": list(v.shape), "device": str(v.device)}
        else:
            summary[k] = v
    return summary


class _Harness:
    def __init__(self, pool_size_mb: int = 32, pool_credits: int = 16):
        ctx = mp.get_context("spawn")
        self.s_cmd: mp.Queue = ctx.Queue()
        self.s_res: mp.Queue = ctx.Queue()
        self.r_cmd: mp.Queue = ctx.Queue()
        self.r_res: mp.Queue = ctx.Queue()
        dev = torch.accelerator.current_device_index()

        s_cfg = {
            "stage_id": 0,
            "role": "sender",
            "local_device": dev,
            "pool_size_mb": pool_size_mb,
            "pool_credits": pool_credits,
            "tensor_lifetime_sec": 10.0,
            "device": dev,
        }
        r_cfg = {"stage_id": 1, "role": "receiver", "local_device": dev, "device": dev}

        self.sender = ctx.Process(target=_sender_proc, args=(self.s_cmd, self.s_res, s_cfg), daemon=True)
        self.receiver = ctx.Process(target=_receiver_proc, args=(self.r_cmd, self.r_res, r_cfg), daemon=True)
        self.sender.start()
        self.receiver.start()
        assert self.s_res.get(timeout=30)[0] == "ready"
        assert self.r_res.get(timeout=30)[0] == "ready"

    def put(self, fs, ts, key, spec, timeout=10):
        self.s_cmd.put(("put", fs, ts, key, spec))
        r = self.s_res.get(timeout=timeout)
        return r[1], r[2], r[3]

    def get(self, fs, ts, key, meta=None, timeout=10):
        self.r_cmd.put(("get", fs, ts, key, meta))
        r = self.r_res.get(timeout=timeout)
        if r[1] is None:
            return None
        return r[1], r[2]

    def health(self, timeout=5):
        self.s_cmd.put(("health",))
        return self.s_res.get(timeout=timeout)[1]

    def metrics(self, timeout=5):
        self.s_cmd.put(("metrics",))
        return self.s_res.get(timeout=timeout)[1]

    def close(self):
        for q in (self.s_cmd, self.r_cmd):
            try:
                q.put(("quit",))
            except Exception:
                pass
        self.sender.join(timeout=5)
        self.receiver.join(timeout=5)
        for p in (self.sender, self.receiver):
            if p.is_alive():
                p.kill()


@pytest.fixture()
def harness():
    h = _Harness()
    yield h
    h.close()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestCudaIPCFunctional:
    def test_put_then_get(self, harness):
        spec = {"hidden": _tspec((128, 256)), "meta": {"req_id": "r1"}}
        ok, size, meta = harness.put("0", "1", "req_1", spec)
        assert ok and size > 0
        # Pin the ring transport: a stage-id/name mismatch would silently route through the
        # SHM fallback and still "pass", masking a broken ring path.
        assert not meta.get("cpu_fallback"), "expected ring/pool path, got SHM fallback"

        result = harness.get("0", "1", "req_1", meta=meta)
        assert result is not None
        summary, _ = result
        assert summary["hidden"]["shape"] == [128, 256]
        assert "cuda" in summary["hidden"]["device"]
        assert summary["meta"]["req_id"] == "r1"

    def test_multiple_keys(self, harness):
        for i in range(8):
            ok, _, _ = harness.put("0", "1", f"req_{i}", {"h": _tspec((64, 128)), "i": i})
            assert ok

        for i in range(8):
            result = harness.get("0", "1", f"req_{i}")
            assert result is not None
            summary, _ = result
            assert summary["i"] == i

    def test_cpu_fallback_on_overflow(self, harness):
        spec = {"big": _tspec((8 * 1024 * 1024,), "float32")}
        ok, _, meta = harness.put("0", "1", "big_req", spec)
        assert ok
        assert meta.get("cpu_fallback", False)

    def test_health(self, harness):
        h = harness.health()
        assert h["status"] == "healthy"
        assert h["role"] == "sender"
        assert h["pool_credits"] == 16

    def test_supports_gpu_tensor_flag(self):
        from vllm_omni.distributed.omni_connectors.connectors.cuda_ipc_connector import CudaIPCConnector

        assert CudaIPCConnector.supports_gpu_tensor is True
