# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for ``CudaIPCConnector`` concurrency-safety fixes.

These tests focus on the pure-Python concurrency control logic — the ACK
drain loop and the ``get()`` routing/TOCTOU logic. They construct connector
instances via ``__new__`` so the constructor's CUDA initialization is
skipped, allowing the tests to run on CPU-only CI hosts.

The connector module is loaded directly by file path with stubbed
dependencies, avoiding the full ``vllm_omni`` package-import chain (which
otherwise pulls in heavyweight optional deps such as ``vllm`` patches and
``transformers``).
"""

from __future__ import annotations

import importlib.util
import os
import sys
import threading
import time
import types
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Lightweight module loader: avoid importing the whole ``vllm_omni`` package.
# ---------------------------------------------------------------------------


def _load_connector_module() -> types.ModuleType:
    """Load ``cuda_ipc_connector.py`` with stubbed inter-package deps.

    The connector references three project-internal symbols:
      * ``vllm_omni.entrypoints.stage_utils.shm_read_bytes`` / ``shm_write_bytes``
      * ``..utils.logging.get_connector_logger``
      * ``..utils.serialization.OmniSerializer``
      * ``.base.OmniConnectorBase``

    All of them are easy to stub for unit tests focused on lock/TOCTOU logic.
    """
    project_root = Path(__file__).resolve().parents[3]
    target = project_root / "vllm_omni/distributed/omni_connectors/connectors/cuda_ipc_connector.py"
    assert target.exists(), f"Missing connector source: {target}"

    # --- Stub the project-internal modules the connector imports from. ---
    # ``vllm_omni.entrypoints.stage_utils``
    stage_utils = types.ModuleType("vllm_omni.entrypoints.stage_utils")

    def _stub_shm_write_bytes(*_a, **_k):  # pragma: no cover - not exercised
        raise AssertionError("shm_write_bytes should not be called by these tests")

    def _stub_shm_read_bytes(*_a, **_k):  # pragma: no cover - not exercised
        raise AssertionError("shm_read_bytes should not be called by these tests")

    stage_utils.shm_write_bytes = _stub_shm_write_bytes
    stage_utils.shm_read_bytes = _stub_shm_read_bytes

    pkg_vllm_omni = types.ModuleType("vllm_omni")
    pkg_vllm_omni.__path__ = []  # mark as package
    pkg_entrypoints = types.ModuleType("vllm_omni.entrypoints")
    pkg_entrypoints.__path__ = []
    sys.modules.setdefault("vllm_omni", pkg_vllm_omni)
    sys.modules.setdefault("vllm_omni.entrypoints", pkg_entrypoints)
    sys.modules["vllm_omni.entrypoints.stage_utils"] = stage_utils

    # --- Build a minimal ``connectors`` parent package containing the
    # ``base`` and ``..utils.{logging,serialization}`` siblings used by the
    # connector. ---
    pkg = types.ModuleType("ipc_test_pkg")
    pkg.__path__ = []
    utils_pkg = types.ModuleType("ipc_test_pkg.utils")
    utils_pkg.__path__ = []
    connectors_pkg = types.ModuleType("ipc_test_pkg.connectors")
    connectors_pkg.__path__ = []
    sys.modules["ipc_test_pkg"] = pkg
    sys.modules["ipc_test_pkg.utils"] = utils_pkg
    sys.modules["ipc_test_pkg.connectors"] = connectors_pkg

    # logging stub
    logging_mod = types.ModuleType("ipc_test_pkg.utils.logging")
    import logging as _logging

    def _get_connector_logger(name):
        return _logging.getLogger(name)

    logging_mod.get_connector_logger = _get_connector_logger
    sys.modules["ipc_test_pkg.utils.logging"] = logging_mod

    # serialization stub: a tiny pickle-based serializer is enough; tests
    # never exercise the deserialize-from-SHM code path end-to-end.
    serialization_mod = types.ModuleType("ipc_test_pkg.utils.serialization")
    import pickle as _pickle

    class _OmniSerializer:
        @staticmethod
        def serialize(obj):
            return _pickle.dumps(obj)

        @staticmethod
        def deserialize(data):
            return _pickle.loads(data)

    serialization_mod.OmniSerializer = _OmniSerializer
    sys.modules["ipc_test_pkg.utils.serialization"] = serialization_mod

    # base stub: provide the abstract surface the connector inherits from.
    base_mod = types.ModuleType("ipc_test_pkg.connectors.base")

    class _OmniConnectorBase:
        supports_raw_data = False
        supports_gpu_tensor = False

        @staticmethod
        def serialize_obj(obj):
            return _OmniSerializer.serialize(obj)

        @staticmethod
        def deserialize_obj(data):
            return _OmniSerializer.deserialize(data)

        def __del__(self):
            try:
                self.close()
            except Exception:
                pass

    base_mod.OmniConnectorBase = _OmniConnectorBase
    sys.modules["ipc_test_pkg.connectors.base"] = base_mod

    # --- Load the source file with a synthetic fully-qualified name so the
    # ``..utils.*`` and ``.base`` relative imports resolve to our stubs. ---
    spec = importlib.util.spec_from_file_location(
        "ipc_test_pkg.connectors.cuda_ipc_connector",
        target,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["ipc_test_pkg.connectors.cuda_ipc_connector"] = module
    spec.loader.exec_module(module)
    return module


cuda_ipc_connector = _load_connector_module()
CudaIPCConnector = cuda_ipc_connector.CudaIPCConnector
_IPC_SEGMENT_NOT_PRESENT = cuda_ipc_connector._IPC_SEGMENT_NOT_PRESENT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sender_skeleton(
    *,
    tensor_lifetime_sec: float = 30.0,
    max_held_bytes: int = 16 * 1024**2,
) -> CudaIPCConnector:
    """Build a connector instance without running CUDA-dependent init.

    We bypass ``__init__`` and manually populate the fields the methods we
    are testing actually touch.  This keeps the unit tests free of CUDA /
    libcudart dependencies.
    """
    c = CudaIPCConnector.__new__(CudaIPCConnector)
    c.config = {}
    c.stage_id = 0
    c.role = "sender"
    c.tensor_lifetime_sec = tensor_lifetime_sec
    c.local_device = "cpu"
    c._closed = False
    c._cudart = None
    c._max_held_bytes = max_held_bytes
    c._held_bytes = 0
    c._held_tensors = {}
    c._held_lock = threading.Lock()
    c._stop_event = threading.Event()
    c._ack_thread = None
    c._shm_compat_decode_failures = {}
    c._metrics = {
        "puts": 0,
        "gets": 0,
        "bytes_transferred": 0,
        "gpu_tensors_transferred": 0,
        "acks": 0,
        "ack_timeouts": 0,
        "errors": 0,
        "cpu_fallbacks": 0,
    }
    return c


# ---------------------------------------------------------------------------
# _drain_acks: lock-under-I/O regression tests
# ---------------------------------------------------------------------------


def test_drain_acks_does_not_hold_lock_during_shm_probe():
    """``_has_ack`` (SHM I/O) must run *outside* ``_held_lock``.

    Regression test for the lock-under-I/O pattern: previously, a slow or
    stuck SHM open inside ``_has_ack`` could block all concurrent ``put()``
    callers because they share ``_held_lock``. The fix snapshots state under
    the lock, probes SHM without it, and only re-acquires briefly to mutate.
    """
    c = _make_sender_skeleton()
    c._held_tensors["k1"] = (time.time(), [], 100)
    c._held_bytes = 100

    probe_started = threading.Event()
    release_probe = threading.Event()
    lock_was_free_during_probe = threading.Event()

    def slow_has_ack(_key: str) -> bool:
        probe_started.set()
        # The connector's lock should be free during this I/O.
        if c._held_lock.acquire(blocking=False):
            try:
                lock_was_free_during_probe.set()
            finally:
                c._held_lock.release()
        release_probe.wait(timeout=2.0)
        return False

    drain_thread = threading.Thread(target=c._drain_acks)

    with patch.object(c, "_has_ack", side_effect=slow_has_ack):
        drain_thread.start()
        try:
            assert probe_started.wait(timeout=2.0), "Probe never started"
            assert lock_was_free_during_probe.wait(timeout=2.0), (
                "_held_lock was held during _has_ack — lock-under-I/O regression"
            )
        finally:
            release_probe.set()
            drain_thread.join(timeout=2.0)
            assert not drain_thread.is_alive()


def test_drain_acks_concurrent_put_lock_acquire_not_blocked():
    """A slow ``_has_ack`` must not block holders of ``_held_lock``.

    ``put()`` only briefly takes ``_held_lock``; if ``_drain_acks`` is
    holding the lock during a slow SHM probe, ``put()`` would stall.
    Acquire the lock from the test thread with a short timeout while
    ``_drain_acks`` is mid-probe.
    """
    c = _make_sender_skeleton()
    for i in range(5):
        c._held_tensors[f"k{i}"] = (time.time(), [], 10)
        c._held_bytes += 10

    release_probe = threading.Event()
    probe_in_progress = threading.Event()

    def slow_has_ack(_key: str) -> bool:
        probe_in_progress.set()
        release_probe.wait(timeout=3.0)
        return False

    drain_thread = threading.Thread(target=c._drain_acks)
    with patch.object(c, "_has_ack", side_effect=slow_has_ack):
        drain_thread.start()
        try:
            assert probe_in_progress.wait(timeout=2.0)
            acquired = c._held_lock.acquire(timeout=0.5)
            try:
                assert acquired, "put() would have been blocked by _drain_acks"
            finally:
                if acquired:
                    c._held_lock.release()
        finally:
            release_probe.set()
            drain_thread.join(timeout=3.0)
            assert not drain_thread.is_alive()


def test_drain_acks_releases_acked_entries():
    c = _make_sender_skeleton()
    c._held_tensors["a"] = (time.time(), [], 100)
    c._held_tensors["b"] = (time.time(), [], 200)
    c._held_bytes = 300

    with patch.object(c, "_has_ack", side_effect=lambda k: k == "a"):
        c._drain_acks()

    assert "a" not in c._held_tensors
    assert "b" in c._held_tensors
    assert c._held_bytes == 200
    assert c._metrics["acks"] == 1
    assert c._metrics["ack_timeouts"] == 0


def test_drain_acks_releases_timed_out_entries():
    c = _make_sender_skeleton(tensor_lifetime_sec=0.05)
    old_ts = time.time() - 1.0
    c._held_tensors["stale"] = (old_ts, [], 50)
    c._held_tensors["fresh"] = (time.time(), [], 75)
    c._held_bytes = 125

    with patch.object(c, "_has_ack", return_value=False):
        c._drain_acks()

    assert "stale" not in c._held_tensors
    assert "fresh" in c._held_tensors
    assert c._held_bytes == 75
    assert c._metrics["ack_timeouts"] == 1


def test_drain_acks_does_not_drop_refreshed_entry():
    """If an entry is replaced (different timestamp) between snapshot and
    re-acquire, the new entry must not be dropped.

    Without the snapshot/timestamp re-validation, a slow probe followed by
    a fast new ``put()`` for the same composite key could result in the new
    holder being silently released, leaving a dangling IPC handle.
    """
    c = _make_sender_skeleton()
    old_ts = time.time() - 100.0
    c._held_tensors["k"] = (old_ts, [], 10)
    c._held_bytes = 10

    refreshed = {}

    def has_ack_then_refresh(_key: str) -> bool:
        new_ts = time.time()
        refreshed["ts"] = new_ts
        with c._held_lock:
            c._held_tensors["k"] = (new_ts, [], 10)
        return True  # claim ACK for the *old* entry

    with patch.object(c, "_has_ack", side_effect=has_ack_then_refresh):
        c._drain_acks()

    assert "k" in c._held_tensors, "Refreshed entry was incorrectly dropped"
    assert c._held_tensors["k"][0] == refreshed["ts"]
    assert c._held_bytes == 10
    assert c._metrics["acks"] == 0


def test_drain_acks_empty_state_is_noop():
    c = _make_sender_skeleton()
    with patch.object(c, "_has_ack", side_effect=AssertionError("should not be called")):
        c._drain_acks()
    assert c._held_bytes == 0


def test_drain_acks_concurrent_mutation_is_safe():
    """Hammer with concurrent inserts while ``_drain_acks`` runs."""
    c = _make_sender_skeleton(tensor_lifetime_sec=10.0)
    stop = threading.Event()

    def writer():
        i = 0
        while not stop.is_set():
            key = f"w{i % 32}"
            with c._held_lock:
                prev = c._held_tensors.get(key)
                if prev is not None:
                    c._held_bytes -= prev[2]
                c._held_tensors[key] = (time.time(), [], 1)
                c._held_bytes += 1
            i += 1

    with patch.object(c, "_has_ack", side_effect=lambda k: int(k[1:]) % 2 == 0):
        threads = [threading.Thread(target=writer) for _ in range(4)]
        for t in threads:
            t.start()
        try:
            for _ in range(50):
                c._drain_acks()
        finally:
            stop.set()
            for t in threads:
                t.join(timeout=2.0)

    expected = sum(nbytes for (_ts, _h, nbytes) in c._held_tensors.values())
    assert c._held_bytes == expected


# ---------------------------------------------------------------------------
# get(): TOCTOU & routing tests
# ---------------------------------------------------------------------------


def test_get_routes_to_shm_compat_when_lock_file_absent():
    """No lock file => CPU-fallback / shm_compat path, never opens IPC SHM."""
    c = _make_sender_skeleton()
    c.role = "receiver"

    composite_key = "key1@s0_s1"
    payload_name = c._payload_name(composite_key)
    lock_file = c._lock_file(payload_name)
    if os.path.exists(lock_file):
        os.remove(lock_file)

    called = {"ipc": 0, "compat": 0}

    def fake_ipc(*_a, **_k):
        called["ipc"] += 1
        return None

    def fake_compat(_get_key):
        called["compat"] += 1
        return ({"data": "ok"}, 5)

    with (
        patch.object(c, "_try_get_ipc", side_effect=fake_ipc),
        patch.object(c, "_try_get_shm_compat", side_effect=fake_compat),
    ):
        result = c.get("s0", "s1", "key1")

    assert result == ({"data": "ok"}, 5)
    assert called == {"ipc": 0, "compat": 1}


def test_get_falls_back_to_compat_when_ipc_segment_missing_under_lock():
    """If the lock file exists but the IPC segment is absent under the
    lock, fall back to ``_try_get_shm_compat`` transparently.

    This closes the TOCTOU window in which a sender might be mid-write
    between a lock-free probe and the locked read.
    """
    c = _make_sender_skeleton()
    c.role = "receiver"

    composite_key = "key2@s0_s1"
    payload_name = c._payload_name(composite_key)
    lock_file = c._lock_file(payload_name)

    try:
        with open(lock_file, "wb") as f:
            f.write(b"")

        compat_called = {"n": 0}

        def fake_compat(_k):
            compat_called["n"] += 1
            return ({"fallback": True}, 3)

        with (
            patch.object(c, "_try_get_ipc", return_value=_IPC_SEGMENT_NOT_PRESENT),
            patch.object(c, "_try_get_shm_compat", side_effect=fake_compat),
        ):
            result = c.get("s0", "s1", "key2")

        assert result == ({"fallback": True}, 3)
        assert compat_called["n"] == 1
    finally:
        if os.path.exists(lock_file):
            os.remove(lock_file)


def test_try_get_ipc_returns_sentinel_when_lock_file_missing():
    """Lock file gone => sentinel, no spurious errors."""
    c = _make_sender_skeleton()
    c.role = "receiver"
    composite_key = "missing@s0_s1"
    payload_name = c._payload_name(composite_key)
    lock_file = c._lock_file(payload_name)
    if os.path.exists(lock_file):
        os.remove(lock_file)

    result = c._try_get_ipc("missing", composite_key, payload_name, lock_file)
    assert result is _IPC_SEGMENT_NOT_PRESENT


def test_try_get_ipc_returns_sentinel_when_segment_missing_under_lock():
    """Lock file present but SHM segment absent => sentinel.

    This is the TOCTOU fix: the existence check is performed under the lock,
    so a missing segment never produces a corrupt-read or false-positive.
    """
    c = _make_sender_skeleton()
    c.role = "receiver"
    composite_key = "halfwritten@s0_s1"
    payload_name = c._payload_name(composite_key)
    lock_file = c._lock_file(payload_name)

    try:
        with open(lock_file, "wb") as f:
            f.write(b"")

        result = c._try_get_ipc("halfwritten", composite_key, payload_name, lock_file)
        assert result is _IPC_SEGMENT_NOT_PRESENT
    finally:
        if os.path.exists(lock_file):
            os.remove(lock_file)


def test_get_returns_none_when_closed():
    c = _make_sender_skeleton()
    c.role = "receiver"
    c._closed = True
    assert c.get("s0", "s1", "anything") is None


def test_get_does_not_probe_shm_outside_lock():
    """Regression: ``get()`` itself must not open the IPC SHM segment
    *before* taking the per-payload file lock.

    The original implementation probed ``shm_pkg.SharedMemory(name=...)``
    in ``get()`` *outside* the lock. That probe is exactly the TOCTOU
    surface the reviewer flagged. Verify that ``get()`` no longer opens
    the SHM segment by name unless ``_try_get_ipc`` is invoked (which
    performs the open under the lock).
    """
    c = _make_sender_skeleton()
    c.role = "receiver"

    composite_key = "guard@s0_s1"
    payload_name = c._payload_name(composite_key)
    lock_file = c._lock_file(payload_name)
    if os.path.exists(lock_file):
        os.remove(lock_file)

    # Track every ``shm_pkg.SharedMemory`` constructor call.
    open_calls: list[str] = []
    real_shm_cls = cuda_ipc_connector.shm_pkg.SharedMemory

    class _RecordingSHM:
        def __init__(self, *args, name=None, **kwargs):
            open_calls.append(name)
            raise FileNotFoundError(name)

    with (
        patch.object(cuda_ipc_connector.shm_pkg, "SharedMemory", _RecordingSHM),
        patch.object(c, "_try_get_shm_compat", return_value=None),
    ):
        # Lock file missing => only compat path runs. Nothing should ever
        # invoke ``shm_pkg.SharedMemory(name=payload_name)`` from ``get()``.
        c.get("s0", "s1", "guard")

    # Sanity: still using the real type when not patched (avoid leaking
    # the recording class to other tests).
    assert cuda_ipc_connector.shm_pkg.SharedMemory is real_shm_cls

    # The IPC ``payload_name`` must never have been opened from ``get()``
    # itself (the lock-free probe). It is fine for compat-path internals
    # to open the *get_key*-named segment, but that path is mocked above.
    assert payload_name not in open_calls, (
        "get() opened the IPC segment outside the per-payload file lock — TOCTOU regression"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
