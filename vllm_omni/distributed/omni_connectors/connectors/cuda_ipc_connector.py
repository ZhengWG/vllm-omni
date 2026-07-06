# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CUDA IPC connector: GPU-to-GPU transfer over a pre-allocated pool + per-edge ring.

- Sender allocates one GPU pool, exports its IPC handle once, splits it into credit slots;
  put() copies tensors into a slot (D2D).
- Control plane: a per-edge keyed ring (``CudaIpcControlRing``) carries small payloads inline
  and big-payload pool descriptors, opened once per edge.
- Receiver opens the pool handle once (cached) and D2D-copies from the slot offset.
- Credit release via a shared-memory release board (1 byte/slot) + a TTL sweep.
- Ordering: both ends synchronize their copy stream before hand-off — no async use-after-free.
- Fallback: on credit/ring exhaustion or payload overflow, put() degrades to CPU /dev/shm.

Limitation: no sender live-restart (receiver caches the sender's IPC handles); restart the edge.
"""

import ctypes
import os
import platform
import queue as _queue_mod
import threading
import time as _time_mod
import uuid
from collections import deque
from multiprocessing import shared_memory as shm_pkg
from multiprocessing.resource_tracker import unregister
from typing import Any

import torch

from vllm_omni.entrypoints.stage_utils import shm_write_bytes

from ..utils.logging import get_connector_logger
from ..utils.serialization import OmniSerializer
from .base import OmniConnectorBase
from .cuda_ipc_control_ring import (
    RING_HEADER_BYTES,
    RING_PCLASS_INLINE,
    RING_PCLASS_POOL,
    CudaIpcControlRing,
    RingFullError,
    RingHeader,
    key_hash16,
    make_composite_key,
    ring_shm_name,
    untrack_shm,
)
from .cuda_ipc_runtime import (
    _CUDA_EVENT_DISABLE_TIMING,
    _CUDA_EVENT_INTERPROCESS,
    _CUDA_IPC_MEM_LAZY_ENABLE_PEER_ACCESS,
    _CudaIpcEventHandle,
    _CudaIpcMemHandle,
    load_cudart,
    make_dlpack_view,
    memcpy_async_d2d,
    stream_wait_event,
)

logger = get_connector_logger(__name__)

_GPU_TENSOR_MARKER = "__cuda_ipc_tensor__"
_POOL_MARKER = "__cuda_ipc_pool__"

_POOL_ALIGNMENT = 16  # bytes, for GPU copy efficiency

# Auto-size when pool_size_mb / pool_credits are omitted: credits = max(64, max_num_seqs*2),
# pool_size_mb = credits * slot_size_mb (default 64 MB/slot — fits Qwen3-Omni ~33 MB prefill
# handoff @ input 4000). Explicit pool_size_mb / pool_credits / slot_size_mb override.
_DEFAULT_SLOT_SIZE_MB = 64
_DEFAULT_POOL_CREDITS = 64
_DEFAULT_SMALL_SLOT_KB = 256  # small-slot pool: streaming-chunk sized
_DEFAULT_SMALL_SLOT_COUNT = 128  # +32MB GPU with the defaults
_SMALL_SLOT_SLACK = 4096  # routing headroom for per-tensor alignment padding
_DEFAULT_RECV_STREAMS = 8  # receiver D2D copy streams (round-robined per get)

# Timing constants — overridable via extra config keys of the same name
# (without leading underscore), e.g. ``"credit_wait_sec": 0.01``.
_CREDIT_WAIT_SEC = 0.05  # put() inline reclaim window before CPU fallback
_CREDIT_POLL_SEC = 0.0005  # poll interval within the reclaim window
_RELEASE_FAST_INTERVAL_SEC = 0.001  # board-reclaim thread fast tick
_RELEASE_TTL_EVERY_N_TICKS = 20  # TTL sweep runs every N fast ticks


class _SlotOverflowError(Exception):
    """Raised when tensors exceed a pool slot's capacity."""

    def __init__(self, nbytes: int = 0, slot_size: int = 0):
        super().__init__(f"tensor {nbytes}B exceeds pool slot {slot_size}B")
        self.nbytes = nbytes
        self.slot_size = slot_size


class _PoolSlot:
    """Tracks packing state for tensors within a single pool credit slot."""

    __slots__ = ("_pool", "_base", "_size", "_cursor")

    def __init__(self, pool: torch.Tensor, slot_offset: int, slot_size: int):
        self._pool = pool
        self._base = slot_offset
        self._size = slot_size
        self._cursor = 0

    def pack(self, tensor: torch.Tensor) -> int:
        """Copy tensor into the pool slot via PyTorch .copy_(), return byte offset."""
        nbytes = tensor.nbytes
        padding = (-self._cursor) % _POOL_ALIGNMENT
        aligned = self._cursor + padding
        if aligned + nbytes > self._size:
            raise _SlotOverflowError(nbytes, self._size)
        src_bytes = tensor.view(torch.uint8).reshape(-1)
        self._pool[self._base + aligned : self._base + aligned + nbytes].copy_(src_bytes)
        self._cursor = aligned + nbytes
        return aligned


class CudaIPCConnector(OmniConnectorBase):
    """CUDA IPC connector with pre-allocated memory pool.

    Sender pre-allocates a GPU memory pool, registers its IPC handle once,
    and divides it into credit-managed slots. Each put() copies tensors into
    a slot and sends the offset via SHM. The receiver opens the pool handle
    once (cached) and copies from the offset — no per-tensor IPC overhead.
    """

    supports_gpu_tensor: bool = True
    request_scoped_cleanup: bool = True  # zero-copy slot lease release; no-op otherwise

    # --- Init ---

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self._parse_config(config)
        self._init_runtime_state()
        self._init_cuda()
        self._init_lazy_defaults()
        self._reclaim_stale_ring_at_init()
        logger.info(
            "CudaIPCConnector initialized: role=%s, local_device=%s, replica_id=%s, "
            "pool=%dMB (%d credits x %dMB slots) — edge resources allocate lazily on first use",
            self.role,
            self.local_device,
            self._replica_id,
            self._pool_size // (1024 * 1024),
            self._pool_credits,
            self._slot_size // 1024 // 1024,
        )
        logger.debug("CudaIPCConnector config_keys=%s", sorted(config.keys()))

    def _parse_config(self, config: dict[str, Any]) -> None:
        """Resolve role, edge identity, thresholds, timing, and pool sizing."""
        self.stage_id = int(config.get("stage_id", -1))
        self.role = str(config.get("role", "sender")).lower()
        if self.role not in {"sender", "receiver"}:
            raise ValueError(f"Invalid role={self.role!r}. Expected 'sender' or 'receiver'.")
        self.tensor_lifetime_sec = float(config.get("tensor_lifetime_sec", 30.0))
        # replica_id (per same-host replica) from VLLM_OMNI_REPLICA_ID, set by the engine-core
        # process. Aligned 1:1 edges: sender and receiver resolve the same value.
        _rid = config.get("replica_id")
        if _rid is None:
            _rid = os.environ.get("VLLM_OMNI_REPLICA_ID", 0)
        self._replica_id = max(0, int(_rid or 0))
        # TP>1: only the data-transfer rank owns the per-edge ring (non-transfer ranks never
        # transmit, so must not create a same-named ring). Injected by the stage worker.
        self._is_transfer_rank = bool(config.get("is_transfer_rank", True))
        # keep_on_gpu=false: transport-only mode — ring/pool still used, but
        # payload builders keep tensors on the CPU async-output pipeline.
        self.supports_gpu_tensor = bool(config.get("keep_on_gpu", True))
        # Tiered placement (L2): keep only tensors >= this many bytes on GPU.
        # 0 keeps everything (classic keep_on_gpu); pair with keep_on_gpu: true.
        # Stable per-key GPU placement: listed key roots stay on GPU for every
        # chunk (per-instance size decisions flap devices and break stream cat).
        _keys = config.get("keep_on_gpu_keys") or []
        self._keep_on_gpu_keys = frozenset(str(k) for k in _keys)
        # Zero-copy receive: slots consumed as DLPack views; release deferred
        # to request cleanup + TTL backstop.
        self._zero_copy_recv = bool(config.get("zero_copy_recv", False))
        self._inline_threshold = int(config.get("inline_threshold_bytes", 16384))
        self._ring_entries_cfg = int(config.get("ring_entries", 0))  # 0 => auto from credits
        self._ring_body_max = int(config.get("ring_body_max", 524288))
        self.local_device = self._resolve_local_device(config.get("local_device", "auto"))
        # Pool sizing: credits (concurrency) and slot_size_mb (max single pool put) are independent;
        # pool_size_mb defaults to credits * slot_size_mb unless explicitly set.
        max_num_seqs = int(config.get("max_num_seqs", 0))
        if "pool_credits" in config:
            auto_credits = int(config["pool_credits"])
        elif max_num_seqs > 0:
            # ~1 pool credit per req (chunk-0 prefill); x2 headroom for board-reclaim lag.
            auto_credits = max(_DEFAULT_POOL_CREDITS, max_num_seqs * 2)
        else:
            auto_credits = _DEFAULT_POOL_CREDITS
        slot_size_mb = int(config.get("slot_size_mb", _DEFAULT_SLOT_SIZE_MB))
        self._pool_credits = int(config.get("pool_credits", auto_credits))
        # Small-slot pool for streaming chunks (e.g. codec frames): payloads far
        # below the big-slot size stop burning a whole big credit, which keeps
        # big credits available for prefill handoffs under load. Disable with
        # small_slot_count: 0.
        self._small_slot_size = int(config.get("small_slot_kb", _DEFAULT_SMALL_SLOT_KB)) * 1024
        self._small_slot_count = int(config.get("small_slot_count", _DEFAULT_SMALL_SLOT_COUNT))
        if self._small_slot_size <= 0:
            self._small_slot_count = 0
        if "pool_size_mb" in config:
            auto_size_mb = int(config["pool_size_mb"])
        else:
            auto_size_mb = self._pool_credits * slot_size_mb
        self._pool_size = auto_size_mb * 1024 * 1024
        self._slot_size = self._pool_size // self._pool_credits
        # Timing overrides via extra config.
        self._credit_wait_sec = float(config.get("credit_wait_sec", _CREDIT_WAIT_SEC))
        self._credit_poll_sec = float(config.get("credit_poll_sec", _CREDIT_POLL_SEC))
        self._release_interval_sec = float(config.get("release_fast_interval_sec", _RELEASE_FAST_INTERVAL_SEC))
        self._release_ttl_every = int(config.get("release_ttl_every_n_ticks", _RELEASE_TTL_EVERY_N_TICKS))

    @property
    def gpu_tensor_keys(self) -> frozenset | None:
        """Stable per-key GPU placement set (None when unfiltered/disabled)."""
        if not self.supports_gpu_tensor or not self._keep_on_gpu_keys:
            return None
        return self._keep_on_gpu_keys

    def _init_runtime_state(self) -> None:
        """Locks, ring/receiver caches, metrics, lifecycle flags."""
        self._ring: CudaIpcControlRing | None = None
        self._opened_rings: dict[tuple[str, str], CudaIpcControlRing] = {}
        self._ring_edge_handles: dict[tuple[str, str], tuple[bytes, bytes, str]] = {}
        # inode tracking detects sender-side ring replacement (see refresh).
        self._ring_inodes: dict[tuple[str, str], int] = {}
        self._ring_miss_counts: dict[tuple[str, str], int] = {}
        self._opened_pools: dict[bytes, ctypes.c_void_p] = {}
        self._opened_boards: dict[str, shm_pkg.SharedMemory] = {}
        self._opened_events: dict[bytes, ctypes.c_void_p] = {}
        self._closed = False
        self._cudart = None
        self._held_lock = threading.Lock()
        self._open_lock = threading.Lock()
        self._lazy_init_lock = threading.Lock()
        # Packs published before their D2D completes: (event, payload_ref)
        # keeps sources alive; drained by the release loop.
        self._inflight_packs: deque[tuple[torch.cuda.Event, Any]] = deque()
        self._inflight_lock = threading.Lock()
        # Board releases deferred until the slot D2D completes; drained per get.
        self._pending_board_marks: deque[tuple[torch.cuda.Event, str, int]] = deque()
        self._board_mark_lock = threading.Lock()
        # Zero-copy receive: DLPack struct keepalive + per-request slot holds.
        self._dlpack_keepalive: list = []
        self._zero_copy_holds: dict[str, list[tuple[str, int]]] = {}
        self._stop_event = threading.Event()
        self._release_thread: threading.Thread | None = None
        self._shm_compat_decode_failures: dict[str, int] = {}
        self._metrics = {
            "puts": 0,
            "gets": 0,
            "bytes_transferred": 0,
            "gpu_tensors_transferred": 0,
            "board_releases": 0,
            "ttl_releases": 0,
            "errors": 0,
            "cpu_fallbacks": 0,
            # per-reason breakdown so ops can see WHY fallbacks spike without grepping logs
            "fallback_ring_full": 0,
            "fallback_credits_exhausted": 0,
            "fallback_slot_overflow": 0,
            "fallback_descriptor_too_big": 0,
            "fallback_inline_too_big": 0,
        }

    def _init_cuda(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CudaIPCConnector requires CUDA runtime.")
        self._cudart = load_cudart()
        if torch.accelerator.device_count() > 1:
            self._validate_p2p_access()

    def _init_sender_resources(self) -> None:
        """Sender data-transfer rank: GPU pool, IPC event, credits, release board, per-edge ring."""
        total_pool_bytes = self._pool_size + self._small_slot_count * self._small_slot_size
        with torch.cuda.device(self.local_device):
            # One tensor, one IPC handle: big slots at [0, pool_size), small
            # slots at [pool_size, total). slot_offset stays an absolute pool
            # offset on the wire, so the receiver decode path is unchanged.
            self._pool = torch.zeros(total_pool_bytes, dtype=torch.uint8, device=self.local_device)
            self._copy_stream = torch.cuda.Stream(device=self.local_device)
            self._copy_done_event = torch.cuda.Event()
        self._pool_handle = self._get_ipc_handle(self._pool.data_ptr())
        self._ipc_event = ctypes.c_void_p()
        flags = ctypes.c_uint(_CUDA_EVENT_INTERPROCESS | _CUDA_EVENT_DISABLE_TIMING)
        ret = self._cudart.cudaEventCreateWithFlags(ctypes.byref(self._ipc_event), flags)
        if ret != 0:
            raise RuntimeError(f"cudaEventCreateWithFlags failed: {ret}")
        ipc_evt_handle = _CudaIpcEventHandle()
        ret = self._cudart.cudaIpcGetEventHandle(ctypes.byref(ipc_evt_handle), self._ipc_event)
        if ret != 0:
            raise RuntimeError(f"cudaIpcGetEventHandle failed: {ret}")
        self._ipc_event_handle_bytes = bytes(ipc_evt_handle)
        self._credit_queue: _queue_mod.Queue[int] = _queue_mod.Queue(maxsize=self._pool_credits)
        for i in range(self._pool_credits):
            self._credit_queue.put_nowait(i * self._slot_size)
        self._small_credit_queue: _queue_mod.Queue[int] = _queue_mod.Queue(maxsize=max(1, self._small_slot_count))
        for j in range(self._small_slot_count):
            self._small_credit_queue.put_nowait(self._pool_size + j * self._small_slot_size)
        self._held_credits: dict[str, tuple[float, int]] = {}
        # CPU-fallback /dev/shm segments {name: ts}: receiver unlinks on read, else the
        # release loop TTL-sweeps them (the adapter never calls connector cleanup()).
        self._fallback_segs: dict[str, float] = {}
        self._board_name = f"cudaipc_board_{uuid.uuid4().hex[:16]}"
        n_board = self._pool_credits + self._small_slot_count
        self._board = shm_pkg.SharedMemory(create=True, size=n_board, name=self._board_name)
        self._board.buf[:n_board] = bytes(n_board)

        # Seqlock ring is fenceless — only correct on x86 TSO; warn loudly on weak-memory hosts.
        if platform.machine() not in ("x86_64", "AMD64"):
            logger.warning(
                "CudaIpcControlRing seqlock is fenceless and only correct on x86 TSO; on %s "
                "(ARM/POWER) a torn read is possible — memory fences needed before production use.",
                platform.machine(),
            )
        n_slots = self._ring_entries_cfg or max(64, self._pool_credits * 4)
        body_max = max(self._ring_body_max, self._inline_threshold)
        self._ring = CudaIpcControlRing.create(
            ring_shm_name(self.stage_id, self.stage_id + 1, self._replica_id),
            n_slots,
            body_max,
            header_bytes=RING_HEADER_BYTES,
        )
        self._ring.write_header(self._ring_header_blob())

    def _init_lazy_defaults(self) -> None:
        """Baseline state: no pool/board/ring/recv-streams.

        Edge resources allocate lazily (first put/get), so the adapter and the
        mixin can both construct a connector from the same stage config without
        double-allocating the pool or create/unlinking the same-named ring."""
        self._pool = None
        self._pool_handle = None
        self._credit_queue = None
        self._small_credit_queue = None
        self._held_credits = {}
        self._fallback_segs = {}
        self._board_name = None
        self._board = None
        self._recv_copy_streams = []
        self._recv_stream_idx = 0

    def _ensure_sender_resources(self) -> None:
        """Allocate pool/board/ring + start the release loop on first send."""
        if self._pool is not None:
            return
        with self._lazy_init_lock:
            if self._pool is not None or self._closed:
                return
            self._init_sender_resources()
            self._start_release_thread()
            logger.info(
                "CudaIPCConnector(stage=%s): sender resources allocated (pool=%dMB + %d small slots, ring=%s)",
                self.stage_id,
                self._pool_size // (1024 * 1024),
                self._small_slot_count,
                ring_shm_name(self.stage_id, self.stage_id + 1, self._replica_id),
            )

    def _ensure_recv_streams(self) -> None:
        """Create receiver D2D copy streams/events on first pool-class get()."""
        if self._recv_copy_streams:
            return
        with self._lazy_init_lock:
            if self._recv_copy_streams:
                return
            with torch.cuda.device(self.local_device):
                self._recv_copy_streams = [
                    torch.cuda.Stream(device=self.local_device) for _ in range(_DEFAULT_RECV_STREAMS)
                ]

    def _reclaim_stale_ring_at_init(self) -> None:
        """Sender: unlink a dead previous run's ring BEFORE lazy create.

        Ring names are deterministic; a receiver polling in the init->first-put
        window must not cache a stale segment from a crashed run."""
        if self.role != "sender" or not self._is_transfer_rank:
            return
        name = ring_shm_name(self.stage_id, self.stage_id + 1, self._replica_id)
        if CudaIpcControlRing.reclaim_stale(name):
            logger.info(
                "CudaIPCConnector(stage=%s): reclaimed stale ring %s from a dead previous run", self.stage_id, name
            )

    _RING_REFRESH_EVERY_N_MISSES = 50

    def _maybe_refresh_stale_ring(self, from_stage, to_stage) -> None:
        """Receiver: drop a cached ring whose shm file was replaced/unlinked.

        Cheap inode check every N consecutive misses; next poll reopens live."""
        edge = (from_stage, to_stage)
        if edge not in self._opened_rings:
            return
        n = self._ring_miss_counts.get(edge, 0) + 1
        self._ring_miss_counts[edge] = n
        if n % self._RING_REFRESH_EVERY_N_MISSES:
            return
        name = ring_shm_name(from_stage, to_stage, self._replica_id)
        try:
            inode = os.stat(f"/dev/shm/{name}").st_ino
        except OSError:
            inode = -1  # unlinked with no replacement yet
        if inode == self._ring_inodes.get(edge, inode):
            return
        logger.warning(
            "CudaIPCConnector(stage=%s): ring %s for edge %s was replaced (sender restart or "
            "lazy re-create); reopening.",
            self.stage_id,
            name,
            edge,
        )
        stale = self._opened_rings.pop(edge, None)
        self._ring_edge_handles.pop(edge, None)
        self._ring_inodes.pop(edge, None)
        if stale is not None:
            try:
                stale.close()
            except Exception:
                pass

    def _start_release_thread(self) -> None:
        self._release_thread = threading.Thread(target=self._release_loop, daemon=True, name="cuda-ipc-release-loop")
        self._release_thread.start()

    # --- Device & SHM helpers ---

    @staticmethod
    def _resolve_local_device(local_device_cfg: str | int) -> torch.device:
        if local_device_cfg == "auto":
            return torch.device("cuda", torch.accelerator.current_device_index())
        if isinstance(local_device_cfg, int):
            return torch.device("cuda", local_device_cfg)
        if isinstance(local_device_cfg, str):
            if local_device_cfg.startswith("cuda"):
                return torch.device(local_device_cfg)
            return torch.device("cuda", int(local_device_cfg))
        return torch.device("cuda", torch.accelerator.current_device_index())

    @staticmethod
    def _atomic_shm_write(payload: bytes, name: str) -> dict[str, Any]:
        """Write to SHM atomically: write to temp name, then rename."""
        tmp_name = f"{name}__tmp"
        meta = shm_write_bytes(payload, name=tmp_name)
        os.rename(f"/dev/shm/{tmp_name}", f"/dev/shm/{name}")
        try:
            unregister(f"/{tmp_name}", "shared_memory")
        except KeyError:
            pass
        meta["name"] = name
        return meta

    def _validate_p2p_access(self) -> None:
        n_devices = torch.accelerator.device_count()
        no_p2p_pairs = []
        for i in range(n_devices):
            for j in range(i + 1, n_devices):
                if not torch.cuda.can_device_access_peer(i, j):
                    no_p2p_pairs.append((i, j))
        if no_p2p_pairs:
            logger.warning(
                f"No P2P access between GPU pairs: {no_p2p_pairs}. "
                f"D2D copy will fall back to PCIe staging (slower than NVLink)."
            )

    # --- CUDA IPC handles (ctypes bindings in cuda_ipc_runtime.load_cudart) ---

    def _get_ipc_handle(self, ptr: int) -> bytes:
        """Obtain a 64-byte CUDA IPC memory handle for a device pointer."""
        handle = _CudaIpcMemHandle()
        ret = self._cudart.cudaIpcGetMemHandle(ctypes.byref(handle), ctypes.c_void_p(ptr))
        if ret != 0:
            raise RuntimeError(f"cudaIpcGetMemHandle failed with code {ret}")
        return bytes(handle)

    def _open_ipc_ptr(self, handle_bytes: bytes) -> ctypes.c_void_p:
        """Open a CUDA IPC handle and return the mapped device pointer."""
        handle = _CudaIpcMemHandle.from_buffer_copy(handle_bytes)
        dev_ptr = ctypes.c_void_p()
        ret = self._cudart.cudaIpcOpenMemHandle(
            ctypes.byref(dev_ptr), handle, ctypes.c_uint(_CUDA_IPC_MEM_LAZY_ENABLE_PEER_ACCESS)
        )
        if ret != 0:
            raise RuntimeError(f"cudaIpcOpenMemHandle failed with code {ret}")
        return dev_ptr

    def _close_ipc_ptr(self, dev_ptr: ctypes.c_void_p) -> None:
        ret = self._cudart.cudaIpcCloseMemHandle(dev_ptr)
        if ret != 0:
            logger.warning("cudaIpcCloseMemHandle failed with code %s", ret)

    def _open_pool(self, pool_handle: bytes) -> ctypes.c_void_p:
        """Open a pool IPC handle, cached. Lock-guarded: cudaIpcOpenMemHandle errors on an
        already-open handle, so the check-and-open must be atomic across recv threads."""
        with self._open_lock:
            if pool_handle not in self._opened_pools:
                self._opened_pools[pool_handle] = self._open_ipc_ptr(pool_handle)
            return self._opened_pools[pool_handle]

    def _open_ipc_event(self, handle_bytes: bytes) -> ctypes.c_void_p:
        """Open a CUDA IPC event handle (cached — only opened once per sender)."""
        with self._open_lock:
            if handle_bytes not in self._opened_events:
                handle = _CudaIpcEventHandle.from_buffer_copy(handle_bytes)
                event = ctypes.c_void_p()
                ret = self._cudart.cudaIpcOpenEventHandle(ctypes.byref(event), handle)
                if ret != 0:
                    raise RuntimeError(f"cudaIpcOpenEventHandle failed: {ret}")
                self._opened_events[handle_bytes] = event
            return self._opened_events[handle_bytes]

    # --- Control plane: ring wiring ---

    def _ring_header_blob(self) -> bytes:
        return RingHeader(self._pool_handle, self._ipc_event_handle_bytes, self._board_name).pack()

    def _open_ring_receiver(self, from_stage, to_stage):
        edge = (from_stage, to_stage)
        ring = self._opened_rings.get(edge)
        if ring is None:
            name = ring_shm_name(from_stage, to_stage, self._replica_id)
            try:
                ring = CudaIpcControlRing.open(name)
            except FileNotFoundError:
                return None  # sender not up yet; poll loop tolerates None
            self._opened_rings[edge] = ring
            try:
                self._ring_inodes[edge] = os.stat(f"/dev/shm/{name}").st_ino
            except OSError:
                pass
        # Cache the parsed header only once a valid (magic+version) one is present — never
        # cache zero handles from a ring whose sender hasn't written the header yet.
        if edge not in self._ring_edge_handles:
            hdr = RingHeader.try_unpack(ring.read_header(RING_HEADER_BYTES))
            if hdr is not None:
                self._ring_edge_handles[edge] = (hdr.pool_handle, hdr.event_handle, hdr.board_name)
        return ring

    def _estimate_nbytes(self, obj: Any) -> int:
        """Sum GPU-tensor bytes (the part that would go to the pool) WITHOUT a
        serialize/D2H — used to route inline vs pool."""
        if isinstance(obj, torch.Tensor):
            return obj.nbytes if obj.is_cuda else 0
        if isinstance(obj, dict):
            return sum(self._estimate_nbytes(v) for v in obj.values())
        if isinstance(obj, list | tuple):
            return sum(self._estimate_nbytes(v) for v in obj)
        if hasattr(obj, "__struct_fields__"):
            return sum(
                self._estimate_nbytes(getattr(obj, f)) for f in obj.__struct_fields__ if getattr(obj, f) is not None
            )
        return 0

    # --- Pool slot codec ---

    def _walk_encode_pool(self, obj: Any, slot: _PoolSlot) -> Any:
        """Recursively replace CUDA tensors with pool offset metadata."""
        if isinstance(obj, torch.Tensor):
            if obj.is_cuda:
                t = obj.detach().contiguous()
                tensor_offset = slot.pack(t)
                return {
                    _GPU_TENSOR_MARKER: True,
                    "shape": list(t.shape),
                    "dtype": str(t.dtype).removeprefix("torch."),
                    "nbytes": int(t.nbytes),
                    "pool_offset": tensor_offset,
                }
            return obj
        if isinstance(obj, dict):
            return {k: self._walk_encode_pool(v, slot) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._walk_encode_pool(v, slot) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self._walk_encode_pool(v, slot) for v in obj)
        if hasattr(obj, "__struct_fields__"):
            # Drop None struct fields — matches data_entry_keys.to_dict (the SHM/inline wire
            # contract), so pool and SHM paths produce the same dict keys downstream.
            return {
                f: self._walk_encode_pool(getattr(obj, f), slot)
                for f in obj.__struct_fields__
                if getattr(obj, f) is not None
            }
        return obj

    def _decode_pool_tensor(
        self,
        meta: dict[str, Any],
        pool_ptr: ctypes.c_void_p,
        slot_offset: int,
        stream: torch.cuda.Stream | None = None,
    ) -> torch.Tensor:
        """Decode a tensor from a cached pool mapping on *stream*.

        If *stream* is None, falls back to the current (compute) stream for
        backward compatibility.
        """
        shape = tuple(meta["shape"])
        dtype = getattr(torch, meta["dtype"])
        nbytes = int(meta["nbytes"])
        tensor_offset = int(meta["pool_offset"])
        if self._zero_copy_recv:
            # In-place DLPack view; caller fences on the packer's IPC event and
            # the slot stays leased until request cleanup.
            view = make_dlpack_view(
                pool_ptr.value + slot_offset + tensor_offset,
                list(shape),
                meta["dtype"],
                "cuda",
                self.local_device.index or 0,
                self._dlpack_keepalive,
            )
            self._metrics["gpu_tensors_transferred"] += 1
            return view
        target_stream = stream if stream is not None else torch.cuda.current_stream(self.local_device)
        # Allocate dst on target_stream (the D2D copy stream) to avoid an alloc-vs-copy
        # cross-stream race at write time.
        with torch.cuda.stream(target_stream):
            dst = torch.empty(shape, dtype=dtype, device=self.local_device)
        memcpy_async_d2d(
            self._cudart,
            dst.data_ptr(),
            pool_ptr.value + slot_offset + tensor_offset,
            nbytes,
            target_stream.cuda_stream,
        )
        # dst is consumed downstream on the model/default stream and may be cached across
        # steps; record it there so the allocator won't reuse its memory while that use is live.
        dst.record_stream(torch.cuda.current_stream(self.local_device))
        self._metrics["gpu_tensors_transferred"] += 1
        return dst

    def _walk_decode_pool(
        self,
        obj: Any,
        pool_ptr: ctypes.c_void_p,
        slot_offset: int,
        stream: torch.cuda.Stream | None = None,
    ) -> Any:
        """Recursively restore tensors from pool offset metadata."""
        if isinstance(obj, dict) and obj.get(_GPU_TENSOR_MARKER):
            return self._decode_pool_tensor(obj, pool_ptr, slot_offset, stream=stream)
        if isinstance(obj, dict):
            return {k: self._walk_decode_pool(v, pool_ptr, slot_offset, stream=stream) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._walk_decode_pool(v, pool_ptr, slot_offset, stream=stream) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self._walk_decode_pool(v, pool_ptr, slot_offset, stream=stream) for v in obj)
        return obj

    # --- Credit pool (sender) ---

    def _slot_index_for_offset(self, slot_offset: int) -> int:
        """Board index for an absolute pool offset (big region first, then small)."""
        if slot_offset < self._pool_size:
            return slot_offset // self._slot_size
        return self._pool_credits + (slot_offset - self._pool_size) // self._small_slot_size

    def _slot_capacity_for_offset(self, slot_offset: int) -> int:
        return self._slot_size if slot_offset < self._pool_size else self._small_slot_size

    def _queue_for_offset(self, slot_offset: int) -> "_queue_mod.Queue[int]":
        return self._credit_queue if slot_offset < self._pool_size else self._small_credit_queue

    def _acquire_credit(self, prefer_small: bool = False) -> int | None:
        """Get a free slot offset, reclaiming board credits inline.

        ``prefer_small`` tries the small-slot pool first (streaming chunks)
        and falls back to a big slot when small credits are exhausted, so a
        burst of small chunks degrades to big slots instead of /dev/shm.
        Bounded wait (``credit_wait_sec``) before giving up; returns None to
        trigger the CPU fallback.
        """
        if prefer_small and self._small_slot_count > 0:
            queues = (self._small_credit_queue, self._credit_queue)
        else:
            queues = (self._credit_queue,)
        for q in queues:
            try:
                return q.get_nowait()
            except _queue_mod.Empty:
                pass
        deadline = _time_mod.monotonic() + self._credit_wait_sec
        while _time_mod.monotonic() < deadline:
            self._reclaim_board_credits()
            for q in queues:
                try:
                    return q.get_nowait()
                except _queue_mod.Empty:
                    pass
            _time_mod.sleep(self._credit_poll_sec)
        return None

    def _reclaim_board_credits(self) -> None:
        """Sender: reclaim credits whose board byte was set by the receiver."""
        if self._board is None:
            return
        buf = self._board.buf
        with self._held_lock:
            released = [
                (key, slot_offset)
                for key, (_ts, slot_offset) in self._held_credits.items()
                if buf[self._slot_index_for_offset(slot_offset)] == 1
            ]
            for key, slot_offset in released:
                self._held_credits.pop(key, None)
                buf[self._slot_index_for_offset(slot_offset)] = 0
                self._queue_for_offset(slot_offset).put_nowait(slot_offset)
                self._metrics["board_releases"] += 1

    def _release_expired_credits(self) -> None:
        """TTL sweep: reclaim slots whose receiver never marked the board
        (e.g. the request was aborted or the receiver died)."""
        now = _time_mod.monotonic()
        with self._held_lock:
            expired = [
                (key, slot_offset)
                for key, (ts, slot_offset) in self._held_credits.items()
                if now - ts > self.tensor_lifetime_sec
            ]
            for key, slot_offset in expired:
                self._held_credits.pop(key, None)
                self._board.buf[self._slot_index_for_offset(slot_offset)] = 0
                self._queue_for_offset(slot_offset).put_nowait(slot_offset)
                self._metrics["ttl_releases"] += 1
        # also TTL-sweep orphaned CPU-fallback /dev/shm segments (receiver aborted before
        # reading; normally the receiver unlinks on read so these are already gone).
        segs = getattr(self, "_fallback_segs", None)
        if segs:
            stale = [name for name, ts in segs.items() if now - ts > self.tensor_lifetime_sec]
            for name in stale:
                segs.pop(name, None)
                try:
                    seg = shm_pkg.SharedMemory(name=name)
                    seg.close()
                    seg.unlink()
                    self._metrics["fallback_seg_reclaims"] = self._metrics.get("fallback_seg_reclaims", 0) + 1
                except FileNotFoundError:
                    pass  # receiver already consumed + unlinked it (the common case)
                except Exception as e:
                    logger.debug("fallback seg unlink %s: %s", name, e)

    def _drain_inflight_packs(self) -> None:
        """Sender: drop payload refs whose pool D2D has completed."""
        with self._inflight_lock:
            dq = self._inflight_packs
            while dq and dq[0][0].query():
                dq.popleft()

    def _drain_pending_board_marks(self) -> None:
        """Receiver: mark the release board for completed slot D2Ds.

        Deferred marking keeps get() free of host syncs. A mark that never
        drains (traffic stopped) is covered by the sender's TTL sweep.
        """
        with self._board_mark_lock:
            ready = []
            dq = self._pending_board_marks
            while dq and dq[0][0].query():
                ready.append(dq.popleft())
        for _ev, board_name, slot_index in ready:
            self._mark_board_release(board_name, slot_index)

    def _release_loop(self) -> None:
        tick = 0
        while not self._stop_event.is_set():
            try:
                self._reclaim_board_credits()
                self._drain_inflight_packs()
                tick += 1
                if tick % self._release_ttl_every == 0:
                    self._release_expired_credits()
            except Exception as e:
                logger.warning("Release loop error: %s", e, exc_info=True)
            self._stop_event.wait(timeout=self._release_interval_sec)

    # --- put() ---

    def put(
        self,
        from_stage: str,
        to_stage: str,
        put_key: str,
        data: Any,
        compute_event: "torch.cuda.Event | None" = None,
    ) -> tuple[bool, int, dict[str, Any] | None]:
        if self._closed:
            return False, 0, None
        if not self._is_transfer_rank:
            # Inert non-transfer-rank sender: no ring/pool. Guard against a stray call.
            return False, 0, None

        composite_key = make_composite_key(put_key, from_stage, to_stage)
        return self._put_control_plane(from_stage, to_stage, put_key, composite_key, data, compute_event)

    def _put_control_plane(self, from_stage, to_stage, put_key, composite_key, data, compute_event=None):
        """Primary send path: route small payloads inline vs large via GPU pool (both publish to ring)."""
        try:
            self._ensure_sender_resources()
            kh = key_hash16(composite_key)
            if self._estimate_nbytes(data) < self._inline_threshold:
                return self._put_inline(from_stage, to_stage, put_key, composite_key, data, kh, compute_event)
            return self._put_pool(from_stage, to_stage, put_key, composite_key, data, kh, compute_event)
        except Exception as e:
            self._metrics["errors"] += 1
            logger.error("CudaIPCConnector control-plane put failed for %s: %s", put_key, e, exc_info=True)
            return False, 0, None

    def _put_inline(self, from_stage, to_stage, put_key, composite_key, data, kh, compute_event=None):
        # Serialize (a cheap D2H for a tiny GPU tensor) straight into the ring body.
        # Order the D2H after the producer's writes — same fence contract as the
        # pool path; without it a small keep_on_gpu tensor could serialize stale
        # bytes when the producer wrote on a non-ambient stream.
        if compute_event is not None:
            torch.cuda.current_stream(self.local_device).wait_event(compute_event)
        payload = self.serialize_obj(data)
        if len(payload) > self._ring.body_max:
            return self._put_cpu_fallback(
                from_stage, to_stage, put_key, composite_key, data, reason=f"inline_too_big {len(payload)}"
            )
        try:
            self._ring.publish(
                kh, RING_PCLASS_INLINE, payload, ts=int(_time_mod.time()), ttl_sec=int(self.tensor_lifetime_sec)
            )
        except RingFullError:
            return self._put_cpu_fallback(from_stage, to_stage, put_key, composite_key, data, reason="ring_full")
        self._metrics["puts"] += 1
        self._metrics["bytes_transferred"] += len(payload)
        return True, len(payload), {"ring": True, "size": len(payload)}

    def _put_pool(self, from_stage, to_stage, put_key, composite_key, data, kh, compute_event=None, force_big=False):
        # Acquire a credit, D2D-pack into the slot, then publish a small descriptor
        # (slot_offset/slot_index + tensor layout) to the ring.
        # TODO: support acquiring multiple contiguous credits/slots when a single payload
        # exceeds _slot_size (currently: _SlotOverflowError → CPU fallback).
        # Route by payload size: streaming-sized payloads take a small slot so a
        # codec chunk burst does not burn 64MB big credits. The estimate ignores
        # per-tensor alignment padding, hence the slack; a small-slot overflow
        # retries once with a big slot (force_big) before the CPU fallback.
        prefer_small = (
            not force_big
            and self._small_slot_count > 0
            and self._estimate_nbytes(data) <= self._small_slot_size - _SMALL_SLOT_SLACK
        )
        slot_offset = self._acquire_credit(prefer_small=prefer_small)
        if slot_offset is None:
            return self._put_cpu_fallback(
                from_stage, to_stage, put_key, composite_key, data, reason="credits_exhausted"
            )
        credit_returned = False
        try:
            self._board.buf[self._slot_index_for_offset(slot_offset)] = 0
            slot = _PoolSlot(self._pool, slot_offset, self._slot_capacity_for_offset(slot_offset))
            # Fence the pack after the producer's writes (event when supplied,
            # else the ambient/legacy default stream — PTDS-off assumption).
            if compute_event is not None:
                self._copy_stream.wait_event(compute_event)
            else:
                self._copy_stream.wait_stream(torch.cuda.current_stream(self.local_device))
            try:
                with torch.cuda.stream(self._copy_stream):
                    encoded_obj = self._walk_encode_pool(data, slot)
            except _SlotOverflowError as e:
                self._copy_done_event.record(self._copy_stream)
                self._copy_done_event.synchronize()
                credit_returned = True
                self._queue_for_offset(slot_offset).put_nowait(slot_offset)
                if slot_offset >= self._pool_size:
                    # Small slot underestimated (alignment/metadata) — retry with a big slot.
                    return self._put_pool(
                        from_stage, to_stage, put_key, composite_key, data, kh, compute_event, force_big=True
                    )
                return self._put_cpu_fallback(
                    from_stage,
                    to_stage,
                    put_key,
                    composite_key,
                    data,
                    reason=f"slot_overflow nbytes={e.nbytes} slot={e.slot_size}",
                )
            ret = self._cudart.cudaEventRecord(self._ipc_event, ctypes.c_void_p(self._copy_stream.cuda_stream))
            if ret != 0:
                logger.warning("cudaEventRecord (IPC) failed: %d", ret)
            # No host sync: receiver orders on the IPC event; payload refs are
            # held in _inflight_packs until the pack completes.
            pack_done = torch.cuda.Event()
            pack_done.record(self._copy_stream)
            with self._inflight_lock:
                self._inflight_packs.append((pack_done, data))
            descriptor = OmniSerializer.serialize(
                {
                    _POOL_MARKER: True,
                    "slot_offset": slot_offset,
                    "slot_index": self._slot_index_for_offset(slot_offset),
                    "payload": encoded_obj,
                }
            )
        except Exception:
            if not credit_returned:
                self._queue_for_offset(slot_offset).put_nowait(slot_offset)
            raise
        # Descriptor grows with sequence length; if it overflows the ring body,
        # degrade to the CPU fallback (read via _try_get_shm_compat) — never crash.
        if len(descriptor) > self._ring.body_max:
            self._queue_for_offset(slot_offset).put_nowait(slot_offset)
            return self._put_cpu_fallback(
                from_stage,
                to_stage,
                put_key,
                composite_key,
                data,
                reason=f"descriptor_too_big {len(descriptor)}>{self._ring.body_max}",
            )
        with self._held_lock:
            # monotonic() to match the TTL sweep's clock (NTP-step safe; see _release_expired_credits).
            self._held_credits[composite_key] = (_time_mod.monotonic(), slot_offset)
        try:
            self._ring.publish(
                kh, RING_PCLASS_POOL, descriptor, ts=int(_time_mod.time()), ttl_sec=int(self.tensor_lifetime_sec)
            )
        except RingFullError:
            with self._held_lock:
                self._held_credits.pop(composite_key, None)
            self._queue_for_offset(slot_offset).put_nowait(slot_offset)
            return self._put_cpu_fallback(from_stage, to_stage, put_key, composite_key, data, reason="ring_full")
        self._metrics["puts"] += 1
        self._metrics["bytes_transferred"] += len(descriptor)
        return True, len(descriptor), {"ring": True, "size": len(descriptor)}

    def _put_cpu_fallback(
        self,
        from_stage: str,
        to_stage: str,
        put_key: str,
        composite_key: str,
        data: Any,
        reason: str = "",
    ) -> tuple[bool, int, dict[str, Any] | None]:
        logger.warning(
            "CudaIPCConnector CPU fallback for %s (from_stage=%s to_stage=%s): %s",
            put_key,
            from_stage,
            to_stage,
            reason or "pool credits exhausted or slot overflow",
        )
        self._metrics["cpu_fallbacks"] += 1
        # Categorize by the reason's leading token (ring_full / credits_exhausted / ...).
        cat = f"fallback_{reason.split(maxsplit=1)[0]}" if reason else "fallback_other"
        self._metrics[cat] = self._metrics.get(cat, 0) + 1
        payload = self.serialize_obj(data)
        size = len(payload)

        meta = self._atomic_shm_write(payload, name=put_key)
        # Track for TTL cleanup in case the receiver aborts and never reads/unlinks it.
        if getattr(self, "_fallback_segs", None) is not None:
            self._fallback_segs[put_key] = _time_mod.monotonic()

        self._metrics["puts"] += 1
        self._metrics["bytes_transferred"] += size
        return True, size, {"shm": meta, "size": size, "cpu_fallback": True}

    # --- get() ---

    def get(
        self,
        from_stage: str,
        to_stage: str,
        get_key: str,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Any, int] | None:
        if self._closed:
            return None

        composite_key = make_composite_key(get_key, from_stage, to_stage)
        return self._get_control_plane(from_stage, to_stage, get_key, composite_key)

    def _get_control_plane(self, from_stage, to_stage, get_key, composite_key):
        """Primary recv path: poll ring (inline or pool); on miss try CPU-fallback /dev/shm."""
        try:
            self._drain_pending_board_marks()
            ring = self._open_ring_receiver(from_stage, to_stage)
            if ring is None:
                return None
            if (from_stage, to_stage) not in self._ring_edge_handles:
                # Header not ready — don't poll (poll consumes; a pool entry would be lost). Retry.
                return None
            r = ring.poll(key_hash16(composite_key))
            if r is None:
                # Ring miss: chunk not published yet (poll retry), the sender replaced the
                # ring (staleness refresh below), or the sender took the CPU fallback
                # (/dev/shm by put_key) — checked last so the consumer never hangs.
                self._maybe_refresh_stale_ring(from_stage, to_stage)
                return self._try_get_shm_compat(get_key)
            self._ring_miss_counts.pop((from_stage, to_stage), None)
            pclass, body = r
            if pclass == RING_PCLASS_INLINE:
                # Return CPU tensors; the downstream model does the H2D (parity with SHM).
                # Doing it here was a redundant device-wide sync on the talker forward.
                obj = self.deserialize_obj(body)
                self._metrics["gets"] += 1
                self._metrics["bytes_transferred"] += len(body)
                return obj, len(body)

            # POOL: descriptor from the entry body; handles either self-contained
            # (pre-packed, L3) or from the ring header (classic path — the poll
            # above is gated on _ring_edge_handles so they are present).
            raw = OmniSerializer.deserialize(body)
            obj = self._consume_pool_descriptor(
                raw, fallback_handles=self._ring_edge_handles[(from_stage, to_stage)], hold_key=get_key
            )
            self._metrics["gets"] += 1
            self._metrics["bytes_transferred"] += len(body)
            return obj, len(body)
        except Exception as e:
            self._metrics["errors"] += 1
            logger.error("CudaIPCConnector control-plane get failed for %s: %s", get_key, e, exc_info=True)
            return None

    def _consume_pool_descriptor(self, raw: dict, fallback_handles=None, hold_key: str | None = None) -> Any:
        """Decode a pool descriptor into tensors (ring or shm-compat delivery).

        Self-contained (pre-packed) descriptors carry their own pool/event
        handles + board identity; classic descriptors resolve them from the
        ring header via ``fallback_handles``. The D2D runs on a round-robined
        copy stream fenced on the packer's IPC event; the consuming (ambient)
        stream is fenced on the copy, and the board mark is deferred until the
        copy completes (see _drain_pending_board_marks).
        """
        pool_handle, event_handle, board_name = fallback_handles
        pool_ptr = self._open_pool(bytes(pool_handle))
        if self._zero_copy_recv:
            # No copy: fence the consuming stream on the packer's IPC event;
            # slot released at cleanup(request_id), TTL backstop.
            if event_handle and event_handle != bytes(len(event_handle)):
                ipc_event = self._open_ipc_event(event_handle)
                stream_wait_event(self._cudart, torch.cuda.current_stream(self.local_device).cuda_stream, ipc_event)
            obj = self._walk_decode_pool(raw["payload"], pool_ptr, raw["slot_offset"], stream=None)
            with self._board_mark_lock:
                self._zero_copy_holds.setdefault(hold_key or "", []).append((board_name, int(raw["slot_index"])))
            return obj
        self._ensure_recv_streams()
        idx = self._recv_stream_idx % len(self._recv_copy_streams)
        self._recv_stream_idx += 1
        copy_stream = self._recv_copy_streams[idx]
        if event_handle and event_handle != bytes(len(event_handle)):
            ipc_event = self._open_ipc_event(event_handle)
            stream_wait_event(self._cudart, copy_stream.cuda_stream, ipc_event)
        copy_stream.wait_stream(torch.cuda.current_stream(self.local_device))
        obj = self._walk_decode_pool(raw["payload"], pool_ptr, raw["slot_offset"], stream=copy_stream)
        # Fence the consuming stream on the D2D (no host stall); fresh event per
        # get keeps deferred board marks independent.
        copy_done_event = torch.cuda.Event()
        copy_done_event.record(copy_stream)
        torch.cuda.current_stream(self.local_device).wait_event(copy_done_event)
        with self._board_mark_lock:
            self._pending_board_marks.append((copy_done_event, board_name, int(raw["slot_index"])))
        return obj

    def _mark_board_release(self, board_name: str, slot_index: int) -> None:
        """Receiver: flip the slot byte on the sender's release board."""
        board = self._opened_boards.get(board_name)
        if board is None:
            try:
                board = shm_pkg.SharedMemory(name=board_name)
            except FileNotFoundError:
                logger.warning("Release board %s not found; sender will rely on TTL.", board_name)
                return
            untrack_shm(board_name)  # non-owner: never unlink the sender's board at exit
            self._opened_boards[board_name] = board
        if not 0 <= slot_index < board.size:
            logger.warning("Release board %s: slot_index %d out of range.", board_name, slot_index)
            return
        board.buf[slot_index] = 1

    def _try_get_shm_compat(self, get_key: str) -> tuple[Any, int] | None:
        try:
            seg = shm_pkg.SharedMemory(name=get_key)
        except FileNotFoundError:
            return None
        # Hold this one handle through read AND unlink — never close-then-reopen-by-name,
        # which races a same-key segment the sender may have just rewritten.
        try:
            data_bytes = bytes(seg.buf[: seg.size])
            try:
                obj = self.deserialize_obj(data_bytes)
            except Exception as de:
                n = self._shm_compat_decode_failures.get(get_key, 0) + 1
                self._shm_compat_decode_failures[get_key] = n
                logger.warning(
                    "CudaIPCConnector shm_compat decode failed for %s: %s (attempt=%d, bytes=%d)",
                    get_key,
                    de,
                    n,
                    len(data_bytes),
                )
                if n >= 3:
                    seg.unlink()
                return None

            self._shm_compat_decode_failures.pop(get_key, None)
            seg.unlink()
            obj = self._move_to_device(obj)
            size = len(data_bytes)
            self._metrics["gets"] += 1
            self._metrics["bytes_transferred"] += size
            return obj, size
        except Exception as e:
            logger.warning("CudaIPCConnector shm_compat get failed for %s: %s", get_key, e)
            return None
        finally:
            try:
                seg.close()
            except Exception:
                pass

    def _move_to_device(self, obj: Any, non_blocking: bool = True) -> Any:
        """Move CPU tensors to local GPU so CUDA graph replay is safe."""
        if isinstance(obj, torch.Tensor) and obj.device.type == "cpu":
            return obj.to(self.local_device, non_blocking=non_blocking)
        if isinstance(obj, dict):
            return {k: self._move_to_device(v, non_blocking) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._move_to_device(v, non_blocking) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self._move_to_device(v, non_blocking) for v in obj)
        return obj

    # --- Lifecycle ---

    def cleanup(self, request_id: str) -> None:
        """Release zero-copy slot holds for a finished request.

        Non-zero-copy transfers need no per-request cleanup (board marks are
        posted at copy completion; TTL sweeps the rest). Zero-copy views keep
        their slot leased for the request lifetime — this releases every hold
        whose key belongs to ``request_id`` (get keys embed the request id).
        """
        if not self._zero_copy_holds:
            return
        with self._board_mark_lock:
            keys = [k for k in self._zero_copy_holds if request_id in k]
            entries = [e for k in keys for e in self._zero_copy_holds.pop(k)]
        for board_name, slot_index in entries:
            self._mark_board_release(board_name, slot_index)

    def health(self) -> dict[str, Any]:
        return {
            "status": "healthy" if not self._closed else "closed",
            "role": self.role,
            "local_device": str(self.local_device),
            "replica_id": self._replica_id,
            "pool_size_mb": self._pool_size // (1024 * 1024),
            "pool_credits": self._pool_credits,
            "small_slot_kb": self._small_slot_size // 1024,
            "small_slot_count": self._small_slot_count,
            "held_credits": len(self._held_credits),
            **self._metrics,
        }

    @staticmethod
    def _try_or_warn(fn, label: str) -> None:
        """Run a cleanup step, downgrading any failure to a warning (shutdown best-effort)."""
        try:
            fn()
        except Exception as e:
            logger.warning("%s failed: %s", label, e)

    def close(self) -> None:
        if getattr(self, "_closed", True):
            return
        self._closed = True
        logger.info("Closing CudaIPCConnector...")

        self._stop_event.set()
        if self._release_thread is not None and self._release_thread.is_alive():
            self._release_thread.join(timeout=1.0)
        # Flush deferred completions best-effort: mark finished slot D2Ds on the
        # board and drop in-flight pack refs (pending GPU work is torn down with
        # the process; the sender's TTL sweep covers anything unmarked).
        self._try_or_warn(self._drain_pending_board_marks, "drain pending board marks")
        with self._inflight_lock:
            self._inflight_packs.clear()
        with self._board_mark_lock:
            self._pending_board_marks.clear()
            self._zero_copy_holds.clear()
        with self._held_lock:
            self._held_credits.clear()

        for pool_ptr in self._opened_pools.values():
            self._try_or_warn(lambda p=pool_ptr: self._close_ipc_ptr(p), "close pool mapping")
        self._opened_pools.clear()

        # Destroy CUDA IPC events: sender's own + receiver-opened (cudaEventDestroy).
        if self._cudart is not None:
            own_event = getattr(self, "_ipc_event", None)
            if own_event is not None:
                self._try_or_warn(lambda: self._cudart.cudaEventDestroy(own_event), "destroy sender IPC event")
                self._ipc_event = None
            for evt in self._opened_events.values():
                self._try_or_warn(lambda e=evt: self._cudart.cudaEventDestroy(e), "destroy opened IPC event")
            self._opened_events.clear()

        if self._board is not None:
            self._try_or_warn(self._board.close, "release board close")
            self._try_or_warn(self._board.unlink, "release board unlink")
            self._board = None

        # Unlink any CPU-fallback shm still tracked (FileNotFoundError = receiver already took it).
        for name in list(getattr(self, "_fallback_segs", {}) or {}):

            def _unlink(n=name):
                try:
                    seg = shm_pkg.SharedMemory(name=n)
                except FileNotFoundError:
                    return
                seg.close()
                seg.unlink()

            self._try_or_warn(_unlink, f"unlink fallback seg {name}")
        if getattr(self, "_fallback_segs", None) is not None:
            self._fallback_segs.clear()

        for board in self._opened_boards.values():
            self._try_or_warn(board.close, "close opened board")
        self._opened_boards.clear()

        # Ring control plane: sender unlinks its ring; receiver closes opens.
        if self._ring is not None:
            self._ring.close()  # owner -> unlinks
            self._ring = None
        for ring in self._opened_rings.values():
            self._try_or_warn(ring.close, "close opened ring")
        self._opened_rings.clear()

        self._pool = None
        if torch.cuda.is_available():
            self._try_or_warn(torch.cuda.ipc_collect, "torch.cuda.ipc_collect")
        logger.info("CudaIPCConnector closed.")
