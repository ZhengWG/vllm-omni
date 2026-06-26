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
import hashlib
import os
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
    RING_PCLASS_SHM,
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
    _CudaIpcEventHandle,
    _CudaIpcMemHandle,
    event_query,
    load_cudart,
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
_DEFAULT_RECV_STREAMS = 8  # receiver D2D copy streams (round-robined per get)
_DEFAULT_PUT_POOL_COPY_STREAMS = 1
_DEFAULT_PUT_POOL_ASYNC_INFLIGHT_FACTOR = 4
_DEFAULT_PUT_POOL_ASYNC_INFLIGHT_MIN = 8

# Timing constants — overridable via extra config keys of the same name
# (without leading underscore), e.g. ``"credit_wait_sec": 0.01``.
_CREDIT_WAIT_SEC = 0.05  # put() inline reclaim window before CPU fallback
_CREDIT_POLL_SEC = 0.0005  # poll interval within the reclaim window
_RELEASE_FAST_INTERVAL_SEC = 0.001  # board-reclaim thread fast tick
_RELEASE_TTL_EVERY_N_TICKS = 20  # TTL sweep runs every N fast ticks


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; fallback to %.3f", name, raw, default)
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; fallback to %d", name, raw, default)
        return default


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

    # --- Init ---

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self._parse_config(config)
        self._init_runtime_state()
        self._init_cuda()
        if self._is_sender_owner:
            self._init_sender_resources()
        else:
            self._init_inert_state()
        if self._is_sender_owner:
            self._start_release_thread()
        logger.info(
            "CudaIPCConnector initialized: role=%s, local_device=%s, "
            "replica_id=%s, pool=%dMB (%d credits x %dMB slots)",
            self.role,
            self.local_device,
            self._replica_id,
            self._pool_size // (1024 * 1024),
            self._pool_credits,
            self._slot_size // 1024 // 1024,
        )
        logger.debug("CudaIPCConnector config_keys=%s", sorted(config.keys()))

    @property
    def _is_sender_owner(self) -> bool:
        """The sender's data-transfer rank — the only one that owns a pool/board/ring."""
        return self.role == "sender" and self._is_transfer_rank

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
        self._inline_threshold = int(config.get("inline_threshold_bytes", 16384))
        # Inline route serializes payload as CPU bytes (SHM-compatible semantics).
        # Keep a guard knob for workloads that prefer forcing CUDA payloads to pool.
        self._inline_cuda_tensors = bool(
            config.get(
                "inline_cuda_tensors",
                _env_bool("VLLM_OMNI_CUDA_IPC_INLINE_CUDA_TENSORS", default=False),
            )
        )
        self._inline_use_shm = bool(
            config.get(
                "inline_use_shm",
                _env_bool("VLLM_OMNI_CUDA_IPC_INLINE_USE_SHM", default=True),
            )
        )
        # Size-based CUDA-inline gate (defaults to inline threshold): payloads
        # <= this size can take inline route and be normalized to CPU before
        # serialization, matching SHM connector behavior.
        raw_inline_cuda_max_cfg = config.get("inline_cuda_max_bytes")
        if raw_inline_cuda_max_cfg is None:
            env_raw = os.environ.get("VLLM_OMNI_CUDA_IPC_INLINE_CUDA_MAX_BYTES")
            raw_inline_cuda_max_cfg = self._inline_threshold if env_raw is None else env_raw
        try:
            raw_inline_cuda_max_bytes = int(raw_inline_cuda_max_cfg)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid inline_cuda_max_bytes=%r; fallback to inline_threshold=%d",
                raw_inline_cuda_max_cfg,
                self._inline_threshold,
            )
            raw_inline_cuda_max_bytes = self._inline_threshold
        self._inline_cuda_max_bytes = max(0, min(raw_inline_cuda_max_bytes, self._inline_threshold))
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
        # On ring miss, legacy behavior probes /dev/shm compatibility path on every poll.
        # In dedicated IPC deployments where sender fallback is guaranteed absent, disable
        # this probe to avoid repeated SharedMemory open/miss syscalls and exceptions.
        self._enable_shm_compat_on_ring_miss = bool(
            config.get(
                "enable_shm_compat_on_ring_miss",
                _env_bool("VLLM_OMNI_CUDA_IPC_SHM_COMPAT_ON_RING_MISS", default=True),
            )
        )
        # Optional profiling logs for critical put/get path bottlenecks.
        # Enable with config ``profile_log=true`` or env
        # ``VLLM_OMNI_CUDA_IPC_PROFILE_LOG=1``.
        self._profile_log_enabled = bool(
            config.get("profile_log", _env_bool("VLLM_OMNI_CUDA_IPC_PROFILE_LOG", default=False))
        )
        self._profile_log_threshold_ms = float(
            config.get(
                "profile_log_threshold_ms",
                _env_float("VLLM_OMNI_CUDA_IPC_PROFILE_LOG_THRESHOLD_MS", default=2.0),
            )
        )
        self._profile_log_every_n = max(
            1,
            int(
                config.get(
                    "profile_log_every_n",
                    _env_int("VLLM_OMNI_CUDA_IPC_PROFILE_LOG_EVERY_N", default=32),
                )
            ),
        )
        # Sender-side pool put synchronization mode:
        # - True  (default): block on copy_stream.synchronize() for strict behavior.
        # - False: publish descriptor after enqueue + IPC event record, relying on
        #          stream/event ordering and record_stream lifetime tracking.
        self._put_pool_blocking_sync = bool(
            config.get(
                "put_pool_blocking_sync",
                _env_bool("VLLM_OMNI_CUDA_IPC_PUT_POOL_BLOCKING_SYNC", default=True),
            )
        )
        # Sender-side pool put parallelism / queue-depth controls in non-blocking mode:
        # - put_pool_copy_streams: number of sender pack/copy streams to round-robin.
        # - put_pool_async_inflight_limit: max published-but-not-locally-complete puts.
        #   <=0 uses an auto limit (bounded by pool_credits).
        self._put_pool_copy_streams = max(
            1,
            int(
                config.get(
                    "put_pool_copy_streams",
                    _env_int("VLLM_OMNI_CUDA_IPC_PUT_POOL_COPY_STREAMS", default=_DEFAULT_PUT_POOL_COPY_STREAMS),
                )
            ),
        )
        raw_async_inflight_limit = int(
            config.get(
                "put_pool_async_inflight_limit",
                _env_int("VLLM_OMNI_CUDA_IPC_PUT_POOL_ASYNC_INFLIGHT_LIMIT", default=0),
            )
        )
        if raw_async_inflight_limit <= 0 and not self._put_pool_blocking_sync:
            raw_async_inflight_limit = max(
                _DEFAULT_PUT_POOL_ASYNC_INFLIGHT_MIN,
                self._put_pool_copy_streams * _DEFAULT_PUT_POOL_ASYNC_INFLIGHT_FACTOR,
            )
        self._put_pool_async_inflight_limit = 0
        if not self._put_pool_blocking_sync:
            self._put_pool_async_inflight_limit = min(max(0, raw_async_inflight_limit), self._pool_credits)
        # Receiver-side pool get behavior:
        # - True  (default): force copy stream to wait current stream before D2D decode.
        # - False: enqueue D2D decode directly on copy stream for better overlap.
        self._get_pool_wait_current_stream = bool(
            config.get(
                "get_pool_wait_current_stream",
                _env_bool("VLLM_OMNI_CUDA_IPC_GET_POOL_WAIT_CURRENT_STREAM", default=True),
            )
        )
        # Optional deep profiling for receiver pool-get wait split.
        # When enabled, add a synchronized wait-probe event to split
        # copy_finish stall into:
        #   - event_wait_sync_ms (waiting on sender event)
        #   - decode_finish_sync_ms (decode/copy stream completion wait)
        self._profile_wait_split = bool(
            config.get(
                "profile_wait_split",
                _env_bool("VLLM_OMNI_CUDA_IPC_PROFILE_WAIT_SPLIT", default=False),
            )
        )
        self._defer_unready_pool_get = bool(
            config.get(
                "defer_unready_pool_get",
                _env_bool("VLLM_OMNI_CUDA_IPC_DEFER_UNREADY_POOL_GET", default=False),
            )
        )

    def _init_runtime_state(self) -> None:
        """Locks, ring/receiver caches, metrics, lifecycle flags."""
        self._ring: CudaIpcControlRing | None = None
        self._opened_rings: dict[tuple[str, str], CudaIpcControlRing] = {}
        self._ring_edge_handles: dict[tuple[str, str], tuple[bytes, bytes, str]] = {}
        self._opened_pools: dict[bytes, ctypes.c_void_p] = {}
        self._opened_boards: dict[str, shm_pkg.SharedMemory] = {}
        self._opened_events: dict[bytes, ctypes.c_void_p] = {}
        self._pending_pool_gets: dict[str, dict[str, Any]] = {}
        self._closed = False
        self._cudart = None
        self._held_lock = threading.Lock()
        self._open_lock = threading.Lock()
        self._ring_publish_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._release_thread: threading.Thread | None = None
        self._shm_compat_decode_failures: dict[str, int] = {}
        # Optional producer-side stream registered by the model runner. When set,
        # ``_put_pool`` orders its pool D2D after operations queued on this stream
        # (typically ``_omni_payload_copy_stream`` from the AR model runner).
        # When ``None``, ``_put_pool`` falls back to recording an event on the
        # save thread's ambient stream — only correct when PTDS is off and the
        # producer wrote on the legacy default stream. See ``register_producer_stream``.
        self._producer_stream: torch.cuda.Stream | None = None
        self._producer_event: torch.cuda.Event | None = None
        self._producer_fallback_warned: bool = False
        self._last_credit_poll_iters: int = 0
        self._producer_order_lock = threading.Lock()
        self._sender_copy_streams: list[torch.cuda.Stream] = []
        self._sender_copy_stream_idx: int = 0
        self._sender_copy_stream_lock = threading.Lock()
        self._put_pool_async_inflight_events: deque[torch.cuda.Event] = deque()
        self._put_pool_async_inflight_lock = threading.Lock()
        self._profile_log_counter = 0
        self._pool_event_not_ready_counter = 0
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
            "ring_misses": 0,
            "shm_compat_checks": 0,
            "pool_event_not_ready": 0,
        }

    def _init_cuda(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CudaIPCConnector requires CUDA runtime.")
        self._cudart = load_cudart()
        if torch.accelerator.device_count() > 1:
            self._validate_p2p_access()

    def _init_sender_resources(self) -> None:
        """Sender data-transfer rank: GPU pool, IPC event, release board, per-edge ring."""
        with torch.cuda.device(self.local_device):
            self._pool = torch.zeros(self._pool_size, dtype=torch.uint8, device=self.local_device)
            self._sender_copy_streams = [
                torch.cuda.Stream(device=self.local_device) for _ in range(self._put_pool_copy_streams)
            ]
            self._copy_stream = self._sender_copy_streams[0]
            self._compute_event = torch.cuda.Event()
        self._sender_copy_stream_idx = 0
        with self._put_pool_async_inflight_lock:
            self._put_pool_async_inflight_events.clear()
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
        # In async sender mode (non-blocking put_pool), one shared event for all
        # slots can create cross-slot waits on receiver side because each new
        # cudaEventRecord overwrites the same event timeline marker. Use one IPC
        # event per slot and carry that handle in descriptor metadata.
        self._slot_ipc_events: list[ctypes.c_void_p] = []
        self._slot_ipc_event_handle_bytes: list[bytes] = []
        if not self._put_pool_blocking_sync:
            for _ in range(self._pool_credits):
                slot_event = ctypes.c_void_p()
                ret = self._cudart.cudaEventCreateWithFlags(ctypes.byref(slot_event), flags)
                if ret != 0:
                    raise RuntimeError(f"cudaEventCreateWithFlags (slot) failed: {ret}")
                slot_evt_handle = _CudaIpcEventHandle()
                ret = self._cudart.cudaIpcGetEventHandle(ctypes.byref(slot_evt_handle), slot_event)
                if ret != 0:
                    raise RuntimeError(f"cudaIpcGetEventHandle (slot) failed: {ret}")
                self._slot_ipc_events.append(slot_event)
                self._slot_ipc_event_handle_bytes.append(bytes(slot_evt_handle))
        self._credit_queue: _queue_mod.Queue[int] = _queue_mod.Queue(maxsize=self._pool_credits)
        for i in range(self._pool_credits):
            self._credit_queue.put_nowait(i * self._slot_size)
        self._held_credits: dict[str, tuple[float, int]] = {}
        # CPU-fallback /dev/shm segments {name: ts}: receiver unlinks on read, else the
        # release loop TTL-sweeps them (the adapter never calls connector cleanup()).
        self._fallback_segs: dict[str, float] = {}
        self._board_name = f"cudaipc_board_{uuid.uuid4().hex[:16]}"
        self._board = shm_pkg.SharedMemory(create=True, size=self._pool_credits, name=self._board_name)
        self._board.buf[: self._pool_credits] = bytes(self._pool_credits)
        n_slots = self._ring_entries_cfg or max(64, self._pool_credits * 4)
        body_max = max(self._ring_body_max, self._inline_threshold)
        self._ring = CudaIpcControlRing.create(
            ring_shm_name(self.stage_id, self.stage_id + 1, self._replica_id),
            n_slots,
            body_max,
            header_bytes=RING_HEADER_BYTES,
        )
        self._ring.write_header(self._ring_header_blob())

    def _init_inert_state(self) -> None:
        """Receiver, or an inert non-transfer-rank sender: no pool/board/ring."""
        self._pool = None
        self._pool_handle = None
        self._credit_queue = None
        self._held_credits = {}
        self._fallback_segs = {}
        self._board_name = None
        self._board = None
        self._slot_ipc_events = []
        self._slot_ipc_event_handle_bytes = []
        self._sender_copy_streams = []
        self._copy_stream = None
        if self.role == "receiver":
            with torch.cuda.device(self.local_device):
                self._recv_copy_streams = [
                    torch.cuda.Stream(device=self.local_device) for _ in range(_DEFAULT_RECV_STREAMS)
                ]
                self._recv_copy_events = [torch.cuda.Event() for _ in range(_DEFAULT_RECV_STREAMS)]
                self._recv_wait_probe_events = (
                    [torch.cuda.Event() for _ in range(_DEFAULT_RECV_STREAMS)] if self._profile_wait_split else []
                )
        else:
            self._recv_copy_streams = []
            self._recv_copy_events = []
            self._recv_wait_probe_events = []
        self._recv_stream_idx = 0

    def _start_release_thread(self) -> None:
        self._release_thread = threading.Thread(target=self._release_loop, daemon=True, name="cuda-ipc-release-loop")
        self._release_thread.start()

    def _should_profile_log(self, elapsed_ms: float) -> bool:
        if elapsed_ms >= self._profile_log_threshold_ms:
            return True
        if not self._profile_log_enabled:
            return False
        self._profile_log_counter += 1
        return (self._profile_log_counter % self._profile_log_every_n) == 0

    def _profile_log(self, phase: str, elapsed_ms: float, **fields: Any) -> None:
        if not self._should_profile_log(elapsed_ms):
            return
        details = " ".join(f"{k}={v}" for k, v in sorted(fields.items()))
        logger.info(
            "CudaIPCConnector profile phase=%s elapsed_ms=%.3f role=%s stage=%s %s",
            phase,
            elapsed_ms,
            self.role,
            self.stage_id,
            details,
        )

    # --- Producer stream registration ---

    def _maybe_warn_ambient_fallback(self) -> None:
        """One-shot warning when ``_put_pool`` falls back to ambient-stream ordering.

        The fallback is only correct when PTDS is off and the producer wrote
        on the legacy default stream. We don't raise — older code paths still
        rely on this — but we surface a warning on the first put() so a
        silently-broken assumption (e.g. a future PTDS rollout) is at least
        observable in logs.
        """
        if self._producer_fallback_warned:
            return
        self._producer_fallback_warned = True
        # Detect PTDS heuristically: when enabled, ``current_stream()`` on a
        # newly-spawned thread returns a per-thread default stream whose
        # ``cuda_stream`` pointer is non-zero (legacy default has 0). We
        # cannot reliably probe from this thread, so we just warn that the
        # caller did not register a producer stream and explain the
        # consequences. Callers using the keep_on_gpu snapshot path should
        # call ``register_producer_stream`` to opt into the correct path.
        logger.warning(
            "CudaIPCConnector: _put_pool taking ambient-stream fallback "
            "ordering. Correct only when PTDS is off and the producer "
            "wrote on the legacy default stream. Call "
            "register_producer_stream(_omni_payload_copy_stream) from the "
            "model runner to use safe stream-based ordering."
        )

    def register_producer_stream(self, producer_stream: torch.cuda.Stream | None) -> None:
        """Register the stream the producer uses to write payload tensors.

        When set, ``_put_pool`` orders its pool D2D after operations queued on
        this stream via ``copy_stream.wait_stream(producer_stream)``. This is
        the correct ordering primitive when ``put()`` runs on a thread other
        than the producer (e.g. the chunk_transfer save thread) and the
        producer writes on a non-default stream — for example the AR model
        runner's ``_omni_payload_copy_stream`` used by the keep_on_gpu
        snapshot path.

        Without this registration, ``_put_pool`` falls back to recording an
        event on the save thread's ambient stream, which is only correct
        when PTDS is off and the producer also wrote on the legacy default
        stream. We log a warning on the first put() in that mode so the
        fragile assumption can't break silently.

        Args:
            producer_stream: A ``torch.cuda.Stream`` instance, or ``None`` to
                clear a previous registration and revert to the legacy
                fallback.
        """
        self._producer_stream = producer_stream
        # Reset the warn-once latch so a re-registration after clear can
        # surface a new warning if the fallback path is triggered later.
        self._producer_fallback_warned = False

    def register_producer_event(self, producer_event: torch.cuda.Event | None) -> None:
        """Register one-shot producer-ready event for the next put().

        When set, _put_pool will order copy_stream by wait_event(producer_event)
        (narrow fence: up to the recorded point) instead of wait_stream(producer_stream)
        (broad fence: all currently queued producer stream work).
        """
        with self._producer_order_lock:
            self._producer_event = producer_event

    def _consume_producer_event(self) -> torch.cuda.Event | None:
        with self._producer_order_lock:
            evt = self._producer_event
            self._producer_event = None
            return evt

    def _next_sender_copy_stream(self) -> tuple[torch.cuda.Stream, int]:
        streams = self._sender_copy_streams
        if not streams:
            if self._copy_stream is None:
                raise RuntimeError("Sender copy stream is not initialized.")
            return self._copy_stream, 0
        if len(streams) == 1:
            return streams[0], 0
        with self._sender_copy_stream_lock:
            idx = self._sender_copy_stream_idx % len(streams)
            self._sender_copy_stream_idx += 1
        return streams[idx], idx

    def _wait_put_pool_async_window(self) -> tuple[float, int]:
        limit = int(self._put_pool_async_inflight_limit)
        if limit <= 0:
            return 0.0, 0
        wait_ms = 0.0
        waited_events = 0
        while True:
            wait_evt = None
            with self._put_pool_async_inflight_lock:
                while self._put_pool_async_inflight_events and self._put_pool_async_inflight_events[0].query():
                    self._put_pool_async_inflight_events.popleft()
                if len(self._put_pool_async_inflight_events) < limit:
                    break
                wait_evt = self._put_pool_async_inflight_events.popleft()
            if wait_evt is None:
                break
            wait_t0 = _time_mod.perf_counter()
            wait_evt.synchronize()
            wait_ms += (_time_mod.perf_counter() - wait_t0) * 1000.0
            waited_events += 1
        return wait_ms, waited_events

    def _track_put_pool_async_event(self, copy_stream: torch.cuda.Stream) -> int:
        limit = int(self._put_pool_async_inflight_limit)
        if limit <= 0:
            return 0
        done_evt = torch.cuda.Event()
        done_evt.record(copy_stream)
        with self._put_pool_async_inflight_lock:
            self._put_pool_async_inflight_events.append(done_evt)
            return len(self._put_pool_async_inflight_events)

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
    def _safe_name(prefix: str, key: str) -> str:
        """Generate a short, collision-free SHM name via SHA1 hash."""
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:20]
        return f"{prefix}_{digest}"

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
        ret = self._cudart.cudaIpcOpenMemHandle(ctypes.byref(dev_ptr), handle, ctypes.c_uint(1))
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
            try:
                ring = CudaIpcControlRing.open(ring_shm_name(from_stage, to_stage, self._replica_id))
            except FileNotFoundError:
                return None  # sender not up yet; poll loop tolerates None
            self._opened_rings[edge] = ring
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
        if isinstance(obj, (list, tuple)):
            return sum(self._estimate_nbytes(v) for v in obj)
        if hasattr(obj, "__struct_fields__"):
            return sum(
                self._estimate_nbytes(getattr(obj, f)) for f in obj.__struct_fields__ if getattr(obj, f) is not None
            )
        return 0

    def _count_cuda_tensors(self, obj: Any) -> int:
        if isinstance(obj, torch.Tensor):
            return 1 if obj.is_cuda else 0
        if isinstance(obj, dict):
            return sum(self._count_cuda_tensors(v) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return sum(self._count_cuda_tensors(v) for v in obj)
        if hasattr(obj, "__struct_fields__"):
            return sum(
                self._count_cuda_tensors(getattr(obj, f))
                for f in obj.__struct_fields__
                if getattr(obj, f) is not None
            )
        return 0

    # --- Pool slot codec ---

    def _walk_encode_pool(self, obj: Any, slot: _PoolSlot, copy_stream: torch.cuda.Stream) -> Any:
        """Recursively replace CUDA tensors with pool offset metadata."""
        if isinstance(obj, torch.Tensor):
            if obj.is_cuda:
                if not self._put_pool_blocking_sync:
                    # Allocator-lifetime hint only: prevents caching-allocator
                    # reuse of allocator-managed source storage until this
                    # copy stream is done. Correctness ordering is established
                    # by producer_event / producer_stream fences; graph-static
                    # buffers are not protected by record_stream.
                    obj.record_stream(copy_stream)
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
            return {k: self._walk_encode_pool(v, slot, copy_stream) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._walk_encode_pool(v, slot, copy_stream) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self._walk_encode_pool(v, slot, copy_stream) for v in obj)
        if hasattr(obj, "__struct_fields__"):
            # Drop None struct fields — matches data_entry_keys.to_dict (the SHM/inline wire
            # contract), so pool and SHM paths produce the same dict keys downstream.
            return {
                f: self._walk_encode_pool(getattr(obj, f), slot, copy_stream)
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

    def _acquire_credit(self) -> int | None:
        """Get a free slot offset, reclaiming board credits inline.

        Bounded wait (``credit_wait_sec``) before giving up; returns None to
        trigger the CPU fallback.
        """
        poll_iters = 0
        try:
            slot = self._credit_queue.get_nowait()
            self._last_credit_poll_iters = 0
            return slot
        except _queue_mod.Empty:
            pass
        deadline = _time_mod.monotonic() + self._credit_wait_sec
        while _time_mod.monotonic() < deadline:
            poll_iters += 1
            self._reclaim_board_credits()
            try:
                slot = self._credit_queue.get_nowait()
                self._last_credit_poll_iters = poll_iters
                return slot
            except _queue_mod.Empty:
                _time_mod.sleep(self._credit_poll_sec)
        self._last_credit_poll_iters = poll_iters
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
                if buf[slot_offset // self._slot_size] == 1
            ]
            for key, slot_offset in released:
                self._held_credits.pop(key, None)
                buf[slot_offset // self._slot_size] = 0
                self._credit_queue.put_nowait(slot_offset)
                self._metrics["board_releases"] += 1

    def _release_expired_credits(self) -> None:
        """TTL sweep: reclaim slots whose receiver never marked the board
        (e.g. the request was aborted or the receiver died)."""
        now = _time_mod.time()
        with self._held_lock:
            expired = [
                (key, slot_offset)
                for key, (ts, slot_offset) in self._held_credits.items()
                if now - ts > self.tensor_lifetime_sec
            ]
            for key, slot_offset in expired:
                self._held_credits.pop(key, None)
                self._board.buf[slot_offset // self._slot_size] = 0
                self._credit_queue.put_nowait(slot_offset)
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

    def _release_loop(self) -> None:
        tick = 0
        while not self._stop_event.is_set():
            try:
                self._reclaim_board_credits()
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
    ) -> tuple[bool, int, dict[str, Any] | None]:
        if self._closed:
            return False, 0, None
        if not self._is_transfer_rank:
            # Inert non-transfer-rank sender: no ring/pool. Guard against a stray call.
            return False, 0, None

        composite_key = make_composite_key(put_key, from_stage, to_stage)
        return self._put_control_plane(from_stage, to_stage, put_key, composite_key, data)

    def put_with_producer_event(
        self,
        from_stage: str,
        to_stage: str,
        put_key: str,
        data: Any,
        producer_event: torch.cuda.Event | None = None,
    ) -> tuple[bool, int, dict[str, Any] | None]:
        if self._closed:
            return False, 0, None
        if not self._is_transfer_rank:
            return False, 0, None
        composite_key = make_composite_key(put_key, from_stage, to_stage)
        return self._put_control_plane(
            from_stage,
            to_stage,
            put_key,
            composite_key,
            data,
            producer_event=producer_event,
        )

    def _put_control_plane(
        self,
        from_stage,
        to_stage,
        put_key,
        composite_key,
        data,
        producer_event: torch.cuda.Event | None = None,
    ):
        """Primary send path: route small payloads inline vs large via GPU pool (both publish to ring)."""
        t0 = _time_mod.perf_counter()
        try:
            kh = key_hash16(composite_key)
            est_nbytes = self._estimate_nbytes(data)
            inline_cuda_by_size = (
                est_nbytes > 0 and self._inline_cuda_max_bytes > 0 and est_nbytes <= self._inline_cuda_max_bytes
            )
            force_pool_for_cuda = est_nbytes > 0 and not self._inline_cuda_tensors and not inline_cuda_by_size
            if producer_event is None:
                producer_event = self._consume_producer_event()
            route_inline = est_nbytes < self._inline_threshold and not force_pool_for_cuda
            if route_inline:
                if self._inline_use_shm:
                    result = self._put_shm_inline(from_stage, to_stage, put_key, composite_key, data, kh)
                    route = "shm_inline"
                else:
                    result = self._put_inline(
                        from_stage,
                        to_stage,
                        put_key,
                        composite_key,
                        data,
                        kh,
                        producer_event=producer_event,
                        est_nbytes=est_nbytes,
                    )
                    route = "inline"
            else:
                result = self._put_pool(
                    from_stage,
                    to_stage,
                    put_key,
                    composite_key,
                    data,
                    kh,
                    producer_event,
                )
                route = "pool"
            elapsed_ms = (_time_mod.perf_counter() - t0) * 1000.0
            meta = result[2] if len(result) >= 3 else None
            self._profile_log(
                "put_control_plane",
                elapsed_ms,
                key=put_key,
                route=route,
                est_nbytes=est_nbytes,
                force_pool_for_cuda=force_pool_for_cuda,
                inline_cuda_by_size=inline_cuda_by_size,
                inline_cuda_max_bytes=self._inline_cuda_max_bytes,
                inline_use_shm=self._inline_use_shm,
                ok=result[0],
                wire_size=result[1],
                cpu_fallback=bool(isinstance(meta, dict) and meta.get("cpu_fallback")),
                fallback_producer_order_mode=(meta.get("producer_order_mode") if isinstance(meta, dict) else None),
                fallback_producer_order_wait_ms=(
                    round(float(meta.get("producer_order_wait_ms", 0.0)), 3) if isinstance(meta, dict) else None
                ),
            )
            return result
        except Exception as e:
            self._metrics["errors"] += 1
            logger.error("CudaIPCConnector control-plane put failed for %s: %s", put_key, e, exc_info=True)
            return False, 0, None

    def _put_inline(
        self,
        from_stage,
        to_stage,
        put_key,
        composite_key,
        data,
        kh,
        producer_event: torch.cuda.Event | None = None,
        est_nbytes: int | None = None,
    ):
        t0 = _time_mod.perf_counter()
        producer_order_mode = "none"
        producer_order_wait_ms = 0.0
        inline_cuda_to_cpu_ms = 0.0
        inline_cuda_tensors = 0
        inline_cuda_nbytes = 0
        serialize_ms = 0.0
        ring_publish_ms = 0.0
        inline_payload_obj = data
        inline_cuda_nbytes = est_nbytes if isinstance(est_nbytes, int) and est_nbytes >= 0 else self._estimate_nbytes(data)
        if inline_cuda_nbytes > 0:
            # Inline is SHM-like: serialize_obj handles tensor.detach().cpu()
            # in one pass. Do not pre-copy here; that duplicates work relative
            # to SharedMemoryConnector and was measured as the dominant per-token
            # inline cost for tiny CUDA payloads.
            producer_order_mode, producer_order_wait_ms = self._order_current_stream_after_producer(producer_event)
            inline_cuda_tensors = self._count_cuda_tensors(data)
        # Serialize (a cheap D2H for a tiny GPU tensor) straight into the ring body.
        serialize_t0 = _time_mod.perf_counter()
        payload = self.serialize_obj(inline_payload_obj)
        serialize_ms = (_time_mod.perf_counter() - serialize_t0) * 1000.0
        if len(payload) > self._ring.body_max:
            result = self._put_cpu_fallback(
                from_stage,
                to_stage,
                put_key,
                composite_key,
                inline_payload_obj,
                reason=f"inline_too_big {len(payload)}",
            )
            self._profile_log(
                "put_inline",
                (_time_mod.perf_counter() - t0) * 1000.0,
                key=put_key,
                payload_bytes=len(payload),
                outcome="cpu_fallback",
                producer_order_mode=producer_order_mode,
                producer_order_wait_ms=round(producer_order_wait_ms, 3),
                inline_cuda_to_cpu_ms=round(inline_cuda_to_cpu_ms, 3),
                serialize_ms=round(serialize_ms, 3),
                inline_cuda_tensors=inline_cuda_tensors,
                inline_cuda_nbytes=inline_cuda_nbytes,
            )
            return result
        try:
            ring_publish_t0 = _time_mod.perf_counter()
            with self._ring_publish_lock:
                self._ring.publish(
                    kh, RING_PCLASS_INLINE, payload, ts=int(_time_mod.time()), ttl_sec=int(self.tensor_lifetime_sec)
                )
            ring_publish_ms = (_time_mod.perf_counter() - ring_publish_t0) * 1000.0
        except RingFullError:
            result = self._put_cpu_fallback(
                from_stage,
                to_stage,
                put_key,
                composite_key,
                inline_payload_obj,
                reason="ring_full",
            )
            self._profile_log(
                "put_inline",
                (_time_mod.perf_counter() - t0) * 1000.0,
                key=put_key,
                payload_bytes=len(payload),
                outcome="ring_full_fallback",
                producer_order_mode=producer_order_mode,
                producer_order_wait_ms=round(producer_order_wait_ms, 3),
                inline_cuda_to_cpu_ms=round(inline_cuda_to_cpu_ms, 3),
                serialize_ms=round(serialize_ms, 3),
                ring_publish_ms=round(ring_publish_ms, 3),
                inline_cuda_tensors=inline_cuda_tensors,
                inline_cuda_nbytes=inline_cuda_nbytes,
            )
            return result
        self._metrics["puts"] += 1
        self._metrics["bytes_transferred"] += len(payload)
        self._profile_log(
            "put_inline",
            (_time_mod.perf_counter() - t0) * 1000.0,
            key=put_key,
            payload_bytes=len(payload),
            outcome="ring_publish",
            producer_order_mode=producer_order_mode,
            producer_order_wait_ms=round(producer_order_wait_ms, 3),
            inline_cuda_to_cpu_ms=round(inline_cuda_to_cpu_ms, 3),
            serialize_ms=round(serialize_ms, 3),
            ring_publish_ms=round(ring_publish_ms, 3),
            inline_cuda_tensors=inline_cuda_tensors,
            inline_cuda_nbytes=inline_cuda_nbytes,
        )
        return True, len(payload), {"ring": True, "size": len(payload)}

    def _put_shm_inline(self, from_stage, to_stage, put_key, composite_key, data, kh):
        """SHM-backed small-payload path announced through the ring.

        This keeps IPC's ring control-plane notification while using the same
        CPU-byte payload transport shape as SharedMemoryConnector for frequent
        small chunks.
        """
        t0 = _time_mod.perf_counter()
        serialize_t0 = _time_mod.perf_counter()
        payload = self.serialize_obj(data)
        serialize_ms = (_time_mod.perf_counter() - serialize_t0) * 1000.0
        shm_t0 = _time_mod.perf_counter()
        meta = self._atomic_shm_write(payload, name=put_key)
        shm_write_ms = (_time_mod.perf_counter() - shm_t0) * 1000.0
        if getattr(self, "_fallback_segs", None) is not None:
            self._fallback_segs[put_key] = _time_mod.time()
        descriptor = OmniSerializer.serialize({"shm": meta, "size": len(payload)})
        ring_publish_ms = 0.0
        try:
            ring_publish_t0 = _time_mod.perf_counter()
            with self._ring_publish_lock:
                self._ring.publish(
                    kh,
                    RING_PCLASS_SHM,
                    descriptor,
                    ts=int(_time_mod.time()),
                    ttl_sec=int(self.tensor_lifetime_sec),
                )
            ring_publish_ms = (_time_mod.perf_counter() - ring_publish_t0) * 1000.0
        except RingFullError:
            # Receiver can still discover this through shm-compat ring-miss path.
            self._metrics["fallback_ring_full"] = self._metrics.get("fallback_ring_full", 0) + 1
            self._profile_log(
                "put_shm_inline",
                (_time_mod.perf_counter() - t0) * 1000.0,
                key=put_key,
                payload_bytes=len(payload),
                descriptor_bytes=len(descriptor),
                outcome="ring_full_shm_compat",
                serialize_ms=round(serialize_ms, 3),
                shm_write_ms=round(shm_write_ms, 3),
            )
            return True, len(payload), {"shm": meta, "size": len(payload), "cpu_fallback": True}

        self._metrics["puts"] += 1
        self._metrics["bytes_transferred"] += len(payload)
        self._profile_log(
            "put_shm_inline",
            (_time_mod.perf_counter() - t0) * 1000.0,
            key=put_key,
            payload_bytes=len(payload),
            descriptor_bytes=len(descriptor),
            outcome="ring_publish",
            serialize_ms=round(serialize_ms, 3),
            shm_write_ms=round(shm_write_ms, 3),
            ring_publish_ms=round(ring_publish_ms, 3),
        )
        return True, len(payload), {"ring": True, "shm": meta, "size": len(payload)}

    def _put_pool(
        self,
        from_stage,
        to_stage,
        put_key,
        composite_key,
        data,
        kh,
        producer_event: torch.cuda.Event | None = None,
    ):
        # Acquire a credit, D2D-pack into the slot, then publish a small descriptor
        # (slot_offset/slot_index + tensor layout) to the ring.
        # TODO: support acquiring multiple contiguous credits/slots when a single payload
        # exceeds _slot_size (currently: _SlotOverflowError → CPU fallback).
        total_t0 = _time_mod.perf_counter()
        async_backpressure_wait_ms = 0.0
        async_backpressure_wait_events = 0
        async_inflight_depth = 0
        if not self._put_pool_blocking_sync:
            # In non-blocking mode, bound sender queue lead so receiver doesn't
            # stall on events that are far behind in sender stream queues.
            async_backpressure_wait_ms, async_backpressure_wait_events = self._wait_put_pool_async_window()
        sender_copy_stream_count = max(1, len(self._sender_copy_streams))
        sender_copy_stream_idx = 0
        credit_t0 = _time_mod.perf_counter()
        slot_offset = self._acquire_credit()
        credit_wait_ms = (_time_mod.perf_counter() - credit_t0) * 1000.0
        credit_poll_iters = int(getattr(self, "_last_credit_poll_iters", 0))
        if slot_offset is None:
            result = self._put_cpu_fallback(
                from_stage,
                to_stage,
                put_key,
                composite_key,
                data,
                reason="credits_exhausted",
                producer_event=producer_event,
            )
            self._profile_log(
                "put_pool",
                (_time_mod.perf_counter() - total_t0) * 1000.0,
                key=put_key,
                outcome="credits_exhausted_fallback",
                credit_wait_ms=round(credit_wait_ms, 3),
                credit_poll_iters=credit_poll_iters,
                async_backpressure_wait_ms=round(async_backpressure_wait_ms, 3),
                async_backpressure_wait_events=async_backpressure_wait_events,
                async_inflight_limit=self._put_pool_async_inflight_limit,
                sender_copy_streams=sender_copy_stream_count,
            )
            return result
        copy_stream, sender_copy_stream_idx = self._next_sender_copy_stream()
        credit_returned = False
        descriptor_bytes = 0
        pack_sync_ms = 0.0
        descriptor_ser_ms = 0.0
        producer_order_wait_ms = 0.0
        producer_order_mode = "ambient"
        event_record_ms = 0.0
        ring_publish_ms = 0.0
        slot_used_bytes = 0
        try:
            self._board.buf[slot_offset // self._slot_size] = 0
            slot = _PoolSlot(self._pool, slot_offset, self._slot_size)
            slot_idx = slot_offset // self._slot_size
            # Order the pack after the producer's writes. Two paths:
            #
            # 1. Preferred: ``register_producer_stream`` was called by the model
            #    runner with the actual producer stream — for the AR runner's
            #    keep_on_gpu snapshot path that's the model's compute stream
            #    (the snapshot clone has to live there to be CUDA-graph safe;
            #    see _snapshot_omni_output_tensors_for_async_output).
            #    ``copy_stream.wait_stream(producer_stream)`` is the correct
            #    primitive on any host: it fences the pool D2D past every op
            #    currently queued on ``producer_stream``, including the
            #    snapshot clone, regardless of which thread queued it.
            # 2. Fallback (legacy): record an event on the save thread's
            #    ambient stream and wait on it. Only correct when PTDS is off
            #    AND the producer wrote on the legacy default stream — e.g.
            #    the synchronous, non-async-output path. Warn once when this
            #    path is taken with PTDS enabled, so the silent-corruption
            #    assumption can't break unnoticed.
            producer_wait_t0 = _time_mod.perf_counter()
            if producer_event is not None:
                copy_stream.wait_event(producer_event)
                producer_order_mode = "event"
            elif self._producer_stream is not None:
                copy_stream.wait_stream(self._producer_stream)
                producer_order_mode = "stream"
            else:
                self._maybe_warn_ambient_fallback()
                self._compute_event.record()
                copy_stream.wait_event(self._compute_event)
                producer_order_mode = "ambient"
            producer_order_wait_ms = (_time_mod.perf_counter() - producer_wait_t0) * 1000.0
            try:
                pack_t0 = _time_mod.perf_counter()
                with torch.cuda.stream(copy_stream):
                    encoded_obj = self._walk_encode_pool(data, slot, copy_stream)
                slot_used_bytes = slot._cursor
            except _SlotOverflowError as e:
                copy_stream.synchronize()
                credit_returned = True
                self._credit_queue.put_nowait(slot_offset)
                result = self._put_cpu_fallback(
                    from_stage,
                    to_stage,
                    put_key,
                    composite_key,
                    data,
                    reason=f"slot_overflow nbytes={e.nbytes} slot={e.slot_size}",
                    producer_event=producer_event,
                )
                self._profile_log(
                    "put_pool",
                    (_time_mod.perf_counter() - total_t0) * 1000.0,
                    key=put_key,
                    outcome="slot_overflow_fallback",
                    credit_wait_ms=round(credit_wait_ms, 3),
                    credit_poll_iters=credit_poll_iters,
                    async_backpressure_wait_ms=round(async_backpressure_wait_ms, 3),
                    async_backpressure_wait_events=async_backpressure_wait_events,
                    async_inflight_limit=self._put_pool_async_inflight_limit,
                    sender_copy_stream_idx=sender_copy_stream_idx,
                    sender_copy_streams=sender_copy_stream_count,
                )
                return result
            event_to_record = self._ipc_event
            slot_event_handle: bytes | None = None
            if not self._put_pool_blocking_sync and self._slot_ipc_events:
                event_to_record = self._slot_ipc_events[slot_idx]
                slot_event_handle = self._slot_ipc_event_handle_bytes[slot_idx]
            event_record_t0 = _time_mod.perf_counter()
            ret = self._cudart.cudaEventRecord(event_to_record, ctypes.c_void_p(copy_stream.cuda_stream))
            event_record_ms = (_time_mod.perf_counter() - event_record_t0) * 1000.0
            if ret != 0:
                logger.warning("cudaEventRecord (IPC) failed: %d", ret)
            # In default mode, block until the pack D2D finishes.
            # In async mode, rely on IPC event ordering + record_stream source
            # lifetime tracking to avoid a sender-side hard sync.
            if self._put_pool_blocking_sync:
                copy_stream.synchronize()
            pack_sync_ms = (_time_mod.perf_counter() - pack_t0) * 1000.0
            descriptor_t0 = _time_mod.perf_counter()
            sender_desc_done_ns = _time_mod.perf_counter_ns()
            descriptor = OmniSerializer.serialize(
                {
                    _POOL_MARKER: True,
                    "slot_offset": slot_offset,
                    "slot_index": slot_idx,
                    "payload": encoded_obj,
                    "sender_desc_done_ns": sender_desc_done_ns,
                    # Optional per-slot event handle for async sender mode.
                    # Receiver falls back to ring-header event handle when absent.
                    **({"event_handle": slot_event_handle} if slot_event_handle is not None else {}),
                }
            )
            descriptor_ser_ms = (_time_mod.perf_counter() - descriptor_t0) * 1000.0
            descriptor_bytes = len(descriptor)
        except Exception:
            if not credit_returned:
                self._credit_queue.put_nowait(slot_offset)
            raise
        # Descriptor grows with sequence length; if it overflows the ring body,
        # degrade to the CPU fallback (read via _try_get_shm_compat) — never crash.
        if len(descriptor) > self._ring.body_max:
            self._credit_queue.put_nowait(slot_offset)
            result = self._put_cpu_fallback(
                from_stage,
                to_stage,
                put_key,
                composite_key,
                data,
                reason=f"descriptor_too_big {len(descriptor)}>{self._ring.body_max}",
                producer_event=producer_event,
            )
            self._profile_log(
                "put_pool",
                (_time_mod.perf_counter() - total_t0) * 1000.0,
                key=put_key,
                outcome="descriptor_too_big_fallback",
                descriptor_bytes=len(descriptor),
                slot_used_bytes=slot_used_bytes,
                credit_wait_ms=round(credit_wait_ms, 3),
                credit_poll_iters=credit_poll_iters,
                pack_sync_ms=round(pack_sync_ms, 3),
                descriptor_ser_ms=round(descriptor_ser_ms, 3),
                producer_order_wait_ms=round(producer_order_wait_ms, 3),
                event_record_ms=round(event_record_ms, 3),
                async_backpressure_wait_ms=round(async_backpressure_wait_ms, 3),
                async_backpressure_wait_events=async_backpressure_wait_events,
                async_inflight_limit=self._put_pool_async_inflight_limit,
                sender_copy_stream_idx=sender_copy_stream_idx,
                sender_copy_streams=sender_copy_stream_count,
            )
            return result
        with self._held_lock:
            self._held_credits[composite_key] = (_time_mod.time(), slot_offset)
        try:
            ring_publish_t0 = _time_mod.perf_counter()
            with self._ring_publish_lock:
                self._ring.publish(
                    kh, RING_PCLASS_POOL, descriptor, ts=int(_time_mod.time()), ttl_sec=int(self.tensor_lifetime_sec)
                )
            ring_publish_ms = (_time_mod.perf_counter() - ring_publish_t0) * 1000.0
        except RingFullError:
            with self._held_lock:
                self._held_credits.pop(composite_key, None)
            self._credit_queue.put_nowait(slot_offset)
            result = self._put_cpu_fallback(
                from_stage,
                to_stage,
                put_key,
                composite_key,
                data,
                reason="ring_full",
                producer_event=producer_event,
            )
            self._profile_log(
                "put_pool",
                (_time_mod.perf_counter() - total_t0) * 1000.0,
                key=put_key,
                outcome="ring_full_fallback",
                descriptor_bytes=descriptor_bytes,
                slot_used_bytes=slot_used_bytes,
                credit_wait_ms=round(credit_wait_ms, 3),
                credit_poll_iters=credit_poll_iters,
                pack_sync_ms=round(pack_sync_ms, 3),
                descriptor_ser_ms=round(descriptor_ser_ms, 3),
                producer_order_wait_ms=round(producer_order_wait_ms, 3),
                event_record_ms=round(event_record_ms, 3),
                async_backpressure_wait_ms=round(async_backpressure_wait_ms, 3),
                async_backpressure_wait_events=async_backpressure_wait_events,
                async_inflight_limit=self._put_pool_async_inflight_limit,
                sender_copy_stream_idx=sender_copy_stream_idx,
                sender_copy_streams=sender_copy_stream_count,
            )
            return result
        if not self._put_pool_blocking_sync:
            async_inflight_depth = self._track_put_pool_async_event(copy_stream)
        self._metrics["puts"] += 1
        self._metrics["bytes_transferred"] += len(descriptor)
        self._profile_log(
            "put_pool",
            (_time_mod.perf_counter() - total_t0) * 1000.0,
            key=put_key,
            outcome="ring_publish",
            descriptor_bytes=descriptor_bytes,
            slot_used_bytes=slot_used_bytes,
            slot_idx=slot_idx,
            credit_wait_ms=round(credit_wait_ms, 3),
            credit_poll_iters=credit_poll_iters,
            pack_sync_ms=round(pack_sync_ms, 3),
            descriptor_ser_ms=round(descriptor_ser_ms, 3),
            producer_order_wait_ms=round(producer_order_wait_ms, 3),
            producer_order_mode=producer_order_mode,
            event_record_ms=round(event_record_ms, 3),
            ring_publish_ms=round(ring_publish_ms, 3),
            producer_stream_registered=self._producer_stream is not None,
            blocking_sync=self._put_pool_blocking_sync,
            async_backpressure_wait_ms=round(async_backpressure_wait_ms, 3),
            async_backpressure_wait_events=async_backpressure_wait_events,
            async_inflight_depth=async_inflight_depth,
            async_inflight_limit=self._put_pool_async_inflight_limit,
            sender_copy_stream_idx=sender_copy_stream_idx,
            sender_copy_streams=sender_copy_stream_count,
        )
        return True, len(descriptor), {"ring": True, "size": len(descriptor)}

    def _order_current_stream_after_producer(
        self,
        producer_event: torch.cuda.Event | None = None,
    ) -> tuple[str, float]:
        wait_t0 = _time_mod.perf_counter()
        if producer_event is not None:
            torch.cuda.current_stream(self.local_device).wait_event(producer_event)
            return "event", (_time_mod.perf_counter() - wait_t0) * 1000.0
        if self._producer_stream is not None:
            torch.cuda.current_stream(self.local_device).wait_stream(self._producer_stream)
            return "stream", (_time_mod.perf_counter() - wait_t0) * 1000.0
        # Legacy/ambient fallback: only correct when producer also writes on the
        # same ambient stream semantics. Keep warning behavior consistent.
        self._maybe_warn_ambient_fallback()
        return "ambient", (_time_mod.perf_counter() - wait_t0) * 1000.0

    def _put_cpu_fallback(
        self,
        from_stage: str,
        to_stage: str,
        put_key: str,
        composite_key: str,
        data: Any,
        reason: str = "",
        producer_event: torch.cuda.Event | None = None,
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
        producer_order_mode = "none"
        producer_order_wait_ms = 0.0
        if self._estimate_nbytes(data) > 0:
            producer_order_mode, producer_order_wait_ms = self._order_current_stream_after_producer(producer_event)
        payload = self.serialize_obj(data)
        size = len(payload)

        meta = self._atomic_shm_write(payload, name=put_key)
        # Track for TTL cleanup in case the receiver aborts and never reads/unlinks it.
        if getattr(self, "_fallback_segs", None) is not None:
            self._fallback_segs[put_key] = _time_mod.time()

        self._metrics["puts"] += 1
        self._metrics["bytes_transferred"] += size
        return True, size, {
            "shm": meta,
            "size": size,
            "cpu_fallback": True,
            "producer_order_mode": producer_order_mode,
            "producer_order_wait_ms": producer_order_wait_ms,
        }

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
        t0 = _time_mod.perf_counter()
        try:
            ring = self._open_ring_receiver(from_stage, to_stage)
            if ring is None:
                return None
            if (from_stage, to_stage) not in self._ring_edge_handles:
                # Header not ready — don't poll (poll consumes; a pool entry would be lost). Retry.
                return None
            pending_pool = self._pending_pool_gets.get(composite_key)
            if pending_pool is None:
                r = ring.poll(key_hash16(composite_key))
            else:
                r = (RING_PCLASS_POOL, pending_pool["body"])
            if r is None:
                self._metrics["ring_misses"] += 1
                if not self._enable_shm_compat_on_ring_miss:
                    self._profile_log(
                        "get_control_plane",
                        (_time_mod.perf_counter() - t0) * 1000.0,
                        key=get_key,
                        pclass="ring_miss_no_shm_compat",
                    )
                    return None
                # Ring miss: chunk not published yet (poll retry), or the sender took the CPU
                # fallback (/dev/shm by put_key) — read that so the consumer never hangs.
                self._metrics["shm_compat_checks"] += 1
                shm_result = self._try_get_shm_compat(get_key)
                if shm_result is not None:
                    self._profile_log(
                        "get_control_plane",
                        (_time_mod.perf_counter() - t0) * 1000.0,
                        key=get_key,
                        pclass="shm_compat",
                        payload_bytes=shm_result[1],
                    )
                return shm_result
            pclass, body = r
            if pclass == RING_PCLASS_INLINE:
                # Return CPU tensors; the downstream model does the H2D (parity with SHM).
                # Doing it here was a redundant device-wide sync on the talker forward.
                obj = self.deserialize_obj(body)
                self._metrics["gets"] += 1
                self._metrics["bytes_transferred"] += len(body)
                self._profile_log(
                    "get_control_plane",
                    (_time_mod.perf_counter() - t0) * 1000.0,
                    key=get_key,
                    pclass="inline",
                    payload_bytes=len(body),
                )
                return obj, len(body)

            if pclass == RING_PCLASS_SHM:
                meta_t0 = _time_mod.perf_counter()
                raw = OmniSerializer.deserialize(body)
                meta_decode_ms = (_time_mod.perf_counter() - meta_t0) * 1000.0
                shm_meta = raw.get("shm") if isinstance(raw, dict) else None
                shm_name = shm_meta.get("name") if isinstance(shm_meta, dict) else get_key
                shm_result = self._try_get_shm_compat(str(shm_name))
                if shm_result is not None:
                    self._profile_log(
                        "get_control_plane",
                        (_time_mod.perf_counter() - t0) * 1000.0,
                        key=get_key,
                        pclass="shm_inline",
                        payload_bytes=shm_result[1],
                        descriptor_bytes=len(body),
                        descriptor_decode_ms=round(meta_decode_ms, 3),
                    )
                return shm_result

            # POOL: handles from the ring header (guaranteed present — the poll above is gated
            # on _ring_edge_handles), descriptor from the entry body.
            pool_handle, event_handle, board_name = self._ring_edge_handles[(from_stage, to_stage)]
            if pending_pool is not None:
                raw = pending_pool["raw"]
                descriptor_decode_ms = 0.0
                event_source = pending_pool.get("event_source", "pending")
            else:
                descriptor_decode_t0 = _time_mod.perf_counter()
                raw = OmniSerializer.deserialize(body)
                descriptor_decode_ms = (_time_mod.perf_counter() - descriptor_decode_t0) * 1000.0
                event_source = "descriptor" if "event_handle" in raw else "header"
            event_handle = raw.get("event_handle", event_handle)
            if event_handle is not None and not isinstance(event_handle, bytes):
                event_handle = bytes(event_handle)
            ipc_event = None
            if event_handle:
                ipc_event = self._open_ipc_event(event_handle)
                if self._defer_unready_pool_get and not event_query(self._cudart, ipc_event):
                    if pending_pool is None:
                        self._pending_pool_gets[composite_key] = {
                            "body": body,
                            "raw": raw,
                            "event_source": event_source,
                        }
                    self._metrics["pool_event_not_ready"] = self._metrics.get("pool_event_not_ready", 0) + 1
                    self._pool_event_not_ready_counter += 1
                    # This path can be hit many times while polling chunk-0
                    # descriptors. Logging every hit perturbs benchmark timing,
                    # so emit only a low-rate signal plus naturally-slow probes.
                    not_ready_ms = (_time_mod.perf_counter() - t0) * 1000.0
                    if not_ready_ms >= self._profile_log_threshold_ms or (
                        self._pool_event_not_ready_counter % max(1024, self._profile_log_every_n * 16) == 0
                    ):
                        self._profile_log(
                            "get_control_plane",
                            not_ready_ms,
                            key=get_key,
                            pclass="pool_event_not_ready",
                            payload_bytes=len(body),
                            event_source=event_source,
                            not_ready_count=self._pool_event_not_ready_counter,
                        )
                    return None
            sender_desc_done_ns = raw.get("sender_desc_done_ns")
            recv_ingress_ms = 0.0
            if isinstance(sender_desc_done_ns, int) and sender_desc_done_ns > 0:
                recv_ingress_ms = (_time_mod.perf_counter_ns() - sender_desc_done_ns) / 1_000_000.0
            open_pool_t0 = _time_mod.perf_counter()
            pool_ptr = self._open_pool(pool_handle)
            open_pool_ms = (_time_mod.perf_counter() - open_pool_t0) * 1000.0
            idx = self._recv_stream_idx % len(self._recv_copy_streams)
            self._recv_stream_idx += 1
            copy_stream = self._recv_copy_streams[idx]
            copy_done_event = self._recv_copy_events[idx]
            event_wait_enqueue_ms = 0.0
            event_wait_sync_ms = 0.0
            if ipc_event is not None:
                event_wait_t0 = _time_mod.perf_counter()
                stream_wait_event(self._cudart, copy_stream.cuda_stream, ipc_event)
                event_wait_enqueue_ms = (_time_mod.perf_counter() - event_wait_t0) * 1000.0
            copy_t0 = _time_mod.perf_counter()
            if self._profile_wait_split and event_handle and self._recv_wait_probe_events:
                # Force a one-time sync point right after the stream-side wait op so
                # we can split receiver stall into upstream-event wait vs decode finish.
                wait_probe_event = self._recv_wait_probe_events[idx]
                wait_probe_event.record(copy_stream)
                wait_sync_t0 = _time_mod.perf_counter()
                wait_probe_event.synchronize()
                event_wait_sync_ms = (_time_mod.perf_counter() - wait_sync_t0) * 1000.0
            copy_wait_current_stream_ms = 0.0
            if self._get_pool_wait_current_stream:
                wait_t0 = _time_mod.perf_counter()
                copy_stream.wait_stream(torch.cuda.current_stream(self.local_device))
                copy_wait_current_stream_ms = (_time_mod.perf_counter() - wait_t0) * 1000.0
            decode_enqueue_t0 = _time_mod.perf_counter()
            obj = self._walk_decode_pool(raw["payload"], pool_ptr, raw["slot_offset"], stream=copy_stream)
            decode_enqueue_ms = (_time_mod.perf_counter() - decode_enqueue_t0) * 1000.0
            copy_done_event.record(copy_stream)
            # Block until the D2D finishes before hand-off (payload is consumed later on the
            # model thread), then mark the board synchronously so TTL can't race the transfer.
            copy_finish_t0 = _time_mod.perf_counter()
            copy_done_event.synchronize()
            copy_finish_sync_ms = (_time_mod.perf_counter() - copy_finish_t0) * 1000.0
            decode_finish_sync_ms = copy_finish_sync_ms if self._profile_wait_split else 0.0
            copy_sync_ms = (_time_mod.perf_counter() - copy_t0) * 1000.0
            board_release_t0 = _time_mod.perf_counter()
            self._mark_board_release(board_name, int(raw["slot_index"]))
            board_release_ms = (_time_mod.perf_counter() - board_release_t0) * 1000.0
            self._metrics["gets"] += 1
            self._metrics["bytes_transferred"] += len(body)
            self._pending_pool_gets.pop(composite_key, None)
            self._profile_log(
                "get_control_plane",
                (_time_mod.perf_counter() - t0) * 1000.0,
                key=get_key,
                pclass="pool",
                payload_bytes=len(body),
                slot_idx=int(raw["slot_index"]),
                copy_sync_ms=round(copy_sync_ms, 3),
                copy_wait_current_stream_ms=round(copy_wait_current_stream_ms, 3),
                copy_finish_sync_ms=round(copy_finish_sync_ms, 3),
                decode_enqueue_ms=round(decode_enqueue_ms, 3),
                descriptor_decode_ms=round(descriptor_decode_ms, 3),
                open_pool_ms=round(open_pool_ms, 3),
                event_wait_enqueue_ms=round(event_wait_enqueue_ms, 3),
                event_wait_sync_ms=round(event_wait_sync_ms, 3),
                board_release_ms=round(board_release_ms, 3),
                recv_ingress_ms=round(recv_ingress_ms, 3),
                decode_finish_sync_ms=round(decode_finish_sync_ms, 3),
                event_source=event_source,
                wait_split_profiled=self._profile_wait_split,
                wait_current_stream=self._get_pool_wait_current_stream,
            )
            return obj, len(body)
        except Exception as e:
            self._metrics["errors"] += 1
            logger.error("CudaIPCConnector control-plane get failed for %s: %s", get_key, e, exc_info=True)
            return None

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
            # Correctness-first fallback path: stage input may be consumed by a
            # different stream/thread shortly after get(). Using non_blocking
            # H2D here without an explicit consumer-stream fence can expose
            # partially copied payloads under fallback pressure.
            h2d_t0 = _time_mod.perf_counter()
            obj = self._move_to_device(obj, non_blocking=False)
            h2d_ms = (_time_mod.perf_counter() - h2d_t0) * 1000.0
            size = len(data_bytes)
            self._metrics["gets"] += 1
            self._metrics["bytes_transferred"] += size
            self._profile_log(
                "get_control_plane",
                h2d_ms,
                key=get_key,
                pclass="shm_compat_h2d",
                payload_bytes=size,
            )
            return obj, size
        except Exception as e:
            logger.warning("CudaIPCConnector shm_compat get failed for %s: %s", get_key, e)
            return None
        finally:
            try:
                seg.close()
            except Exception:
                pass

    def _move_to_device(self, obj: Any, non_blocking: bool = False) -> Any:
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
        # Required by OmniConnectorBase but a no-op: credits are reclaimed by the release
        # board + TTL sweep, not per request_id.
        return

    def health(self) -> dict[str, Any]:
        return {
            "status": "healthy" if not self._closed else "closed",
            "role": self.role,
            "local_device": str(self.local_device),
            "replica_id": self._replica_id,
            "pool_size_mb": self._pool_size // (1024 * 1024),
            "pool_credits": self._pool_credits,
            "held_credits": len(self._held_credits),
            "put_pool_copy_streams": self._put_pool_copy_streams,
            "put_pool_async_inflight_limit": self._put_pool_async_inflight_limit,
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
        with self._held_lock:
            self._held_credits.clear()
        with self._put_pool_async_inflight_lock:
            self._put_pool_async_inflight_events.clear()

        for pool_ptr in self._opened_pools.values():
            self._try_or_warn(lambda p=pool_ptr: self._close_ipc_ptr(p), "close pool mapping")
        self._opened_pools.clear()

        # Destroy CUDA IPC events: sender's own + receiver-opened (cudaEventDestroy).
        if self._cudart is not None:
            own_event = getattr(self, "_ipc_event", None)
            if own_event is not None:
                self._try_or_warn(lambda: self._cudart.cudaEventDestroy(own_event), "destroy sender IPC event")
                self._ipc_event = None
            for evt in getattr(self, "_slot_ipc_events", []) or []:
                self._try_or_warn(lambda e=evt: self._cudart.cudaEventDestroy(e), "destroy sender slot IPC event")
            self._slot_ipc_events = []
            self._slot_ipc_event_handle_bytes = []
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
        self._sender_copy_streams = []
        self._copy_stream = None
        if torch.cuda.is_available():
            self._try_or_warn(torch.cuda.ipc_collect, "torch.cuda.ipc_collect")
        logger.info("CudaIPCConnector closed.")
