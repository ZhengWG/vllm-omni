# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Runs WriteTasks and the step device→host staging pool.

The manager owns request/slot identity and when to submit. This class
owns the staging pool, the GPU-byte budget, the copy queues, and the
single committer that writes into the CPU block pool.

Two device→host paths — the step path does not write `chunk.host` in
the committer:

    JOIN_NEXT_STEP   save already launched a whole-step device→host into
                     a staging slot and set `chunk.host` as views.
                     Committer waits that `step_d2h_event`, then writes
                     the pool.
    JOIN_ON_FINISH   committer copies the device clone → owned host
                     tensors, then writes the pool.

Async: high-priority then low-priority queues, then pool write.
Eager: submit() does wait+write inline (CPU tests / no CUDA).
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, NamedTuple

import torch

from vllm_omni.core.prefix_cache.block_pool import PrefixBlockPool
from vllm_omni.core.prefix_cache.interface import (
    OmniPrefixCacheStagingTimeoutError,
    OmniPrefixCacheUnmatchError,
    PrefixCacheConfig,
    ReqId,
    TensorName,
    Tid,
    WriteSchedule,
)

logger = logging.getLogger(__name__)


class StagingBufferHolder(NamedTuple):
    """One holder of a D2H staging-buffer slot. The slot is free when none remain.

    Not a buffer state — concurrent owners share the same slot:
    - for_step: claimed at save, released when materialize/discard consumes the ctx
    - for_task: bound before WriteTask submit, released when that task completes
    """

    kind: Literal["step", "task"]
    owner_id: int

    @classmethod
    def for_step(cls, step_id: int) -> StagingBufferHolder:
        return cls("step", step_id)

    @classmethod
    def for_task(cls, tid: int) -> StagingBufferHolder:
        return cls("task", tid)


@dataclass
class _SnapshotHolder:
    """GPU-byte-budget charge + refcount for one shared device clone.

    Does not store tensors (those live on ``_WriteChunk.tensors``). Several
    writes share one clone: charge ``nbytes`` once, release when the last
    ``tids`` drops. Immediate writes do not use this.
    """

    nbytes: int
    tids: set[Tid] = field(default_factory=set)


@dataclass
class _WriteChunk:
    """One save's slots + tensors appended onto a WriteTask.

    Controller identity is the parent ``tid``, not a request. Deferred
    writes grow by appending one chunk per save.
    """

    slots_cpu: torch.Tensor  # int64 flat row ids, in token order
    tensors: dict[TensorName, torch.Tensor]  # freeze (device or eager CPU)
    host: dict[TensorName, torch.Tensor] = field(default_factory=dict)  # host view of rows [0:n)
    # Deferred: chunks from the same save share this GPU-byte charge.
    # Immediate: None (bytes tracked on the WriteTask).
    snapshot_holder: _SnapshotHolder | None = None


@dataclass
class WriteTask:
    """One write of (slot, key) rows for a single request.

    Identity: `tid` is the handle in the manager's (slot, key) tables.
    `req_id` + `write_n` mark whose write this is and the nth time that
    request opened a write. One write may cover several keys.

    Pipeline:

        queued / device-staged
            -> copy claimed (`d2h_claimed`)
            -> `host_ready`  (D2H complete; device freeze refs may drop)
            -> scatter
            -> `done`        (in the CPU mirror; `failed` instead on error)

    How `host_ready` is reached:
    - JOIN_NEXT_STEP: `chunk.host` is a staging view set at save.
      `_copy_task` only waits `step_d2h_event` (does not write host).
    - JOIN_ON_FINISH: committer copies device→host into `chunk.host`,
      then sets `host_ready`.

    `JOIN_NEXT_STEP` starts on the high-priority copy queue; the next
    save waits `host_ready` only. `JOIN_ON_FINISH` stays on the
    low-priority queue, or not queued yet (`submit(queued=False)`),
    until finish or GPU-byte-budget pressure moves it to high-priority
    — once. Budget flush takes the unfinished task with the oldest
    `enqueued_time`.

    Concurrent readers/writers:
    - Staging readers (materialize clone, committer pool write) all wait
      the same `step_d2h_event` before touching the view.
    - A later task taking the same (slot, key) records those rows in
      `reassigned`; the old pool write skips them so the two writes do
      not overlap.
    - Append: `append_chunk` loses if copy already claimed or `done`
      — caller opens a fresh task rather than mutating a closed one.
    - `lock` covers `reassigned` / `d2h_claimed` / `append_chunk` /
      host↔freeze / `slot_to_row`. `host_ready` and `done` are their
      own events (`set_host_tensor` / `mark_host_ready` / `mark_failed` /
      `mark_done`). `scatter_rows` snapshots `reassigned`.
    """

    tid: Tid
    req_id: ReqId
    write_n: int  # 1-based: nth write opened by this request
    schedule: WriteSchedule
    chunks: list[_WriteChunk]
    # On-device clone finished on the compute stream. Copy/read streams
    # must wait this before touching `chunks[].tensors`, or they can read
    # the next CUDA-graph static-buffer overwrite.
    freeze_event: object | None = None
    nbytes: int = 0  # GPU-byte-budget accounting only; not a correctness signal
    # Moved from the low-priority queue onto the high-priority one
    # (finish / budget). Once.
    escalated: bool = False
    # Only one thread may run the copy stage. Staging: wait `step_d2h_event`.
    # Deferred: committer writes `chunk.host`.
    d2h_claimed: bool = False
    # Committer could not finish the write; manager raises on next entry.
    failed: bool = False
    # Slots this write no longer owns (a newer write took them over).
    reassigned: dict[TensorName, torch.Tensor] = field(default_factory=dict)
    # Per-write CPU event: this write's host rows are ready (`join_host_ready`).
    host_ready: threading.Event = field(default_factory=threading.Event)
    # Write into the CPU pool has finished (strictly after host_ready).
    done: threading.Event = field(default_factory=threading.Event)
    # Guards reassigned / d2h_claimed / append / host↔freeze. Not host_ready or done.
    lock: threading.Lock = field(default_factory=threading.Lock)
    # time.monotonic() at submit; cap flush picks the smallest of these.
    enqueued_time: float = field(default_factory=time.monotonic)
    # Immediate path: this write's view into the shared step staging page.
    staging_slot: int | None = None
    # Immediate path: this step's CUDA device→host event (shared). None if deferred.
    step_d2h_event: object | None = None
    # slot -> (which chunk, row in that tensor). Built on demand
    # when a write has more than one `_WriteChunk`; scatter uses slot, the tensor uses row.
    _slot_to_row: dict[int, tuple[int, int]] | None = None

    def add_reassigned(self, key: str, slots: torch.Tensor) -> None:
        with self.lock:
            prev = self.reassigned.get(key)
            self.reassigned[key] = slots.clone() if prev is None else torch.cat([prev, slots])

    def try_claim_d2h(self) -> bool:
        """Only one thread may run the copy stage. True if this caller won."""
        with self.lock:
            if self.d2h_claimed:
                return False
            self.d2h_claimed = True
            return True

    def is_done(self) -> bool:
        return self.done.is_set()

    def is_host_ready(self) -> bool:
        return self.host_ready.is_set()

    def ready_to_scatter(self) -> bool:
        return self.is_host_ready() and not self.is_done()

    def append_chunk(self, chunk: _WriteChunk, freeze_event: object | None = None) -> bool:
        """Grow this write with one save's rows. False if copy already claimed.

        `freeze_event` is stored in the same snapshot: events on one compute
        stream are ordered, so the newest also covers every earlier clone.
        """
        nbytes = sum(t.numel() * t.element_size() for t in chunk.tensors.values())
        with self.lock:
            if self.d2h_claimed or self.is_done():
                return False
            self.chunks.append(chunk)
            self._slot_to_row = None
            if freeze_event is not None:
                self.freeze_event = freeze_event
            # Shared clones are charged on the snapshot holder, not per-view.
            if chunk.snapshot_holder is None:
                self.nbytes += nbytes
            return True

    def get_host_tensor(self, si: int, key: str) -> torch.Tensor | None:
        """`chunks[si]` host if written, else device freeze. One snapshot."""
        with self.lock:
            chunk = self.chunks[si]
            src = chunk.host.get(key)
            if src is None:
                src = chunk.tensors.get(key)
            return src

    def set_host_tensor(self, rows: list[tuple[_WriteChunk, str, torch.Tensor]]) -> None:
        """Write these host tensors, drop the device freeze, set `host_ready`."""
        with self.lock:
            for chunk, key, tensor in rows:
                chunk.host[key] = tensor
            self._clear_tensors()
        self.host_ready.set()

    def mark_host_ready(self) -> None:
        """Host already written (staging views). Wait the step D2H, drop freeze."""
        if self.step_d2h_event is not None:
            self.step_d2h_event.synchronize()
        self.clear_tensors()
        self.host_ready.set()

    def mark_failed(self) -> None:
        """Unblock joiners. Host may be missing; manager fail-fasts."""
        self.failed = True
        self.clear_tensors()
        self.host_ready.set()
        self.done.set()

    def mark_done(self) -> None:
        self.done.set()

    def scatter_rows(self) -> list[tuple[TensorName, torch.Tensor, torch.Tensor]]:
        """`(key, slots, host)` to write. Omits slots in `reassigned`."""
        with self.lock:
            reassigned = {k: s.clone() for k, s in self.reassigned.items()}
        out: list[tuple[TensorName, torch.Tensor, torch.Tensor]] = []
        for chunk in self.chunks:
            for k, host in chunk.host.items():
                taken = reassigned.get(k)
                if taken is not None and taken.numel():
                    keep = ~torch.isin(chunk.slots_cpu, taken)
                    if not bool(keep.any()):
                        continue
                    out.append((k, chunk.slots_cpu[keep], host[keep]))
                else:
                    out.append((k, chunk.slots_cpu, host))
        return out

    def clear_tensors(self) -> None:
        """Drop the device freeze. Host is unchanged (staging wait / fail)."""
        with self.lock:
            self._clear_tensors()

    def _clear_tensors(self) -> None:
        for chunk in self.chunks:
            chunk.tensors = {}

    def keys(self) -> set[TensorName]:
        ks: set[TensorName] = set()
        for chunk in self.chunks:
            ks.update(chunk.tensors.keys())
        return ks

    def slot_to_row(self) -> dict[int, tuple[int, int]]:
        with self.lock:
            if self._slot_to_row is None:
                m: dict[int, tuple[int, int]] = {}
                for si, chunk in enumerate(self.chunks):
                    for ri, slot in enumerate(chunk.slots_cpu.tolist()):
                        m[slot] = (si, ri)
                self._slot_to_row = m
            return self._slot_to_row


class StagingBufferPool:
    """Reusable pinned landing zone for ONE whole-step device→host at save.

    Per-task `chunk.host` is a row-range view into a slot, so the committer
    skips per-task D2H. Slots recycle; this is not the CPU block pool.

    A slot stays busy while anyone still holds it: the step (until
    materialize/discard) and each immediate write that views the page
    (until that write is retired). Prefix hits do not hold a slot —
    they wait for scatter and read the durable pool.

    Leftover-only saves still claim a slot (empty views) so every step
    id shares this bound. A full pool waits; timeout then errors.
    """

    def __init__(self, depth: int, capacity: int):
        self.depth = depth
        self.capacity = capacity  # rows per slot; a larger step fails fast
        self._bufs: dict[TensorName, torch.Tensor] = {}  # [depth*capacity, width]
        self._busy: list[set[StagingBufferHolder]] = [set() for _ in range(depth)]
        self._slot_free_condition = threading.Condition()
        self._closed = False

    def _buf(self, key: str, width: int, dtype: torch.dtype, pin: bool) -> torch.Tensor:
        buf = self._bufs.get(key)
        if buf is None or buf.shape[-1] != width or buf.dtype != dtype:
            buf = torch.empty((self.depth * self.capacity, width), dtype=dtype, pin_memory=pin)
            self._bufs[key] = buf
        return buf

    def claim(self, holder: StagingBufferHolder, timeout: float) -> int:
        """Grab a free slot for `holder`. Waits until one is free, then
        times out. ``timeout<=0`` fails immediately if none are free.
        """
        deadline = time.monotonic() + max(0.0, timeout)
        with self._slot_free_condition:
            while True:
                if self._closed:
                    raise OmniPrefixCacheUnmatchError("staging pool shut down while waiting for a slot")
                for slot in range(self.depth):
                    if not self._busy[slot]:
                        self._busy[slot].add(holder)
                        return slot
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise OmniPrefixCacheStagingTimeoutError(f"timed out waiting for a staging slot after {timeout}s")
                self._slot_free_condition.wait(timeout=remaining)

    def bind(self, slot: int, holder: StagingBufferHolder) -> None:
        with self._slot_free_condition:
            self._busy[slot].add(holder)

    def release(self, slot: int, holder: StagingBufferHolder) -> None:
        with self._slot_free_condition:
            self._busy[slot].discard(holder)
            if not self._busy[slot]:
                self._slot_free_condition.notify()

    def close(self) -> None:
        with self._slot_free_condition:
            self._closed = True
            self._slot_free_condition.notify_all()

    def views(self, slot: int, key: str, n: int, width: int, dtype: torch.dtype, pin: bool) -> torch.Tensor:
        base = slot * self.capacity
        return self._buf(key, width, dtype, pin)[base : base + n]


@dataclass
class StepD2HClaim:
    """One whole-step landing in StagingBufferPool.

    Return of ``stage_step_host``. The manager hangs this on
    ``_StepContext`` until materialize/discard releases the step holder.
    """

    staging_slot: int  # StagingBufferPool index
    views: dict[TensorName, torch.Tensor]  # host rows [0:n)
    event: object | None = None  # torch.cuda.Event; None on eager/CPU


class OmniPrefixCacheController:
    """Staging pool + committer. Step device→host is launched at save;
    this thread waits that event (JOIN_NEXT_STEP) or copies deferred
    rows (JOIN_ON_FINISH), then writes into the CPU pool.
    """

    def __init__(self, pool: PrefixBlockPool, config: PrefixCacheConfig, eager: bool | None = None):
        self._pool = pool
        self._config = config
        self._eager = (not torch.cuda.is_available()) if eager is None else eager
        self._tasks: dict[Tid, WriteTask] = {}
        self._completed: deque[Tid] = deque()  # scattered, awaiting manager drain
        self._failed: deque[Tid] = deque()  # write failed; manager must fail-fast
        self._staged_bytes = 0
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._queue_hi: deque[Tid] = deque()  # JOIN_NEXT_STEP + forced JOIN_ON_FINISH
        self._queue_lo: deque[Tid] = deque()  # JOIN_ON_FINISH waiting for finish/budget
        self._blocked: list[Tid] = []  # host copy done, awaiting pool write
        self._shutdown = False
        self._copy_stream: torch.cuda.Stream | None = None
        self._read_stream: torch.cuda.Stream | None = None
        self._worker: threading.Thread | None = None
        self._staging_pool = StagingBufferPool(config.staging_depth, config.staging_capacity_tokens)
        if not self._eager:
            self._copy_stream = torch.cuda.Stream()
            self._read_stream = torch.cuda.Stream()
            self._worker = threading.Thread(target=self._worker_loop, name="omni-prefix-cache-committer", daemon=True)
            self._worker.start()

    # --------------------------------------------------------- step device→host staging

    def _d2h_on_stream(
        self,
        stream: torch.cuda.Stream,
        freeze_event: object | None,
        copy: Callable[[], None],
    ) -> torch.cuda.Event:
        """Issue D2H on `stream` after freeze; return the done event."""
        with torch.cuda.stream(stream):
            if freeze_event is not None:
                stream.wait_event(freeze_event)
            copy()
            ev = torch.cuda.Event()
            ev.record()
        return ev

    def stage_step_host(
        self, tensors: dict[str, torch.Tensor], n: int, freeze_event: object | None, step_holder: StagingBufferHolder
    ) -> StepD2HClaim:
        """Claim a staging slot and, when `tensors` is non-empty, launch
        ONE whole-step device→host into it.

        Leftover-only saves pass empty `tensors` and still take a slot
        (empty views) so every step id shares this bound. A full pool
        waits for materialize/discard; timeout then errors. A step
        larger than the page overflows the next slot — that is a config
        break. The caller binds tasks after submit; `step_holder` is
        released by materialize/discard via staging_release.
        """
        if tensors and n > self._staging_pool.capacity:
            raise OmniPrefixCacheUnmatchError(
                f"step has {n} tokens; staging capacity is {self._staging_pool.capacity} "
                "(size staging_capacity_tokens to max_num_batched_tokens)"
            )
        try:
            slot = self._staging_pool.claim(step_holder, self._config.staging_claim_timeout_s)
        except OmniPrefixCacheStagingTimeoutError as e:
            raise OmniPrefixCacheStagingTimeoutError(
                f"{e}; leaked slot owners or committer backlog (in_flight_tasks={len(self._tasks)})"
            ) from e
        try:
            pin = not self._eager
            views: dict[str, torch.Tensor] = {}
            event: object | None = None
            if self._eager or all(t.device.type == "cpu" for t in tensors.values()):
                for key, src in tensors.items():
                    v = self._staging_pool.views(slot, key, n, int(src.shape[-1]), src.dtype, pin)
                    v.copy_(src)
                    views[key] = v
            else:

                def _copy_to_staging() -> None:
                    for key, src in tensors.items():
                        v = self._staging_pool.views(slot, key, n, int(src.shape[-1]), src.dtype, pin)
                        v.copy_(src, non_blocking=True)
                        views[key] = v

                event = self._d2h_on_stream(self._copy_stream, freeze_event, _copy_to_staging)
            return StepD2HClaim(staging_slot=slot, views=views, event=event)
        except Exception:
            self._staging_pool.release(slot, step_holder)
            raise

    def staging_bind(self, slot: int, holder: StagingBufferHolder) -> None:
        self._staging_pool.bind(slot, holder)

    def staging_release(self, slot: int, holder: StagingBufferHolder) -> None:
        self._staging_pool.release(slot, holder)

    # ------------------------------------------------------------------ submit

    def submit(self, task: WriteTask, queued: bool = True) -> None:
        """Register a task. queued=False (deferred tasks) stays on the
        GPU clone until finish/abort or the GPU-byte budget forces a copy.

        Caller must reserve() the task bytes first (budget flush can
        block; the manager does that outside the state lock).
        """
        # Immediate (no snapshot_holder): charge/release via task.nbytes.
        # Deferred shared clone: charge the holder once at reserve().
        task.nbytes = sum(
            t.numel() * t.element_size() for s in task.chunks if s.snapshot_holder is None for t in s.tensors.values()
        )
        task.enqueued_time = time.monotonic()
        with self._lock:
            self._tasks[task.tid] = task
        if self._eager:
            if queued:
                self._run_eager(task)
            return
        if queued:
            with self._wake:
                (self._queue_hi if task.schedule is WriteSchedule.JOIN_NEXT_STEP else self._queue_lo).append(task.tid)
                self._wake.notify_all()

    def append_chunk(self, task: WriteTask, chunk: _WriteChunk, freeze_event: object | None = None) -> bool:
        return task.append_chunk(chunk, freeze_event)

    def pin_snapshot_holder(self, holder: _SnapshotHolder, tid: int) -> None:
        """Record that ``tid`` holds a view of this step's deferred snapshot."""
        with self._lock:
            holder.tids.add(tid)

    def reserve(self, nbytes: int, exclude: set[int] | None = None) -> None:
        """Reserve GPU-clone bytes; blocking flush happens here, so callers
        must not hold the manager's state lock."""
        self._reserve_bytes(nbytes, exclude=exclude)

    def _release_staged_bytes(self, task: WriteTask) -> None:
        """Drop this task's GPU-byte-budget charge.

        Shared deferred clones release only when the last holder tid drops.
        Immediate tasks (no ``snapshot_holder``) still release ``task.nbytes``.
        """
        tickets: list[_SnapshotHolder] = []
        seen: set[int] = set()
        for chunk in task.chunks:
            holder = chunk.snapshot_holder
            if holder is not None and id(holder) not in seen:
                seen.add(id(holder))
                tickets.append(holder)
        with self._wake:
            if tickets:
                for holder in tickets:
                    if task.tid in holder.tids:
                        holder.tids.discard(task.tid)
                        if not holder.tids:
                            self._staged_bytes -= holder.nbytes
            else:
                self._staged_bytes -= task.nbytes

    def _reserve_bytes(self, nbytes: int, exclude: set[int] | None = None) -> None:
        # Cap backpressure: force-flush oldest pending tasks until under
        # budget. Bounded block: their D2H has usually long completed.
        exclude = exclude or set()
        while True:
            with self._lock:
                pending = [tid for tid, t in self._tasks.items() if not t.is_done() and tid not in exclude]
                if self._staged_bytes + nbytes <= self._config.gpu_staging_bytes or not pending:
                    # Under budget or no pending tasks; admit reservation.
                    self._staged_bytes += nbytes
                    return
                oldest = min(pending, key=lambda tid: self._tasks[tid].enqueued_time)
            logger.warning("omni prefix cache: staging cap hit, force-flushing task %d", oldest)
            self.escalate([oldest])
            self.join([oldest])

    # ------------------------------------------------------------- lifecycle

    def escalate(self, tids: list[int]) -> None:
        if self._eager:
            for tid in tids:
                task = self._tasks.get(tid)
                if task is not None and not task.is_done():
                    self._run_eager(task)
            return
        with self._wake:
            for tid in tids:
                task = self._tasks.get(tid)
                if task is None or task.escalated or task.is_done():
                    continue
                task.escalated = True
                try:
                    self._queue_lo.remove(tid)
                    self._queue_hi.appendleft(tid)
                except ValueError:
                    # Not in the lazy queue: either an unqueued deferred
                    # task (queue it now) or already claimed/queued-hi.
                    if tid not in self._queue_hi and tid not in self._blocked and not task.d2h_claimed:
                        self._queue_hi.appendleft(tid)
            self._wake.notify_all()

    def join(self, tids: list[int]) -> None:
        """Block until each task has finished scatter (or failed)."""
        for tid in tids:
            task = self._tasks.get(tid)
            if task is not None:
                task.done.wait()

    def join_host_ready(self, tids: list[int]) -> None:
        """Block until each task's D2H is complete (`host_ready`).

        Staging: committer has waited `step_d2h_event`. Deferred: committer
        has written `chunk.host`. Does not wait scatter.
        """
        for tid in tids:
            task = self._tasks.get(tid)
            if task is not None:
                task.host_ready.wait()

    def drain_completed(self) -> list[int]:
        """Pop scattered tasks from `_completed` and drop them from `_tasks`.

        WriteTask holders release HERE — the same locked drain that flips
        state to committed — not at scatter: a hit plan that still sees rows
        in-transit must be able to hold the slot before it is reclaimable.
        """
        out: list[int] = []
        with self._lock:
            while self._completed:
                out.append(self._completed.popleft())
            tasks = [self._tasks.pop(tid, None) for tid in out]
        for task in tasks:
            if task is not None and task.staging_slot is not None:
                self._staging_pool.release(task.staging_slot, StagingBufferHolder.for_task(task.tid))
        return out

    def drain_failed(self) -> list[int]:
        """Pop failed task ids from `_failed`. Does not drop `_tasks`."""
        out: list[int] = []
        with self._lock:
            while self._failed:
                out.append(self._failed.popleft())
        return out

    def get_task(self, tid: int) -> WriteTask | None:
        return self._tasks.get(tid)

    def shutdown(self) -> None:
        self._staging_pool.close()
        with self._wake:
            self._shutdown = True
            self._wake.notify_all()
        if self._worker is not None:
            self._worker.join(timeout=5.0)

    # ------------------------------------------------------------ fetch_host

    @torch.inference_mode()
    def fetch_host(self, task: WriteTask, slots: torch.Tensor, key: str) -> torch.Tensor:
        """Rows for `slots` of one in-flight JOIN_ON_FINISH task.

        `_slot_ref` puts JOIN_NEXT_STEP tids in `join_tids` (join then
        pool). This path reads committer-written `chunk.host`, or the
        device freeze if that D2H has not landed.
        """
        if task.step_d2h_event is not None:
            task.step_d2h_event.synchronize()
        return self._rows_from(task, slots, key)

    def _rows_from(self, task: WriteTask, slots: torch.Tensor, key: str) -> torch.Tensor:
        """Map `slots` to rows across one or more `_WriteChunk`s; preserve caller order."""
        s2r = task.slot_to_row()
        idx = [s2r[int(s)] for s in slots.tolist()]
        parts: list[torch.Tensor] = []
        order: list[int] = []
        pos = 0
        rows_groups: dict[int, list[tuple[int, int]]] = {}
        for si, ri in idx:
            rows_groups.setdefault(si, []).append((pos, ri))
            pos += 1
        for si, items in rows_groups.items():
            src = task.get_host_tensor(si, key)
            if src is None:
                continue
            rows_idx = torch.tensor([ri for _, ri in items], dtype=torch.long)
            picked = self._slice_rows(task, src, rows_idx, host=(src.device.type == "cpu"))
            parts.append(picked)
            order.extend(p for p, _ in items)
        if not parts:
            raise KeyError(f"key {key} not present in task {task.tid}")
        cat = torch.cat(parts, dim=0)
        out = torch.empty_like(cat)
        out[torch.tensor(order, dtype=torch.long)] = cat
        return out

    def _slice_rows(self, task: WriteTask, src: torch.Tensor, rows_idx: torch.Tensor, host: bool) -> torch.Tensor:
        n = int(rows_idx.numel())
        # Ascending-run check without materializing an arange: endpoints plus
        # a monotonic diff are enough, and the common case is one long run.
        contiguous = (
            n > 0 and int(rows_idx[-1]) - int(rows_idx[0]) == n - 1 and (n < 2 or bool((rows_idx.diff() == 1).all()))
        )

        def _pick() -> torch.Tensor:
            return src[rows_idx[0] : rows_idx[0] + n] if contiguous else src.index_select(0, rows_idx)

        if host or src.device.type == "cpu":
            return _pick()
        if self._read_stream is None:
            return _pick().detach().cpu()
        copied: list[torch.Tensor] = []

        def _copy_to_cpu() -> None:
            copied.append(_pick().to("cpu", non_blocking=True))

        ev = self._d2h_on_stream(self._read_stream, task.freeze_event, _copy_to_cpu)
        ev.synchronize()
        return copied[0]

    # ------------------------------------------------------------ eager mode

    @torch.inference_mode()
    def _run_eager(self, task: WriteTask) -> None:
        if not task.try_claim_d2h():
            if task.ready_to_scatter():
                self._scatter(task)
            return
        if task.schedule is WriteSchedule.JOIN_NEXT_STEP:
            # Host is already a staging view (copied at save). Drop freeze.
            # `step_d2h_event` is None on CPU; `mark_host_ready` skips wait.
            task.mark_host_ready()
        else:
            # JOIN_ON_FINISH: no copy stream; freeze → owned host inline.
            task.set_host_tensor(
                [
                    (chunk, k, t.detach().cpu() if t.device.type != "cpu" else t.clone())
                    for chunk in task.chunks
                    for k, t in chunk.tensors.items()
                ]
            )
        self._release_staged_bytes(task)
        self._scatter(task)

    # ---------------------------------------------------------- worker loop

    def _worker_loop(self) -> None:
        # A dying committer would strand every join() forever, so the loop
        # never propagates: it fails the offending task and keeps serving.
        while True:
            tid = None
            try:
                with self._wake:
                    while not self._shutdown and not self._queue_hi and not self._queue_lo:
                        if self._blocked:
                            break
                        # submit / escalate / shutdown all notify.
                        self._wake.wait()
                    if self._shutdown and not self._queue_hi and not self._queue_lo and not self._blocked:
                        return
                    if self._queue_hi:
                        tid = self._queue_hi.popleft()
                    elif self._queue_lo:
                        tid = self._queue_lo.popleft()
                if tid is not None:
                    task = self._tasks.get(tid)
                    if task is not None:
                        self._copy_task(task)
                        with self._wake:
                            if tid not in self._blocked:
                                self._blocked.append(tid)
                self._scatter_host_ready()
            except BaseException:
                logger.exception("omni prefix cache committer failed on task %s; releasing waiters", tid)
                self._fail_task(tid)

    @torch.inference_mode()
    def _copy_task(self, task: WriteTask) -> None:
        """Reach `host_ready`. Staging: wait the save-time D2H event.
        Deferred: this is the D2H into owned `chunk.host` tensors.
        """
        if not task.try_claim_d2h():
            return
        if task.schedule is WriteSchedule.JOIN_NEXT_STEP:
            # `chunk.host` is already a staging view; D2H flew at save.
            # `mark_host_ready` waits `step_d2h_event` if one was recorded.
            # No per-task copy. Scatter is `_scatter_host_ready`.
            task.mark_host_ready()
            self._release_staged_bytes(task)
            return
        chunk_bytes = self._config.copy_chunk_bytes
        pending_host: list[tuple[_WriteChunk, str, torch.Tensor]] = []
        pending_cats: list[tuple[_WriteChunk, str, list[torch.Tensor]]] = []
        with torch.cuda.stream(self._copy_stream):
            if task.freeze_event is not None:
                self._copy_stream.wait_event(task.freeze_event)
            for chunk in task.chunks:
                for k, src in chunk.tensors.items():
                    if src.numel() * src.element_size() > chunk_bytes:
                        rows_per_chunk = max(1, chunk_bytes // max(1, src.shape[-1] * src.element_size()))
                        parts = [
                            src[start : start + rows_per_chunk].to("cpu", non_blocking=True)
                            for start in range(0, src.shape[0], rows_per_chunk)
                        ]
                        pending_cats.append((chunk, k, parts))
                    else:
                        pending_host.append((chunk, k, src.to("cpu", non_blocking=True)))
            ev = torch.cuda.Event()
            ev.record()
        ev.synchronize()
        task.set_host_tensor([(chunk, k, torch.cat(parts, dim=0)) for chunk, k, parts in pending_cats] + pending_host)
        self._release_staged_bytes(task)

    def _fail_task(self, tid: int | None) -> None:
        """Release waiters for a task the committer could not complete.

        Idempotent, and only releases bytes the copy stage has not already
        released: a raise AFTER a successful _copy_task (host_ready set)
        must not subtract this task's bytes a second time.
        """
        task = self._tasks.get(tid) if tid is not None else None
        if task is None or task.is_done():
            return
        if not task.is_host_ready():
            self._release_staged_bytes(task)
        with self._wake:
            if tid in self._blocked:
                self._blocked.remove(tid)
        task.mark_failed()
        if task.staging_slot is not None:
            self._staging_pool.release(task.staging_slot, StagingBufferHolder.for_task(task.tid))
        with self._lock:
            # Publish the failure: rows behind already-published block hashes
            # never landed, which the manager must raise on (hiding it would
            # crash on every future hit that touches these slots).
            self._failed.append(task.tid)

    @torch.inference_mode()
    def _scatter_host_ready(self) -> None:
        """Write `_blocked` tasks whose `host_ready` is set into the CPU pool."""
        with self._wake:
            ready = [tid for tid in self._blocked if (t := self._tasks.get(tid)) and t.is_host_ready()]
            for tid in ready:
                self._blocked.remove(tid)
        for tid in ready:
            task = self._tasks.get(tid)
            if task is None:
                continue
            try:
                self._scatter(task)
            except BaseException:
                # Attribute the failure to THIS task: letting it propagate
                # would fail whichever entry the worker loop happened to be
                # copying, double-release its bytes, and strand this one's
                # join() forever.
                logger.exception("omni prefix cache scatter failed on task %s; releasing waiters", tid)
                self._fail_task(tid)

    @torch.inference_mode()
    def _scatter(self, task: WriteTask) -> None:
        for key, slots, host in task.scatter_rows():
            self._pool.write(key, slots, host)
        task.mark_done()
        with self._lock:
            self._completed.append(task.tid)
