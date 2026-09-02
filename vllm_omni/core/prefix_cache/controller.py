# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Runs WriteTasks and the step D2H staging pool.

The manager owns request/slot identity and when to submit. This class
owns the staging pool, the GPU-byte cap, the copy queues, and the
single committer that scatters into the CPU block pool.

Two D2H paths — there is no per-task `publish_host` on the step path:

    JOIN_NEXT_STEP   save already launched a whole-step D2H into a
                     staging slot and hung `seg.host` as views.
                     Committer waits that `host_event`, then scatters.
    JOIN_ON_FINISH   committer copies GPU freeze → owned host tensors,
                     then scatters.

Async: hi/lo queues, then scatter.
Eager: submit() does wait+scatter inline (CPU tests / no CUDA).
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Literal, NamedTuple

import torch

from vllm_omni.core.prefix_cache.block_pool import PrefixBlockPool
from vllm_omni.core.prefix_cache.interface import (
    OmniPrefixCacheUnmatchError,
    PrefixCacheConfig,
    WriteSchedule,
)

logger = logging.getLogger(__name__)


class StagingBufferHolder(NamedTuple):
    """One holder of a D2H staging-buffer slot. The slot is free when none remain.

    Not a buffer state — concurrent owners share the same slot:
    - step: claimed at save, released when materialize/discard consumes the ctx
    - task: bound before WriteTask submit, released when that tid drains
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
class _GpuFreezeAlloc:
    """One unique GPU freeze allocation (a whole-step clone).

    C→1 hangs per-req views on this storage. Charge ``_staged_bytes``
    once; release when the last holder tid drops. Same refcount idea as
    ``StagingBufferHolder``, but this is the GPU freeze, not a host slot.
    """

    nbytes: int
    holders: set[int] = field(default_factory=set)


@dataclass
class _Segment:
    """One contiguous save's rows: slots + per-key frozen tensors."""

    slots_cpu: torch.Tensor  # int64 flat row ids, in token order
    tensors: dict[str, torch.Tensor]  # frozen (GPU) or eager CPU tensors
    # JOIN_NEXT_STEP: view into the staging page, hung at save (D2H may
    # still be in flight; wait `host_event`). JOIN_ON_FINISH: owned CPU
    # tensor written by the committer after its D2H.
    host: dict[str, torch.Tensor] = field(default_factory=dict)
    # Shared C→1 clone this view hangs on (deferred). None on the
    # immediate path, which still charges/releases via ``task.nbytes``.
    gpu_alloc: _GpuFreezeAlloc | None = None


@dataclass
class WriteTask:
    """One write of (slot, key) rows for a single request.

    Identity: `tid` is the handle in the manager's (slot, key) tables.
    `req_id` + `write_n` mark whose write this is and the nth time that
    request opened a write. One task may carry several keys.

    Pipeline (one direction, no rollback)::

        queued / GPU-staged
            -> copy claimed (`d2h_claimed`)
            -> `host_ready`  (D2H complete; GPU freeze refs may drop)
            -> scatter
            -> `done`        (in the CPU mirror; `failed` instead on error)

    How `host_ready` is reached:
    - JOIN_NEXT_STEP: `seg.host` is a staging view hung at save.
      `_copy_task` only waits `host_event` (does not write host).
    - JOIN_ON_FINISH: committer D2H writes owned tensors into
      `seg.host`, then sets `host_ready`.

    `JOIN_NEXT_STEP` starts on the hi copy queue and is joined at the
    next save (`host_ready` only). `JOIN_ON_FINISH` stays on the lo
    queue, or unqueued (deferred `submit(queued=False)`), until finish
    or cap pressure `escalate`s it onto hi — once. Cap flush takes the
    unfinished task with the oldest `enqueued_time`.

    Concurrent readers/writers:
    - Staging readers (materialize clone, fetch_host, committer scatter)
      all wait the same `host_event` before touching the view.
      `d2h_claimed` is the single-claimer so only one `_copy_task`
      runs; it does not publish host on this path.
    - Remount: a later task taking the same (slot, key) pushes those
      rows into `skip`; the old scatter omits them so the two writes
      stay disjoint (no join edge).
    - Append: `append_segment` loses if copy already claimed or `done`
      — caller opens a fresh task rather than mutating a closed one.
    - `lock` covers `skip` / `d2h_claimed` / segment append only.
      `host_ready` and `done` are their own events.
    """

    tid: int
    req_id: str
    write_n: int  # 1-based: nth write opened by this request
    schedule: WriteSchedule
    segments: list[_Segment]
    # Compute-stream D2D freeze completion. Copy/read streams must wait
    # this before touching `segments[].tensors`, or they can read the
    # next CUDA-graph static-buffer overwrite.
    freeze_event: object | None = None  # torch.cuda.Event
    nbytes: int = 0  # GPU-staging cap accounting only; not a correctness signal
    # Promoted from the lazy queue onto the hi queue (finish / cap). Once.
    escalated: bool = False
    # Single claimer for the copy stage. Staging: wait `host_event`.
    # Deferred: committer writes `seg.host`. Not a host publisher on
    # the staging path.
    d2h_claimed: bool = False
    # Committer could not land the write; manager fail-fasts on next entry.
    failed: bool = False
    # (slot, key) pairs reassigned to a newer task: the scatter skips them,
    # making old and new mirror writes disjoint (no ordering edges needed).
    skip: dict[str, torch.Tensor] = field(default_factory=dict)
    # D2H complete (staging event done, or deferred host tensors written).
    # GPU freeze refs may drop; join_host_ready can return.
    host_ready: threading.Event = field(default_factory=threading.Event)
    # Scatter into the CPU mirror has finished (strictly after host_ready).
    done: threading.Event = field(default_factory=threading.Event)
    # Guards skip / d2h_claimed / segment append. Not host_ready or done.
    lock: threading.Lock = field(default_factory=threading.Lock)
    # time.monotonic() at submit; cap flush picks the smallest of these.
    enqueued_time: float = field(default_factory=time.monotonic)
    # uses staging views and a shared D2H (host_event) instead of per-task copies.
    staging_slot: int | None = None
    host_event: object | None = None  # torch.cuda.Event of the step D2H
    # slot -> (which segment, row in that segment's tensor). Built on demand
    # for multi-segment tasks; scatter uses slot, the tensor uses row.
    _slot_to_row: dict[int, tuple[int, int]] | None = None

    def num_rows(self) -> int:
        return sum(int(s.slots_cpu.numel()) for s in self.segments)

    def add_skip(self, key: str, slots: torch.Tensor) -> None:
        with self.lock:
            prev = self.skip.get(key)
            self.skip[key] = slots.clone() if prev is None else torch.cat([prev, slots])

    def keys(self) -> set[str]:
        ks: set[str] = set()
        for s in self.segments:
            ks.update(s.tensors.keys())
        return ks

    def slot_to_row(self) -> dict[int, tuple[int, int]]:
        if self._slot_to_row is None:
            m: dict[int, tuple[int, int]] = {}
            for si, seg in enumerate(self.segments):
                for ri, slot in enumerate(seg.slots_cpu.tolist()):
                    m[slot] = (si, ri)
            self._slot_to_row = m
        return self._slot_to_row


class StagingBufferPool:
    """Reusable pinned landing zone for ONE whole-step D2H at save.

    Per-task `seg.host` is a row-range view into a slot, so the committer
    skips per-task D2H. Readers (materialize, committer, fetch_host) all
    wait the same `host_event` before touching the view. Slots recycle;
    this is not the CPU block pool.

    A slot is busy while any StagingBufferHolder is in `_busy`: one per
    WriteTask (freed when the manager drains its completion) and one for
    the step context (freed when materialize/discard consumes it). Hit
    reads of JOIN_NEXT_STEP rows join scatter and then read the pool, so
    they do not pin this slot. No free slot is a contract break
    (fail-fast), not a second D2H path.
    """

    def __init__(self, depth: int, capacity: int):
        self.depth = depth
        self.capacity = capacity  # rows per slot; a larger step fails fast
        self._bufs: dict[str, torch.Tensor] = {}  # key -> [depth*capacity, width]
        self._busy: list[set[StagingBufferHolder]] = [set() for _ in range(depth)]
        self._lock = threading.Lock()

    def _buf(self, key: str, width: int, dtype: torch.dtype, pin: bool) -> torch.Tensor:
        buf = self._bufs.get(key)
        if buf is None or buf.shape[-1] != width or buf.dtype != dtype:
            buf = torch.empty((self.depth * self.capacity, width), dtype=dtype, pin_memory=pin)
            self._bufs[key] = buf
        return buf

    def try_claim(self, holder: StagingBufferHolder) -> int | None:
        """Grab a free slot for `holder` (the step holder); None if all busy."""
        with self._lock:
            for slot in range(self.depth):
                if not self._busy[slot]:
                    self._busy[slot].add(holder)
                    return slot
        return None

    def bind(self, slot: int, holder: StagingBufferHolder) -> None:
        with self._lock:
            self._busy[slot].add(holder)

    def release(self, slot: int, holder: StagingBufferHolder) -> None:
        with self._lock:
            self._busy[slot].discard(holder)

    def views(self, slot: int, key: str, n: int, width: int, dtype: torch.dtype, pin: bool) -> torch.Tensor:
        base = slot * self.capacity
        return self._buf(key, width, dtype, pin)[base : base + n]


class OmniPrefixCacheController:
    """Staging pool + committer. Step D2H is launched at save; this
    thread waits that event (JOIN_NEXT_STEP) or copies deferred rows
    (JOIN_ON_FINISH), then scatters into the CPU pool.
    """

    def __init__(self, pool: PrefixBlockPool, config: PrefixCacheConfig, eager: bool | None = None):
        self._pool = pool
        self._config = config
        self._eager = (not torch.cuda.is_available()) if eager is None else eager
        self._tasks: dict[int, WriteTask] = {}
        self._completed: deque[int] = deque()  # scattered, awaiting manager drain
        self._failed: deque[int] = deque()  # write failed; manager must fail-fast
        self._staged_bytes = 0
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._queue_hi: deque[int] = deque()  # join-next-step + escalated
        self._queue_lo: deque[int] = deque()  # join-on-finish trickle
        self._blocked: list[int] = []  # copy done, awaiting scatter
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

    # --------------------------------------------------------- step D2H staging

    def stage_step_host(
        self, tensors: dict[str, torch.Tensor], n: int, freeze_event: object | None, step_holder: StagingBufferHolder
    ) -> tuple[int, dict[str, torch.Tensor], object | None]:
        """Launch ONE whole-step D2H into a staging slot, ahead of consumption.

        Returns (slot, key -> host view of rows [0:n), d2h event). The
        caller only invokes this for a non-empty packed step. Too large a
        step, or no free slot, is a contract break — not a second D2H
        path. The caller binds task holders after submit; `step_holder`
        is released by materialize/discard via staging_release.
        """
        if not tensors or n <= 0:
            raise OmniPrefixCacheUnmatchError(
                f"stage_step_host called with n={n} keys={list(tensors)}; only a non-empty packed step may launch D2H"
            )
        if n > self._staging_pool.capacity:
            raise OmniPrefixCacheUnmatchError(
                f"step has {n} tokens; staging capacity is {self._staging_pool.capacity} "
                "(size staging_capacity_tokens to max_num_batched_tokens)"
            )
        slot = self._staging_pool.try_claim(step_holder)
        if slot is None:
            raise OmniPrefixCacheUnmatchError(
                "D2H staging pool exhausted; unconsumed steps, leaked holders, "
                f"or committer backlog (in_flight_tasks={len(self._tasks)})"
            )
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
                assert self._copy_stream is not None
                with torch.cuda.stream(self._copy_stream):
                    if freeze_event is not None:
                        self._copy_stream.wait_event(freeze_event)
                    for key, src in tensors.items():
                        v = self._staging_pool.views(slot, key, n, int(src.shape[-1]), src.dtype, pin)
                        v.copy_(src, non_blocking=True)
                        views[key] = v
                    event = torch.cuda.Event()
                    event.record()
            return slot, views, event
        except Exception:
            self._staging_pool.release(slot, step_holder)
            raise

    def staging_bind(self, slot: int, holder: StagingBufferHolder) -> None:
        self._staging_pool.bind(slot, holder)

    def staging_release(self, slot: int, holder: StagingBufferHolder) -> None:
        self._staging_pool.release(slot, holder)

    # ------------------------------------------------------------------ submit

    def submit(self, task: WriteTask, queued: bool = True) -> None:
        """Register a task. queued=False (deferred tasks) stays GPU-staged
        until escalated or cap-flushed.

        Caller must reserve() the task bytes first (cap flush can block;
        the manager does that outside the state lock).
        """
        # Immediate (no gpu_alloc): charge/release via task.nbytes.
        # Deferred C→1 views: charge the shared alloc once at reserve().
        task.nbytes = sum(
            t.numel() * t.element_size() for s in task.segments if s.gpu_alloc is None for t in s.tensors.values()
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

    def append_segment(self, task: WriteTask, seg: _Segment) -> bool:
        """Grow a deferred (JOIN_ON_FINISH) task with one step's rows.

        Returns False when the task already started copying (cap flush or a
        finish/conflict escalation got there first); the caller then opens a
        fresh task instead of mutating a closed one. Bytes for this segment
        must already be reserve()'d; on False they stay reserved for the
        replacement submit.
        """
        nbytes = sum(t.numel() * t.element_size() for t in seg.tensors.values())
        with task.lock:
            if task.d2h_claimed or task.done.is_set():
                return False
            task.segments.append(seg)
            task._slot_to_row = None
            # Shared clones are charged on the alloc, not per-view.
            if seg.gpu_alloc is None:
                task.nbytes += nbytes
        return True

    def pin_gpu_freeze(self, alloc: _GpuFreezeAlloc, tid: int) -> None:
        """Record that ``tid`` holds a view of this step's deferred clone."""
        with self._lock:
            alloc.holders.add(tid)

    def reserve(self, nbytes: int, exclude: set[int] | None = None) -> None:
        """Public cap reservation; blocking flush happens here, so callers
        must not hold the manager's state lock."""
        self._reserve_bytes(nbytes, exclude=exclude)

    def _release_staged_bytes(self, task: WriteTask) -> None:
        """Drop this task's GPU-freeze charge.

        Shared C→1 clones release only when the last holder tid drops.
        Immediate tasks (no ``gpu_alloc``) still release ``task.nbytes``.
        """
        allocs: list[_GpuFreezeAlloc] = []
        seen: set[int] = set()
        for seg in task.segments:
            alloc = seg.gpu_alloc
            if alloc is not None and id(alloc) not in seen:
                seen.add(id(alloc))
                allocs.append(alloc)
        with self._wake:
            if allocs:
                for alloc in allocs:
                    if task.tid in alloc.holders:
                        alloc.holders.discard(task.tid)
                        if not alloc.holders:
                            self._staged_bytes -= alloc.nbytes
            else:
                self._staged_bytes -= task.nbytes

    def _reserve_bytes(self, nbytes: int, exclude: set[int] | None = None) -> None:
        # Cap backpressure: force-flush oldest pending tasks until under
        # budget. Bounded block: their D2H has usually long completed.
        exclude = exclude or set()
        while True:
            with self._lock:
                pending = [tid for tid, t in self._tasks.items() if not t.done.is_set() and tid not in exclude]
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
                if task is not None and not task.done.is_set():
                    self._run_eager(task)
            return
        with self._wake:
            for tid in tids:
                task = self._tasks.get(tid)
                if task is None or task.escalated or task.done.is_set():
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

        Staging: committer has waited `host_event`. Deferred: committer
        has written `seg.host`. Does not wait scatter.
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
        with self._wake:
            self._shutdown = True
            self._wake.notify_all()
        if self._worker is not None:
            self._worker.join(timeout=5.0)

    # ------------------------------------------------------------ fetch_host

    @torch.inference_mode()
    def fetch_host(self, task: WriteTask, slots: torch.Tensor, key: str) -> torch.Tensor:
        """Rows for `slots` of one in-flight task.

        JOIN_NEXT_STEP hangs `seg.host` as a staging view at save, before
        D2H finishes. Wait that step's `host_event` (same event materialize
        and the committer wait; record-once, wait-many) so a prefetch
        cannot slice a half-written page. No `host_event` means either
        eager (copy already done) or deferred (empty host falls through
        to a sync D2H from the GPU freeze).
        """
        if task.host_event is not None:
            task.host_event.synchronize()
        if len(task.segments) == 1:
            return self._rows_single_segment(task, slots, key)
        s2r = task.slot_to_row()
        idx = [s2r[int(s)] for s in slots.tolist()]
        return self._rows_from(task, idx, key)

    def _rows_from(self, task: WriteTask, idx: list[tuple[int, int]], key: str) -> torch.Tensor:
        # Group by segment, preserve caller order via position bookkeeping.
        parts: list[torch.Tensor] = []
        order: list[int] = []
        pos = 0
        seg_groups: dict[int, list[tuple[int, int]]] = {}
        for si, ri in idx:
            seg_groups.setdefault(si, []).append((pos, ri))
            pos += 1
        for si, items in seg_groups.items():
            seg = task.segments[si]
            # Staging: host[k] is a view hung at save (waited above).
            # Deferred: host[k] is set only after the committer D2H; else
            # the GPU freeze. (`or` would bool() a Tensor and raise.)
            src = seg.host.get(key)
            if src is None:
                src = seg.tensors.get(key)
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

    def _rows_single_segment(self, task: WriteTask, slots: torch.Tensor, key: str) -> torch.Tensor:
        """Map `slots` to rows in this task's one packed segment."""
        seg = task.segments[0]
        # Staging: host[k] is a view hung at save (waited above).
        # Deferred: host[k] is set only after the committer D2H; else
        # the GPU freeze. (`or` would bool() a Tensor and raise.)
        src = seg.host.get(key)
        if src is None:
            src = seg.tensors.get(key)
        if src is None:
            raise KeyError(f"key {key} not present in task {task.tid}")

        seg_slots = seg.slots_cpu
        n = int(slots.numel())
        # Same token order: this-step rows or a prefix hit → 0..n-1.
        if n <= seg_slots.numel() and bool(torch.equal(slots, seg_slots[:n])):
            rows_idx = torch.arange(n, dtype=torch.int64)
        else:
            loc = {int(s): i for i, s in enumerate(seg_slots.tolist())}
            try:
                rows_idx = torch.tensor([loc[int(s)] for s in slots.tolist()], dtype=torch.int64)
            except KeyError:
                raise KeyError(f"slots not covered by task {task.tid}") from None
        return self._slice_rows(task, src, rows_idx, host=(src.device.type == "cpu"))

    def _slice_rows(self, task: WriteTask, src: torch.Tensor, rows_idx: torch.Tensor, host: bool) -> torch.Tensor:
        n = int(rows_idx.numel())
        # Ascending-run check without materializing an arange: endpoints plus
        # a monotonic diff are enough, and the common case is one long run.
        contiguous = (
            n > 0 and int(rows_idx[-1]) - int(rows_idx[0]) == n - 1 and (n < 2 or bool((rows_idx.diff() == 1).all()))
        )
        if host or src.device.type == "cpu":
            return src[rows_idx[0] : rows_idx[0] + rows_idx.numel()] if contiguous else src.index_select(0, rows_idx)
        if self._read_stream is None:
            picked = src[rows_idx[0] : rows_idx[0] + rows_idx.numel()] if contiguous else src.index_select(0, rows_idx)
            return picked.detach().cpu()
        with torch.cuda.stream(self._read_stream):
            if task.freeze_event is not None:
                self._read_stream.wait_event(task.freeze_event)
            picked = src[rows_idx[0] : rows_idx[0] + rows_idx.numel()] if contiguous else src.index_select(0, rows_idx)
            cpu = picked.to("cpu", non_blocking=True)
            ev = torch.cuda.Event()
            ev.record()
        ev.synchronize()
        return cpu

    # ------------------------------------------------------------ eager mode

    @torch.inference_mode()
    def _run_eager(self, task: WriteTask) -> None:
        with task.lock:
            if task.d2h_claimed:
                if not task.done.is_set() and task.host_ready.is_set():
                    self._scatter(task)
                return
            task.d2h_claimed = True
        if task.staging_slot is None:
            for seg in task.segments:
                for k, t in seg.tensors.items():
                    seg.host[k] = t.detach().cpu() if t.device.type != "cpu" else t.clone()
        task.host_ready.set()
        self._release_staged_bytes(task)
        for seg in task.segments:
            seg.tensors = {}
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
                self._scatter_ready()
            except BaseException:
                logger.exception("omni prefix cache committer failed on task %s; releasing waiters", tid)
                self._fail_task(tid)

    @torch.inference_mode()
    def _copy_task(self, task: WriteTask) -> None:
        """Reach `host_ready`. Staging: wait the save-time D2H event.
        Deferred: this is the D2H into owned `seg.host` tensors.
        """
        with task.lock:
            if task.d2h_claimed:
                return
            task.d2h_claimed = True
        if task.host_event is not None:
            # Staging: seg.host was pre-filled with slot views and the
            # whole step's D2H was launched at save. Wait that one event
            # (shared across the step's tasks; idempotent), release the GPU
            # refs, done.
            task.host_event.synchronize()
            task.host_ready.set()
            self._release_staged_bytes(task)
            for seg in task.segments:
                seg.tensors = {}
            return
        assert self._copy_stream is not None
        chunk_bytes = self._config.copy_chunk_bytes
        pending_host: list[tuple[_Segment, str, torch.Tensor]] = []
        pending_cats: list[tuple[_Segment, str, list[torch.Tensor]]] = []
        with torch.cuda.stream(self._copy_stream):
            if task.freeze_event is not None:
                self._copy_stream.wait_event(task.freeze_event)
            for seg in task.segments:
                for k, src in seg.tensors.items():
                    if task.schedule is WriteSchedule.JOIN_ON_FINISH and src.numel() * src.element_size() > chunk_bytes:
                        rows_per_chunk = max(1, chunk_bytes // max(1, src.shape[-1] * src.element_size()))
                        parts = [
                            src[start : start + rows_per_chunk].to("cpu", non_blocking=True)
                            for start in range(0, src.shape[0], rows_per_chunk)
                        ]
                        pending_cats.append((seg, k, parts))
                    else:
                        pending_host.append((seg, k, src.to("cpu", non_blocking=True)))
            ev = torch.cuda.Event()
            ev.record()
        ev.synchronize()
        # Publish host only after D2H; readers treat host[k] as complete.
        for seg, k, parts in pending_cats:
            seg.host[k] = torch.cat(parts, dim=0)
        for seg, k, cpu in pending_host:
            seg.host[k] = cpu
        task.host_ready.set()
        self._release_staged_bytes(task)
        for seg in task.segments:
            seg.tensors = {}

    def _fail_task(self, tid: int | None) -> None:
        """Release waiters for a task the committer could not complete.

        Idempotent, and only releases bytes the copy stage has not already
        released: a raise AFTER a successful _copy_task (host_ready set)
        must not subtract this task's bytes a second time.
        """
        task = self._tasks.get(tid) if tid is not None else None
        if task is None or task.done.is_set():
            return
        task.failed = True
        if not task.host_ready.is_set():
            self._release_staged_bytes(task)
        with self._wake:
            if tid in self._blocked:
                self._blocked.remove(tid)
        for seg in task.segments:
            seg.tensors = {}
        task.host_ready.set()
        task.done.set()
        if task.staging_slot is not None:
            self._staging_pool.release(task.staging_slot, StagingBufferHolder.for_task(task.tid))
        with self._lock:
            # Publish the failure: rows behind already-published block hashes
            # never landed, which the manager must turn into a fail-fast (a
            # silently poisoned span would crash on every future hit instead).
            self._failed.append(task.tid)

    @torch.inference_mode()
    def _scatter_ready(self) -> None:
        with self._wake:
            ready = [tid for tid in self._blocked if (t := self._tasks.get(tid)) and t.host_ready.is_set()]
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
        with task.lock:
            # Slots a later task took over (block reuse); do not write those.
            skip = {k: s.clone() for k, s in task.skip.items()}
        for seg in task.segments:
            for k, host in seg.host.items():
                skipped = skip.get(k)
                if skipped is not None and skipped.numel():
                    keep = ~torch.isin(seg.slots_cpu, skipped)
                    if not bool(keep.any()):
                        continue
                    self._pool.write(k, seg.slots_cpu[keep], host[keep])
                else:
                    self._pool.write(k, seg.slots_cpu, host)
        task.done.set()
        with self._lock:
            self._completed.append(task.tid)
