"""Runs WriteTasks: GPU-stage, host copy, scatter into the CPU block pool.

The manager owns request/slot identity and when to submit. This class
owns the queues, the staging-byte cap, and the single committer that
writes the pool.

Async: hi/lo copy queues, then scatter.
Eager: submit() does both inline (CPU tests / no CUDA).
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import torch

from vllm_omni.core.tensor_cache.block_pool import TensorBlockPool
from vllm_omni.core.tensor_cache.interface import TensorCacheConfig, WriteSchedule

logger = logging.getLogger(__name__)


@dataclass
class _Segment:
    """One contiguous save's rows: slots + per-key frozen tensors."""

    slots_cpu: torch.Tensor  # int64 flat row ids, in token order
    tensors: dict[str, torch.Tensor]  # frozen (GPU) or eager CPU tensors
    host: dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass
class WriteTask:
    """One write of (slot, key) rows for a single request.

    Identity: `tid` is the handle in the manager's (slot, key) tables.
    `req_id` + `write_n` mark whose write this is and the nth time that
    request opened a write. One task may carry several keys.

    Pipeline (one direction, no rollback)::

        queued / GPU-staged
            -> copy claimed (`d2h_claimed`)
            -> `host_ready`  (host published; GPU refs may drop)
            -> scatter
            -> `done`        (in the CPU mirror; `failed` instead on error)

    `JOIN_NEXT_STEP` starts on the hi copy queue and is joined at the
    next save (`host_ready` only). `JOIN_ON_FINISH` stays on the lo
    queue, or unqueued (deferred `submit(queued=False)`), until finish
    or cap pressure `escalate`s it onto hi — once. Cap flush takes the
    unfinished task with the oldest `enqueued_time`.

    Concurrent readers/writers:
    - Copy: `publish_host` and the committer race `d2h_claimed`;
      only the winner publishes `seg.host`.
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
    # Single publisher for the host copy: publish_host and the
    # committer D2H race this; only the winner may write `seg.host`.
    d2h_claimed: bool = False
    # Committer could not land the write; manager fail-fasts on next entry.
    failed: bool = False
    # (slot, key) pairs reassigned to a newer task: the scatter skips them,
    # making old and new mirror writes disjoint (no ordering edges needed).
    skip: dict[str, torch.Tensor] = field(default_factory=dict)
    # Host copy published; GPU refs may drop; join_copied can return.
    host_ready: threading.Event = field(default_factory=threading.Event)
    # Scatter into the CPU mirror has finished (strictly after host_ready).
    done: threading.Event = field(default_factory=threading.Event)
    # Guards skip / d2h_claimed / segment append. Not host_ready or done.
    lock: threading.Lock = field(default_factory=threading.Lock)
    # time.monotonic() at submit; cap flush picks the smallest of these.
    enqueued_time: float = field(default_factory=time.monotonic)
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


class OmniTensorCacheController:
    def __init__(self, pool: TensorBlockPool, config: TensorCacheConfig, eager: bool | None = None):
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
        if not self._eager:
            self._copy_stream = torch.cuda.Stream()
            self._read_stream = torch.cuda.Stream()
            self._worker = threading.Thread(target=self._worker_loop, name="omni-tensor-cache-committer", daemon=True)
            self._worker.start()

    # ------------------------------------------------------------------ submit

    def submit(self, task: WriteTask, queued: bool = True) -> None:
        """Register a task. queued=False (deferred tasks) stays GPU-staged
        until escalated or cap-flushed.

        Caller must reserve() the task bytes first (cap flush can block;
        the manager does that outside the state lock).
        """
        task.nbytes = sum(t.numel() * t.element_size() for s in task.segments for t in s.tensors.values())
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
            task.nbytes += nbytes
        return True

    def reserve(self, nbytes: int, exclude: set[int] | None = None) -> None:
        """Public cap reservation; blocking flush happens here, so callers
        must not hold the manager's state lock."""
        self._reserve_bytes(nbytes, exclude=exclude)

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
        """Block until each task's host copy is published."""
        for tid in tids:
            task = self._tasks.get(tid)
            if task is not None:
                task.host_ready.wait()

    def drain_completed(self) -> list[int]:
        """Pop scattered tasks from `_completed` and drop them from `_tasks`."""
        out: list[int] = []
        with self._lock:
            while self._completed:
                out.append(self._completed.popleft())
            for tid in out:
                self._tasks.pop(tid, None)
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

        Host copy if published; otherwise slice sync D2H on the read stream.
        """
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
            # host[k] is published only after D2H; else the freeze. (`or`
            # would bool() a Tensor and raise.)
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
        # host[k] is published only after D2H; else the freeze. (`or`
        # would bool() a Tensor and raise.)
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

    @torch.inference_mode()
    def snapshot_host(self, src: torch.Tensor, freeze_event: object | None) -> torch.Tensor:
        """One whole-tensor D2H of a frozen step clone (read stream, single
        sync). CPU tensors pass through."""
        if src.device.type == "cpu":
            return src
        if self._read_stream is None:
            return src.detach().cpu()
        with torch.cuda.stream(self._read_stream):
            if freeze_event is not None:
                self._read_stream.wait_event(freeze_event)
            cpu = src.to("cpu", non_blocking=True)
            ev = torch.cuda.Event()
            ev.record()
        ev.synchronize()
        return cpu

    def publish_host(self, task: WriteTask, host: dict[str, torch.Tensor]) -> bool:
        """Attach a ready host copy to a single-segment task:
        the background committer skips its own D2H and just scatters.
        Loses the claim race -> False, caller's copy is
        discarded (single-publisher rule)."""
        with task.lock:
            if task.d2h_claimed or task.done.is_set() or len(task.segments) != 1:
                return False
            task.d2h_claimed = True
        seg = task.segments[0]
        missing = [k for k in seg.tensors if k not in host]
        for k, rows in host.items():
            seg.host[k] = rows
        for k in missing:
            seg.host[k] = self._slice_rows(task, seg.tensors[k], torch.arange(seg.tensors[k].shape[0]), host=False)
        task.host_ready.set()
        released = sum(t.numel() * t.element_size() for t in seg.tensors.values())
        seg.tensors = {}
        with self._wake:
            self._staged_bytes -= released
            for q in (self._queue_hi, self._queue_lo):
                try:
                    q.remove(task.tid)
                except ValueError:
                    pass
            self._blocked.append(task.tid)
            self._wake.notify_all()
        if self._eager:
            self._scatter(task)
        return True

    # ------------------------------------------------------------ eager mode

    @torch.inference_mode()
    def _run_eager(self, task: WriteTask) -> None:
        with task.lock:
            if task.d2h_claimed:
                if not task.done.is_set() and task.host_ready.is_set():
                    self._scatter(task)
                return
            task.d2h_claimed = True
        for seg in task.segments:
            for k, t in seg.tensors.items():
                seg.host[k] = t.detach().cpu() if t.device.type != "cpu" else t.clone()
        task.host_ready.set()
        with self._lock:
            self._staged_bytes -= task.nbytes
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
                        # submit / escalate / publish_host / shutdown all notify.
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
        """
        Asynchronously copy tensor data for a WriteTask from device (GPU) to host (CPU).
        """
        with task.lock:
            if task.d2h_claimed:
                return
            task.d2h_claimed = True
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
        with self._wake:
            self._staged_bytes -= task.nbytes
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
        with self._wake:
            if not task.host_ready.is_set():
                self._staged_bytes = max(0, self._staged_bytes - task.nbytes)
            if tid in self._blocked:
                self._blocked.remove(tid)
        for seg in task.segments:
            seg.tensors = {}
        task.host_ready.set()
        task.done.set()
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
