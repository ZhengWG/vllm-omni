"""OmniTensorCacheController: Task execution Controller.

Two-stage pipeline per:

1. A copy queue (D2H issuance — class
priority, chunked Class A trickle, cap accounting) and a scatter stage
(runs tasks whose copy finished). The committer thread is the mirror's
single writer; slot-reuse needs no ordering edges — a reassigned
(slot, key) is pushed into the old task's skip set, so old and new
mirror writes are disjoint by construction.

2. Eager mode (no background thread) serves CPU-only tests and non-CUDA
platforms: submit() completes the whole pipeline synchronously.
"""

import logging
import threading
from collections import deque
from dataclasses import dataclass, field

import torch

from vllm_omni.core.tensor_cache.block_pool import TensorBlockPool
from vllm_omni.core.tensor_cache.interface import TensorCacheConfig

logger = logging.getLogger(__name__)

CLASS_A = "A"
CLASS_B = "B"


@dataclass
class _Segment:
    """One contiguous save's rows: slots + per-key frozen tensors."""

    slots_cpu: torch.Tensor  # int64 flat row ids, in token order
    tensors: dict[str, torch.Tensor]  # frozen (GPU) or eager CPU tensors
    host: dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass
class WriteTask:
    """One write: a set of (slot, key) rows for one request.

    `tid` is the compact handle stored in the manager's state tables; the
    RFC's logical id is (req_id, key, seq) — recoverable via the metadata
    here for every key the task carries.
    """

    tid: int
    req_id: str
    seq: int
    klass: str
    segments: list[_Segment]
    freeze_event: object | None = None  # torch.cuda.Event
    nbytes: int = 0
    escalated: bool = False
    d2h_claimed: bool = False
    failed: bool = False
    # (slot, key) pairs reassigned to a newer task: the scatter skips them,
    # making old and new mirror writes disjoint (no ordering edges needed).
    skip: dict[str, torch.Tensor] = field(default_factory=dict)
    host_ready: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    _slot_to_row: dict[int, tuple[int, int]] | None = None  # slot -> (seg, row)

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
        self._queue_hi: deque[int] = deque()  # Class B + escalated
        self._queue_lo: deque[int] = deque()  # Class A trickle
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

    def submit(self, task: WriteTask, queued: bool = True, reserved: bool = False) -> None:
        """Register a task. queued=False (deferred entries) stays GPU-staged
        until escalated or cap-flushed.

        reserved=True: the caller already ran reserve() for these bytes (so
        the blocking cap flush stays outside the manager's state lock).
        """
        task.nbytes = sum(t.numel() * t.element_size() for s in task.segments for t in s.tensors.values())
        if not reserved:
            self._reserve_bytes(task.nbytes)
        with self._lock:
            self._tasks[task.tid] = task
        if self._eager:
            if queued:
                self._run_eager(task)
            return
        if queued:
            with self._wake:
                (self._queue_hi if task.klass == CLASS_B else self._queue_lo).append(task.tid)
                self._wake.notify_all()

    def append_segment(self, task: WriteTask, seg: _Segment, reserved: bool = False) -> bool:
        """Grow a deferred (Class A) entry with one step's rows.

        Returns False when the entry already started copying (cap flush or a
        finish/conflict escalation got there first); the caller then opens a
        fresh entry instead of mutating a closed one.

        reserved=True: bytes were reserved by the caller; on False they stay
        reserved and carry over to the fresh entry submitted with
        reserved=True.
        """
        nbytes = sum(t.numel() * t.element_size() for t in seg.tensors.values())
        if not reserved:
            # Exclude this task from the cap flush: flushing the very entry
            # we are appending to would close it under us.
            self._reserve_bytes(nbytes, exclude={task.tid})
        with task.lock:
            if task.d2h_claimed or task.done.is_set():
                if not reserved:
                    with self._lock:
                        self._staged_bytes = max(0, self._staged_bytes - nbytes)
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
        # Cap backpressure: force-flush oldest pending entries until under
        # budget. Bounded block: their D2H has usually long completed.
        exclude = exclude or set()
        while True:
            with self._lock:
                pending = [eid for eid, t in self._tasks.items() if not t.done.is_set() and eid not in exclude]
                if self._staged_bytes + nbytes <= self._config.gpu_staging_bytes or not pending:
                    self._staged_bytes += nbytes
                    return
                oldest = min(pending)
            logger.warning("tensor_cache: staging cap hit, force-flushing entry %d", oldest)
            self.escalate([oldest])
            self.join([oldest])

    # ------------------------------------------------------------- lifecycle

    def escalate(self, entry_ids: list[int]) -> None:
        if self._eager:
            for eid in entry_ids:
                task = self._tasks.get(eid)
                if task is not None and not task.done.is_set():
                    self._run_eager(task)
            return
        with self._wake:
            for eid in entry_ids:
                task = self._tasks.get(eid)
                if task is None or task.escalated or task.done.is_set():
                    continue
                task.escalated = True
                try:
                    self._queue_lo.remove(eid)
                    self._queue_hi.appendleft(eid)
                except ValueError:
                    # Not in the lazy queue: either an unqueued deferred
                    # entry (queue it now) or already claimed/queued-hi.
                    if eid not in self._queue_hi and eid not in self._blocked and not task.d2h_claimed:
                        self._queue_hi.appendleft(eid)
            self._wake.notify_all()

    def join(self, entry_ids: list[int]) -> None:
        for eid in entry_ids:
            task = self._tasks.get(eid)
            if task is not None:
                task.done.wait()

    def join_host_ready(self, tids: list[int]) -> None:
        """Bounded-depth wait: waits only for in-flight GPU staging
        to complete, i.e., for host buffer to be ready. Waiting for
        the scatter phase as in join() is unnecessary for correctness."""
        for tid in tids:
            task = self._tasks.get(tid)
            if task is not None:
                task.host_ready.wait()

    def drain_completed(self) -> list[int]:
        """Manager-side fixed-point drain; returns scattered entry ids."""
        out: list[int] = []
        with self._lock:
            while self._completed:
                out.append(self._completed.popleft())
            for eid in out:
                self._tasks.pop(eid, None)
        return out

    def drain_failed(self) -> list[int]:
        """Entry ids whose write failed; the manager fail-fasts on these."""
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
        """Rows for `slots` of one in-flight entry, resolved shortest-path.

        pre-staged -> host buffer rows; otherwise slice sync D2H on the read
        stream. A full-coverage read that wins the claim feeds its result
        back as the pre-staged host copy (skips the background D2H).
        """
        if len(task.segments) == 1:
            # Fast path: one segment (every main entry) needs no regroup or
            # reorder — resolve rows with a single tensor op.
            rows = self._rows_single_segment(task, slots, key)
            seg = task.segments[0]
            if (
                not task.host_ready.is_set()
                and slots.numel() == seg.slots_cpu.numel()
                and bool(torch.equal(slots, seg.slots_cpu))
            ):
                # Full in-order read: publish it as the pre-staged host copy
                # so the background committer skips its own D2H (PCIe once).
                self._try_feed_back(task, key, rows, [])
            return rows

        s2r = task.slot_to_row()
        idx = [s2r[int(s)] for s in slots.tolist()]
        if task.host_ready.is_set():
            return self._rows_from(task, idx, key, host=True)

        full_cover = len(idx) == task.num_rows()
        rows = self._rows_from(task, idx, key, host=False)  # sync D2H inside
        if full_cover:
            self._try_feed_back(task, key, rows, idx)
        return rows

    def _rows_from(self, task: WriteTask, idx: list[tuple[int, int]], key: str, host: bool) -> torch.Tensor:
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
            src = seg.host.get(key) if host else seg.tensors.get(key)
            if src is None and not host:
                # Copy thread may have published host bufs and dropped the
                # GPU refs between our host_ready check and here.
                src = seg.host.get(key)
            if src is None:
                continue
            rows_idx = torch.tensor([ri for _, ri in items], dtype=torch.long)
            picked = self._slice_rows(task, src, rows_idx, host)
            parts.append(picked)
            order.extend(p for p, _ in items)
        if not parts:
            raise KeyError(f"key {key} not present in entry {task.tid}")
        cat = torch.cat(parts, dim=0)
        out = torch.empty_like(cat)
        out[torch.tensor(order, dtype=torch.long)] = cat
        return out

    def _rows_single_segment(self, task: WriteTask, slots: torch.Tensor, key: str) -> torch.Tensor:
        """Row lookup for a single-segment entry, without per-token Python.

        Row index = position of each slot inside the entry's slot list. The
        entry's slots are the step's slot mapping, so a searchsorted over its
        sorted order maps requested slots to rows in one vectorized step.
        """
        seg = task.segments[0]
        host_ready = task.host_ready.is_set()
        src = seg.host.get(key) if host_ready else seg.tensors.get(key)
        if src is None:
            src = seg.host.get(key)
        if src is None:
            raise KeyError(f"key {key} not present in entry {task.tid}")

        entry_slots = seg.slots_cpu
        # Reading the whole entry in order is the common case (this step's own
        # rows): the row indices are just 0..n-1, so skip the lookup entirely.
        if slots.numel() == entry_slots.numel() and bool(torch.equal(slots, entry_slots)):
            return self._slice_rows(
                task, src, torch.arange(entry_slots.numel(), dtype=torch.int64), host=(src.device.type == "cpu")
            )

        order = torch.argsort(entry_slots)
        pos = torch.searchsorted(entry_slots[order], slots).clamp(max=entry_slots.numel() - 1)
        rows_idx = order[pos]
        if not bool(torch.equal(entry_slots[rows_idx], slots)):
            raise KeyError(f"slots not covered by entry {task.tid}")
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

    def _try_feed_back(self, task: WriteTask, key: str, rows_cpu: torch.Tensor, idx: list[tuple[int, int]]) -> None:
        # Feed back only for single-segment entries with all keys covered by
        # separate fetches is over-engineering; claim per-entry, attach the
        # fetched key, and D2H the remaining keys inline (rare, same window).
        with task.lock:
            if task.d2h_claimed or len(task.segments) != 1:
                return
            task.d2h_claimed = True
        seg = task.segments[0]
        seg.host[key] = rows_cpu
        for k, src in seg.tensors.items():
            if k not in seg.host:
                seg.host[k] = self._slice_rows(task, src, torch.arange(src.shape[0]), host=False)
        task.host_ready.set()
        released = sum(t.numel() * t.element_size() for t in seg.tensors.values())
        seg.tensors = {}
        with self._wake:
            self._staged_bytes -= released
            # Move out of the copy queues; scatter stage picks it up.
            for q in (self._queue_hi, self._queue_lo):
                try:
                    q.remove(task.tid)
                except ValueError:
                    pass
            self._blocked.append(task.tid)
            self._wake.notify_all()

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
            eid = None
            try:
                with self._wake:
                    while not self._shutdown and not self._queue_hi and not self._queue_lo:
                        if self._blocked:
                            break
                        # submit / escalate / feed-back / shutdown all notify.
                        self._wake.wait()
                    if self._shutdown and not self._queue_hi and not self._queue_lo and not self._blocked:
                        return
                    if self._queue_hi:
                        eid = self._queue_hi.popleft()
                    elif self._queue_lo:
                        eid = self._queue_lo.popleft()
                if eid is not None:
                    task = self._tasks.get(eid)
                    if task is not None:
                        self._copy_task(task)
                        with self._wake:
                            if eid not in self._blocked:
                                self._blocked.append(eid)
                self._scatter_ready()
            except BaseException:
                logger.exception("tensor_cache committer failed on entry %s; releasing waiters", eid)
                self._fail_task(eid)

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
        pending_cats: list[tuple[_Segment, str, list[torch.Tensor]]] = []
        with torch.cuda.stream(self._copy_stream):
            if task.freeze_event is not None:
                self._copy_stream.wait_event(task.freeze_event)
            for seg in task.segments:
                for k, src in seg.tensors.items():
                    if task.klass == CLASS_A and src.numel() * src.element_size() > chunk_bytes:
                        rows_per_chunk = max(1, chunk_bytes // max(1, src.shape[-1] * src.element_size()))
                        parts = [
                            src[start : start + rows_per_chunk].to("cpu", non_blocking=True)
                            for start in range(0, src.shape[0], rows_per_chunk)
                        ]
                        pending_cats.append((seg, k, parts))
                    else:
                        seg.host[k] = src.to("cpu", non_blocking=True)
            ev = torch.cuda.Event()
            ev.record()
        ev.synchronize()
        for seg, k, parts in pending_cats:
            seg.host[k] = torch.cat(parts, dim=0)
        # host bufs complete -> host_ready -> drop GPU refs. Do not reorder.
        task.host_ready.set()
        with self._wake:
            self._staged_bytes -= task.nbytes
        for seg in task.segments:
            seg.tensors = {}

    def _fail_task(self, entry_id: int | None) -> None:
        """Release waiters for a task the committer could not complete.

        Idempotent, and only releases bytes the copy stage has not already
        released: a raise AFTER a successful _copy_task (host_ready set)
        must not subtract this task's bytes a second time.
        """
        task = self._tasks.get(entry_id) if entry_id is not None else None
        if task is None or task.done.is_set():
            return
        task.failed = True
        with self._wake:
            if not task.host_ready.is_set():
                self._staged_bytes = max(0, self._staged_bytes - task.nbytes)
            if entry_id in self._blocked:
                self._blocked.remove(entry_id)
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
            ready = [eid for eid in self._blocked if (t := self._tasks.get(eid)) and t.host_ready.is_set()]
            for eid in ready:
                self._blocked.remove(eid)
        for eid in ready:
            task = self._tasks.get(eid)
            if task is None:
                continue
            try:
                self._scatter(task)
            except BaseException:
                # Attribute the failure to THIS task: letting it propagate
                # would fail whichever entry the worker loop happened to be
                # copying, double-release its bytes, and strand this one's
                # join() forever.
                logger.exception("tensor_cache scatter failed on entry %s; releasing waiters", eid)
                self._fail_task(eid)

    @torch.inference_mode()
    def _scatter(self, task: WriteTask) -> None:
        with task.lock:
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
