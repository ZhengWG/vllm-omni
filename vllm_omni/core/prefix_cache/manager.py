# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Omni prefix cache, manager side.

Owns slot occupancy, the request-task table, the hit/span registry,
per-step snapshots, and the merge. The controller owns the staging
pool, copy queues, and pool scatter. The state lock covers those
tables only — never a join, a cap flush, or a copy.

Helper docstrings mark ``_state_lock`` (non-reentrant):

    Caller holds   already inside a critical section; do not acquire
    Takes          acquires here (``@_locked`` or ``with``)

Two host stores:

    StagingBufferPool   reusable step-sized pages. save launches ONE
                        whole-step D2H here for immediately-cached keys
                        (hidden + non-deferred mm). Per-task `chunk.host`
                        is a view into that page, not a second copy.
    PrefixBlockPool     durable (kv_slot, key) prefix cache. The
                        committer only scatters into it.

Two write paths (schedule is the key split, not token count):

    JOIN_NEXT_STEP      immediately-cached keys. D2H is already in
                        flight at submit; the committer waits
                        `step_d2h_event` then H2H-scatters. Joined at the
                        next save (`host_ready` only).
    JOIN_ON_FINISH      deferred mm. Stays on the device freeze; the
                        committer does that D2H, then scatters.
                        Escalated on finish/abort or cap pressure.

Per real scheduler_output, engine-thread order:

    new_step_starts   before _update_states drops finished requests
                      (register hits, start prefix prefetch)
    forward
    save_outputs      device snapshot + launch staging D2H; returns step id
    materialize XOR discard_step   consume that id exactly once

materialize may run on the async output builder while the engine is
already in the next step. Warmup/dummy runs are never fed.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Iterable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Any, NamedTuple

import torch

from vllm_omni.core.prefix_cache.block_pool import PrefixBlockPool
from vllm_omni.core.prefix_cache.controller import (
    OmniPrefixCacheController,
    StagingBufferHolder,
    StepD2HClaim,
    WriteTask,
    _SnapshotHolder,
    _WriteChunk,
)
from vllm_omni.core.prefix_cache.interface import (
    ModelCachePolicy,
    OmniPrefixCacheUnmatchError,
    PrefixCacheConfig,
    ReqId,
    StageCacheOutputs,
    StepId,
    TensorName,
    Tid,
    WriteSchedule,
    is_hidden_key,
    without_hidden,
)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

    from vllm_omni.core.prefix_cache.group_view import FullAttentionGroupView

logger = logging.getLogger(__name__)


class _Occupancy(IntEnum):
    ABSENT = 0
    IN_TRANSIT = 1
    COMMITTED = 2


def _is_step_token_tensor(val: Any, n: int, padded: int) -> bool:
    """2D+ tensor whose first dim is this step's token count (``n`` or padded).

    True means callers may take ``val[:n]``. Leftover tensors (``codes.ref``),
    lists, and other shapes are False.
    """
    return isinstance(val, torch.Tensor) and val.ndim >= 2 and int(val.shape[0]) in (n, padded)


def _snapshot_leftover_mm_cpu(
    mm_outputs: dict[str, Any],
    device_snapshot_keys: set[str],
    num_tokens_unpadded: int,
    num_tokens_padded: int | None = None,
) -> dict[str, Any]:
    """CPU copy of mm that did not go through staging D2H.

    Skip ``device_snapshot_keys`` (those already have a staging page). Copy
    the rest — deferred mm, lists, ``codes.ref`` — so materialize can
    run after the next forward overwrites graph buffers. Slice ``[:n]``
    only when ``shape[0] == n``; ``>= n`` would clip ``codes.ref``.
    """
    n = num_tokens_unpadded
    padded = n if num_tokens_padded is None else int(num_tokens_padded)

    def _copy(val: Any) -> Any:
        if isinstance(val, torch.Tensor):
            t = val[:n] if _is_step_token_tensor(val, n, padded) and int(val.shape[0]) == n else val
            copied = t.detach()
            # .cpu() copies device tensors; CPU/pinned views still share storage.
            copied = copied.cpu() if copied.device.type != "cpu" else copied.clone()
            return copied.contiguous()
        if isinstance(val, Mapping):
            return {k: _copy(v) for k, v in val.items()}
        if isinstance(val, list):
            return [_copy(v) for v in val]
        if isinstance(val, tuple):
            return tuple(_copy(v) for v in val)
        return val

    return {
        key: _copy(val) for key, val in mm_outputs.items() if key not in device_snapshot_keys and not is_hidden_key(key)
    }


def _locked(fn):
    """Serialize facade entry points: the async output builder calls
    materialize() while the engine thread is in the next step."""

    def wrapper(self, *args, **kwargs):
        with self._state_lock:
            return fn(self, *args, **kwargs)

    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


@dataclass
class _SlotRef:
    """Where one (req, key) span's slots live: planned under the lock, fetched outside it.

    Schedule split, not a single read tier:
    - JOIN_NEXT_STEP in-transit: ``join_tids``. Fetch waits ``done``,
      drains, then reads the pool. Staging views are never sliced.
    - JOIN_ON_FINISH in-transit: ``staged_list`` task refs. Fetch uses
      ``fetch_host`` (device freeze / committer host).
    - Already scattered: ``already_staged`` → pool.

    A JOIN_NEXT_STEP task may disappear between plan and join (another
    facade already drained it); ``join`` no-ops and the pool rows persist.
    """

    slots: torch.Tensor  # KV slot ids (PrefixBlockPool rows)
    key: TensorName
    req_id: ReqId
    already_staged: bool  # this key is already in the CPU pool
    staged_list: list[tuple[WriteTask, torch.Tensor]]  # JOIN_ON_FINISH only
    join_tids: list[Tid] = field(default_factory=list)  # JOIN_NEXT_STEP in-transit


@dataclass(kw_only=True)
class _StepContext:
    """Save-time snapshot of one step; consumed exactly once.

    Built on the engine thread in save_outputs so materialize (possibly on
    the async builder) never reads the live batch. Host rows live on `d2h`
    (None only if the step had no rows). materialize XOR discard_step pops
    it; leaking contexts fails fast at a later save.
    """

    # Packed layout in batch order: req -> [start, end) of this step's rows.
    spans: dict[ReqId, tuple[int, int]]
    num_tokens_unpadded: int = 0

    # Hits snapshotted at new_step_starts. Prefetch fills [hit | empty tail]
    # during forward; materialize writes the tail.
    hits: dict[ReqId, tuple[int, list[int] | None]]  # (hit_upto, blocks)
    hit_prefetch: dict[ReqId, dict[TensorName, Future]] = field(default_factory=dict)

    # Key split frozen at save: recompute at materialize races ensure_key.
    cached_keys: set[TensorName] = field(default_factory=set)
    # Leftover mm copied to CPU at save (deferred tails + uncached
    # passthrough). Never live graph-buffer refs: materialize may run
    # after the next forward has overwritten those buffers.
    mm_cpu_snapshot: dict[TensorName, Any] = field(default_factory=dict)

    # Staging claim if this step had rows.
    d2h: StepD2HClaim | None = None


class _SlotStatus(NamedTuple):
    """Occupancy row for one tensor name (views into the table, not copies)."""

    state: torch.Tensor  # int8[num_slots]
    tids: torch.Tensor  # Tid per kv slot; 0 = none


class _SlotStatusTable:
    """Per (KV slot, tensor name): empty, being written, or already in the pool.

    Hidden and a deferred mm field on the same slot are independent.
    ``map_slots`` marks a write in flight; if another task still owns the
    slot, the manager tells that task to skip those rows. ``commit``
    runs after scatter: still-owned slots become committed.
    """

    def __init__(self, num_slots: int) -> None:
        self.num_slots = num_slots
        self.state: dict[TensorName, torch.Tensor] = {}  # int8[num_slots]
        self.tids: dict[TensorName, torch.Tensor] = {}  # Tid per kv slot; 0 = none
        self.task_slots: dict[Tid, torch.Tensor] = {}  # kv slots
        self.task_keys: dict[Tid, tuple[TensorName, ...]] = {}

    def init_table(self, key: TensorName) -> None:
        """Allocate occupancy tensors for ``key`` if they do not exist."""
        if key in self.state:
            return
        self.state[key] = torch.zeros(self.num_slots, dtype=torch.int8)
        self.tids[key] = torch.zeros(self.num_slots, dtype=torch.int64)

    def get_slot_status(self, key: TensorName) -> _SlotStatus:
        """Occupancy tensors for ``key``. ``init_table`` must have run."""
        return _SlotStatus(state=self.state[key], tids=self.tids[key])

    def map_slots(
        self, slots: torch.Tensor, tid: Tid, keys: Iterable[TensorName]
    ) -> list[tuple[Tid, TensorName, torch.Tensor]]:
        """Hang ``tid`` on these (slot, key). Return stolen in-transit rows."""
        keys = tuple(keys)
        stolen: list[tuple[Tid, TensorName, torch.Tensor]] = []
        for key in keys:
            status = self.get_slot_status(key)
            cur = status.tids[slots]
            stale = (status.state[slots] == _Occupancy.IN_TRANSIT) & (cur != tid) & (cur != 0)
            if bool(stale.any()):
                for old in {int(o) for o in cur[stale].tolist()}:
                    stolen.append((old, key, slots[stale & (cur == old)]))
            status.state[slots] = _Occupancy.IN_TRANSIT
            status.tids[slots] = tid
        prev = self.task_slots.get(tid)
        if prev is None:
            self.task_slots[tid] = slots
            self.task_keys[tid] = keys
        else:
            # Deferred tasks grow one `_WriteChunk` per step.
            self.task_slots[tid] = torch.cat([prev, slots])
            self.task_keys[tid] = tuple(dict.fromkeys(self.task_keys[tid] + keys))
        return stolen

    def commit(self, tids: Iterable[Tid]) -> None:
        """Flip still-owned slots to COMMITTED and drop the reverse index."""
        for tid in tids:
            slots = self.task_slots.pop(tid, None)
            keys = self.task_keys.pop(tid, ())
            if slots is None:
                continue
            for key in keys:
                status = self.get_slot_status(key)
                still_ours = status.tids[slots] == tid
                idx = slots[still_ours]
                status.state[idx] = _Occupancy.COMMITTED
                status.tids[idx] = 0


class _RequestTaskTable:
    """Per request: still live, which WriteTasks it opened, deferred task.

    Also allocates ``tid`` and increments per-request ``write_n``.
    Copy and scatter stay on the controller.
    """

    def __init__(self) -> None:
        self._next_tid: Tid = 1
        self.write_n: dict[ReqId, int] = {}  # last write_n issued
        self.tasks: dict[ReqId, set[Tid]] = {}
        self.deferred: dict[ReqId, WriteTask] = {}
        self.live_reqs: set[ReqId] = set()

    def alloc_tid(self) -> Tid:
        tid = self._next_tid
        self._next_tid += 1
        return tid

    def increment_write_n(self, req_id: ReqId) -> int:
        n = self.write_n.get(req_id, 0) + 1
        self.write_n[req_id] = n
        return n

    def track(self, req_id: ReqId, tid: Tid) -> None:
        self.tasks.setdefault(req_id, set()).add(tid)

    def finish(self, req_id: ReqId) -> tuple[set[Tid], WriteTask | None]:
        """Drop this request's rows. Returns owned tids + deferred task."""
        self.live_reqs.discard(req_id)
        self.write_n.pop(req_id, None)
        tids = self.tasks.pop(req_id, set())
        dtask = self.deferred.pop(req_id, None)
        return tids, dtask

    def drop_completed(self, tids: Iterable[Tid]) -> None:
        done = set(tids)
        if not done:
            return
        for req_tids in self.tasks.values():
            req_tids -= done


class OmniPrefixCacheManager:
    def __init__(
        self,
        config: PrefixCacheConfig,
        view: FullAttentionGroupView,
        *,
        eager: bool | None = None,
    ):
        self._config = config
        self._view = view
        self._pool = PrefixBlockPool(config)
        self._controller = OmniPrefixCacheController(self._pool, config, eager=eager)
        self._policy = ModelCachePolicy()
        # Serializes engine vs async-builder facade entries. Non-reentrant:
        # those entries never call each other, and the lock must not cover
        # a join, cap flush, or D2H.
        self._state_lock = threading.Lock()

        self._slot_status = _SlotStatusTable(config.num_blocks * config.block_size)
        # Hidden is known from the default policy; mm rows at init_table.
        if (hk := self._policy.hidden_key) is not None:
            self._slot_status.init_table(hk)
        self._request_tasks = _RequestTaskTable()
        # Join worklists — not occupancy, not the request-task table.
        self._join_next_step_tids: list[Tid] = []
        self._join_finished_tids: set[Tid] = set()  # escalated on finish/abort

        # This step's hits (copied into _StepContext at save).
        self._cur_num_scheduled: dict[ReqId, int] = {}
        self._hit_spans: dict[ReqId, tuple[int, list[int]]] = {}  # (upto, blocks)
        self._hit_prefetch: dict[ReqId, dict[TensorName, Future]] = {}

        # Prefix gather during forward (CPU work releases the GIL).
        # One worker: complete in submit order; reap from the head.
        self._prefetch_pool = ThreadPoolExecutor(1, thread_name_prefix="omni-prefix-cache-prefetch")
        self._prefetch_queue: deque[tuple[Future, _SlotRef]] = deque()

        # Consume-exactly-once snapshots (materialize XOR discard_step).
        self._next_step_id: StepId = 1
        self._step_ctxs: dict[StepId, _StepContext] = {}

    # ------------------------------------------------------------- facade

    def register_policy(self, policy: ModelCachePolicy) -> None:
        self._policy = policy
        if (hk := policy.hidden_key) is not None:
            self._slot_status.init_table(hk)

    @_locked
    @torch.inference_mode()
    def new_step_starts(self, scheduler_output: SchedulerOutput) -> None:
        """Consume one scheduler_output (lifecycle stream).

        Engine thread only; before _update_states removes finished
        requests; exactly once per real step. Registers new-request prefix
        hits (snapshotting their block tables) and escalates the writes of
        finished/aborted requests — a block hash that entered the batch
        must land in the cache, abort included.
        """
        # 1. Publish writes the committer has already scattered.
        self._commit_drained_writes()

        # 2. Finished/aborted reqs: escalate leftover writes now.
        #    join_host_ready waits at the next save.
        finished = getattr(scheduler_output, "finished_req_ids", None) or ()
        for req_id in finished:
            tids, dtask = self._request_tasks.finish(req_id)
            if dtask is not None:
                tids.add(dtask.tid)
            pending_tasks = [tid for tid in tids if self._controller.get_task(tid) is not None]
            if pending_tasks:
                # Abort too: those block hashes are already in vLLM.
                # Dropping the write would leave future hits ABSENT.
                self._controller.escalate(pending_tasks)
                self._join_finished_tids.update(pending_tasks)

        # 3. Copy this arrival's prefix-hit block ids. scheduled_new_reqs
        #    is the only place they appear; after _update_states they sit
        #    on the live request and grow as decode allocates more blocks.
        #    materialize (async builder) must not reread that live table.
        self._clear_hit_infos()
        for new_req in getattr(scheduler_output, "scheduled_new_reqs", ()) or ():
            req_id = new_req.req_id
            if req_id in self._request_tasks.live_reqs:
                # Streaming continuation (async_chunk): parity with legacy —
                # no hit marking; span/delivered_upto refinement is Phase 2.
                continue
            self._request_tasks.live_reqs.add(req_id)
            num_computed = int(getattr(new_req, "num_computed_tokens", 0) or 0)
            if num_computed > 0:
                # block_ids is per-kv-group; group 0 only.
                blocks = getattr(new_req, "block_ids", None)
                if blocks is not None and len(blocks) > 0 and not isinstance(blocks[0], int):
                    blocks = blocks[0]
                if not blocks:
                    # Fail at the cause: a hit we cannot snapshot now would
                    # crash at materialize time with less context (materialize is
                    # forbidden from reading the live batch).
                    raise OmniPrefixCacheUnmatchError(
                        f"prefix hit for req {req_id} ({num_computed} tokens) carries no block_ids"
                    )
                hit_blocks = list(blocks[: num_computed // self._config.block_size])
                self._hit_spans[req_id] = (num_computed, hit_blocks)

        # 4. Gather those spans on the prefetch thread; overlaps this forward.
        while self._prefetch_queue and self._prefetch_queue[0][0].done():
            self._prefetch_queue.popleft()
        self._cur_num_scheduled = dict(scheduler_output.num_scheduled_tokens)
        if self._hit_spans:
            self._prefetch_hit_spans()

    @torch.inference_mode()
    def save_outputs(
        self,
        hidden_states: torch.Tensor | None,
        mm_outputs: dict[str, Any] | None,
        *,
        num_tokens_unpadded: int,
        num_tokens_padded: int,
    ) -> int:
        """Write this step's outputs into the cache; returns the step id.

        Engine thread only, after the forward and before materialize.
        Immediately-cached rows: one D2D freeze, one whole-step D2H into
        the staging pool, then one JOIN_NEXT_STEP WriteTask per request
        whose `chunk.host` is a view of that page. Deferred rows stay on
        the device freeze (JOIN_ON_FINISH); the committer copies them later.
        Leftover mm (deferred tails, uncached passthrough) is copied to
        CPU here so materialize never reads live graph buffers.
        Snapshots everything materialize needs. The returned step id MUST
        be consumed exactly once — by materialize() or discard_step();
        leaking contexts fails fast at a later save.

        The state lock never covers a blocking wait: the previous step's
        JOIN_NEXT_STEP join, the clone build, and the cap reservation (which may
        flush) all run unlocked.
        """
        # 1. Join the previous step's host copies (unlocked).
        self._wait_for_host_ready()

        # 2. Packed batch layout for this step (req -> [start, end)).
        req_order = self._view.batch_req_ids()
        num_sched = {r: int(self._cur_num_scheduled.get(r, 0)) for r in req_order}
        query_start: dict[str, int] = {}
        current_start_idx = 0
        for req_id in req_order:
            query_start[req_id] = current_start_idx
            current_start_idx += num_sched[req_id]

        slots_cpu: torch.Tensor | None = None
        mm_outputs = mm_outputs or {}
        device_snapshot: dict[str, torch.Tensor] = {}
        deferred_chunks: list[tuple[str, _WriteChunk]] = []
        freeze_event = None

        # 3. Immediate device snapshot (D2D) and reserve staging bytes.
        if num_tokens_unpadded > 0:
            # Derive the slot mapping on CPU: reading the device one back
            # would need a stream sync that waits on the whole forward.
            slots_cpu = self._view.step_slots_cpu(req_order, num_sched)
            if int(slots_cpu.numel()) != num_tokens_unpadded:
                # Fail at the cause: skipping the save would leave rows absent
                # behind hashes vLLM already published — a delayed crash at
                # some future hit instead of a debuggable one here.
                raise OmniPrefixCacheUnmatchError(
                    f"slot mapping covers {int(slots_cpu.numel())} of {num_tokens_unpadded} scheduled tokens; "
                    "CPU-side slot derivation out of sync with the batch"
                )
            device_snapshot = self._get_device_snapshot(
                hidden_states, mm_outputs, num_tokens_unpadded, num_tokens_padded
            )
            deferred_chunks = self._build_deferred_chunks(
                mm_outputs,
                slots_cpu,
                req_order,
                num_sched,
                query_start,
                num_tokens_unpadded,
                num_tokens_padded,
            )

            freezed_tensors = [t for t in device_snapshot.values()] + [
                t for _, chunk in deferred_chunks for t in chunk.tensors.values()
            ]
            if freezed_tensors:
                if torch.cuda.is_available() and any(t.is_cuda for t in freezed_tensors):
                    freeze_event = torch.cuda.Event()
                    freeze_event.record()
                # Charge unique allocations: immediate clones + one deferred
                # C→1 clone. Do not sum per-req views — they share storage.
                immediate_bytes = sum(t.numel() * t.element_size() for t in device_snapshot.values())
                deferred_holder = deferred_chunks[0][1].snapshot_holder if deferred_chunks else None
                deferred_bytes = deferred_holder.nbytes if deferred_holder is not None else 0
                # Cap reservation may block on a flush: outside the lock. The
                # flush must not close the deferred entries we are about to
                # append to (main-thread-only reads, safe unlocked).
                exclude = {
                    self._request_tasks.deferred[r].tid for r, _ in deferred_chunks if r in self._request_tasks.deferred
                }
                self._controller.reserve(immediate_bytes + deferred_bytes, exclude=exclude)

        # 4. Leftover mm onto CPU (unlocked). Keys already in device_snapshot
        #    go through staging D2H; everything else must be on CPU before
        #    the next forward overwrites graph buffers.
        leftover_mm = _snapshot_leftover_mm_cpu(
            mm_outputs, set(device_snapshot), num_tokens_unpadded, num_tokens_padded
        )

        # 5. Bound unconsumed step ctxs *before* claiming a staging slot —
        #    otherwise a full pool raises holder-exhaustion and hides the ids.
        #    Then optional D2H (unlocked), then submit + hang ctx (locked).
        #
        # TODO: every sid-issuing save should claim an in-flight ticket
        # (leftover-only included). Full → wait for materialize/discard,
        # timeout then error. Then delete this raise.
        self._raise_if_unconsumed_ctxs_at_capacity()
        d2h_claim: StepD2HClaim | None = None
        step_holder = StagingBufferHolder.for_step(self._next_step_id)
        transferred = False
        bound_tids: list[int] = []
        try:
            if device_snapshot:
                d2h_claim = self._controller.stage_step_host(
                    device_snapshot, num_tokens_unpadded, freeze_event, step_holder
                )

            step_id = self._publish_saved_step(
                req_order=req_order,
                query_start=query_start,
                num_sched=num_sched,
                num_tokens_unpadded=num_tokens_unpadded,
                device_snapshot=device_snapshot,
                slots_cpu=slots_cpu,
                leftover_mm=leftover_mm,
                mm_keys=set(mm_outputs.keys()),
                deferred_chunks=deferred_chunks,
                freeze_event=freeze_event,
                d2h_claim=d2h_claim,
                bound_tids=bound_tids,
            )
            transferred = True
            return step_id
        finally:
            # Slot claim is outside the lock; a later raise must drop holders.
            if not transferred and d2h_claim is not None:
                self._release_staging_on_failed_save(d2h_claim.staging_slot, step_holder, bound_tids)

    @torch.inference_mode()
    def materialize(self, step_id: int, req_ids: list[str]) -> StageCacheOutputs:
        """Per-request merged outputs for the step saved as `step_id`.

        Any thread. `req_ids` must be (a subset of) the save-time snapshot;
        an outside id means the caller is reading the live batch.
        A request without a hit is a plain miss and gets exactly
        this step's rows — normal path, nothing logged. A hit span that
        resolves to absent rows raises OmniPrefixCacheUnmatchError: fatal
        by contract, never a degrade.

        Two phases: under the lock, drain completions and pin every row
        source (task refs + masks, absent checks included) — the storage
        tier is NOT baked in. Unlocked: wait this step's `step_d2h_event`,
        clone the staging views (then drop the step holder), and merge.
        The engine thread never waits on this thread's PCIe.
        """
        ctx = None
        step_released = False
        try:
            with self._state_lock:
                ctx = self._take_step_ctx(step_id)
                self._commit_drained_writes()

                # The builder must pass (a subset of) the req list captured at
                # save time — an id outside the snapshot means it is reading the
                # live batch, which the contract forbids (debug assert, not a
                # degrade path).
                assert set(req_ids) <= set(ctx.spans), (
                    f"materialize(step {step_id}) got req ids outside the save snapshot: "
                    f"{sorted(set(req_ids) - set(ctx.spans))[:8]}"
                )

                if self._policy.hidden_key is None and not ctx.hits:
                    # Nothing will read the views; drop the step holder here
                    # or the slot leaks (consume-exactly-once ends with us).
                    self._release_step_staging(ctx, step_id)
                    step_released = True
                    return StageCacheOutputs(hidden_states=None, mm_outputs={})

                cached_keys = ctx.cached_keys

                hit_sources: dict[tuple[str, str], _SlotRef | Future] = {}
                try:
                    for req_id in req_ids:
                        hit = ctx.hits.get(req_id)
                        if not hit:
                            continue
                        hit_upto, hit_blocks = hit
                        prefetched = ctx.hit_prefetch.get(req_id, {})
                        slots = self._get_hit_slots(req_id, hit_upto, hit_blocks)
                        keys = self._policy.get_hit_keys(cached_keys)
                        for key in keys:
                            fut = prefetched.get(key)
                            if fut is not None:
                                hit_sources[(req_id, key)] = fut
                                continue
                            hit_sources[(req_id, key)] = self._slot_ref(slots, key, req_id)
                except Exception:
                    logger.critical("omni prefix cache unmatch during materialize", exc_info=True)
                    raise

            # ---- unlocked: data movement + merge ----
            current: dict[str, torch.Tensor] = {}
            if ctx.d2h is not None:
                # Whole-step D2H was launched at save. One event wait (usually
                # already complete), then a contiguous copy-out per key — the
                # copy detaches consumers from the reusable staging slot.
                if ctx.d2h.event is not None:
                    ctx.d2h.event.synchronize()
                current = {k: v.clone() for k, v in ctx.d2h.views.items()}
                self._release_step_staging(ctx, step_id)
                step_released = True

            hidden_out: dict[str, torch.Tensor] | None = None
            hidden_key = self._policy.hidden_key
            if hidden_key is not None and hidden_key in current:
                hidden_out = {}
                for req_id in req_ids:
                    hidden_out[req_id] = self._merge_cached_for_req(
                        ctx, req_id, hidden_key, current[hidden_key], hit_sources
                    )

            mm_out: dict[str, dict[str, Any]] = {}
            for key in cached_keys:
                cur = current.get(key)
                if cur is None:
                    val = ctx.mm_cpu_snapshot.get(key)
                    if not isinstance(val, torch.Tensor):
                        continue
                    # Leftover snapshot already classified; do not re-slice.
                    cur = val
                mm_out[key] = {
                    req_id: self._merge_cached_for_req(ctx, req_id, key, cur, hit_sources) for req_id in req_ids
                }

            self._merge_uncached_mm(ctx, req_ids, cached_keys, mm_out)
            return StageCacheOutputs(hidden_states=hidden_out, mm_outputs=mm_out)
        finally:
            if ctx is not None and not step_released:
                self._release_step_staging(ctx, step_id)

    @_locked
    def discard_step(self, step_id: int) -> None:
        """Consume the step context when nothing will materialize it.

        Any thread; same exactly-once contract as materialize (unknown or
        duplicate id fails fast). Only the read-side snapshot is dropped —
        the cache write proceeds unchanged.
        """
        ctx = self._take_step_ctx(step_id)
        self._release_step_staging(ctx, step_id)

    def shutdown(self) -> None:
        self._prefetch_pool.shutdown(wait=False, cancel_futures=True)
        self._prefetch_queue.clear()
        self._controller.shutdown()

    # ------------------------------------------------------ new_step

    def _clear_hit_infos(self) -> None:
        """Drop the live hit / prefetch tables. Caller holds ``_state_lock``."""
        self._hit_spans.clear()
        self._hit_prefetch.clear()

    def _prefetch_hit_spans(self) -> None:
        """Caller holds ``_state_lock``. Plan each hit span and gather it on
        the prefetch thread, overlapping the forward. A span that fails to
        plan — same-step hits resolve rows this step's save has not
        registered yet — is left to materialize, which owns the fail-fast.
        """
        keys = self._policy.get_hit_keys(self._pool.keys())
        for req_id, (hit_upto, hit_blocks) in self._hit_spans.items():
            n_new = int(self._cur_num_scheduled.get(req_id, 0))
            slots = self._get_hit_slots(req_id, hit_upto, hit_blocks)
            futs: dict[str, Future] = {}
            for key in keys:
                try:
                    src = self._slot_ref(slots, key, req_id)
                except (OmniPrefixCacheUnmatchError, KeyError):
                    continue
                fut = self._prefetch_pool.submit(self._prefetch_hit, src, n_new)
                self._prefetch_queue.append((fut, src))
                futs[key] = fut
            if futs:
                self._hit_prefetch[req_id] = futs

    @torch.inference_mode()
    def _prefetch_hit(self, src: _SlotRef, n_new: int) -> torch.Tensor:
        """Prefetch thread: gather the hit span and pre-build the merged
        buffer with the prefix filled. materialize writes only this step's
        rows at the tail — the gather AND the prefix copy both happen while
        the forward runs, and the cat leaves the critical path."""
        rows = self._fetch_source(src)
        out = torch.empty((rows.shape[0] + n_new, rows.shape[-1]), dtype=rows.dtype)
        out[: rows.shape[0]] = rows
        return out

    # ---------------------------------------------------------- save

    def _wait_for_host_ready(self) -> None:
        """Pop last step's join worklists, then wait ``host_ready`` unlocked.

        Lock covers only the pop. ``join_host_ready`` may block on D2H.
        """
        with self._state_lock:
            join_ids = list(self._join_finished_tids)
            join_ids.extend(self._join_next_step_tids)
            self._join_finished_tids.clear()
            self._join_next_step_tids.clear()
        if join_ids:
            self._controller.join_host_ready(join_ids)

    @_locked
    def _publish_saved_step(
        self,
        *,
        req_order: list[str],
        query_start: dict[str, int],
        num_sched: dict[str, int],
        num_tokens_unpadded: int,
        device_snapshot: dict[str, torch.Tensor],
        slots_cpu: torch.Tensor | None,
        leftover_mm: dict[str, Any],
        mm_keys: set[str],
        deferred_chunks: list[tuple[str, _WriteChunk]],
        freeze_event: object | None,
        d2h_claim: StepD2HClaim | None,
        bound_tids: list[int],
    ) -> StepId:
        """Takes ``_state_lock``. Submit this step's writes and hang the
        consume-once ctx. Copies live hits into the ctx, then clears them.
        D2H and cap flush stay outside.
        """
        self._commit_drained_writes()
        if device_snapshot:
            assert slots_cpu is not None and d2h_claim is not None
            self._submit_step_writes(
                req_order,
                query_start,
                num_sched,
                device_snapshot,
                slots_cpu,
                d2h_claim.views,
                freeze_event,
                d2h_claim.staging_slot,
                d2h_claim.event,
                bound_tids,
            )
        self._stage_deferred(deferred_chunks, freeze_event)
        step_id = self._next_step_id
        self._next_step_id += 1
        self._step_ctxs[step_id] = _StepContext(
            spans={r: (query_start[r], query_start[r] + num_sched[r]) for r in req_order},
            num_tokens_unpadded=num_tokens_unpadded,
            hits=dict(self._hit_spans),
            hit_prefetch=dict(self._hit_prefetch),
            cached_keys=without_hidden(self._pool.keys()) & mm_keys,
            mm_cpu_snapshot=leftover_mm,
            d2h=d2h_claim,
        )
        self._clear_hit_infos()
        return step_id

    def _get_device_snapshot(
        self,
        hidden_states: torch.Tensor | None,
        mm_outputs: dict[str, Any],
        num_tokens_unpadded: int,
        num_tokens_padded: int,
    ) -> dict[str, torch.Tensor]:
        """D2D-clone this step's immediately-cached rows off live buffers.

        Hidden + token-major mm (unpadded or CUDA-graph padded). Deferred
        keys are a separate clone (``_build_deferred_chunks``). Talker
        ``codes.audio`` stays unpadded while hidden is padded; both must
        open a pool key. Lists and other shapes stay leftover.
        """
        n = num_tokens_unpadded
        out: dict[str, torch.Tensor] = {}
        if hidden_states is not None and (hk := self._policy.hidden_key) is not None:
            if hidden_states.ndim < 2 or hidden_states.shape[0] < n:
                rows = 0 if hidden_states.ndim < 2 else int(hidden_states.shape[0])
                raise OmniPrefixCacheUnmatchError(f"hidden_states has {rows} rows, need {n}")
            self._ensure_cache_key(hk, hidden_states.dtype, int(hidden_states.shape[-1]))
            out[hk] = hidden_states[:n].clone()

        for key, val in mm_outputs.items():
            if self._policy.skip_immediate_mm(key):
                continue
            if not _is_step_token_tensor(val, n, num_tokens_padded):
                continue
            self._ensure_cache_key(key, val.dtype, int(val.shape[-1]))
            out[key] = val[:n].clone()
        return out

    def _build_deferred_chunks(
        self,
        mm_outputs: dict[str, Any],
        slots_cpu: torch.Tensor,
        req_order: list[str],
        num_sched: dict[str, int],
        query_start: dict[str, int],
        num_tokens_unpadded: int,
        num_tokens_padded: int,
    ) -> list[tuple[str, _WriteChunk]]:
        """Clone this step's deferred rows (build phase, no lock held).

        One whole-step D2D per key that is a step token tensor, then per-req
        views — same pattern as ``_get_device_snapshot``. List-valued deferred
        keys (Higgs ``codes.audio``) stay leftover.
        """
        n = num_tokens_unpadded
        deferred_tensors: dict[str, torch.Tensor] = {}
        for key in self._policy.deferred_keys:
            val = mm_outputs.get(key)
            if not _is_step_token_tensor(val, n, num_tokens_padded):
                continue
            if not self._pool.has_key(key):
                self._ensure_cache_key(key, val.dtype, int(val.shape[-1]))
            deferred_tensors[key] = val[:n].clone()
        if not deferred_tensors:
            return []
        holder = _SnapshotHolder(nbytes=sum(t.numel() * t.element_size() for t in deferred_tensors.values()))
        out: list[tuple[str, _WriteChunk]] = []
        for req_id in req_order:
            sched = num_sched[req_id]
            if sched <= 0:
                continue
            start = query_start[req_id]
            end = start + sched
            out.append(
                (
                    req_id,
                    _WriteChunk(
                        slots_cpu=slots_cpu[start:end],
                        tensors={k: v[start:end] for k, v in deferred_tensors.items()},
                        snapshot_holder=holder,
                    ),
                )
            )
        return out

    def _submit_step_writes(
        self,
        req_order: list[str],
        query_start: dict[str, int],
        num_sched: dict[str, int],
        device_snapshot: dict[str, torch.Tensor],
        slots_cpu: torch.Tensor,
        host_views: dict[str, torch.Tensor],
        freeze_event,
        staging_slot: int,
        step_d2h_event,
        bound_tids: list[int],
    ) -> None:
        """Caller holds ``_state_lock``. One queued WriteTask per request.

        Per-req views of the shared device snapshot: one D2D clone, req-scoped
        finish/abort, skip masks, and completion. Appends bound tids to
        `bound_tids` as it goes so a mid-loop raise still unwinds holders.
        """
        for req_id in req_order:
            start = query_start[req_id]
            end = start + num_sched[req_id]
            if end == start:
                continue
            tensors = {k: v[start:end] for k, v in device_snapshot.items()}
            tid = self._request_tasks.alloc_tid()
            chunk = _WriteChunk(slots_cpu=slots_cpu[start:end], tensors=tensors)
            # Host rows are views into the slot; the committer only waits
            # the shared step event. D2H is already in flight.
            chunk.host = {k: v[start:end] for k, v in host_views.items()}
            task = WriteTask(
                tid=tid,
                req_id=req_id,
                write_n=self._request_tasks.increment_write_n(req_id),
                schedule=WriteSchedule.JOIN_NEXT_STEP,
                chunks=[chunk],
                freeze_event=freeze_event,
                staging_slot=staging_slot,
                step_d2h_event=step_d2h_event,
            )
            self._map_slots(slots_cpu[start:end], tid, tensors.keys())
            # Bind before submit: the slot must never be holder-free
            # while the task is live (freed at completion drain).
            self._controller.staging_bind(staging_slot, StagingBufferHolder.for_task(tid))
            bound_tids.append(tid)
            self._controller.submit(task)
            self._request_tasks.track(req_id, tid)
            self._join_next_step_tids.append(tid)

    def _stage_deferred(self, deferred_chunks: list[tuple[str, _WriteChunk]], freeze_event) -> None:
        """Caller holds ``_state_lock``. Register pre-built deferred `_WriteChunk`s
        (bytes already reserved by save_outputs)."""
        for req_id, chunk in deferred_chunks:
            task = self._request_tasks.deferred.get(req_id)
            if task is not None and not self._controller.append_chunk(task, chunk, freeze_event):
                # Entry closed under us (cap flush / escalation): start a new one.
                task = None
            if task is None:
                task = WriteTask(
                    tid=self._request_tasks.alloc_tid(),
                    req_id=req_id,
                    write_n=self._request_tasks.increment_write_n(req_id),
                    schedule=WriteSchedule.JOIN_ON_FINISH,
                    chunks=[chunk],
                    freeze_event=freeze_event,
                )
                self._request_tasks.deferred[req_id] = task
                self._request_tasks.track(req_id, task.tid)
                self._controller.submit(task, queued=False)
            if chunk.snapshot_holder is not None:
                self._controller.pin_snapshot_holder(chunk.snapshot_holder, task.tid)
            # Block reuse across deferred tenants (preemption path) is
            # handled inside _map_slots: the old tenant's rows are skipped.
            self._map_slots(chunk.slots_cpu, task.tid, chunk.tensors.keys())

    # ----------------------------------------------------- occupancy

    def _ensure_cache_key(self, key: TensorName, dtype: torch.dtype, feat: int) -> None:
        """Open the pool slab and the occupancy row for ``key``."""
        self._pool.ensure_key(key, dtype, feat)
        self._slot_status.init_table(key)

    def _map_slots(self, slots: torch.Tensor, tid: int, keys: Iterable[str]) -> None:
        """Caller holds ``_state_lock``. Hang `tid` on these (slot, key);
        remounts go on the old WriteTask."""
        for old, key, stolen in self._slot_status.map_slots(slots, tid, keys):
            old_task = self._controller.get_task(old)
            if old_task is not None:
                old_task.add_reassigned(key, stolen)

    @torch.inference_mode()
    def _commit_drained_writes(self) -> None:
        """Fold completed/failed writes into occupancy. Caller holds ``_state_lock``."""
        failed = self._controller.drain_failed()
        if failed:
            # A failed write leaves rows absent behind hashes vLLM already
            # published — unservable and unrecoverable, so fatal. Raise here,
            # once, at the earliest facade entry instead of poisoning every
            # future hit that touches these slots.
            raise OmniPrefixCacheUnmatchError(
                f"prefix cache write failed for task(s) {failed}; cached rows lost behind published hashes"
            )
        drained = self._controller.drain_completed()
        if drained:
            self._slot_status.commit(drained)
            self._request_tasks.drop_completed(drained)

    # --------------------------------------------------- consume-once

    def _take_step_ctx(self, step_id: int) -> _StepContext:
        """Pop the context for this step id (consume-exactly-once). Caller holds ``_state_lock``."""
        ctx = self._step_ctxs.pop(step_id, None)
        if ctx is None:
            raise OmniPrefixCacheUnmatchError(
                f"step context {step_id} missing (have {sorted(self._step_ctxs)}); already consumed or never saved"
            )
        return ctx

    def _raise_if_unconsumed_ctxs_at_capacity(self) -> None:
        """Takes ``_state_lock``. Unconsumed contexts at `staging_depth` means
        the runner skipped both materialize and discard_step. Checked
        before claiming a staging slot so a full pool does not hide the
        leaked ids.

        This bound currently equals the slot-pool depth (same config field).
        They are different failures. TODO: claim a ticket on every sid
        save (including leftover-only); wait + timeout instead of raise.
        """
        with self._state_lock:
            if len(self._step_ctxs) >= self._config.staging_depth:
                raise OmniPrefixCacheUnmatchError(
                    f"{len(self._step_ctxs)} unconsumed step contexts (ids={sorted(self._step_ctxs)}); "
                    "runner violated the consume-exactly-once contract"
                )

    def _release_step_staging_slot(self, slot: int, step_id: int) -> None:
        self._controller.staging_release(slot, StagingBufferHolder.for_step(step_id))

    def _release_step_staging(self, ctx: _StepContext, step_id: int) -> None:
        """Drop this step's staging hold. Does not require ``_state_lock``.

        materialize/discard: task holds leave with drain.
        """
        if ctx.d2h is not None:
            self._release_step_staging_slot(ctx.d2h.staging_slot, step_id)

    def _release_staging_on_failed_save(
        self, slot: int, step_holder: StagingBufferHolder, bound_tids: list[int]
    ) -> None:
        """save raised after claiming the slot: drop the step hold and any
        task holds. Does not require ``_state_lock``.
        """
        self._release_step_staging_slot(slot, step_holder.owner_id)
        for tid in bound_tids:
            self._controller.staging_release(slot, StagingBufferHolder.for_task(tid))

    # -------------------------------------------------- slot ref / fetch

    def _get_hit_slots(self, req_id: str, hit_upto: int, hit_blocks: list[int]) -> torch.Tensor:
        """Prefix-hit block ids → KV slot ids. No table access; does not
        require ``_state_lock``.
        """
        bs = self._config.block_size
        assert hit_upto % bs == 0, (
            f"prefix hit not block aligned (req={req_id}, hit_upto={hit_upto}, block_size={bs}); "
            "vLLM invariant violated"
        )
        block_ids = torch.tensor(hit_blocks, dtype=torch.int64)
        return (block_ids.unsqueeze(1) * bs + torch.arange(bs)).reshape(-1)[:hit_upto]

    def _slot_ref(self, slots: torch.Tensor, key: str, req_id: str) -> _SlotRef:
        """Caller holds ``_state_lock``. Pin a ``_SlotRef`` for `slots` (no data movement).

        In-transit rows win over the mirror: their rows may not have been
        scattered yet, and a mirror read would return zero/stale values.
        JOIN_NEXT_STEP tasks go in ``join_tids`` (join-then-pool at
        fetch). JOIN_ON_FINISH tasks stay as refs for fetch_host.

        Hidden rejects any ABSENT hole (prefetch swallows; materialize
        logs). Other keys only need a source — holes fall to the mirror.
        """
        status = self._slot_status.get_slot_status(key)
        states = status.state[slots]
        tids = status.tids[slots]
        staged_mask = states == _Occupancy.IN_TRANSIT

        staged: list[tuple[WriteTask, torch.Tensor]] = []
        join_tids: list[int] = []
        for tid in {int(t) for t in tids[staged_mask].tolist()}:
            task = self._controller.get_task(tid) if tid != 0 else None
            if task is None:
                raise OmniPrefixCacheUnmatchError(
                    f"(slot, {key}) rows of req {req_id} are in-transit but entry {tid} cannot serve them"
                )
            if task.schedule is WriteSchedule.JOIN_NEXT_STEP:
                join_tids.append(task.tid)
            else:
                staged.append((task, staged_mask & (tids == tid)))

        already_staged = self._pool.has_key(key)
        has_source = already_staged or bool(staged) or bool(join_tids)
        if is_hidden_key(key):
            n_abs = int((states == _Occupancy.ABSENT).sum())
            if n_abs or not has_source:
                raise OmniPrefixCacheUnmatchError(
                    f"hit span for req {req_id} key={key} is not readable ({n_abs} absent slots)"
                )
        elif not has_source:
            raise KeyError(f"key {key} has no cache mirror")
        return _SlotRef(
            slots=slots,
            key=key,
            req_id=req_id,
            already_staged=already_staged,
            staged_list=staged,
            join_tids=join_tids,
        )

    def _fetch_source(self, src: _SlotRef) -> torch.Tensor:
        """Fetch a planned row source (execute phase, no lock).

        One key is one schedule: ``join_tids`` (JOIN_NEXT_STEP) and
        ``staged_list`` (JOIN_ON_FINISH) do not coexist. Immediate: wait
        ``done``, drain, read the pool. Deferred: pool rows already
        scattered, overlay ``fetch_host`` on the in-transit mask.
        """
        assert not (src.join_tids and src.staged_list), (
            f"{src.key}: JOIN_NEXT_STEP and JOIN_ON_FINISH in-transit on the same span"
        )
        # For JOIN_NEXT_STEP, wait `done`, drain, read the pool
        if src.join_tids:
            self._controller.join(src.join_tids)
            with self._state_lock:
                self._commit_drained_writes()
            out = self._pool.rows(src.key, src.slots)
            self._ensure_not_reassigned(src.slots, src.key, req_id=src.req_id)
            return out

        # For JOIN_ON_FINISH, pool rows already scattered, overlay `fetch_host` on the in-transit mask
        n = int(src.slots.numel())
        out: torch.Tensor | None = None
        if src.already_staged:
            out = self._pool.rows(src.key, src.slots)
        in_transit = None
        for task, mask in src.staged_list:
            try:
                rows = self._controller.fetch_host(task, src.slots[mask], src.key)
            except KeyError:
                raise OmniPrefixCacheUnmatchError(
                    f"(slot, {src.key}) rows of req {src.req_id} are staged in entry "
                    f"{task.tid} (req {task.req_id}, write_n {task.write_n}) but the task cannot serve them"
                ) from None
            if out is None:
                out = torch.zeros((n, rows.shape[-1]), dtype=rows.dtype)
            out[mask] = rows
            in_transit = mask if in_transit is None else in_transit | mask
        self._ensure_not_reassigned(src.slots, src.key, in_transit_mask=in_transit, req_id=src.req_id)
        return out

    def _ensure_not_reassigned(
        self,
        slots: torch.Tensor,
        key: str,
        *,
        in_transit_mask: torch.Tensor | None = None,
        req_id: str = "?",
    ) -> None:
        """Takes ``_state_lock``. Post-fetch check: pool rows read unlocked
        may have been remounted mid-read (block reuse). A torn pool read
        must fail-fast. JOIN_ON_FINISH slots already in-transit at plan
        time are excluded; JOIN_NEXT_STEP slots must be COMMITTED after drain.
        """
        with self._state_lock:
            status = self._slot_status.get_slot_status(key)
            violated = status.state[slots] == _Occupancy.IN_TRANSIT
            if in_transit_mask is not None:
                violated &= ~in_transit_mask
            if bool(violated.any()):
                raise OmniPrefixCacheUnmatchError(
                    f"(slot, {key}) rows of req {req_id} were reassigned to a new entry during "
                    f"materialize ({int(violated.sum())} slots; block reuse mid-read)"
                )

    # ---------------------------------------------------------- merge

    def _merge_cached_for_req(
        self,
        ctx: _StepContext,
        req_id: str,
        key: str,
        current_cpu: torch.Tensor,
        hit_sources: dict[tuple[str, str], _SlotRef | Future],
    ) -> torch.Tensor:
        """Hit prefix + this step's rows for one (req, key).

        No hit → this step's slice only. Prefetch Future → write the
        slice into the reserved tail. Else cat(fetch, new).
        """
        if req_id not in ctx.spans:
            # The caller passed a req the step context never saw: a silent
            # empty slice here would ship a zero-row payload downstream.
            raise OmniPrefixCacheUnmatchError(f"req {req_id} not in this step's context (had {list(ctx.spans)[:8]})")
        start, end = ctx.spans[req_id]
        new_rows = current_cpu[start:end]
        src = hit_sources.get((req_id, key))
        if src is None:
            return new_rows
        if isinstance(src, Future):
            # Prefetched during the forward, prefix already in place; only
            # this step's rows land here. result() re-raises fetch/validation
            # errors — the fail-fast contract survives the thread hop.
            merged = src.result()
            merged[merged.shape[0] - new_rows.shape[0] :] = new_rows
            return merged
        cached = self._fetch_source(src)
        return torch.cat([cached, new_rows], dim=0)

    def _merge_uncached_mm(
        self,
        ctx: _StepContext,
        req_ids: list[str],
        cached_keys: set[str],
        mm_out: dict[str, dict[str, Any]],
    ) -> None:
        """Write mm keys that are not in the prefix cache into mm_out.

        No hit concat: leftover mm was already copied to CPU at save
        (``ctx.mm_cpu_snapshot``). cached_keys already went through
        _merge_cached_for_req. ``req_ids`` is a subset of ``ctx.spans``.
        """
        uncached = {k: v for k, v in ctx.mm_cpu_snapshot.items() if k not in cached_keys and not is_hidden_key(k)}
        if not uncached:
            return
        from vllm_omni.utils.mm_outputs import to_payload_element

        order = list(ctx.spans)
        total_length = sum(e - s for s, e in ctx.spans.values())
        for key, val in uncached.items():
            per_req: dict[str, Any] = {}
            for req_id in req_ids:
                idx = order.index(req_id)
                start, end = ctx.spans[req_id]
                per_req[req_id] = to_payload_element(
                    val,
                    idx,
                    start=start,
                    end=end,
                    pass_lists_through=True,
                    seq_len=total_length,
                )
            mm_out[key] = per_req
