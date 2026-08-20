"""Omni prefix cache, manager side.

Owns the (slot, key) state tables, per-step snapshots, hit/span registry,
and the merge math. Data movement belongs to OmniTensorCacheController;
its completion events are drained here at fixed points. One non-reentrant
lock guards the state tables only — it never covers a join, a cap flush,
or a D2H.

Runner contract, per real scheduler_output and in order:
new_step_starts (before _update_states removes finished requests) ->
forward -> save_outputs -> materialize OR discard_step (the returned
step id is consumed exactly once). Warmup/dummy runs are never fed.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from vllm_omni.core.tensor_cache.block_pool import TensorBlockPool
from vllm_omni.core.tensor_cache.controller import (
    OmniTensorCacheController,
    WriteTask,
    _Segment,
)
from vllm_omni.core.tensor_cache.group_view import KVCacheGroupView
from vllm_omni.core.tensor_cache.interface import (
    HIDDEN_KEY,
    ModelCachePolicy,
    OmniTensorCacheUnmatchError,
    StageCacheOutputs,
    TensorCacheConfig,
    WriteSchedule,
)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

logger = logging.getLogger(__name__)

_ABSENT, _IN_TRANSIT, _COMMITTED = 0, 1, 2


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
class _RowSource:
    """Row sources resolved under the state lock, fetched outside it.

    Holds pinned task references, never a storage tier: a task may commit
    between plan and fetch, and fetch_host re-resolves where the rows live
    at call time (host buffers survive commit).
    """

    slots: torch.Tensor
    key: str
    req_id: str
    baseline_mirror: bool
    staged: list[tuple[WriteTask, torch.Tensor]]  # (task, mask over slots)
    staged_any: torch.Tensor | None = None  # union of staged masks (validation)


@dataclass
class _StepContext:
    task_ids: dict[str, int]  # req_id -> this step's WriteTask handle
    slots_cpu: torch.Tensor | None
    hits: dict[str, tuple[int, list[int] | None]]  # req_id -> (hit_upto, hit block ids)
    num_scheduled: dict[str, int]
    req_order: list[str]
    query_start: dict[str, int]
    frozen_mm_keys: set[str]
    mm_flat_refs: dict[str, Any] = field(default_factory=dict)
    num_tokens_unpadded: int = 0
    # Cached-key classification frozen at save time: recomputing it at
    # materialize would race a later save's ensure_key and reclassify a
    # passthrough key into an (all-absent) cached one mid-flight.
    cached_keys: set[str] = field(default_factory=set)
    # Whole-step frozen clones (the per-request tasks hold views of these).
    # materialize reads them with ONE D2H per key and feeds the host copy
    # back into the per-request tasks — without this, a cc8 decode step pays
    # 8 separate slice-D2H syncs on the read path and 8 more on the copy
    # thread (measured −25% cc8 tok/s).
    step_tensors: dict[str, torch.Tensor] = field(default_factory=dict)
    freeze_event: Any = None


class OmniTensorCacheManager:
    def __init__(
        self,
        config: TensorCacheConfig,
        view: KVCacheGroupView,
        *,
        eager: bool | None = None,
    ):
        self._config = config
        self._view = view
        self._pool = TensorBlockPool(config)
        self._controller = OmniTensorCacheController(self._pool, config, eager=eager)
        self._policy = ModelCachePolicy()
        # materialize() runs on the async output builder thread while the
        # engine thread is already in the next step's new_step_starts /
        # save_outputs, and all three mutate the slot tables. Non-reentrant
        # on purpose: facade entry points never call each other, and the lock
        # must never cover a blocking wait (join / cap flush / D2H).
        self._state_lock = threading.Lock()
        self._num_slots = config.num_blocks * config.block_size
        # (slot, key) state: per-key tables, lazily created. hidden may be
        # committed while a deferred key on the same slot is still in-transit;
        # a slot-keyed table cannot represent that (and made "never had this
        # key" indistinguishable from "write went missing").
        self._key_state: dict[str, torch.Tensor] = {}  # key -> int8[num_slots]
        self._key_owner: dict[str, torch.Tensor] = {}  # key -> int64[num_slots]
        self._task_slots: dict[int, torch.Tensor] = {}
        self._task_keys: dict[int, tuple[str, ...]] = {}
        self._req_tasks: dict[str, set[int]] = {}
        self._deferred_tasks: dict[str, WriteTask] = {}  # req_id -> task
        self._next_tid = 1
        self._join_next_step_tids: list[int] = []
        self._seq: dict[str, int] = {}  # req_id -> physical-task counter
        self._finished_join: set[int] = set()
        self._pending_hits: dict[str, int] = {}
        self._live_reqs: set[str] = set()
        # step_id -> context; each entry is consumed exactly once, by
        # materialize or discard_step (unknown/duplicate id = fail-fast).
        self._next_step_id = 1
        self._step_ctxs: dict[int, _StepContext] = {}
        self._cur_num_scheduled: dict[str, int] = {}

    # ------------------------------------------------------------- facade

    def register_policy(self, policy: ModelCachePolicy) -> None:
        self._policy = policy

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
        self._apply_completions()

        finished = getattr(scheduler_output, "finished_req_ids", None) or ()
        for req_id in finished:
            self._live_reqs.discard(req_id)
            self._seq.pop(req_id, None)
            eids = self._req_tasks.pop(req_id, set())
            dtask = self._deferred_tasks.pop(req_id, None)
            if dtask is not None:
                eids.add(dtask.tid)
            pending = [e for e in eids if self._controller.get_task(e) is not None]
            if pending:
                # Abort included: complete the writes (never roll back) so
                # still-hashed blocks stay servable.
                self._controller.escalate(pending)
                self._finished_join.update(pending)

        self._pending_hits.clear()
        for new_req in getattr(scheduler_output, "scheduled_new_reqs", ()) or ():
            req_id = new_req.req_id
            if req_id in self._live_reqs:
                # Streaming continuation (async_chunk): parity with legacy —
                # no hit marking; span/delivered_upto refinement is Phase 2.
                continue
            self._live_reqs.add(req_id)
            num_computed = int(getattr(new_req, "num_computed_tokens", 0) or 0)
            if num_computed > 0:
                # Snapshot the hit block table now: materialize must not read
                # the live input_batch (it may have advanced under the async
                # builder). block_ids is per-kv-group; group 0 only.
                blocks = getattr(new_req, "block_ids", None)
                if blocks is not None and len(blocks) > 0 and not isinstance(blocks[0], int):
                    blocks = blocks[0]
                if not blocks:
                    # Fail at the cause: a hit we cannot snapshot now would
                    # crash at materialize time with less context (materialize is
                    # forbidden from reading the live batch).
                    raise OmniTensorCacheUnmatchError(
                        f"prefix hit for req {req_id} ({num_computed} tokens) carries no block_ids"
                    )
                hit_blocks = list(blocks[: num_computed // self._config.block_size])
                self._pending_hits[req_id] = (num_computed, hit_blocks)

        self._cur_num_scheduled = dict(scheduler_output.num_scheduled_tokens)

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
        Freezes the step's rows (one D2D clone), submits one WriteTask per
        request, and snapshots everything materialize needs. The returned
        step id MUST be consumed exactly once — by materialize() or
        discard_step(); leaking contexts fails fast at a later save.

        The state lock never covers a blocking wait: the previous step's
        JOIN_NEXT_STEP join, the clone build, and the cap reservation (which may
        flush) all run unlocked.
        """
        with self._state_lock:
            # Step-boundary join (consume-then-schedule): the previous
            # step's JOIN_NEXT_STEP tasks plus tasks escalated for requests
            # finished this step.
            join_ids = list(self._finished_join)
            join_ids.extend(self._join_next_step_tids)
            self._finished_join.clear()
            self._join_next_step_tids = []
        if join_ids:
            # Copy-completion wait only: the join bounds in-flight GPU staging;
            # scatter completion is the committer's business.
            self._controller.join_host_ready(join_ids)

        n = num_tokens_unpadded
        req_order = self._view.batch_req_ids()
        num_sched = {r: int(self._cur_num_scheduled.get(r, 0)) for r in req_order}
        query_start: dict[str, int] = {}
        acc = 0
        for r in req_order:
            query_start[r] = acc
            acc += num_sched[r]

        slots_cpu: torch.Tensor | None = None
        frozen_mm_keys: set[str] = set()
        mm_flat = mm_outputs or {}
        tensors: dict[str, torch.Tensor] = {}
        deferred_segs: list[tuple[str, _Segment]] = []
        freeze_event = None

        if n > 0:
            # Derive the slot mapping on CPU: reading the device one back
            # would need a stream sync that waits on the whole forward.
            slots_cpu = self._view.step_slots_cpu(req_order, num_sched)
            if int(slots_cpu.numel()) != n:
                # Fail at the cause: skipping the save would leave rows absent
                # behind hashes vLLM already published — a delayed crash at
                # some future hit instead of a debuggable one here.
                raise OmniTensorCacheUnmatchError(
                    f"slot mapping covers {int(slots_cpu.numel())} of {n} scheduled tokens; "
                    "CPU-side slot derivation out of sync with the batch"
                )
            if hidden_states is not None and self._policy.needs_full_hidden_states:
                self._pool.ensure_key(HIDDEN_KEY, hidden_states.dtype, int(hidden_states.shape[-1]))
                tensors[HIDDEN_KEY] = hidden_states[:n].clone()
            for key, val in mm_flat.items():
                if key in self._policy.skip_keys or key in self._policy.deferred_keys:
                    continue
                if not isinstance(val, torch.Tensor) or val.ndim < 2:
                    continue
                if val.shape[0] == num_tokens_padded and not self._pool.has_key(key):
                    self._pool.ensure_key(key, val.dtype, int(val.shape[-1]))
                if not self._pool.has_key(key) or val.shape[0] < n:
                    continue
                tensors[key] = val[:n].clone()
                frozen_mm_keys.add(key)
            deferred_segs = self._build_deferred_segments(mm_flat, slots_cpu, req_order, num_sched, query_start)

            staged = [t for t in tensors.values()] + [t for _, seg in deferred_segs for t in seg.tensors.values()]
            if staged:
                if torch.cuda.is_available() and any(t.is_cuda for t in staged):
                    freeze_event = torch.cuda.Event()
                    freeze_event.record()
                # Cap reservation may block on a flush: outside the lock. The
                # flush must not close the deferred entries we are about to
                # append to (main-thread-only reads, safe unlocked).
                exclude = {self._deferred_tasks[r].tid for r, _ in deferred_segs if r in self._deferred_tasks}
                self._controller.reserve(sum(t.numel() * t.element_size() for t in staged), exclude=exclude)

        with self._state_lock:
            self._apply_completions()
            task_ids: dict[str, int] = {}
            if tensors:
                # One WriteTask per request (task_id = (req_id, key, seq)):
                # per-req views of the shared frozen clone, so the D2D freeze
                # stays a single kernel while finish/abort escalation, skip
                # masks, and completion validation are all req-scoped.
                for r in req_order:
                    s = query_start[r]
                    e = s + num_sched[r]
                    if e == s:
                        continue
                    r_tensors = {k: v[s:e] for k, v in tensors.items()}
                    tid = self._alloc_tid()
                    task = WriteTask(
                        tid=tid,
                        req_id=r,
                        seq=self._next_seq(r),
                        schedule=WriteSchedule.JOIN_ON_FINISH
                        if (e - s) > self._config.join_on_finish_min_tokens
                        else WriteSchedule.JOIN_NEXT_STEP,
                        segments=[_Segment(slots_cpu=slots_cpu[s:e], tensors=r_tensors)],
                        freeze_event=freeze_event,
                    )
                    self._map_slots(slots_cpu[s:e], tid, r_tensors.keys())
                    self._controller.submit(task, reserved=True)
                    self._req_tasks.setdefault(r, set()).add(tid)
                    task_ids[r] = tid
                    if task.schedule is WriteSchedule.JOIN_NEXT_STEP:
                        self._join_next_step_tids.append(tid)

            self._stage_deferred(deferred_segs, freeze_event)

            step_id = self._next_step_id
            self._next_step_id += 1
            self._step_ctxs[step_id] = _StepContext(
                task_ids=task_ids,
                slots_cpu=slots_cpu,
                hits=dict(self._pending_hits),
                num_scheduled=num_sched,
                req_order=req_order,
                query_start=query_start,
                frozen_mm_keys=frozen_mm_keys,
                mm_flat_refs=dict(mm_flat),
                num_tokens_unpadded=n,
                cached_keys=(self._pool.keys() - {HIDDEN_KEY}) & set(mm_flat.keys()),
                step_tensors=dict(tensors),
                freeze_event=freeze_event,
            )
            self._pending_hits.clear()
            if len(self._step_ctxs) > 4:
                # The async builder runs at most one step behind; more
                # unconsumed contexts means the runner is leaking them (a
                # consume path skipping both materialize and discard_step).
                raise OmniTensorCacheUnmatchError(
                    f"{len(self._step_ctxs)} unconsumed step contexts (ids={sorted(self._step_ctxs)}); "
                    "runner violated the consume-exactly-once contract"
                )
            return step_id

    def _take_step_ctx(self, step_id: int) -> _StepContext:
        """Pop the context for this step id (consume-exactly-once)."""
        ctx = self._step_ctxs.pop(step_id, None)
        if ctx is None:
            raise OmniTensorCacheUnmatchError(
                f"step context {step_id} missing (have {sorted(self._step_ctxs)}); already consumed or never saved"
            )
        return ctx

    @torch.inference_mode()
    def materialize(self, step_id: int, req_ids: list[str]) -> StageCacheOutputs:
        """Per-request merged outputs for the step saved as `step_id`.

        Any thread. `req_ids` must be (a subset of) the save-time snapshot;
        an outside id means the caller is reading the live batch (debug
        assert). A request without a hit is a plain miss and gets exactly
        this step's rows — normal path, nothing logged. A hit span that
        resolves to absent rows raises OmniTensorCacheUnmatchError: fatal
        by contract, never a degrade.

        Two phases: under the lock, drain completions and pin every row
        source (task refs + masks, absent checks included) — the storage
        tier is NOT baked in. Unlocked, fetch and merge: fetch_host
        re-resolves the tier at call time and pinned tasks stay readable
        after commit, so the engine thread never waits on this thread's
        PCIe.
        """
        with self._state_lock:
            ctx = self._take_step_ctx(step_id)
            self._apply_completions()

            # The builder must pass (a subset of) the req list captured at
            # save time — an id outside the snapshot means it is reading the
            # live batch, which the contract forbids (debug assert, not a
            # degrade path).
            assert set(req_ids) <= set(ctx.query_start), (
                f"materialize(step {step_id}) got req ids outside the save snapshot: "
                f"{sorted(set(req_ids) - set(ctx.query_start))[:8]}"
            )

            if not self._policy.needs_full_hidden_states and not ctx.hits:
                return StageCacheOutputs(hidden_states=None, mm_outputs={})

            served = list(req_ids)
            own_tasks = {r: self._controller.get_task(t) for r, t in ctx.task_ids.items()}
            want_hidden = self._policy.needs_full_hidden_states
            cached_keys = ctx.cached_keys

            hit_sources: dict[tuple[str, str], _RowSource] = {}
            for req_id in served:
                hit = ctx.hits.get(req_id)
                if not hit:
                    continue
                hit_upto, hit_blocks = hit
                keys = ([HIDDEN_KEY] if want_hidden else []) + sorted(cached_keys)
                for key in keys:
                    hit_sources[(req_id, key)] = self._plan_hit_rows(
                        req_id, hit_upto, hit_blocks, key, strict=(key == HIDDEN_KEY)
                    )

        # ---- unlocked: data movement + merge ----
        current: dict[str, torch.Tensor] = {}
        if ctx.slots_cpu is not None and ctx.task_ids:
            # Tier order matters: when the async builder runs a step late the
            # committer has already copied every task (host bufs) — reading
            # them is sync-free. snapshot_host is the SAME-STEP fallback
            # (eager consumers) and must wait for this step's forward on the
            # GPU, so preferring it on the builder would re-serialize the
            # async pipeline (measured: +3.4ms/step, itl p90 24ms vs 15ms).
            # Drained (None) means committed: the mirror read is sync-free too.
            all_ready = all((t := own_tasks.get(r)) is None or t.host_ready.is_set() for r in ctx.task_ids)
            for key in [HIDDEN_KEY, *ctx.frozen_mm_keys]:
                src = ctx.step_tensors.get(key)
                if src is not None and not all_ready:
                    # One D2H for the whole step (the per-req tasks hold
                    # views of this same frozen clone).
                    current[key] = self._controller.snapshot_host(src, ctx.freeze_event)
                else:
                    rows = self._step_rows(ctx, own_tasks, key)
                    if rows is not None:
                        current[key] = rows
            if ctx.step_tensors and not all_ready:
                # D6 feed-back, per-request form: hand each task its host
                # rows so the committer skips its own D2H (PCIe once).
                hosts = {k: current[k] for k in ctx.step_tensors if k in current}
                if hosts:
                    for r, _tid in ctx.task_ids.items():
                        task = own_tasks.get(r)
                        if task is None or task.host_ready.is_set():
                            continue
                        s = ctx.query_start[r]
                        e = s + ctx.num_scheduled.get(r, 0)
                        self._controller.publish_host(task, {k: v[s:e] for k, v in hosts.items()})

        hidden_out: dict[str, torch.Tensor] | None = None
        if want_hidden and HIDDEN_KEY in current:
            hidden_out = {}
            for req_id in served:
                hidden_out[req_id] = self._merged_for_req(ctx, req_id, HIDDEN_KEY, current[HIDDEN_KEY], hit_sources)

        mm_out: dict[str, dict[str, Any]] = {}
        for key in cached_keys:
            cur = current.get(key)
            if cur is None:
                val = ctx.mm_flat_refs.get(key)
                if not isinstance(val, torch.Tensor):
                    continue
                cur = val[: ctx.num_tokens_unpadded].detach().cpu()
            mm_out[key] = {req_id: self._merged_for_req(ctx, req_id, key, cur, hit_sources) for req_id in served}

        self._merge_passthrough(ctx, served, cached_keys, mm_out)
        return StageCacheOutputs(hidden_states=hidden_out, mm_outputs=mm_out)

    @_locked
    def discard_step(self, step_id: int) -> None:
        """Consume the step context when nothing will materialize it.

        Any thread; same exactly-once contract as materialize (unknown or
        duplicate id fails fast). Only the read-side snapshot is dropped —
        the cache write proceeds unchanged.
        """
        self._take_step_ctx(step_id)

    def shutdown(self) -> None:
        self._controller.shutdown()

    # ------------------------------------------------------------ internals

    def _alloc_tid(self) -> int:
        tid = self._next_tid
        self._next_tid += 1
        return tid

    def _next_seq(self, req_id: str) -> int:
        seq = self._seq.get(req_id, 0) + 1
        self._seq[req_id] = seq
        return seq

    def _tables(self, key: str) -> tuple[torch.Tensor, torch.Tensor]:
        state = self._key_state.get(key)
        if state is None:
            state = self._key_state[key] = torch.zeros(self._num_slots, dtype=torch.int8)
            self._key_owner[key] = torch.zeros(self._num_slots, dtype=torch.int64)
        return state, self._key_owner[key]

    def _map_slots(self, slots: torch.Tensor, tid: int, keys: Iterable[str]) -> None:
        """Hang `tid` on these (slot, key); reassignment = task swap.

        A (slot, key) still in-transit under another task means block reuse
        (the old request was preempted/aborted): push those rows into the old
        task's skip set — its mirror write skips them, ours lands, so the
        writes are disjoint and need no ordering edge.
        """
        keys = tuple(keys)
        for key in keys:
            state, owner = self._tables(key)
            cur = owner[slots]
            stale = (state[slots] == _IN_TRANSIT) & (cur != tid) & (cur != 0)
            if bool(stale.any()):
                for old in {int(o) for o in cur[stale].tolist()}:
                    old_task = self._controller.get_task(old)
                    if old_task is not None:
                        old_task.add_skip(key, slots[stale & (cur == old)])
            state[slots] = _IN_TRANSIT
            owner[slots] = tid
        prev = self._task_slots.get(tid)
        if prev is None:
            self._task_slots[tid] = slots
            self._task_keys[tid] = keys
        else:
            # Deferred tasks grow one segment per step.
            self._task_slots[tid] = torch.cat([prev, slots])
            self._task_keys[tid] = tuple(dict.fromkeys(self._task_keys[tid] + keys))

    def _apply_completions(self) -> None:
        failed = self._controller.drain_failed()
        if failed:
            # A failed write leaves rows absent behind hashes vLLM already
            # published — unservable and unrecoverable, so fatal. Raise here,
            # once, at the earliest facade entry instead of poisoning every
            # future hit that touches these slots.
            raise OmniTensorCacheUnmatchError(
                f"tensor cache write failed for task(s) {failed}; cached rows lost behind published hashes"
            )
        drained = self._controller.drain_completed()
        for tid in drained:
            slots = self._task_slots.pop(tid, None)
            keys = self._task_keys.pop(tid, ())
            if slots is None:
                continue
            for key in keys:
                state, owner = self._tables(key)
                # Only publish slots we still own: a later entry may have
                # taken them over (block reuse), and its write must win.
                still_owned = owner[slots] == tid
                idx = slots[still_owned]
                state[idx] = _COMMITTED
                owner[idx] = 0
        if drained:
            # Drop completed ids so per-request sets stay bounded over long
            # streaming requests.
            done = set(drained)
            for req_id, eids in self._req_tasks.items():
                eids -= done

    def _build_deferred_segments(
        self,
        mm_flat: dict[str, Any],
        slots_cpu: torch.Tensor,
        req_order: list[str],
        num_sched: dict[str, int],
        query_start: dict[str, int],
    ) -> list[tuple[str, _Segment]]:
        """Clone this step's deferred rows (build phase, no lock held)."""
        deferred_keys = [k for k in self._policy.deferred_keys if isinstance(mm_flat.get(k), torch.Tensor)]
        if not deferred_keys:
            return []
        segs: list[tuple[str, _Segment]] = []
        for req_id in req_order:
            sched = num_sched[req_id]
            if sched <= 0:
                continue
            start = query_start[req_id]
            end = start + sched
            tensors: dict[str, torch.Tensor] = {}
            for key in deferred_keys:
                val = mm_flat[key]
                if val.ndim < 2 or val.shape[0] < end:
                    continue
                if not self._pool.has_key(key):
                    self._pool.ensure_key(key, val.dtype, int(val.shape[-1]))
                tensors[key] = val[start:end].clone()
            if tensors:
                segs.append((req_id, _Segment(slots_cpu=slots_cpu[start:end], tensors=tensors)))
        return segs

    def _stage_deferred(self, segs: list[tuple[str, _Segment]], freeze_event) -> None:
        """Register pre-built deferred segments (locked phase; bytes already
        reserved by save_outputs)."""
        for req_id, seg in segs:
            task = self._deferred_tasks.get(req_id)
            if task is not None and not self._controller.append_segment(task, seg, reserved=True):
                # Entry closed under us (cap flush / escalation): start a new one.
                task = None
            if task is not None and freeze_event is not None:
                # Events on one compute stream are ordered: the newest freeze
                # event also covers every earlier segment's clone, so the
                # background copy never races this step's in-flight D2D.
                task.freeze_event = freeze_event
            if task is None:
                task = WriteTask(
                    tid=self._alloc_tid(),
                    req_id=req_id,
                    seq=self._next_seq(req_id),
                    schedule=WriteSchedule.JOIN_ON_FINISH,
                    segments=[seg],
                    freeze_event=freeze_event,
                )
                self._deferred_tasks[req_id] = task
                self._req_tasks.setdefault(req_id, set()).add(task.tid)
                self._controller.submit(task, queued=False, reserved=True)
            # Block reuse across deferred tenants (preemption path) is
            # handled inside _map_slots: the old tenant's rows are skipped.
            self._map_slots(seg.slots_cpu, task.tid, seg.tensors.keys())

    def _step_rows(self, ctx: _StepContext, own_tasks: dict[str, WriteTask | None], key: str) -> torch.Tensor | None:
        """Whole-step rows for `key`, assembled from the per-request tasks
        (execute phase, no lock). Offsets match ctx.query_start."""
        parts: list[torch.Tensor] = []
        for r in ctx.req_order:
            s = ctx.query_start[r]
            e = s + ctx.num_scheduled.get(r, 0)
            if e == s:
                continue
            rows = self._fetch_entry_rows(own_tasks.get(r), ctx.slots_cpu[s:e], key, ctx.task_ids.get(r))
            if rows is None:
                return None
            parts.append(rows)
        return torch.cat(parts, dim=0) if parts else None

    def _fetch_entry_rows(
        self, task: WriteTask | None, slots: torch.Tensor, key: str, tid: int | None
    ) -> torch.Tensor | None:
        """This step's own rows (execute phase, no lock).

        The pinned task stays readable after commit; a drained (or absent)
        entry resolves via the mirror instead — validated against mid-read
        block reuse (only our own entry may hold these rows in-transit).
        """
        if task is not None:
            try:
                return self._controller.fetch_host(task, slots, key)
            except KeyError:
                pass
        if self._pool.has_key(key):
            rows = self._pool.rows(key, slots)
            self._ensure_not_reassigned(slots, key, allow_tid=tid)
            return rows
        return None

    def _plan_hit_rows(
        self, req_id: str, hit_upto: int, hit_blocks: list[int] | None, key: str, strict: bool
    ) -> _RowSource:
        """Resolve a hit span into a row-source plan (locked phase)."""
        bs = self._config.block_size
        assert hit_upto % bs == 0, (
            f"prefix hit not block aligned (req={req_id}, hit_upto={hit_upto}, block_size={bs}); "
            "vLLM invariant violated"
        )
        if hit_blocks is None:
            # The hit block table is snapshotted in new_step_starts precisely
            # so materialize never reads the live batch; a
            # missing snapshot is a registration bug, not a fallback case.
            raise OmniTensorCacheUnmatchError(
                f"no hit-block snapshot for req {req_id} (hit_upto={hit_upto}); hit registered without block_ids"
            )
        block_ids = torch.tensor(hit_blocks, dtype=torch.int64)
        slots = (block_ids.unsqueeze(1) * bs + torch.arange(bs)).reshape(-1)[:hit_upto]

        state = self._key_state.get(key)
        states = state[slots] if state is not None else torch.zeros(int(slots.numel()), dtype=torch.int8)
        if strict and bool((states == _ABSENT).any()):
            absent = slots[states == _ABSENT]
            logger.critical(
                "omni tensor cache unmatch: req=%s key=%s hit_upto=%d absent_slots=%s states=%s",
                req_id,
                key,
                hit_upto,
                absent[:32].tolist(),
                states[:64].tolist(),
            )
            raise OmniTensorCacheUnmatchError(
                f"hit span for req {req_id} resolved to {int((states == _ABSENT).sum())} absent slots"
            )

        return self._plan_rows(slots, key, strict, req_id, states)

    def _plan_rows(
        self,
        slots: torch.Tensor,
        key: str,
        strict: bool,
        req_id: str,
        states: torch.Tensor | None = None,
    ) -> _RowSource:
        """Pin the row sources for `slots` (locked phase; no data movement).

        In-transit rows win over the mirror: their rows may not have been
        scattered yet, and a mirror read would return zero/stale values.
        Per-key absent semantics: rows never registered for this key fall to
        the mirror baseline (legitimate for sparse/deferred keys — the strict
        caller has already rejected absent); rows registered in-transit whose
        entry cannot serve them are a bookkeeping error, never silent zeros.
        The plan holds task references, not storage tiers: fetch_host
        re-resolves the tier when the fetch actually runs.
        """
        n = int(slots.numel())
        if states is None:
            state = self._key_state.get(key)
            states = state[slots] if state is not None else torch.zeros(n, dtype=torch.int8)
        owner_table = self._key_owner.get(key)
        owners = owner_table[slots] if owner_table is not None else torch.zeros(n, dtype=torch.int64)
        staged_mask = states == _IN_TRANSIT

        staged: list[tuple[WriteTask, torch.Tensor]] = []
        for owner in {int(o) for o in owners[staged_mask].tolist()}:
            task = self._controller.get_task(owner) if owner != 0 else None
            if task is None:
                # in-transit implies a live owner entry (state flips to
                # committed in the same locked drain that retires the task).
                # Zeros here would ship silently downstream.
                raise OmniTensorCacheUnmatchError(
                    f"(slot, {key}) rows of req {req_id} are in-transit but entry {owner} cannot serve them"
                )
            staged.append((task, staged_mask & (owners == owner)))

        baseline = self._pool.has_key(key)
        if not baseline and not staged:
            if strict:
                raise OmniTensorCacheUnmatchError(f"no data source for hit span of req {req_id}, key {key}")
            raise KeyError(f"key {key} has no cache mirror")
        return _RowSource(
            slots=slots, key=key, req_id=req_id, baseline_mirror=baseline, staged=staged, staged_any=staged_mask
        )

    def _fetch_source(self, src: _RowSource) -> torch.Tensor:
        """Fetch a planned row source (execute phase, no lock)."""
        n = int(src.slots.numel())
        out: torch.Tensor | None = None
        if src.baseline_mirror:
            out = self._pool.rows(src.key, src.slots)
        for task, mask in src.staged:
            try:
                rows = self._controller.fetch_host(task, src.slots[mask], src.key)
            except KeyError:
                raise OmniTensorCacheUnmatchError(
                    f"(slot, {src.key}) rows of req {src.req_id} are staged in entry "
                    f"{task.tid} (req {task.req_id}, seq {task.seq}) but the task cannot serve them"
                ) from None
            if out is None:
                out = torch.zeros((n, rows.shape[-1]), dtype=rows.dtype)
            out[mask] = rows
        assert out is not None  # _plan_rows guarantees a source
        self._ensure_not_reassigned(src.slots, src.key, planned_staged=src.staged_any, req_id=src.req_id)
        return out

    def _ensure_not_reassigned(
        self,
        slots: torch.Tensor,
        key: str,
        *,
        planned_staged: torch.Tensor | None = None,
        allow_tid: int | None = None,
        req_id: str = "?",
    ) -> None:
        """Post-fetch validation: baseline (mirror) rows read without a lock
        may have been re-registered by a new tenant mid-read (block reuse
        after abort/preemption of the hit request) — the scatter races the
        mirror read, so a torn read must fail loudly, never ship."""
        with self._state_lock:
            state = self._key_state.get(key)
            if state is None:
                return
            violated = state[slots] == _IN_TRANSIT
            if planned_staged is not None:
                violated &= ~planned_staged
            if allow_tid is not None and bool(violated.any()):
                violated &= self._key_owner[key][slots] != allow_tid
            if bool(violated.any()):
                raise OmniTensorCacheUnmatchError(
                    f"(slot, {key}) rows of req {req_id} were reassigned to a new entry during "
                    f"materialize ({int(violated.sum())} slots; block reuse mid-read)"
                )

    def _resolve_rows(
        self,
        slots: torch.Tensor,
        key: str,
        strict: bool,
        req_id: str,
        states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Plan + fetch in one call (tests / debugging)."""
        with self._state_lock:
            src = self._plan_rows(slots, key, strict, req_id, states)
        return self._fetch_source(src)

    def _merged_for_req(
        self,
        ctx: _StepContext,
        req_id: str,
        key: str,
        current_cpu: torch.Tensor,
        hit_sources: dict[tuple[str, str], _RowSource],
    ) -> torch.Tensor:
        if req_id not in ctx.query_start:
            # The caller passed a req the step context never saw: a silent
            # empty slice here would ship a zero-row payload downstream.
            raise OmniTensorCacheUnmatchError(
                f"req {req_id} not in this step's context (had {list(ctx.query_start)[:8]})"
            )
        start = ctx.query_start[req_id]
        end = start + ctx.num_scheduled.get(req_id, 0)
        new_rows = current_cpu[start:end]
        src = hit_sources.get((req_id, key))
        if src is None:
            return new_rows
        cached = self._fetch_source(src)
        return torch.cat([cached, new_rows], dim=0)

    def _merge_passthrough(
        self,
        ctx: _StepContext,
        req_ids: list[str],
        cached_keys: set[str],
        mm_out: dict[str, dict[str, Any]],
    ) -> None:
        passthrough = {k: v for k, v in ctx.mm_flat_refs.items() if k not in cached_keys and k != HIDDEN_KEY}
        if not passthrough:
            return
        from vllm_omni.utils.mm_outputs import build_mm_cpu, to_payload_element

        mm_cpu = build_mm_cpu(multimodal_outputs=passthrough)
        total = sum(ctx.num_scheduled.get(r, 0) for r in ctx.req_order)
        for key, val in mm_cpu.items():
            per_req: dict[str, Any] = {}
            for req_id in req_ids:
                idx = ctx.req_order.index(req_id) if req_id in ctx.req_order else 0
                start = ctx.query_start.get(req_id, 0)
                end = start + ctx.num_scheduled.get(req_id, 0)
                per_req[req_id] = to_payload_element(
                    val,
                    idx,
                    start=start,
                    end=end,
                    pass_lists_through=True,
                    seq_len=total,
                )
            mm_out[key] = per_req
