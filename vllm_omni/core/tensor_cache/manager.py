"""OmniTensorCacheManager: facade over the omni tensor cache.

Owns the block/slot semantic domain: slot-state authority, hit/span
registry, merge math, and the scheduler_output lifecycle stream. Data
movement is delegated to OmniTensorCacheController; state transitions are
applied here at fixed points (step join / materialize / drain), so req-level
consistency never depends on locks.

Runner contract (invariant 6): new_step_starts() before _update_states
removes finished requests; every real scheduler_output exactly once, in
order (warmup/dummy runs must not be fed).
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from vllm_omni.core.tensor_cache.block_pool import TensorBlockPool
from vllm_omni.core.tensor_cache.controller import (
    CLASS_A,
    CLASS_B,
    EntryWriteTask,
    OmniTensorCacheController,
    _Segment,
)
from vllm_omni.core.tensor_cache.group_view import KVCacheGroupView
from vllm_omni.core.tensor_cache.interface import (
    HIDDEN_KEY,
    ModelCachePolicy,
    OmniTensorCacheUnmatchError,
    StageCacheOutputs,
    TensorCacheConfig,
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
class _StepContext:
    entry_id: int | None
    slots_cpu: torch.Tensor | None
    hits: dict[str, tuple[int, list[int] | None]]  # req_id -> (hit_upto, hit block ids)
    num_scheduled: dict[str, int]
    req_order: list[str]
    query_start: dict[str, int]
    frozen_mm_keys: set[str]
    mm_flat_refs: dict[str, Any] = field(default_factory=dict)
    num_tokens_unpadded: int = 0


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
        # save_outputs, and all three mutate the slot tables.
        self._state_lock = threading.RLock()
        num_slots = config.num_blocks * config.block_size
        self._slot_state = torch.zeros(num_slots, dtype=torch.int8)
        self._slot_owner = torch.zeros(num_slots, dtype=torch.int64)
        self._entry_slots: dict[int, torch.Tensor] = {}
        self._req_entries: dict[str, set[int]] = {}
        self._deferred_tasks: dict[str, EntryWriteTask] = {}  # req_id -> task
        self._deferred_slot_owner: dict[str, dict[int, int]] = {}  # key -> slot -> eid
        self._next_entry_id = 1
        self._prev_class_b: int | None = None
        self._finished_join: set[int] = set()
        self._pending_hits: dict[str, int] = {}
        self._live_reqs: set[str] = set()
        self._step_ctxs: deque[_StepContext] = deque()
        self._cur_num_scheduled: dict[str, int] = {}

    # ------------------------------------------------------------- facade

    def register_policy(self, policy: ModelCachePolicy) -> None:
        self._policy = policy

    @_locked
    @torch.inference_mode()
    def new_step_starts(self, scheduler_output: SchedulerOutput) -> None:
        """Consume the lifecycle stream. Must run before _update_states."""
        self._apply_completions()

        finished = getattr(scheduler_output, "finished_req_ids", None) or ()
        for req_id in finished:
            self._live_reqs.discard(req_id)
            eids = self._req_entries.pop(req_id, set())
            dtask = self._deferred_tasks.pop(req_id, None)
            if dtask is not None:
                eids.add(dtask.entry_id)
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
                hit_blocks = list(blocks[: num_computed // self._config.block_size]) if blocks else None
                self._pending_hits[req_id] = (num_computed, hit_blocks)

        self._cur_num_scheduled = dict(scheduler_output.num_scheduled_tokens)

    @_locked
    @torch.inference_mode()
    def save_outputs(
        self,
        hidden_states: torch.Tensor | None,
        mm_outputs: dict[str, Any] | None,
        *,
        num_tokens_unpadded: int,
        num_tokens_padded: int,
    ) -> None:
        # Step-boundary join (consume-then-schedule): previous Class B task
        # plus tasks escalated for requests that finished this step.
        join_ids = list(self._finished_join)
        if self._prev_class_b is not None:
            join_ids.append(self._prev_class_b)
        if join_ids:
            self._controller.join(join_ids)
        self._finished_join.clear()
        self._prev_class_b = None
        self._apply_completions()

        n = num_tokens_unpadded
        req_order = self._view.batch_req_ids()
        num_sched = {r: int(self._cur_num_scheduled.get(r, 0)) for r in req_order}
        query_start: dict[str, int] = {}
        acc = 0
        for r in req_order:
            query_start[r] = acc
            acc += num_sched[r]

        slots_cpu: torch.Tensor | None = None
        entry_id: int | None = None
        frozen_mm_keys: set[str] = set()
        mm_flat = mm_outputs or {}

        if n > 0:
            # Derive the slot mapping on CPU: reading the device one back
            # would need a stream sync that waits on the whole forward.
            slots_cpu = self._view.step_slots_cpu(req_order, num_sched)
            if int(slots_cpu.numel()) != n:
                # Padding/clamping mismatch: fall back rather than miss-map rows.
                logger.warning("tensor_cache: slot mapping covers %d of %d tokens; skipping save", slots_cpu.numel(), n)
                self._step_ctxs.append(
                    _StepContext(
                        None, None, dict(self._pending_hits), num_sched, req_order, query_start, set(), dict(mm_flat), n
                    )
                )
                self._pending_hits.clear()
                return
            tensors: dict[str, torch.Tensor] = {}
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

            if tensors:
                freeze_event = None
                if torch.cuda.is_available() and any(t.is_cuda for t in tensors.values()):
                    freeze_event = torch.cuda.Event()
                    freeze_event.record()
                entry_id = self._alloc_entry_id()
                task = EntryWriteTask(
                    entry_id=entry_id,
                    klass=CLASS_A if n > self._config.class_a_min_tokens else CLASS_B,
                    segments=[_Segment(slots_cpu=slots_cpu, tensors=tensors)],
                    freeze_event=freeze_event,
                )
                task.deps = self._conflict_deps(slots_cpu, entry_id)
                if task.deps:
                    # Old tenant must land first AND must actually run: an
                    # unqueued (deferred) or lazy dep would otherwise stall
                    # this entry's join until its request finishes.
                    self._controller.escalate(sorted(task.deps))
                self._map_slots(slots_cpu, entry_id)
                self._controller.submit(task)
                for r in req_order:
                    self._req_entries.setdefault(r, set()).add(entry_id)
                if task.klass == CLASS_B:
                    self._prev_class_b = entry_id

            self._stage_deferred(mm_flat, slots_cpu, req_order, num_sched, query_start)

        self._step_ctxs.append(
            _StepContext(
                entry_id=entry_id,
                slots_cpu=slots_cpu,
                hits=dict(self._pending_hits),
                num_scheduled=num_sched,
                req_order=req_order,
                query_start=query_start,
                frozen_mm_keys=frozen_mm_keys,
                mm_flat_refs=dict(mm_flat),
                num_tokens_unpadded=n,
            )
        )
        self._pending_hits.clear()
        # Bounded per-step context FIFO: depth 2 covers the one-step-behind
        # async builder; anything older means a step skipped materialize.
        while len(self._step_ctxs) > 2:
            stale = self._step_ctxs.popleft()
            logger.warning("tensor_cache: dropping stale step context (entry=%s)", stale.entry_id)

    def _take_step_ctx(self, req_ids: list[str]) -> _StepContext | None:
        """Pop the context these req_ids belong to, or None.

        Steps that save but never materialize (no pooler payload, an early
        return, an aborted step) would otherwise shift the FIFO and make the
        next merge slice the wrong step's rows, so match on identity rather
        than trusting the order. Returning None means "no cached view for
        this step" — a capability gap the caller degrades around, unlike a
        hit span with absent slots, which is a data inconsistency and stays
        fail-fast.
        """
        want = set(req_ids)
        while self._step_ctxs:
            head = self._step_ctxs[0]
            # The output builder's request set can be a superset of the one
            # captured at save time (a request joins the batch between the
            # forward and the payload build). Own the step as long as the
            # contexts overlap; requests outside it simply have no cached
            # rows this step and are filled from the fresh slice instead.
            if want & set(head.req_order):
                return self._step_ctxs.popleft()
            stale = self._step_ctxs.popleft()
            logger.warning(
                "tensor_cache: dropping unrelated step context (entry=%s, ctx_reqs=%d, want=%d)",
                stale.entry_id,
                len(stale.req_order),
                len(want),
            )
        logger.warning("tensor_cache: no step context for %d requests; using uncached outputs", len(want))
        return None

    @_locked
    @torch.inference_mode()
    def materialize(self, req_ids: list[str]) -> StageCacheOutputs | None:
        """Merged outputs for this step, or None when no cached view covers
        these requests (caller falls back to the uncached path)."""
        ctx = self._take_step_ctx(req_ids)
        if ctx is None:
            return None
        self._apply_completions()

        if not self._policy.needs_full_hidden_states and not ctx.hits:
            return StageCacheOutputs(hidden_states=None, mm_outputs={})

        current: dict[str, torch.Tensor] = {}
        if ctx.entry_id is not None and ctx.slots_cpu is not None:
            for key in [HIDDEN_KEY, *ctx.frozen_mm_keys]:
                rows = self._entry_rows(ctx.entry_id, ctx.slots_cpu, key, mirror_fallback=True)
                if rows is not None:
                    current[key] = rows

        # Requests the step context does not cover produced no rows this step;
        # the caller fills those from the fresh slice.
        served = [r for r in req_ids if r in ctx.query_start]

        hidden_out: dict[str, torch.Tensor] | None = None
        if self._policy.needs_full_hidden_states and HIDDEN_KEY in current:
            hidden_out = {}
            for req_id in served:
                hidden_out[req_id] = self._merged_for_req(ctx, req_id, HIDDEN_KEY, current[HIDDEN_KEY], strict=True)

        mm_out: dict[str, dict[str, Any]] = {}
        cached_keys = (self._pool.keys() - {HIDDEN_KEY}) & set(ctx.mm_flat_refs.keys())
        for key in cached_keys:
            cur = current.get(key)
            if cur is None:
                val = ctx.mm_flat_refs.get(key)
                if not isinstance(val, torch.Tensor):
                    continue
                cur = val[: ctx.num_tokens_unpadded].detach().cpu()
            mm_out[key] = {req_id: self._merged_for_req(ctx, req_id, key, cur, strict=False) for req_id in served}

        self._merge_passthrough(ctx, served, cached_keys, mm_out)
        return StageCacheOutputs(hidden_states=hidden_out, mm_outputs=mm_out)

    @_locked
    def discard_step(self) -> None:
        """Drop the oldest step context when no consumer will materialize it
        (e.g. no pooler payload this step). The cache write still proceeds."""
        if self._step_ctxs:
            self._step_ctxs.popleft()

    def has_new_req_hits(self) -> bool:
        if self._step_ctxs:
            return bool(self._step_ctxs[0].hits)
        return bool(self._pending_hits)

    def shutdown(self) -> None:
        self._controller.shutdown()

    # ------------------------------------------------------------ internals

    def _alloc_entry_id(self) -> int:
        eid = self._next_entry_id
        self._next_entry_id += 1
        return eid

    def _conflict_deps(self, slots: torch.Tensor, entry_id: int) -> set[int]:
        states = self._slot_state[slots]
        mask = states == _IN_TRANSIT
        if not bool(mask.any()):
            return set()
        owners = set(self._slot_owner[slots][mask].tolist())
        owners.discard(entry_id)
        # Block reuse while the previous tenant is still in flight: old
        # entry must scatter first so the new rows win.
        return {o for o in owners if self._controller.get_task(o) is not None}

    def _map_slots(self, slots: torch.Tensor, entry_id: int) -> None:
        self._slot_state[slots] = _IN_TRANSIT
        self._slot_owner[slots] = entry_id
        self._entry_slots[entry_id] = slots

    def _apply_completions(self) -> None:
        drained = self._controller.drain_completed()
        for eid in drained:
            slots = self._entry_slots.pop(eid, None)
            if slots is None:
                continue
            # Only publish slots we still own: a later entry may have taken
            # them over (block reuse), and its write must win.
            still_owned = self._slot_owner[slots] == eid
            idx = slots[still_owned]
            self._slot_state[idx] = _COMMITTED
            self._slot_owner[idx] = 0
        if drained:
            # Drop completed ids so per-request sets stay bounded over long
            # streaming requests.
            done = set(drained)
            for req_id, eids in self._req_entries.items():
                eids -= done
            for key, owner_map in self._deferred_slot_owner.items():
                if any(e in done for e in owner_map.values()):
                    self._deferred_slot_owner[key] = {s: e for s, e in owner_map.items() if e not in done}

    def _stage_deferred(
        self,
        mm_flat: dict[str, Any],
        slots_cpu: torch.Tensor,
        req_order: list[str],
        num_sched: dict[str, int],
        query_start: dict[str, int],
    ) -> None:
        deferred_keys = [k for k in self._policy.deferred_keys if isinstance(mm_flat.get(k), torch.Tensor)]
        if not deferred_keys:
            return
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
            if not tensors:
                continue
            seg = _Segment(slots_cpu=slots_cpu[start:end], tensors=tensors)
            task = self._deferred_tasks.get(req_id)
            if task is not None and not self._controller.append_segment(task, seg):
                # Entry closed under us (cap flush / escalation): start a new one.
                task = None
            if task is None:
                task = EntryWriteTask(entry_id=self._alloc_entry_id(), klass=CLASS_A, segments=[seg])
                self._deferred_tasks[req_id] = task
                self._req_entries.setdefault(req_id, set()).add(task.entry_id)
                self._controller.submit(task, queued=False)
            for key in tensors:
                owner = self._deferred_slot_owner.setdefault(key, {})
                stale_owners: set[int] = set()
                for s in seg.slots_cpu.tolist():
                    prev = owner.get(s)
                    if prev is not None and prev != task.entry_id and self._controller.get_task(prev) is not None:
                        stale_owners.add(prev)
                    owner[s] = task.entry_id
                if stale_owners:
                    # Block reuse across deferred tenants (preemption path):
                    # old rows must scatter before ours so the mirror's final
                    # value belongs to the newest tenant.
                    task.deps.update(stale_owners)
                    self._controller.escalate(sorted(stale_owners))

    def _entry_rows(
        self, entry_id: int, slots: torch.Tensor, key: str, *, mirror_fallback: bool = False
    ) -> torch.Tensor | None:
        """Rows from a staged entry.

        mirror_fallback: for reads of this step's own rows, a completed entry
        must still resolve (via the mirror). Hit-span reads pass False so a
        committed entry falls through to the caller's mirror baseline.
        """
        task = self._controller.get_task(entry_id)
        if task is not None:
            try:
                return self._controller.fetch_host(task, slots, key)
            except KeyError:
                return None
        if mirror_fallback and self._pool.has_key(key):
            return self._pool.rows(key, slots)
        return None

    def _hit_rows(
        self, req_id: str, hit_upto: int, hit_blocks: list[int] | None, key: str, strict: bool
    ) -> torch.Tensor:
        bs = self._config.block_size
        assert hit_upto % bs == 0, (
            f"prefix hit not block aligned (req={req_id}, hit_upto={hit_upto}, block_size={bs}); "
            "vLLM invariant violated"
        )
        if hit_blocks is not None:
            block_ids = torch.tensor(hit_blocks, dtype=torch.int64)
        else:
            # Fallback for scheduler outputs without block_ids: only safe for
            # same-step (eager) materialize.
            block_ids = self._view.cached_block_ids(req_id).to(torch.int64)
        slots = (block_ids.unsqueeze(1) * bs + torch.arange(bs)).reshape(-1)[:hit_upto]

        states = self._slot_state[slots]
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

        return self._resolve_rows(slots, key, strict, req_id, states)

    def _resolve_rows(
        self,
        slots: torch.Tensor,
        key: str,
        strict: bool,
        req_id: str,
        states: torch.Tensor,
    ) -> torch.Tensor:
        """Resolve rows for `slots`, freshest tier first.

        Staged entries (in-transit main entries and deferred keys) win over
        the mirror: their rows may not have been scattered yet, so a mirror
        read would return zero/stale values.
        """
        n = int(slots.numel())
        deferred_owner = self._deferred_slot_owner.get(key) or {}
        # slot -> owning entry, preferring the staged tiers.
        owners = self._slot_owner[slots].clone()
        if deferred_owner:
            for i, s in enumerate(slots.tolist()):
                eid = deferred_owner.get(int(s))
                if eid is not None and self._controller.get_task(eid) is not None:
                    owners[i] = eid
        staged_mask = (states == _IN_TRANSIT) | torch.tensor(
            [bool(deferred_owner.get(int(s)) is not None) for s in slots.tolist()] if deferred_owner else [False] * n
        )

        out: torch.Tensor | None = None
        if self._pool.has_key(key):
            out = self._pool.rows(key, slots)

        for owner in {int(o) for o in owners[staged_mask].tolist()}:
            if owner == 0:
                continue
            own_mask = staged_mask & (owners == owner)
            rows = self._entry_rows(owner, slots[own_mask], key)
            if rows is None:
                continue
            if out is None:
                out = torch.zeros((n, rows.shape[-1]), dtype=rows.dtype)
            out[own_mask] = rows

        if out is None:
            if strict:
                raise OmniTensorCacheUnmatchError(f"no data source for hit span of req {req_id}, key {key}")
            raise KeyError(f"key {key} has no cache mirror")
        return out

    def _merged_for_req(
        self,
        ctx: _StepContext,
        req_id: str,
        key: str,
        current_cpu: torch.Tensor,
        strict: bool,
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
        hit = ctx.hits.get(req_id)
        if not hit:
            return new_rows
        hit_upto, hit_blocks = hit
        cached = self._hit_rows(req_id, hit_upto, hit_blocks, key, strict)
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
