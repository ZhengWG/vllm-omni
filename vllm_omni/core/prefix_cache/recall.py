# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Reading a prefix hit back out of the cache, in two typed phases.

    plan(slots, key, req_id) -> RecallPlan     caller holds the state lock;
                                               pins row sources, moves no data
    recall(plan) -> Tensor                     no lock held on entry; waits,
                                               reads, then re-checks under
                                               the lock

The split is the lock discipline made into types: the only way to get
rows is through a ``RecallPlan``, and the only place a plan is built is
inside the lock. A caller cannot fetch under the lock by accident.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import NoReturn

import torch

from vllm_omni.core.prefix_cache.block_pool import PrefixBlockPool
from vllm_omni.core.prefix_cache.controller import OmniPrefixCacheController, WriteTask
from vllm_omni.core.prefix_cache.interface import (
    OmniPrefixCacheUnmatchError,
    ReqId,
    TensorName,
    Tid,
    WriteSchedule,
    is_hidden_key,
)
from vllm_omni.core.prefix_cache.occupancy import Occupancy, SlotStatusTable


def raise_unreadable_hit(req_id: str, key: str, why: str) -> NoReturn:
    raise OmniPrefixCacheUnmatchError(f"hit span for req {req_id} key={key} is not readable ({why})")


@dataclass
class RecallPlan:
    """Where one (req, key) span's slots live: built under the lock, read outside it.

    Schedule split, not a single read tier:
    - JOIN_NEXT_STEP in-transit: ``join_tids``. Recall waits ``done``,
      publishes, then reads the pool. Staging views are never sliced.
    - JOIN_ON_FINISH in-transit: ``deferred`` task refs + masks. Recall
      uses ``fetch_host`` (device freeze / committer host).
    - Already in the CPU pool: ``in_pool`` → pool.

    A JOIN_NEXT_STEP task may disappear between plan and join (another
    entry already published it into the pool); ``join`` no-ops and the
    pool rows persist.
    """

    slots: torch.Tensor  # KV slot ids (PrefixBlockPool rows)
    key: TensorName
    req_id: ReqId
    in_pool: bool  # this key already has pool storage
    deferred: list[tuple[WriteTask, torch.Tensor]]  # JOIN_ON_FINISH only
    join_tids: list[Tid] = field(default_factory=list)  # JOIN_NEXT_STEP in-transit


class HitRecaller:
    """Plans and executes prefix-hit reads against the occupancy table.

    ``publish_writes`` is the manager's drain-and-commit; it expects the
    caller to hold ``lock`` and is what turns a joined JOIN_NEXT_STEP write
    into COMMITTED occupancy before the pool is read.
    """

    def __init__(
        self,
        *,
        pool: PrefixBlockPool,
        slot_status: SlotStatusTable,
        controller: OmniPrefixCacheController,
        lock: threading.Lock,
        publish_writes: Callable[[], None],
        block_size: int,
    ) -> None:
        self._pool = pool
        self._slot_status = slot_status
        self._controller = controller
        self._lock = lock
        self._publish_writes = publish_writes
        self._block_size = block_size

    def hit_slots(self, hit_upto: int, hit_blocks: list[int]) -> torch.Tensor:
        """Prefix-hit block ids → KV slot ids. Alignment is checked at
        ``new_step_starts``. No lock needed."""
        bs = self._block_size
        block_ids = torch.tensor(hit_blocks, dtype=torch.int64)
        return (block_ids.unsqueeze(1) * bs + torch.arange(bs)).reshape(-1)[:hit_upto]

    # ------------------------------------------------------------- plan

    def plan(self, slots: torch.Tensor, key: TensorName, req_id: ReqId) -> RecallPlan:
        """Caller holds ``lock``. Pin a plan for `slots` (no data movement).

        Rows still being written win over the CPU pool: they may not have
        landed yet, and a pool read would return zero/stale values.
        JOIN_NEXT_STEP tasks go in ``join_tids`` (wait-then-pool at
        recall). JOIN_ON_FINISH tasks stay as refs for fetch_host.

        Hidden rejects any empty hole (prefetch skips; materialize
        raises). Other keys only need a source — holes fall to the pool.
        """
        status = self._slot_status.get_slot_status(key)
        states = status.state[slots]
        tids = status.tids[slots]
        in_transit_mask = states == Occupancy.IN_TRANSIT

        deferred: list[tuple[WriteTask, torch.Tensor]] = []
        join_tids: list[Tid] = []
        for tid in {int(t) for t in tids[in_transit_mask].tolist()}:
            task = self._controller.get_task(tid) if tid != 0 else None
            if task is None:
                raise_unreadable_hit(req_id, key, f"in-transit entry {tid} cannot serve them")
            if task.schedule is WriteSchedule.JOIN_NEXT_STEP:
                join_tids.append(task.tid)
            else:
                deferred.append((task, in_transit_mask & (tids == tid)))

        in_pool = self._pool.has_key(key)
        has_source = in_pool or bool(deferred) or bool(join_tids)
        if is_hidden_key(key):
            n_abs = int((states == Occupancy.ABSENT).sum())
            if n_abs or not has_source:
                raise_unreadable_hit(req_id, key, f"{n_abs} absent slots")
        return RecallPlan(slots=slots, key=key, req_id=req_id, in_pool=in_pool, deferred=deferred, join_tids=join_tids)

    # ----------------------------------------------------------- recall

    def recall(self, plan: RecallPlan) -> torch.Tensor:
        """Execute a plan. Called without ``lock``; takes it only to publish
        and to re-check occupancy after the unlocked read.

        One key is one schedule: ``join_tids`` (JOIN_NEXT_STEP) and
        ``deferred`` (JOIN_ON_FINISH) do not coexist. Immediate: wait
        ``done``, publish, read the pool. Deferred: pool rows already
        written, overlay ``fetch_host`` on the still-in-progress mask.
        """
        if plan.join_tids:
            self._controller.join(plan.join_tids)
            with self._lock:
                self._publish_writes()
            pool_rows = self._pool.rows(plan.key, plan.slots)
            self._ensure_not_reassigned(plan)
            return pool_rows

        n = int(plan.slots.numel())
        out: torch.Tensor | None = None
        if plan.in_pool:
            out = self._pool.rows(plan.key, plan.slots)
        in_transit: torch.Tensor | None = None
        for task, mask in plan.deferred:
            try:
                rows = self._controller.fetch_host(task, plan.slots[mask], plan.key)
            except KeyError:
                raise_unreadable_hit(
                    plan.req_id,
                    plan.key,
                    f"entry {task.tid} (req {task.req_id}, write_n {task.write_n}) cannot serve them",
                )
            if out is None:
                out = torch.zeros((n, rows.shape[-1]), dtype=rows.dtype)
            out[mask] = rows
            in_transit = mask if in_transit is None else in_transit | mask
        self._ensure_not_reassigned(plan, in_transit_mask=in_transit)
        if out is None:
            raise_unreadable_hit(plan.req_id, plan.key, "no source")
        return out

    def _ensure_not_reassigned(self, plan: RecallPlan, *, in_transit_mask: torch.Tensor | None = None) -> None:
        """Takes ``lock``. Post-read check: pool rows read unlocked may have
        been given to a newer write mid-read (block reuse). A torn pool read
        must raise. JOIN_ON_FINISH slots already in-transit at plan time are
        excluded; JOIN_NEXT_STEP slots must be COMMITTED after the
        wait-then-publish.
        """
        with self._lock:
            status = self._slot_status.get_slot_status(plan.key)
            violated = status.state[plan.slots] == Occupancy.IN_TRANSIT
            if in_transit_mask is not None:
                violated &= ~in_transit_mask
            if bool(violated.any()):
                raise_unreadable_hit(
                    plan.req_id, plan.key, f"reassigned during materialize ({int(violated.sum())} slots)"
                )
