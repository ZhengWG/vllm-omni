# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Per (KV slot, tensor name) occupancy: absent, being written, or in the pool.

Owned by the manager and read under its ``_state_lock``. The recaller
plans reads against it; the manager commits drained writes into it.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import IntEnum
from typing import NamedTuple

import torch

from vllm_omni.core.prefix_cache.interface import TensorName, Tid


class Occupancy(IntEnum):
    ABSENT = 0
    IN_TRANSIT = 1
    COMMITTED = 2


class SlotStatus(NamedTuple):
    """Occupancy row for one tensor name (views into the table, not copies)."""

    state: torch.Tensor  # int8[num_slots]
    tids: torch.Tensor  # Tid per kv slot; 0 = none


class SlotStatusTable:
    """Per (KV slot, tensor name): empty, being written, or already in the pool.

    Hidden and a deferred mm field on the same slot are independent.
    ``map_slots`` marks a write in progress; if another task still owns
    the slot, the manager records those rows as no longer owned by it.
    ``commit`` runs after the pool write: still-owned slots become
    committed.
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

    def get_slot_status(self, key: TensorName) -> SlotStatus:
        """Occupancy tensors for ``key``. ``init_table`` must have run."""
        return SlotStatus(state=self.state[key], tids=self.tids[key])

    def map_slots(
        self, slots: torch.Tensor, tid: Tid, keys: Iterable[TensorName]
    ) -> list[tuple[Tid, TensorName, torch.Tensor]]:
        """Record ``tid`` on these (slot, key). Return in-transit rows
        another write still owned (caller marks them skipped on it)."""
        keys = tuple(keys)
        stolen: list[tuple[Tid, TensorName, torch.Tensor]] = []
        for key in keys:
            status = self.get_slot_status(key)
            cur = status.tids[slots]
            stale = (status.state[slots] == Occupancy.IN_TRANSIT) & (cur != tid) & (cur != 0)
            if bool(stale.any()):
                for old in {int(o) for o in cur[stale].tolist()}:
                    stolen.append((old, key, slots[stale & (cur == old)]))
            status.state[slots] = Occupancy.IN_TRANSIT
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
                status.state[idx] = Occupancy.COMMITTED
                status.tids[idx] = 0
