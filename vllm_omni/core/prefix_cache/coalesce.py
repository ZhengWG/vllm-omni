# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Block-granular prefix-cache writes for decode rows.

vLLM caches only full KV blocks, so a decode row becomes reachable by a hit
only once its block is full. On a pure decode step the runner parks each row
in a per-request device ring instead of running the per-step save; a block
that fills is written through ``save_outputs`` in one call, and every other
step writes all parked rows before its own save. Parked rows of a request that
finished, aborted, resumed or was preempted are dropped: they belong to a
partial block vLLM never caches, and its blocks may already be reassigned.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch

from vllm_omni.core.prefix_cache.interface import OmniPrefixCacheUnmatchError


@dataclass(frozen=True)
class ParkedRows:
    """Rows taken out of the rings, packed in ``req_ids`` order."""

    req_ids: tuple[str, ...]
    counts: tuple[int, ...]
    slots_cpu: torch.Tensor  # int64 KV slot per row
    rows: dict[str, torch.Tensor]  # key -> [sum(counts), width], on the producing device


class DecodeRowCoalescer:
    """Per-request rings of at most one block of not-yet-written decode rows."""

    def __init__(self, block_size: int, capacity: int) -> None:
        self.block_size = block_size
        self.capacity = capacity
        self._rings: dict[str, torch.Tensor] = {}  # key -> [capacity * block_size, width]
        self._ring_row: dict[str, int] = {}
        self._free_rows = list(range(capacity - 1, -1, -1))
        # req -> (slot of the block's first position, first parked offset, next offset)
        self._parked: dict[str, tuple[int, int, int]] = {}

    def has_parked(self) -> bool:
        return bool(self._parked)

    def keys(self) -> set[str]:
        return set(self._rings)

    def reset_keys(self) -> None:
        """Drop the rings; only valid with nothing parked."""
        assert not self._parked
        self._rings.clear()

    def has_room(self, req_ids: Iterable[str]) -> bool:
        new = sum(1 for req_id in req_ids if req_id not in self._ring_row)
        return new <= len(self._free_rows)

    def drop(self, req_ids: Iterable[str]) -> None:
        for req_id in req_ids:
            self._parked.pop(req_id, None)
            row = self._ring_row.pop(req_id, None)
            if row is not None:
                self._free_rows.append(row)

    def park(self, req_ids: list[str], slots: np.ndarray, rows: dict[str, torch.Tensor]) -> list[str]:
        """Park row ``i`` of every tensor for ``req_ids[i]`` at KV slot ``slots[i]``.

        Returns the requests whose block this step filled.
        """
        bs = self.block_size
        n = len(req_ids)
        ring_index = np.empty(n, dtype=np.int64)
        filled: list[str] = []
        for i, req_id in enumerate(req_ids):
            slot = int(slots[i])
            base, offset = slot - slot % bs, slot % bs
            parked = self._parked.get(req_id)
            if parked is None:
                parked = (base, offset, offset + 1)
            elif parked[0] != base or parked[2] != offset:
                raise OmniPrefixCacheUnmatchError(
                    f"decode row for req {req_id} at slot {slot} does not continue its parked block "
                    f"(base={parked[0]}, next offset={parked[2]})"
                )
            else:
                parked = (base, parked[1], offset + 1)
            self._parked[req_id] = parked
            row = self._ring_row.get(req_id)
            if row is None:
                row = self._free_rows.pop()
                self._ring_row[req_id] = row
            ring_index[i] = row * bs + offset
            if offset == bs - 1:
                filled.append(req_id)
        index = torch.from_numpy(ring_index)
        for key, val in rows.items():
            ring = self._rings.get(key)
            if ring is None:
                ring = torch.empty((self.capacity * bs, int(val.shape[-1])), dtype=val.dtype, device=val.device)
                self._rings[key] = ring
            ring.index_copy_(0, _on(index, ring.device), val[:n].reshape(n, -1))
        return filled

    def take(self, req_ids: Iterable[str]) -> ParkedRows | None:
        """Remove and return the parked rows of ``req_ids`` (those with none are skipped)."""
        bs = self.block_size
        taken = [(req_id, self._parked.pop(req_id)) for req_id in req_ids if req_id in self._parked]
        if not taken:
            return None
        counts: list[int] = []
        slot_parts: list[np.ndarray] = []
        ring_parts: list[np.ndarray] = []
        for req_id, (base, lo, hi) in taken:
            counts.append(hi - lo)
            slot_parts.append(np.arange(base + lo, base + hi, dtype=np.int64))
            start = self._ring_row[req_id] * bs
            ring_parts.append(np.arange(start + lo, start + hi, dtype=np.int64))
        index = torch.from_numpy(np.concatenate(ring_parts))
        rows = {key: ring.index_select(0, _on(index, ring.device)) for key, ring in self._rings.items()}
        return ParkedRows(
            req_ids=tuple(req_id for req_id, _ in taken),
            counts=tuple(counts),
            slots_cpu=torch.from_numpy(np.concatenate(slot_parts)),
            rows=rows,
        )

    def take_all(self) -> ParkedRows | None:
        return self.take(list(self._parked))


def _on(index: torch.Tensor, device: torch.device) -> torch.Tensor:
    """CPU index tensor on ``device``; a pinned staging copy keeps the H2D async."""
    if device.type == "cpu":
        return index
    return index.pin_memory().to(device, non_blocking=True)
