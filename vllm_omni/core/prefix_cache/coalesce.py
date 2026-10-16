# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Block-granular prefix-cache writes for decode rows.

vLLM caches only full KV blocks, so a decode row becomes reachable by a hit
only once its block is full, and only through a request that starts with that
block as a prefix hit. The runner therefore parks decode rows instead of
writing them every step, and writes full blocks in batches: when a request
has parked ``max_full_blocks`` full blocks, before a step whose new requests
hit a parked block, and when a request leaves. A departed request's partial
block is dropped: vLLM never caches it, and its blocks may be reassigned.

A parked row is a reference into a once-per-step clone of the step's outputs,
so parking costs one device copy per key and no host-to-device traffic.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable
from dataclasses import dataclass, field

import numpy as np
import torch

from vllm_omni.core.prefix_cache.interface import OmniPrefixCacheUnmatchError


@dataclass(frozen=True)
class ParkedRows:
    """Rows taken out of the coalescer, packed in ``req_ids`` order."""

    req_ids: tuple[str, ...]
    counts: tuple[int, ...]
    slots_cpu: torch.Tensor  # int64 KV slot per row
    rows: dict[str, torch.Tensor]  # key -> [sum(counts), width], on the producing device


@dataclass
class _Block:
    base: int  # KV slot of the block's first position
    lo: int  # first parked offset
    hi: int  # next offset
    refs: list[tuple[dict[str, torch.Tensor], int]] = field(default_factory=list)  # (step clone, row)


class DecodeRowCoalescer:
    """Parked decode rows per request, as consecutive blocks."""

    def __init__(self, block_size: int, max_full_blocks: int = 4) -> None:
        self.block_size = block_size
        self.max_full_blocks = max_full_blocks
        self._blocks: dict[str, list[_Block]] = {}
        self._keys: frozenset[str] = frozenset()

    def has_parked(self) -> bool:
        return bool(self._blocks)

    def keys(self) -> frozenset[str]:
        return self._keys

    def open_bases(self) -> dict[str, int]:
        """Base slot of each request's last block while it is still filling."""
        return {r: b[-1].base for r, b in self._blocks.items() if b[-1].hi < self.block_size}

    def park(
        self,
        req_ids: list[str],
        slots: np.ndarray,
        rows: dict[str, torch.Tensor],
        row_index: list[int] | None = None,
    ) -> bool:
        """Park one row per request: ``rows[key][row_index[i]]`` at KV slot ``slots[i]``.

        ``rows`` is cloned once. Returns True when some request now holds
        ``max_full_blocks`` full blocks.
        """
        if self._blocks and frozenset(rows) != self._keys:
            raise OmniPrefixCacheUnmatchError(f"parked keys {sorted(self._keys)} differ from {sorted(rows)}")
        self._keys = frozenset(rows)
        bs = self.block_size
        step = {key: val.clone() for key, val in rows.items()}
        index = row_index if row_index is not None else range(len(req_ids))
        flush = False
        for req_id, slot, row in zip(req_ids, (int(s) for s in slots), index):
            base, offset = slot - slot % bs, slot % bs
            blocks = self._blocks.get(req_id)
            if not blocks:
                self._blocks[req_id] = blocks = [_Block(base, offset, offset + 1)]
            elif blocks[-1].base == base and blocks[-1].hi == offset:
                blocks[-1].hi += 1
            elif blocks[-1].hi == bs and offset == 0:
                blocks.append(_Block(base, 0, 1))
            else:
                last = blocks[-1]
                raise OmniPrefixCacheUnmatchError(
                    f"decode row for req {req_id} at slot {slot} does not continue its parked rows "
                    f"(block base={last.base}, next offset={last.hi})"
                )
            blocks[-1].refs.append((step, row))
            if offset == bs - 1 and len(blocks) >= self.max_full_blocks:
                flush = True
        return flush

    def take(
        self,
        req_ids: Iterable[str] | None = None,
        *,
        full_only: bool = False,
        bases: Collection[int] | None = None,
    ) -> ParkedRows | None:
        """Remove and return parked rows: all, only full blocks, or only blocks at ``bases``."""
        bs = self.block_size
        req_order: list[str] = []
        counts: list[int] = []
        slots: list[np.ndarray] = []
        parts: dict[str, list[torch.Tensor]] = {key: [] for key in self._keys}
        for req_id in list(self._blocks) if req_ids is None else list(req_ids):
            blocks = self._blocks.get(req_id)
            if not blocks:
                continue
            kept: list[_Block] = []
            count = 0
            for block in blocks:
                if (full_only and block.hi < bs) or (bases is not None and block.base not in bases):
                    kept.append(block)
                    continue
                count += block.hi - block.lo
                slots.append(np.arange(block.base + block.lo, block.base + block.hi, dtype=np.int64))
                for key in self._keys:
                    parts[key].extend(step[key][row : row + 1] for step, row in block.refs)
            if count:
                req_order.append(req_id)
                counts.append(count)
            if kept:
                self._blocks[req_id] = kept
            else:
                del self._blocks[req_id]
        if not req_order:
            return None
        return ParkedRows(
            req_ids=tuple(req_order),
            counts=tuple(counts),
            slots_cpu=torch.from_numpy(np.concatenate(slots)),
            rows={key: torch.cat(rows) for key, rows in parts.items()},
        )

    def drop(self, req_ids: Iterable[str]) -> None:
        for req_id in req_ids:
            self._blocks.pop(req_id, None)
