# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Classify one step's outputs for the cache: immediate / deferred / leftover.

Pure with respect to cache state — the only input besides the tensors is
``ModelCachePolicy``. Pool keys a step needs are *reported* on
``StepOutputs.keys_to_open`` and opened by the manager under its lock, so
capture never mutates the pool or the occupancy table.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import torch

from vllm_omni.core.prefix_cache.controller import _BudgetTicket, _WriteChunk
from vllm_omni.core.prefix_cache.interface import (
    ModelCachePolicy,
    OmniPrefixCacheUnmatchError,
    ReqId,
    TensorName,
    is_hidden_key,
)


def is_step_token_tensor(val: Any, n: int, padded: int) -> bool:
    """2D+ tensor whose first dim is this step's token count (``n`` or padded).

    True means callers may take ``val[:n]``. Leftover tensors (``codes.ref``),
    lists, and other shapes are False.
    """
    return isinstance(val, torch.Tensor) and val.ndim >= 2 and int(val.shape[0]) in (n, padded)


def snapshot_leftover_mm_cpu(
    mm_outputs: dict[str, Any],
    device_snapshot_keys: set[str],
    num_tokens_unpadded: int,
) -> dict[str, Any]:
    """CPU copy of mm that did not land on the staging page.

    Skip ``device_snapshot_keys`` (those already have a device→host page).
    Copy the rest — deferred mm, lists, ``codes.ref`` — so materialize can
    run after the next forward overwrites graph buffers. Slice ``[:n]``
    only when ``shape[0] == n``; ``>= n`` would clip ``codes.ref``.
    """
    n = num_tokens_unpadded

    def _copy(val: Any) -> Any:
        if isinstance(val, torch.Tensor):
            t = val[:n] if val.ndim >= 2 and int(val.shape[0]) == n else val
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


class KeySpec(NamedTuple):
    """Pool storage one captured key needs (opened by the manager, idempotent)."""

    key: TensorName
    dtype: torch.dtype
    feat: int


@dataclass
class StepOutputs:
    """This step's outputs split by consumer.

    A tensor whose first dim equals this step's token count (or the
    CUDA-graph padded count) can be sliced ``[:n]`` into per-token rows.
    ``codes.ref`` and lists are not that shape.

    ``immediate``: those per-token rows copied device→host this step
    (hidden + non-deferred mm). ``deferred_chunks``: per-token deferred
    mm, packed per request for JOIN_ON_FINISH. ``leftover``: CPU replica
    for this step's materialize of everything that did not get a staging
    page.
    A deferred per-token key is in both ``deferred_chunks`` (later cache
    write) and ``leftover`` (this-step read) — two consumers, not a
    duplicate store.

    ``immediate_budget`` charges the immediate clones once; every
    JOIN_NEXT_STEP task of the step pins it. Deferred chunks carry their
    own shared ticket. ``keys_to_open`` lists the pool keys these rows
    will be written under.
    """

    immediate: dict[TensorName, torch.Tensor]
    deferred_chunks: list[tuple[ReqId, _WriteChunk]]
    leftover: dict[str, Any]
    immediate_budget: _BudgetTicket | None = None
    keys_to_open: list[KeySpec] = field(default_factory=list)

    @property
    def deferred_budget(self) -> _BudgetTicket | None:
        return self.deferred_chunks[0][1].budget if self.deferred_chunks else None

    def freeze_targets(self) -> list[torch.Tensor]:
        """Device clones the freeze event must cover."""
        return list(self.immediate.values()) + [t for _, c in self.deferred_chunks for t in c.tensors.values()]

    def budget_bytes(self) -> int:
        return sum(t.nbytes for t in (self.immediate_budget, self.deferred_budget) if t is not None)


class OutputCapturer:
    """Splits a step's ``hidden_states`` / ``mm_outputs`` by ``ModelCachePolicy``.

    Runs unlocked on the engine thread inside ``save_outputs``. Clones the
    per-token rows off the live buffers (immediate and deferred), CPU-copies
    everything else (leftover), and packs deferred rows per request.
    """

    def __init__(self, policy: ModelCachePolicy) -> None:
        self._policy = policy

    def capture(
        self,
        hidden_states: torch.Tensor | None,
        mm_outputs: dict[str, Any],
        *,
        num_tokens_unpadded: int,
        num_tokens_padded: int,
        slots_cpu: torch.Tensor | None,
        req_order: list[ReqId],
        num_sched: dict[ReqId, int],
        query_start: dict[ReqId, int],
    ) -> StepOutputs:
        """One pass over ``mm_outputs``. ``n==0`` has only leftover mm (no
        device clones). A deferred key whose first dim is this step's token
        count is cloned for the JOIN_ON_FINISH write and CPU-copied into
        leftover for this-step materialize. Talker ``codes.audio`` stays
        unpadded while hidden is padded; both must open a pool key. Lists
        and other shapes stay leftover.
        """
        n = num_tokens_unpadded
        policy = self._policy
        immediate: dict[TensorName, torch.Tensor] = {}
        deferred_tensors: dict[TensorName, torch.Tensor] = {}
        keys_to_open: list[KeySpec] = []
        if n > 0:
            if hidden_states is not None and (hk := policy.hidden_key) is not None:
                if hidden_states.ndim < 2 or hidden_states.shape[0] < n:
                    rows = 0 if hidden_states.ndim < 2 else int(hidden_states.shape[0])
                    raise OmniPrefixCacheUnmatchError(f"hidden_states has {rows} rows, need {n}")
                keys_to_open.append(KeySpec(hk, hidden_states.dtype, int(hidden_states.shape[-1])))
                immediate[hk] = hidden_states[:n].clone()
            for key, val in mm_outputs.items():
                is_step_rows = is_step_token_tensor(val, n, num_tokens_padded)
                if key in policy.deferred_keys:
                    if is_step_rows:
                        keys_to_open.append(KeySpec(key, val.dtype, int(val.shape[-1])))
                        deferred_tensors[key] = val[:n].clone()
                    continue
                if policy.skip_immediate_mm(key) or not is_step_rows:
                    continue
                keys_to_open.append(KeySpec(key, val.dtype, int(val.shape[-1])))
                immediate[key] = val[:n].clone()
        leftover = snapshot_leftover_mm_cpu(mm_outputs, set(immediate), n)
        deferred_chunks: list[tuple[ReqId, _WriteChunk]] = []
        if deferred_tensors:
            assert slots_cpu is not None
            deferred_chunks = self._pack_deferred_chunks(deferred_tensors, slots_cpu, req_order, num_sched, query_start)
        immediate_budget = (
            _BudgetTicket(nbytes=sum(t.numel() * t.element_size() for t in immediate.values())) if immediate else None
        )
        return StepOutputs(
            immediate=immediate,
            deferred_chunks=deferred_chunks,
            leftover=leftover,
            immediate_budget=immediate_budget,
            keys_to_open=keys_to_open,
        )

    @staticmethod
    def _pack_deferred_chunks(
        deferred_tensors: dict[TensorName, torch.Tensor],
        slots_cpu: torch.Tensor,
        req_order: list[ReqId],
        num_sched: dict[ReqId, int],
        query_start: dict[ReqId, int],
    ) -> list[tuple[ReqId, _WriteChunk]]:
        """Per-req views of already-cloned deferred tensors. No further clone."""
        ticket = _BudgetTicket(nbytes=sum(t.numel() * t.element_size() for t in deferred_tensors.values()))
        out: list[tuple[ReqId, _WriteChunk]] = []
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
                        budget=ticket,
                    ),
                )
            )
        return out
