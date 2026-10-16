# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Decode-row parking: rows reach the pool when their block fills, and a
later hit reads the same rows as the per-step save writes. CPU only."""

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_omni.core.prefix_cache.adapter import PrefixCacheSchedulerAdapter
from vllm_omni.core.prefix_cache.coalesce import DecodeRowCoalescer
from vllm_omni.core.prefix_cache.interface import HIDDEN_KEY, OmniPrefixCacheUnmatchError, PrefixCacheConfig
from vllm_omni.core.prefix_cache.manager import OmniPrefixCacheManager
from vllm_omni.core.prefix_cache.runner_mixin import PrefixCacheRunnerMixin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

BS = 4
WIDTH = 3


def _row(req_id: str, pos: int) -> torch.Tensor:
    return torch.full((WIDTH,), 1000.0 * (ord(req_id[0]) - ord("a") + 1) + pos)


def test_coalescer_reports_filled_blocks_and_returns_parked_rows():
    co = DecodeRowCoalescer(BS, capacity=2)
    filled = []
    for pos in range(5, 8):  # block 1 holds positions 4..7; position 4 was saved per step
        filled = co.park(["a"], np.array([1 * BS + pos % BS]), {HIDDEN_KEY: _row("a", pos).unsqueeze(0)})
    assert filled == ["a"]
    parked = co.take(filled)
    assert parked is not None
    assert parked.req_ids == ("a",) and parked.counts == (3,)
    assert parked.slots_cpu.tolist() == [5, 6, 7]
    assert torch.equal(parked.rows[HIDDEN_KEY], torch.stack([_row("a", p) for p in (5, 6, 7)]))
    assert co.take(["a"]) is None and not co.has_parked()


def test_coalescer_rejects_a_gap_and_frees_ring_rows_on_drop():
    co = DecodeRowCoalescer(BS, capacity=1)
    co.park(["a"], np.array([8]), {HIDDEN_KEY: _row("a", 8).unsqueeze(0)})
    with pytest.raises(OmniPrefixCacheUnmatchError, match="does not continue"):
        co.park(["a"], np.array([10]), {HIDDEN_KEY: _row("a", 10).unsqueeze(0)})
    assert not co.has_room(["b"])
    co.drop(["a"])
    assert co.has_room(["b"]) and not co.has_parked()


def _patch_pp(monkeypatch) -> None:
    mod = ModuleType("vllm.distributed.parallel_state")
    setattr(mod, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True))
    if "vllm.distributed" not in sys.modules:
        monkeypatch.setitem(sys.modules, "vllm.distributed", ModuleType("vllm.distributed"))
    monkeypatch.setitem(sys.modules, "vllm.distributed.parallel_state", mod)


class _View:
    """Group view over plain dicts: per request its group-0 blocks and the
    first position scheduled this step."""

    block_size = BS

    def __init__(self):
        self.order: list[str] = []
        self.blocks: dict[str, list[int]] = {}
        self.start: dict[str, int] = {}

    def _slot(self, req_id: str, pos: int) -> int:
        return self.blocks[req_id][pos // BS] * BS + pos % BS

    def batch_req_ids(self):
        return list(self.order)

    def step_slots_cpu(self, req_ids, num_scheduled):
        slots = [self._slot(r, p) for r in req_ids for p in range(self.start[r], self.start[r] + num_scheduled[r])]
        return torch.tensor(slots, dtype=torch.long)

    def decode_slots_cpu(self, req_ids):
        return np.array([self._slot(r, self.start[r]) for r in req_ids], dtype=np.int64)


class _Runner(PrefixCacheRunnerMixin):
    def __init__(self, coalesce: bool):
        self.omni_prefix_cache = OmniPrefixCacheManager(PrefixCacheConfig(num_blocks=16, block_size=BS), eager=True)
        self._prefix_cache_adapter = PrefixCacheSchedulerAdapter()
        self._prefix_cache_group_view = _View()
        self._prefix_cache_coalescer = DecodeRowCoalescer(BS, capacity=4) if coalesce else None
        self._omni_prefix_cache_cfg = None
        self.is_pooling_model = False
        self.saves = 0
        save = self.omni_prefix_cache.save_outputs

        def counting_save(*args, **kwargs):
            self.saves += 1
            return save(*args, **kwargs)

        self.omni_prefix_cache.save_outputs = counting_save  # type: ignore[method-assign]

    def step(self, reqs, *, new=(), hits=None, finished=(), preempted=()):
        """reqs: req -> (blocks, first scheduled position, scheduled tokens)."""
        view = self._prefix_cache_group_view
        view.order = list(reqs)
        for req_id, (blocks, start, _) in reqs.items():
            view.blocks[req_id], view.start[req_id] = blocks, start
        sched = SimpleNamespace(
            scheduled_new_reqs=[
                SimpleNamespace(req_id=r, num_computed_tokens=(hits or {}).get(r, 0), block_ids=[reqs[r][0]])
                for r in new
            ],
            scheduled_cached_reqs=None,
            finished_req_ids=set(finished),
            preempted_req_ids=set(preempted),
            num_scheduled_tokens={r: n for r, (_, _, n) in reqs.items()},
        )
        self._prefix_cache_step_begin(sched)
        hidden = torch.stack([_row(r, p) for r, (_, s, n) in reqs.items() for p in range(s, s + n)])
        n = int(hidden.shape[0])
        step_id = self._prefix_cache_save_step(hidden, None, num_tokens_unpadded=n, num_tokens_padded=n)
        out = self._prefix_cache_materialize(step_id, list(reqs))[0] if step_id is not None else None
        return step_id, out

    def written(self, slots) -> list[bool]:
        state = self.omni_prefix_cache._slot_status.get_slot_status(HIDDEN_KEY).state
        return [bool(state[s] != 0) for s in slots]


A_BLOCKS = [0, 1, 2, 3]


def _decode_a(runner: _Runner, positions) -> list[int | None]:
    return [runner.step({"a": (A_BLOCKS, p, 1)})[0] for p in positions]


@pytest.mark.parametrize("coalesce", [False, True])
def test_hit_on_decode_written_blocks_reads_the_produced_rows(monkeypatch, coalesce):
    _patch_pp(monkeypatch)
    r = _Runner(coalesce)
    r.step({"a": (A_BLOCKS, 0, 6)}, new=["a"])  # prefill 0..5
    ids = _decode_a(r, range(6, 12))  # decode fills block 1 (at 7) and block 2 (at 11)
    assert all(i is None for i in ids) == coalesce
    # b shares a's first 12 tokens: three full blocks, two of them partly or fully decode-written.
    _, out = r.step({"a": (A_BLOCKS, 12, 1), "b": ([0, 1, 2, 5], 12, 1)}, new=["b"], hits={"b": 12})
    assert torch.equal(out["b"], torch.stack([_row("a", p) for p in range(12)] + [_row("b", 12)]))
    assert torch.equal(out["a"], _row("a", 12).unsqueeze(0))


def test_parking_saves_once_per_filled_block(monkeypatch):
    _patch_pp(monkeypatch)
    per_step, parked = _Runner(False), _Runner(True)
    for r in (per_step, parked):
        r.step({"a": (A_BLOCKS, 0, 4)}, new=["a"])
        _decode_a(r, range(4, 12))
    assert per_step.saves == 9
    assert parked.saves == 3  # prefill + blocks 1 and 2


def test_arrival_writes_parked_rows_before_its_own_save(monkeypatch):
    _patch_pp(monkeypatch)
    r = _Runner(True)
    r.step({"a": (A_BLOCKS, 0, 4)}, new=["a"])
    _decode_a(r, (4, 5))  # parked, block 1 still open
    assert r.written([4, 5]) == [False, False]
    r.step({"a": (A_BLOCKS, 6, 1), "c": ([6], 0, 2)}, new=["c"])  # an arrival is not a pure decode step
    assert r.written([4, 5, 6]) == [True, True, True]
    _decode_a(r, (7,))  # a's next row is the per-step save's block-filling row
    _, out = r.step({"a": (A_BLOCKS, 8, 1), "b": ([0, 1, 7], 8, 1)}, new=["b"], hits={"b": 8})
    assert torch.equal(out["b"], torch.stack([_row("a", p) for p in range(8)] + [_row("b", 8)]))


@pytest.mark.parametrize("how", ["finished", "preempted"])
def test_partial_block_of_a_departed_request_is_dropped(monkeypatch, how):
    _patch_pp(monkeypatch)
    r = _Runner(True)
    r.step({"a": (A_BLOCKS, 0, 4), "z": ([8, 9], 0, 4)}, new=["a", "z"])
    _decode_a(r, (4, 5))
    gone = {how: ["a"]}
    r.step({"z": ([8, 9], 4, 1)}, **gone)
    assert r.written([4, 5]) == [False, False]
    assert not r._prefix_cache_coalescer.has_parked() or "a" not in r._prefix_cache_coalescer._parked
