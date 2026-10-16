# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Decode-row parking: full blocks reach the pool in batches, before a hit
needs them, or when their request leaves; a later hit reads the same rows as
the per-step save writes. CPU only."""

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


def _park(co: DecodeRowCoalescer, req_id: str, block: int, pos: int) -> bool:
    return co.park([req_id], np.array([block * BS + pos % BS]), {HIDDEN_KEY: _row(req_id, pos).unsqueeze(0)})


def test_coalescer_keeps_partial_blocks_and_takes_full_ones():
    co = DecodeRowCoalescer(BS, max_full_blocks=2)
    flushes = [_park(co, "a", 1 if p < 8 else 2, p) for p in range(5, 10)]  # 5..7 close block 1, 8..9 open block 2
    assert flushes == [False] * 5
    full = co.take(full_only=True)
    assert full is not None and full.req_ids == ("a",) and full.counts == (3,)
    assert full.slots_cpu.tolist() == [5, 6, 7]
    assert torch.equal(full.rows[HIDDEN_KEY], torch.stack([_row("a", p) for p in (5, 6, 7)]))
    assert co.open_bases() == {"a": 8}
    rest = co.take()
    assert rest is not None and rest.slots_cpu.tolist() == [8, 9] and not co.has_parked()


def test_coalescer_flags_a_flush_and_selects_blocks_by_base():
    co = DecodeRowCoalescer(BS, max_full_blocks=2)
    flags = [_park(co, "a", 3 + p // BS, p) for p in range(8)]  # blocks 3 and 4, both full
    assert flags[-1] and not any(flags[:-1])
    picked = co.take(bases={4 * BS})
    assert picked is not None and picked.slots_cpu.tolist() == [16, 17, 18, 19]
    assert co.take(full_only=True).slots_cpu.tolist() == [12, 13, 14, 15]


def test_coalescer_rejects_a_gap_and_drops():
    co = DecodeRowCoalescer(BS)
    _park(co, "a", 2, 8)
    with pytest.raises(OmniPrefixCacheUnmatchError, match="does not continue"):
        _park(co, "a", 2, 10)
    co.drop(["a"])
    assert not co.has_parked()


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
    def __init__(self, coalesce: bool, max_full_blocks: int = 2):
        self.omni_prefix_cache = OmniPrefixCacheManager(PrefixCacheConfig(num_blocks=16, block_size=BS), eager=True)
        self._prefix_cache_adapter = PrefixCacheSchedulerAdapter()
        self._prefix_cache_group_view = _View()
        self._prefix_cache_coalescer = DecodeRowCoalescer(BS, max_full_blocks) if coalesce else None
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


def _expect(req_id: str, positions, tail: tuple[str, int]) -> torch.Tensor:
    return torch.stack([_row(req_id, p) for p in positions] + [_row(*tail)])


@pytest.mark.parametrize(("coalesce", "max_full_blocks"), [(False, 2), (True, 2), (True, 8)])
def test_hit_on_decode_written_blocks_reads_the_produced_rows(monkeypatch, coalesce, max_full_blocks):
    _patch_pp(monkeypatch)
    r = _Runner(coalesce, max_full_blocks)
    r.step({"a": (A_BLOCKS, 0, 6)}, new=["a"])  # prefill 0..5
    ids = _decode_a(r, range(6, 12))  # decode fills block 1 (at 7) and block 2 (at 11)
    assert all(i is None for i in ids) == coalesce
    # b shares a's first 12 tokens: three full blocks, two of them partly or fully decode-written;
    # with max_full_blocks=8 they are still parked when b arrives.
    _, out = r.step({"a": (A_BLOCKS, 12, 1), "b": ([0, 1, 2, 5], 12, 1)}, new=["b"], hits={"b": 12})
    assert torch.equal(out["b"], _expect("a", range(12), ("b", 12)))
    assert torch.equal(out["a"], _row("a", 12).unsqueeze(0))


def test_parked_full_blocks_are_written_in_batches(monkeypatch):
    _patch_pp(monkeypatch)
    per_step, parked = _Runner(False), _Runner(True, max_full_blocks=2)
    for r in (per_step, parked):
        r.step({"a": (A_BLOCKS, 0, 4)}, new=["a"])
        _decode_a(r, range(4, 12))
    assert per_step.saves == 9
    assert parked.saves == 2  # prefill + one write of blocks 1 and 2
    assert parked.written(range(4, 12)) == [True] * 8


def test_same_step_hit_on_a_block_completed_by_a_parked_row(monkeypatch):
    _patch_pp(monkeypatch)
    r = _Runner(True, max_full_blocks=8)
    r.step({"a": (A_BLOCKS, 0, 4)}, new=["a"])
    _decode_a(r, (4, 5, 6))  # block 1 parked, still open
    # a's decode of position 7 completes block 1 in the step where b hits it.
    _, out = r.step({"a": (A_BLOCKS, 7, 1), "b": ([0, 1, 6], 8, 1)}, new=["b"], hits={"b": 8})
    assert torch.equal(out["b"], _expect("a", range(8), ("b", 8)))
    assert torch.equal(out["a"], _row("a", 7).unsqueeze(0))


def test_an_arrival_without_a_hit_leaves_running_rows_parked(monkeypatch):
    _patch_pp(monkeypatch)
    r = _Runner(True, max_full_blocks=8)
    r.step({"a": (A_BLOCKS, 0, 4)}, new=["a"])
    _decode_a(r, (4, 5))
    step_id, out = r.step({"a": (A_BLOCKS, 6, 1), "c": ([6], 0, 2)}, new=["c"])
    assert step_id is not None and torch.equal(out["a"], _row("a", 6).unsqueeze(0))
    assert r.written([4, 5, 6]) == [False, False, False]
    assert r.written([24, 25]) == [True, True]  # c's prefill rows


@pytest.mark.parametrize("how", ["finished", "preempted"])
def test_a_departed_request_writes_full_blocks_and_drops_the_partial_one(monkeypatch, how):
    _patch_pp(monkeypatch)
    r = _Runner(True, max_full_blocks=8)
    r.step({"a": (A_BLOCKS, 0, 4), "z": ([8, 9], 0, 4)}, new=["a", "z"])
    _decode_a(r, range(4, 10))  # block 1 full, block 2 holds 8..9
    r.step({"z": ([8, 9], 4, 1)}, **{how: ["a"]})
    assert r.written([4, 5, 6, 7]) == [True] * 4
    assert r.written([8, 9]) == [False, False]
    assert not r._prefix_cache_coalescer.has_parked() or "a" not in r._prefix_cache_coalescer.open_bases()
