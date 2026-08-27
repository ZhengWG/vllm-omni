"""Unit tests for vllm_omni/core/prefix_cache (manager + controller + pool).

CPU-only: the controller runs in eager mode. Uses a fake group view, so no
vLLM runtime is required (runnable without a vllm install via
``pytest --confcutdir=tests/core``).
"""

import logging
import sys
import types
from pathlib import Path

import pytest
import torch

try:  # pragma: no cover - shim only matters on vllm-less dev machines
    import vllm  # noqa: F401
except ModuleNotFoundError:
    # Bypass vllm_omni/__init__ (which imports vllm): register namespace
    # parents so the pure-torch prefix_cache subpackage imports directly.
    _root = Path(__file__).resolve().parents[2]
    for _pkg in ("vllm_omni", "vllm_omni.core"):
        if _pkg not in sys.modules:
            _m = types.ModuleType(_pkg)
            _m.__path__ = [str(_root / _pkg.replace(".", "/"))]
            sys.modules[_pkg] = _m

from vllm_omni.core.prefix_cache.controller import StagingBufferHolder
from vllm_omni.core.prefix_cache.group_view import FullAttentionGroupView
from vllm_omni.core.prefix_cache.interface import (
    HIDDEN_KEY,
    ModelCachePolicy,
    OmniPrefixCacheUnmatchError,
    PrefixCacheConfig,
)
from vllm_omni.core.prefix_cache.manager import OmniPrefixCacheManager

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

NUM_BLOCKS = 16
BLOCK_SIZE = 4
HIDDEN = 8
DTYPE = torch.float32


class FakeView:
    """Duck-typed KVCacheGroupView backed by plain dicts."""

    def __init__(self):
        self.block_size = BLOCK_SIZE
        self.num_blocks = NUM_BLOCKS
        self.req_blocks: dict[str, list[int]] = {}
        self.order: list[str] = []
        self.computed: dict[str, int] = {}
        self.step_slot_mapping: torch.Tensor | None = None

    def slots_for(self, req_id, token_start, token_end):
        blocks = self.req_blocks[req_id]
        slots = []
        for pos in range(token_start, token_end):
            slots.append(blocks[pos // BLOCK_SIZE] * BLOCK_SIZE + pos % BLOCK_SIZE)
        return torch.tensor(slots, dtype=torch.long)

    def cached_block_ids(self, req_id) -> torch.Tensor:
        return torch.tensor(self.req_blocks[req_id], dtype=torch.long)

    def batch_req_ids(self) -> list[str]:
        return list(self.order)

    def step_slots_cpu(self, req_ids, num_scheduled) -> torch.Tensor:
        # Mirrors FullAttentionGroupView: positions start at num_computed.
        parts = []
        for r in req_ids:
            n = int(num_scheduled.get(r, 0))
            if n <= 0:
                continue
            start = self.computed.get(r, 0)
            parts.append(self.slots_for(r, start, start + n))
        return torch.cat(parts) if parts else torch.empty((0,), dtype=torch.long)


class FakeNewReq:
    def __init__(self, req_id, num_computed_tokens=0, block_ids=None):
        self.req_id = req_id
        self.num_computed_tokens = num_computed_tokens
        # Per-kv-group, like vLLM NewRequestData.block_ids.
        self.block_ids = block_ids


class FakeSchedOut:
    def __init__(self, new_reqs=(), finished=(), num_scheduled=None):
        self.scheduled_new_reqs = list(new_reqs)
        self.finished_req_ids = set(finished)
        self.num_scheduled_tokens = dict(num_scheduled or {})


def make_manager(view=None, policy=None, **cfg_kwargs) -> tuple[OmniPrefixCacheManager, FakeView]:
    view = view or FakeView()
    config = PrefixCacheConfig(
        num_blocks=NUM_BLOCKS, block_size=BLOCK_SIZE, hidden_size=HIDDEN, hs_dtype=DTYPE, **cfg_kwargs
    )
    mgr = OmniPrefixCacheManager(config, view, eager=True)
    if policy is not None:
        mgr.register_policy(policy)
    return mgr, view


def run_step(mgr, view, reqs: dict[str, tuple[list[int], int, int]], new_hits=None, finished=(), mm=None) -> int:
    """One step: reqs = req_id -> (blocks, sched_start_pos, sched_tokens).

    Returns the step id save_outputs issued (consume-exactly-once handle).
    """
    view.order = list(reqs.keys())
    new_reqs = []
    num_sched = {}
    slot_parts = []
    hidden_parts = []
    for req_id, (blocks, start_pos, sched) in reqs.items():
        view.req_blocks[req_id] = blocks
        view.computed[req_id] = start_pos
        num_sched[req_id] = sched
        slots = view.slots_for(req_id, start_pos, start_pos + sched)
        slot_parts.append(slots)
        # Value pattern: slot id encoded in every feature -> easy validation.
        hidden_parts.append(slots.to(DTYPE).unsqueeze(1).expand(sched, HIDDEN).clone())
        hit = (new_hits or {}).get(req_id, 0)
        new_reqs.append(FakeNewReq(req_id, num_computed_tokens=hit, block_ids=[list(blocks)]))
    view.step_slot_mapping = torch.cat(slot_parts)
    hidden = torch.cat(hidden_parts)
    sched_out = FakeSchedOut(new_reqs=new_reqs, finished=finished, num_scheduled=num_sched)
    mgr.new_step_starts(sched_out)
    n = int(view.step_slot_mapping.numel())
    return mgr.save_outputs(hidden, mm or {}, num_tokens_unpadded=n, num_tokens_padded=n)


def expected_rows(slots: torch.Tensor) -> torch.Tensor:
    return slots.to(DTYPE).unsqueeze(1).expand(slots.numel(), HIDDEN)


def test_no_hit_passthrough():
    mgr, view = make_manager()
    sid = run_step(mgr, view, {"a": ([0, 1], 0, 8)})
    outs = mgr.materialize(sid, ["a"])
    assert torch.equal(outs.hidden_states["a"], expected_rows(view.slots_for("a", 0, 8)))


def test_hit_merge_from_mirror():
    mgr, view = make_manager()
    # Producer prefills 8 tokens on blocks [0, 1] and finishes.
    s1 = run_step(mgr, view, {"a": ([0, 1], 0, 8)})
    mgr.materialize(s1, ["a"])
    # Consumer hits the full 8 tokens and computes 4 new on block 2.
    s2 = run_step(mgr, view, {"b": ([0, 1, 2], 8, 4)}, new_hits={"b": 8}, finished=["a"])
    outs = mgr.materialize(s2, ["b"])
    merged = outs.hidden_states["b"]
    assert merged.shape == (12, HIDDEN)
    assert torch.equal(merged[:8], expected_rows(view.slots_for("b", 0, 8)))
    assert torch.equal(merged[8:], expected_rows(view.slots_for("b", 8, 12)))


def test_same_step_hit_reads_in_transit():
    # b hits a's blocks in the very step a prefills them: the deferred
    # entry stays GPU-staged (unqueued) in eager mode too, so in-transit
    # reads are exercised via the main-entry fetch path when the entry is
    # still pending. In eager mode the main entry commits synchronously,
    # so simulate the window with a deferred key instead below; here we
    # assert the same-step read is correct through whatever tier serves it.
    mgr, view = make_manager()
    view.req_blocks["a"] = [0, 1]
    sid = run_step(
        mgr,
        view,
        {"a": ([0, 1], 0, 8), "b": ([0, 1, 2], 8, 4)},
        new_hits={"b": 8},
    )
    outs = mgr.materialize(sid, ["a", "b"])
    merged = outs.hidden_states["b"]
    assert torch.equal(merged[:8], expected_rows(view.slots_for("b", 0, 8)))


def test_absent_hit_fails_fast():
    mgr, view = make_manager()
    view.req_blocks["c"] = [5, 6]
    sid = run_step(mgr, view, {"c": ([5, 6, 7], 8, 4)}, new_hits={"c": 8})
    d2h = mgr._step_ctxs[sid].d2h
    with pytest.raises(OmniPrefixCacheUnmatchError):
        mgr.materialize(sid, ["c"])
    # Fail-fast after take_ctx must still drop the step holder.
    if d2h is not None:
        assert not mgr._controller._staging_pool._busy[d2h.slot]


def test_hit_not_block_aligned_asserts():
    mgr, view = make_manager()
    s1 = run_step(mgr, view, {"a": ([0, 1], 0, 8)})
    mgr.materialize(s1, ["a"])
    s2 = run_step(mgr, view, {"b": ([0, 1, 2], 8, 4)}, new_hits={"b": 6})
    with pytest.raises(AssertionError):
        mgr.materialize(s2, ["b"])


def test_mm_cached_key_merge():
    mgr, view = make_manager()
    feat = 3
    mm1 = {"talker.h": torch.arange(8 * feat, dtype=DTYPE).reshape(8, feat)}
    s1 = run_step(mgr, view, {"a": ([0, 1], 0, 8)}, mm=mm1)
    mgr.materialize(s1, ["a"])
    mm2 = {"talker.h": torch.full((4, feat), 7.0)}
    s2 = run_step(mgr, view, {"b": ([0, 1, 2], 8, 4)}, new_hits={"b": 8}, finished=["a"], mm=mm2)
    outs = mgr.materialize(s2, ["b"])
    merged = outs.mm_outputs["talker.h"]["b"]
    assert merged.shape == (12, feat)
    assert torch.equal(merged[:8], mm1["talker.h"])
    assert torch.equal(merged[8:], mm2["talker.h"])


def test_deferred_key_accumulates_and_flushes_on_finish():
    policy = ModelCachePolicy(
        needs_full_hidden_states=False, deferred_keys=frozenset({"codes.audio"}), skip_keys=frozenset({"codes.audio"})
    )
    mgr, view = make_manager(policy=policy)
    feat = 2
    # Two decode steps of one token each, then finish -> flush.
    for pos in range(2):
        sid = run_step(
            mgr,
            view,
            {"a": ([3], pos, 1)},
            mm={"codes.audio": torch.full((1, feat), float(pos + 1))},
        )
        mgr.materialize(sid, ["a"])
    # In-transit read before finish: deferred rows come from staged chunks.
    slots = view.slots_for("a", 0, 2)
    rows = mgr._resolve_rows(slots, "codes.audio", strict=False, req_id="a")
    assert torch.equal(rows[:, 0], torch.tensor([1.0, 2.0]))
    # Finish (abort semantics identical): entry escalates and commits.
    sid = run_step(mgr, view, {"z": ([9], 0, 1)}, finished=["a"])
    mgr.materialize(sid, ["z"])
    mirror = mgr._pool.rows("codes.audio", slots)
    assert torch.equal(mirror[:, 0], torch.tensor([1.0, 2.0]))


def test_cap_forces_flush_of_deferred():
    policy = ModelCachePolicy(
        needs_full_hidden_states=False, deferred_keys=frozenset({"k"}), skip_keys=frozenset({"k"})
    )
    # float32 feat=4 row = 16 bytes; 48 forces a flush by step 4.
    mgr, view = make_manager(policy=policy, gpu_staging_bytes=48)
    for pos in range(4):
        sid = run_step(mgr, view, {"a": ([2, 3], pos, 1)}, mm={"k": torch.full((1, 4), float(pos))})
        mgr.materialize(sid, ["a"])
    # Older rows must have been force-flushed to the mirror.
    early = view.slots_for("a", 0, 1)
    assert float(mgr._pool.rows("k", early)[0, 0]) == 0.0


def test_policy_from_model_shim():
    class M:
        requires_full_prefix_cached_hidden_states = False
        deferred_prefix_cache_mm_keys = {"codes.audio"}

    p = ModelCachePolicy.from_model(M())
    assert p.needs_full_hidden_states is False
    assert p.deferred_keys == frozenset({"codes.audio"})
    assert p.skip_keys == frozenset({"codes.audio"})
    d = ModelCachePolicy.from_model(object())
    assert d.needs_full_hidden_states is True and not d.deferred_keys


def test_group_view_slot_math():
    class TensorWrap:
        def __init__(self, t):
            self.cpu = t

    class Group:
        def __init__(self, t):
            self.block_table = TensorWrap(t)

    class BT:
        def __init__(self, t):
            self._g = Group(t)
            self.block_tables = [self._g.block_table]

        def __getitem__(self, idx):
            assert idx == 0
            return self._g

    class IB:
        def __init__(self, table):
            self.req_ids = ["r1", "r2"]
            self.req_id_to_index = {"r1": 0, "r2": 1}
            self.num_computed_tokens_cpu = torch.tensor([8, 0])
            self.block_table = BT(table)

    table = torch.tensor([[2, 5, 7], [1, 0, 0]])
    view = FullAttentionGroupView(IB(table), block_size=BLOCK_SIZE, num_blocks=NUM_BLOCKS)
    slots = view.slots_for("r1", 2, 6)
    assert slots.tolist() == [2 * 4 + 2, 2 * 4 + 3, 5 * 4 + 0, 5 * 4 + 1]
    # Positions past the block table are clamped (legacy parity).
    slots = view.slots_for("r1", 10, 14)
    assert slots.tolist() == [7 * 4 + 2, 7 * 4 + 3]
    assert view.cached_block_ids("r1").tolist() == [2, 5]
    assert view.slots_for("r1", 3, 3).numel() == 0


def test_join_next_step_previous_save():
    mgr, view = make_manager()
    s1 = run_step(mgr, view, {"a": ([0], 0, 2)})
    assert len(mgr._join_next_step_tids) == 1
    task_id = mgr._join_next_step_tids[0]
    mgr.materialize(s1, ["a"])
    s2 = run_step(mgr, view, {"a": ([0], 2, 1)})
    # Previous B entry joined & drained at the save above.
    assert mgr._controller.get_task(task_id) is None
    assert int(mgr._key_state[HIDDEN_KEY][view.slots_for("a", 0, 2)].min()) == 2
    mgr.materialize(s2, ["a"])


def test_tenant_succession_hit_reads_newest():
    # a occupies block 0, finishes; b reuses block 0 with new values;
    # c hits b's prefix -> must read b's rows, never a's.
    mgr, view = make_manager()
    s1 = run_step(mgr, view, {"a": ([0], 0, 4)})
    mgr.materialize(s1, ["a"])
    s2 = run_step(mgr, view, {"b": ([0], 0, 4)}, finished=["a"])
    mgr.materialize(s2, ["b"])
    b_hidden = expected_rows(view.slots_for("b", 0, 4))
    s3 = run_step(mgr, view, {"c": ([0, 1], 4, 2)}, new_hits={"c": 4}, finished=["b"])
    outs = mgr.materialize(s3, ["c"])
    assert torch.equal(outs.hidden_states["c"][:4], b_hidden)


def test_deferred_tenant_succession_no_stale_wins():
    # Preemption-style reuse on the deferred path: a's staged rows (val 1)
    # still pending when b stages the same slots (val 2). Whatever the
    # finish order, the mirror's final value must be b's.
    policy = ModelCachePolicy(
        needs_full_hidden_states=False, deferred_keys=frozenset({"k"}), skip_keys=frozenset({"k"})
    )
    mgr, view = make_manager(policy=policy)
    s1 = run_step(mgr, view, {"a": ([2], 0, 2)}, mm={"k": torch.full((2, 2), 1.0)})
    mgr.materialize(s1, ["a"])
    # b reuses block 2 while a's deferred entry is still staged (a preempted,
    # not finished). Conflict detection must escalate a's entry first.
    s2 = run_step(mgr, view, {"b": ([2], 0, 2)}, mm={"k": torch.full((2, 2), 2.0)})
    mgr.materialize(s2, ["b"])
    slots = view.slots_for("b", 0, 2)
    # Dangerous order: b finishes (flushes) BEFORE a does.
    s3 = run_step(mgr, view, {"z1": ([9], 0, 1)}, finished=["b"])
    mgr.materialize(s3, ["z1"])
    s4 = run_step(mgr, view, {"z2": ([9], 1, 1)}, finished=["a"])
    mgr.materialize(s4, ["z2"])
    assert torch.equal(mgr._pool.rows("k", slots), torch.full((2, 2), 2.0))


def test_slot_reuse_pushes_skip_to_old_task():
    """Reassignment = task swap: the old tenant's write skips the reassigned
    (slot, key) rows instead of ordering behind a dependency edge."""
    mgr, view = make_manager()
    policy = ModelCachePolicy(
        needs_full_hidden_states=False, deferred_keys=frozenset({"k"}), skip_keys=frozenset({"k"})
    )
    mgr.register_policy(policy)
    # Deferred task holds slots of block 2 in-transit for key "k".
    s1 = run_step(mgr, view, {"a": ([2], 0, 1)}, mm={"k": torch.ones(1, 2)})
    mgr.materialize(s1, ["a"])
    dtask = mgr._deferred_tasks["a"]
    # New request reuses block 2 (same key) while the old task is staged.
    s2 = run_step(mgr, view, {"b": ([2], 0, 2)}, mm={"k": torch.full((2, 2), 2.0)})
    assert mgr._deferred_tasks["b"].tid != dtask.tid
    reused = view.slots_for("b", 0, 1)
    assert "k" in dtask.skip and bool(torch.isin(reused, dtask.skip["k"]).all())
    mgr.materialize(s2, ["b"])


def test_deferred_key_hit_reads_staged_rows_not_mirror():
    """A hit on a still-staged deferred key must serve the staged rows.

    Regression: the mirror is registered on the first stage() call, so a
    has_key-first lookup silently returned zero rows for the whole hit span.
    """
    policy = ModelCachePolicy(
        needs_full_hidden_states=True,
        deferred_keys=frozenset({"k"}),
        skip_keys=frozenset({"k"}),
    )
    mgr, view = make_manager(policy=policy)
    # a stages 4 deferred rows and stays live (entry never flushed).
    s1 = run_step(mgr, view, {"a": ([0], 0, 4)}, mm={"k": torch.full((4, 2), 1.0)})
    mgr.materialize(s1, ["a"])
    s2 = run_step(mgr, view, {"b": ([0, 1], 4, 2)}, new_hits={"b": 4}, mm={"k": torch.full((2, 2), 9.0)})
    rows = mgr.materialize(s2, ["b"]).mm_outputs["k"]["b"]
    assert torch.equal(rows[:4], torch.full((4, 2), 1.0)), rows
    assert torch.equal(rows[4:], torch.full((2, 2), 9.0))


def test_append_to_closed_deferred_entry_opens_new_one():
    """Cap flush / escalation can close a deferred entry mid-request; the
    next step must open a fresh entry instead of asserting."""
    policy = ModelCachePolicy(
        needs_full_hidden_states=False,
        deferred_keys=frozenset({"k"}),
        skip_keys=frozenset({"k"}),
    )
    mgr, view = make_manager(policy=policy)
    s1 = run_step(mgr, view, {"a": ([2], 0, 1)}, mm={"k": torch.full((1, 2), 1.0)})
    mgr.materialize(s1, ["a"])
    first = mgr._deferred_tasks["a"]
    # Force the entry closed the way a cap flush would.
    mgr._controller.escalate([first.tid])
    s2 = run_step(mgr, view, {"a": ([2], 1, 1)}, mm={"k": torch.full((1, 2), 2.0)})
    mgr.materialize(s2, ["a"])
    assert mgr._deferred_tasks["a"].tid != first.tid
    slots = view.slots_for("a", 0, 2)
    rows = mgr._resolve_rows(slots, "k", strict=False, req_id="a")
    assert torch.equal(rows[:, 0], torch.tensor([1.0, 2.0]))


def test_step_slots_cpu_matches_block_table_math():
    """The CPU-derived step slot mapping must equal the per-request slot
    math the view exposes (which mirrors the device slot_mapping)."""

    class TensorWrap:
        def __init__(self, t):
            self.cpu = t

    class Group:
        def __init__(self, t):
            self.block_table = TensorWrap(t)

    class BT:
        def __init__(self, t):
            self._g = Group(t)
            self.block_tables = [self._g.block_table]

        def __getitem__(self, idx):
            return self._g

    class IB:
        def __init__(self, table, computed):
            self.req_ids = ["r1", "r2"]
            self.req_id_to_index = {"r1": 0, "r2": 1}
            self.num_computed_tokens_cpu = torch.tensor(computed)
            self.block_table = BT(table)

    table = torch.tensor([[2, 5, 7], [1, 3, 4]])
    view = FullAttentionGroupView(IB(table, [4, 0]), block_size=BLOCK_SIZE, num_blocks=NUM_BLOCKS)
    num_sched = {"r1": 3, "r2": 5}
    got = view.step_slots_cpu(["r1", "r2"], num_sched)
    want = torch.cat([view.slots_for("r1", 4, 7), view.slots_for("r2", 0, 5)])
    assert torch.equal(got, want), (got, want)
    # A request scheduled for 0 tokens contributes nothing.
    assert torch.equal(view.step_slots_cpu(["r1", "r2"], {"r1": 3, "r2": 0}), view.slots_for("r1", 4, 7))


def test_step_context_consumed_by_id_not_order():
    """Contexts are addressed by step id: a later step consumed first (the
    async-builder N+1-early-return case) must not disturb step N's context.

    Regression: popleft() discarded "the oldest", so N+1's discard threw away
    N's context while N's builder was still pending.
    """
    mgr, view = make_manager()
    s1 = run_step(mgr, view, {"a": ([0], 0, 4)})
    s2 = run_step(mgr, view, {"b": ([1], 0, 4)})
    # N+1 returns early (no consumer) while N's builder is still pending.
    mgr.discard_step(s2)
    outs = mgr.materialize(s1, ["a"])
    assert torch.equal(outs.hidden_states["a"], expected_rows(view.slots_for("a", 0, 4)))
    assert len(mgr._step_ctxs) == 0


def test_step_context_exactly_once():
    """Each context is consumed exactly once; a second consume fails fast."""
    mgr, view = make_manager()
    sid = run_step(mgr, view, {"a": ([0], 0, 4)})
    mgr.materialize(sid, ["a"])
    with pytest.raises(OmniPrefixCacheUnmatchError):
        mgr.materialize(sid, ["a"])
    with pytest.raises(OmniPrefixCacheUnmatchError):
        mgr.discard_step(sid)


def test_unconsumed_contexts_overflow_fails_fast():
    """A runner that leaks step contexts (never materialize/discard) must be
    caught at save time with the leaked ids, before staging-pool exhaustion
    can hide them."""
    mgr, view = make_manager()
    with pytest.raises(OmniPrefixCacheUnmatchError, match="unconsumed step contexts"):
        for pos in range(8):
            run_step(mgr, view, {"a": ([0, 1, 2, 3], pos, 1)})


def test_save_slot_mismatch_fails_fast():
    """A slot mapping that does not cover the scheduled tokens is a poisoned
    save (hashes are already published); it must fail at the cause."""
    mgr, view = make_manager()
    view.order = ["a"]
    view.req_blocks["a"] = [0]
    view.computed["a"] = 0
    mgr.new_step_starts(FakeSchedOut(new_reqs=[FakeNewReq("a")], num_scheduled={"a": 2}))
    hidden = torch.zeros(4, HIDDEN, dtype=DTYPE)
    with pytest.raises(OmniPrefixCacheUnmatchError):
        # num_scheduled says 2 tokens but we claim 4 were produced.
        mgr.save_outputs(hidden, {}, num_tokens_unpadded=4, num_tokens_padded=4)


def test_materialize_rejects_out_of_snapshot_ids():
    """req_ids must be a subset of the save-time snapshot; an outside id
    means the runner is reading the live batch (debug assert, G2)."""
    mgr, view = make_manager()
    s1 = run_step(mgr, view, {"a": ([0, 1], 0, 8)})
    mgr.materialize(s1, ["a"])
    s2 = run_step(mgr, view, {"b": ([0, 1, 2], 8, 4)}, new_hits={"b": 8}, finished=["a"])
    with pytest.raises(AssertionError, match="outside the save snapshot"):
        mgr.materialize(s2, ["b", "late_joiner"])


def test_materialize_unknown_step_id_fails_fast():
    """An unknown step id is a bookkeeping error (the runner lost or reused
    a sid), never a degrade path."""
    mgr, view = make_manager()
    sid = run_step(mgr, view, {"a": ([0], 0, 4)})
    with pytest.raises(OmniPrefixCacheUnmatchError):
        mgr.materialize(sid + 999, ["ghost"])
    mgr.discard_step(sid)


def test_fetch_host_fast_path_matches_general_path():
    """Single-segment fetch matches the general path for this-step identity,
    a shorter prefix hit, and a gapped subsequence (some slots already
    in the mirror). Hit/save order is increasing; no shuffle."""
    from vllm_omni.core.prefix_cache.block_pool import PrefixBlockPool
    from vllm_omni.core.prefix_cache.controller import (
        OmniPrefixCacheController,
        WriteTask,
        _Segment,
    )
    from vllm_omni.core.prefix_cache.interface import HIDDEN_KEY, WriteSchedule

    cfg = PrefixCacheConfig(num_blocks=NUM_BLOCKS, block_size=BLOCK_SIZE, hidden_size=HIDDEN, hs_dtype=DTYPE)
    pool = PrefixBlockPool(cfg)
    pool.ensure_key(HIDDEN_KEY, DTYPE, HIDDEN)
    ctrl = OmniPrefixCacheController(pool, cfg, eager=True)

    # In-order rows well inside the pool (NUM_BLOCKS * BLOCK_SIZE = 64).
    slots = torch.tensor([40, 41, 42, 43, 44], dtype=torch.int64)
    rows = torch.arange(slots.numel() * HIDDEN, dtype=DTYPE).reshape(slots.numel(), HIDDEN)
    task = WriteTask(
        tid=1,
        req_id="r",
        write_n=1,
        schedule=WriteSchedule.JOIN_NEXT_STEP,
        segments=[_Segment(slots_cpu=slots, tensors={HIDDEN_KEY: rows})],
    )
    ctrl.reserve(rows.numel() * rows.element_size())
    ctrl.submit(task)

    for want in (slots, slots[:3], slots[[0, 1, 3, 4]]):
        fast = ctrl.fetch_host(task, want, HIDDEN_KEY)
        s2r = task.slot_to_row()
        general = ctrl._rows_from(task, [s2r[int(s)] for s in want.tolist()], HIDDEN_KEY)
        assert torch.equal(fast, general), (want, fast, general)
        expect = torch.stack([rows[(slots == s).nonzero()[0, 0]] for s in want.tolist()])
        assert torch.equal(fast, expect)


def test_fetch_host_waits_staging_host_event():
    """JOIN_NEXT_STEP hangs seg.host as a staging view at save; D2H may
    still be in flight. fetch_host must wait host_event before slicing."""
    from vllm_omni.core.prefix_cache.block_pool import PrefixBlockPool
    from vllm_omni.core.prefix_cache.controller import (
        OmniPrefixCacheController,
        WriteTask,
        _Segment,
    )
    from vllm_omni.core.prefix_cache.interface import HIDDEN_KEY, WriteSchedule

    cfg = PrefixCacheConfig(num_blocks=NUM_BLOCKS, block_size=BLOCK_SIZE, hidden_size=HIDDEN, hs_dtype=DTYPE)
    pool = PrefixBlockPool(cfg)
    ctrl = OmniPrefixCacheController(pool, cfg, eager=True)

    slots = torch.tensor([0, 1, 2], dtype=torch.int64)
    src = torch.arange(3 * HIDDEN, dtype=DTYPE).reshape(3, HIDDEN)
    landing = torch.zeros_like(src)

    class _HostEvent:
        n = 0

        def synchronize(self):
            self.n += 1
            landing.copy_(src)

    event = _HostEvent()
    seg = _Segment(slots_cpu=slots, tensors={HIDDEN_KEY: src})
    seg.host = {HIDDEN_KEY: landing}
    task = WriteTask(
        tid=1,
        req_id="r",
        write_n=1,
        schedule=WriteSchedule.JOIN_NEXT_STEP,
        segments=[seg],
        staging_slot=0,
        host_event=event,
    )
    rows = ctrl.fetch_host(task, slots, HIDDEN_KEY)
    assert event.n == 1
    assert torch.equal(rows, src)


def test_mm_hit_span_never_registered_serves_mirror_baseline():
    """(slot, key) semantics: a hit span slot on which a sparse mm key was
    never registered is legitimate absence (e.g. a text position with no
    codes) — served from the mirror baseline, not a crash."""
    policy = ModelCachePolicy(
        needs_full_hidden_states=True,
        deferred_keys=frozenset({"k"}),
        skip_keys=frozenset({"k"}),
    )
    mgr, view = make_manager(policy=policy)
    # a stages k on block 0 only; c prefills block 1 with no mm at all.
    s1 = run_step(mgr, view, {"a": ([0], 0, 4)}, mm={"k": torch.full((4, 2), 5.0)})
    mgr.materialize(s1, ["a"])
    s2 = run_step(mgr, view, {"c": ([1], 0, 4)})
    mgr.materialize(s2, ["c"])
    # b hits all 8 tokens (blocks 0+1): k exists on the first 4 slots only.
    s3 = run_step(mgr, view, {"b": ([0, 1, 2], 8, 2)}, new_hits={"b": 8}, mm={"k": torch.full((2, 2), 9.0)})
    rows = mgr.materialize(s3, ["b"]).mm_outputs["k"]["b"]
    assert torch.equal(rows[:4], torch.full((4, 2), 5.0))
    assert torch.equal(rows[4:8], torch.zeros(4, 2))  # never registered -> baseline
    assert torch.equal(rows[8:], torch.full((2, 2), 9.0))


def test_mm_in_transit_unresolvable_fails_fast():
    """Rows registered in-transit whose entry cannot serve them must raise,
    never silently ship zeros (the pre-(slot,key) failure mode)."""
    policy = ModelCachePolicy(
        needs_full_hidden_states=True,
        deferred_keys=frozenset({"k"}),
        skip_keys=frozenset({"k"}),
    )
    mgr, view = make_manager(policy=policy)
    s1 = run_step(mgr, view, {"a": ([0], 0, 4)}, mm={"k": torch.full((4, 2), 1.0)})
    mgr.materialize(s1, ["a"])
    # Corrupt bookkeeping: drop the staged entry without draining it.
    tid = mgr._deferred_tasks["a"].tid
    mgr._controller._tasks.pop(tid)
    s2 = run_step(mgr, view, {"b": ([0, 1], 4, 2)}, new_hits={"b": 4}, mm={"k": torch.full((2, 2), 9.0)})
    with pytest.raises(OmniPrefixCacheUnmatchError):
        mgr.materialize(s2, ["b"])


def test_lock_never_covers_fetch_or_join():
    """Invariant 6: the state lock must not be held across blocking waits —
    controller.join (save) and fetch_host / mirror reads (materialize)."""
    policy = ModelCachePolicy(
        needs_full_hidden_states=True,
        deferred_keys=frozenset({"k"}),
        skip_keys=frozenset({"k"}),
    )
    import threading

    mgr, view = make_manager(policy=policy)
    calls = []

    def probe(kind):
        # locked() sees ANY holder; the prefetch worker fetches by design
        # while the engine thread holds the lock in new_step_starts, so only
        # facade-thread calls prove a violation.
        on_facade = not threading.current_thread().name.startswith("omni-prefix-cache-prefetch")
        calls.append((kind, on_facade and mgr._state_lock.locked()))

    real_fetch = mgr._controller.fetch_host
    real_join = mgr._controller.join_host_ready
    mgr._controller.fetch_host = lambda *a, **kw: (probe("fetch"), real_fetch(*a, **kw))[1]
    mgr._controller.join_host_ready = lambda ids: (probe("join"), real_join(ids))[1]

    s1 = run_step(mgr, view, {"a": ([0], 0, 4)}, mm={"k": torch.full((4, 2), 1.0)})
    mgr.materialize(s1, ["a"])
    # Hit forces a staged fetch; the follow-up save forces a join of the
    # previous step's JOIN_NEXT_STEP tasks.
    s2 = run_step(mgr, view, {"b": ([0, 1], 4, 2)}, new_hits={"b": 4}, mm={"k": torch.full((2, 2), 9.0)})
    mgr.materialize(s2, ["b"])
    assert any(kind == "fetch" for kind, _ in calls)
    assert any(kind == "join" for kind, _ in calls)
    assert all(not locked for _, locked in calls), calls


def test_failed_write_fails_fast_at_next_facade_entry():
    """A committer write failure means rows are lost behind published hashes:
    the next facade call must raise once, at the cause — not poison every
    future hit touching those slots (F2)."""
    mgr, view = make_manager()
    sid = run_step(mgr, view, {"a": ([0], 0, 4)})
    mgr.materialize(sid, ["a"])
    # Simulate a committer failure on a still-registered entry.
    from vllm_omni.core.prefix_cache.controller import WriteTask, _Segment
    from vllm_omni.core.prefix_cache.interface import WriteSchedule

    task = WriteTask(
        tid=999,
        req_id="x",
        write_n=1,
        schedule=WriteSchedule.JOIN_NEXT_STEP,
        segments=[_Segment(slots_cpu=torch.tensor([0]), tensors={})],
    )
    mgr._controller._tasks[999] = task
    mgr._controller._fail_task(999)
    with pytest.raises(OmniPrefixCacheUnmatchError, match="write failed"):
        run_step(mgr, view, {"a": ([0, 1], 4, 1)})


def test_per_request_staging_writes_join_next_step():
    """One save produces one WriteTask per request; staging D2H is already
    in flight, so every queued write is JOIN_NEXT_STEP."""
    mgr, view = make_manager()
    sid = run_step(mgr, view, {"p": ([0, 1], 0, 8), "d": ([2], 0, 1)})
    tp = mgr._controller.get_task(next(iter(mgr._req_tasks["p"])))
    td = mgr._controller.get_task(next(iter(mgr._req_tasks["d"])))
    assert tp is not None and td is not None
    from vllm_omni.core.prefix_cache.interface import WriteSchedule

    assert tp.schedule is WriteSchedule.JOIN_NEXT_STEP
    assert td.schedule is WriteSchedule.JOIN_NEXT_STEP
    assert (tp.req_id, td.req_id) == ("p", "d")
    assert mgr._join_next_step_tids == [tp.tid, td.tid]
    outs = mgr.materialize(sid, ["p", "d"])
    assert torch.equal(outs.hidden_states["p"], expected_rows(view.slots_for("p", 0, 8)))
    assert torch.equal(outs.hidden_states["d"], expected_rows(view.slots_for("d", 0, 1)))


def test_staging_step_prefills_task_host_and_recycles():
    """Staging steps pre-fill per-task host views at save (no per-task copy),
    and slots recycle safely across > depth steps without cross-step
    corruption (outputs are copied out of the reusable slot)."""
    mgr, view = make_manager()
    for i in range(6):  # > staging_depth(4): forces slot reuse
        sid = run_step(mgr, view, {"a": ([i % 8, (i % 8) + 8], i, 1)})
        ctx = mgr._step_ctxs[sid]
        assert ctx.d2h is not None
        expect = expected_rows(view.slots_for("a", i, i + 1))  # capture per step
        outs = mgr.materialize(sid, ["a"])
        assert torch.equal(outs.hidden_states["a"], expect), i


def test_oversized_step_fails_fast():
    """A step larger than staging capacity is a config/contract error."""
    mgr, view = make_manager(staging_capacity_tokens=4)
    with pytest.raises(OmniPrefixCacheUnmatchError, match="staging capacity is 4"):
        run_step(mgr, view, {"a": ([0, 1], 0, 8)})  # 8 > 4


def test_staging_task_never_defers_and_slot_held_until_drain():
    """A staging task's D2H is already in flight -> always JOIN_NEXT_STEP, and
    its slot holder survives the (eager, inline) scatter until the manager
    drains the completion — the window hit readers pin on."""
    from vllm_omni.core.prefix_cache.interface import WriteSchedule

    mgr, view = make_manager()
    sid = run_step(mgr, view, {"p": ([0, 1], 0, 8)})
    ctx = mgr._step_ctxs[sid]
    tid = next(iter(mgr._req_tasks["p"]))
    assert ctx.d2h is not None
    assert mgr._controller.get_task(tid).schedule is WriteSchedule.JOIN_NEXT_STEP
    busy = mgr._controller._staging_pool._busy[ctx.d2h.slot]
    assert StagingBufferHolder.for_task(tid) in busy and StagingBufferHolder.for_step(sid) in busy
    mgr.materialize(sid, ["p"])  # drain releases the task holder; consume releases the step holder
    assert not busy


def test_hit_prefetch_prebuilds_merged_buffer():
    """A registered hit over committed rows is gathered by the prefetch
    thread before the forward finishes; materialize only fills the tail."""
    mgr, view = make_manager()
    s1 = run_step(mgr, view, {"a": ([0, 1], 0, 8)})
    mgr.materialize(s1, ["a"])
    s2 = run_step(mgr, view, {"b": ([0, 1, 2], 8, 4)}, new_hits={"b": 8}, finished=["a"])
    ctx = mgr._step_ctxs[s2]
    fut = ctx.hit_prefetch["b"][HIDDEN_KEY]
    buf = fut.result()
    assert buf.shape == (12, HIDDEN)
    assert torch.equal(buf[:8], expected_rows(view.slots_for("b", 0, 8)))
    merged = mgr.materialize(s2, ["b"]).hidden_states["b"]
    assert merged.data_ptr() == buf.data_ptr()  # the prefetched buffer IS the output
    assert torch.equal(merged[8:], expected_rows(view.slots_for("b", 8, 12)))


def test_same_step_hit_skips_prefetch(caplog):
    """Rows this step's save has not registered yet cannot be planned at
    new_step_starts; the hit falls back to materialize's plan+fetch.
    Prefetch must not log CRITICAL for that expected miss."""
    mgr, view = make_manager()
    view.req_blocks["a"] = [0, 1]
    with caplog.at_level(logging.CRITICAL, logger="vllm_omni.core.prefix_cache.manager"):
        sid = run_step(mgr, view, {"a": ([0, 1], 0, 8), "b": ([0, 1, 2], 8, 4)}, new_hits={"b": 8})
    assert not any("omni prefix cache unmatch" in r.message for r in caplog.records)
    ctx = mgr._step_ctxs[sid]
    assert HIDDEN_KEY not in ctx.hit_prefetch.get("b", {})
    outs = mgr.materialize(sid, ["a", "b"])
    assert torch.equal(outs.hidden_states["b"][:8], expected_rows(view.slots_for("b", 0, 8)))


def test_leftover_mm_snapshot_survives_live_overwrite():
    """Deferred / uncached mm is copied at save; mutating the live buffer
    afterwards must not change materialize (async builder vs next step)."""
    policy = ModelCachePolicy(
        needs_full_hidden_states=True,
        deferred_keys=frozenset({"codes.audio"}),
        skip_keys=frozenset({"codes.audio"}),
    )
    mgr, view = make_manager(policy=policy)
    live = torch.full((2, 2), 1.0)
    sid = run_step(mgr, view, {"a": ([0], 0, 2)}, mm={"codes.audio": live})
    assert "codes.audio" in mgr._step_ctxs[sid].mm_cpu_snapshot
    live.fill_(99.0)
    outs = mgr.materialize(sid, ["a"])
    assert torch.equal(outs.mm_outputs["codes.audio"]["a"], torch.full((2, 2), 1.0))


def test_staging_slot_released_on_no_consumer_early_return():
    """materialize's nothing-to-serve early return must still drop the step
    holder, or the staging pool leaks a slot per step and exhausts."""
    policy = ModelCachePolicy(needs_full_hidden_states=False)
    mgr, view = make_manager(policy=policy)
    for i in range(6):  # > staging_depth(4): leak would exhaust and fail-fast
        sid = run_step(mgr, view, {"a": ([i % 8, (i % 8) + 8], i, 1)}, mm={"k": torch.full((1, 2), float(i))})
        ctx = mgr._step_ctxs[sid]
        assert ctx.d2h is not None, i
        mgr.materialize(sid, ["a"])
        assert not mgr._controller._staging_pool._busy[ctx.d2h.slot], i


def test_save_releases_staging_if_commit_drained_writes_fails():
    """Claim happens outside the lock; a later fail-fast must drop the step holder."""
    mgr, view = make_manager(staging_depth=2)
    calls = {"n": 0}
    real = mgr._commit_drained_writes

    def wrapped():
        calls["n"] += 1
        if calls["n"] == 2:
            raise OmniPrefixCacheUnmatchError("injected fail")
        return real()

    mgr._commit_drained_writes = wrapped
    with pytest.raises(OmniPrefixCacheUnmatchError, match="injected fail"):
        run_step(mgr, view, {"a": ([0], 0, 4)})
    assert all(not busy for busy in mgr._controller._staging_pool._busy)
    mgr._commit_drained_writes = real
    sid = run_step(mgr, view, {"a": ([0], 0, 4)})
    assert mgr._step_ctxs[sid].d2h is not None
    mgr.materialize(sid, ["a"])


def test_from_vllm_config_uses_batched_tokens():
    from types import SimpleNamespace

    cfg = PrefixCacheConfig.from_vllm_config(
        num_blocks=NUM_BLOCKS,
        block_size=BLOCK_SIZE,
        hidden_size=HIDDEN,
        hs_dtype=DTYPE,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8192, max_model_len=32768, max_num_seqs=64),
    )
    assert cfg.staging_capacity_tokens == 8192
    assert cfg.staging_depth == 4
    cfg_fallback = PrefixCacheConfig.from_vllm_config(
        num_blocks=NUM_BLOCKS,
        block_size=BLOCK_SIZE,
        hidden_size=HIDDEN,
        hs_dtype=DTYPE,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=None, max_model_len=4096),
    )
    assert cfg_fallback.staging_capacity_tokens == 4096
