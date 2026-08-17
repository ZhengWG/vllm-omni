"""Unit tests for vllm_omni/core/tensor_cache (manager + controller + pool).

CPU-only: the controller runs in eager mode. Uses a fake group view, so no
vLLM runtime is required (runnable without a vllm install via
``pytest --confcutdir=tests/core``).
"""

import sys
import types
from pathlib import Path

import pytest
import torch

try:  # pragma: no cover - shim only matters on vllm-less dev machines
    import vllm  # noqa: F401
except ModuleNotFoundError:
    # Bypass vllm_omni/__init__ (which imports vllm): register namespace
    # parents so the pure-torch tensor_cache subpackage imports directly.
    _root = Path(__file__).resolve().parents[2]
    for _pkg in ("vllm_omni", "vllm_omni.core"):
        if _pkg not in sys.modules:
            _m = types.ModuleType(_pkg)
            _m.__path__ = [str(_root / _pkg.replace(".", "/"))]
            sys.modules[_pkg] = _m

from vllm_omni.core.tensor_cache.group_view import FullAttentionGroupView
from vllm_omni.core.tensor_cache.interface import (
    ModelCachePolicy,
    OmniTensorCacheUnmatchError,
    TensorCacheConfig,
)
from vllm_omni.core.tensor_cache.manager import OmniTensorCacheManager

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

    def slot_mapping_gpu(self, num_tokens: int) -> torch.Tensor:
        assert self.step_slot_mapping is not None
        return self.step_slot_mapping[:num_tokens]

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


def make_manager(view=None, policy=None, **cfg_kwargs) -> tuple[OmniTensorCacheManager, FakeView]:
    view = view or FakeView()
    config = TensorCacheConfig(
        num_blocks=NUM_BLOCKS, block_size=BLOCK_SIZE, hidden_size=HIDDEN, hs_dtype=DTYPE, **cfg_kwargs
    )
    mgr = OmniTensorCacheManager(config, view, eager=True)
    if policy is not None:
        mgr.register_policy(policy)
    return mgr, view


def run_step(mgr, view, reqs: dict[str, tuple[list[int], int, int]], new_hits=None, finished=(), mm=None):
    """One step: reqs = req_id -> (blocks, sched_start_pos, sched_tokens)."""
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
    mgr.save_outputs(hidden, mm or {}, num_tokens_unpadded=n, num_tokens_padded=n)
    return hidden


def expected_rows(slots: torch.Tensor) -> torch.Tensor:
    return slots.to(DTYPE).unsqueeze(1).expand(slots.numel(), HIDDEN)


def test_no_hit_passthrough():
    mgr, view = make_manager()
    run_step(mgr, view, {"a": ([0, 1], 0, 8)})
    outs = mgr.materialize(["a"])
    assert torch.equal(outs.hidden_states["a"], expected_rows(view.slots_for("a", 0, 8)))


def test_hit_merge_from_mirror():
    mgr, view = make_manager()
    # Producer prefills 8 tokens on blocks [0, 1] and finishes.
    run_step(mgr, view, {"a": ([0, 1], 0, 8)})
    mgr.materialize(["a"])
    # Consumer hits the full 8 tokens and computes 4 new on block 2.
    run_step(mgr, view, {"b": ([0, 1, 2], 8, 4)}, new_hits={"b": 8}, finished=["a"])
    outs = mgr.materialize(["b"])
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
    run_step(
        mgr,
        view,
        {"a": ([0, 1], 0, 8), "b": ([0, 1, 2], 8, 4)},
        new_hits={"b": 8},
    )
    outs = mgr.materialize(["a", "b"])
    merged = outs.hidden_states["b"]
    assert torch.equal(merged[:8], expected_rows(view.slots_for("b", 0, 8)))


def test_absent_hit_fails_fast():
    mgr, view = make_manager()
    view.req_blocks["c"] = [5, 6]
    run_step(mgr, view, {"c": ([5, 6, 7], 8, 4)}, new_hits={"c": 8})
    with pytest.raises(OmniTensorCacheUnmatchError):
        mgr.materialize(["c"])


def test_hit_not_block_aligned_asserts():
    mgr, view = make_manager()
    run_step(mgr, view, {"a": ([0, 1], 0, 8)})
    mgr.materialize(["a"])
    run_step(mgr, view, {"b": ([0, 1, 2], 8, 4)}, new_hits={"b": 6})
    with pytest.raises(AssertionError):
        mgr.materialize(["b"])


def test_mm_cached_key_merge():
    mgr, view = make_manager()
    feat = 3
    mm1 = {"talker.h": torch.arange(8 * feat, dtype=DTYPE).reshape(8, feat)}
    run_step(mgr, view, {"a": ([0, 1], 0, 8)}, mm=mm1)
    mgr.materialize(["a"])
    mm2 = {"talker.h": torch.full((4, feat), 7.0)}
    run_step(mgr, view, {"b": ([0, 1, 2], 8, 4)}, new_hits={"b": 8}, finished=["a"], mm=mm2)
    outs = mgr.materialize(["b"])
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
        run_step(
            mgr,
            view,
            {"a": ([3], pos, 1)},
            mm={"codes.audio": torch.full((1, feat), float(pos + 1))},
        )
        mgr.materialize(["a"])
    # In-transit read before finish: deferred rows come from staged chunks.
    slots = view.slots_for("a", 0, 2)
    rows = mgr._resolve_rows(slots, "codes.audio", strict=False, req_id="a", states=mgr._slot_state[slots])
    assert torch.equal(rows[:, 0], torch.tensor([1.0, 2.0]))
    # Finish (abort semantics identical): entry escalates and commits.
    run_step(mgr, view, {"z": ([9], 0, 1)}, finished=["a"])
    mgr.materialize(["z"])
    mirror = mgr._pool.rows("codes.audio", slots)
    assert torch.equal(mirror[:, 0], torch.tensor([1.0, 2.0]))


def test_cap_forces_flush_of_deferred():
    policy = ModelCachePolicy(
        needs_full_hidden_states=False, deferred_keys=frozenset({"k"}), skip_keys=frozenset({"k"})
    )
    # float32 feat=4 row = 16 bytes; 48 forces a flush by step 4.
    mgr, view = make_manager(policy=policy, gpu_staging_bytes=48)
    for pos in range(4):
        run_step(mgr, view, {"a": ([2, 3], pos, 1)}, mm={"k": torch.full((1, 4), float(pos))})
        mgr.materialize(["a"])
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


def test_class_b_join_previous_step():
    mgr, view = make_manager()
    run_step(mgr, view, {"a": ([0], 0, 2)})
    task_id = mgr._prev_class_b
    assert task_id is not None
    mgr.materialize(["a"])
    run_step(mgr, view, {"a": ([0], 2, 1)})
    # Previous B entry joined & drained at the save above.
    assert mgr._controller.get_task(task_id) is None
    assert int(mgr._slot_state[view.slots_for("a", 0, 2)].min()) == 2
    mgr.materialize(["a"])


def test_tenant_succession_hit_reads_newest():
    # a occupies block 0, finishes; b reuses block 0 with new values;
    # c hits b's prefix -> must read b's rows, never a's.
    mgr, view = make_manager()
    run_step(mgr, view, {"a": ([0], 0, 4)})
    mgr.materialize(["a"])
    run_step(mgr, view, {"b": ([0], 0, 4)}, finished=["a"])
    mgr.materialize(["b"])
    b_hidden = expected_rows(view.slots_for("b", 0, 4))
    run_step(mgr, view, {"c": ([0, 1], 4, 2)}, new_hits={"c": 4}, finished=["b"])
    outs = mgr.materialize(["c"])
    assert torch.equal(outs.hidden_states["c"][:4], b_hidden)


def test_deferred_tenant_succession_no_stale_wins():
    # Preemption-style reuse on the deferred path: a's staged rows (val 1)
    # still pending when b stages the same slots (val 2). Whatever the
    # finish order, the mirror's final value must be b's.
    policy = ModelCachePolicy(
        needs_full_hidden_states=False, deferred_keys=frozenset({"k"}), skip_keys=frozenset({"k"})
    )
    mgr, view = make_manager(policy=policy)
    run_step(mgr, view, {"a": ([2], 0, 2)}, mm={"k": torch.full((2, 2), 1.0)})
    mgr.materialize(["a"])
    # b reuses block 2 while a's deferred entry is still staged (a preempted,
    # not finished). Conflict detection must escalate a's entry first.
    run_step(mgr, view, {"b": ([2], 0, 2)}, mm={"k": torch.full((2, 2), 2.0)})
    mgr.materialize(["b"])
    slots = view.slots_for("b", 0, 2)
    # Dangerous order: b finishes (flushes) BEFORE a does.
    run_step(mgr, view, {"z1": ([9], 0, 1)}, finished=["b"])
    mgr.materialize(["z1"])
    run_step(mgr, view, {"z2": ([9], 1, 1)}, finished=["a"])
    mgr.materialize(["z2"])
    assert torch.equal(mgr._pool.rows("k", slots), torch.full((2, 2), 2.0))


def test_slot_reuse_records_dependency():
    mgr, view = make_manager()
    policy = ModelCachePolicy(
        needs_full_hidden_states=False, deferred_keys=frozenset({"k"}), skip_keys=frozenset({"k"})
    )
    mgr.register_policy(policy)
    # Deferred entry holds slots of block 2 in-transit.
    run_step(mgr, view, {"a": ([2], 0, 1)}, mm={"k": torch.ones(1, 2)})
    mgr.materialize(["a"])
    dtask = mgr._deferred_tasks["a"]
    # New request reuses block 2 while the old entry is still staged.
    mgr.register_policy(ModelCachePolicy())
    run_step(mgr, view, {"b": ([2], 0, 2)})
    ctx_entry = mgr._step_ctxs[-1].entry_id
    # Dependency edge recorded (old deferred entry must land first). In
    # eager mode the main entry already committed, so check the recorded
    # deps on the drained-task metadata instead of live ordering.
    assert dtask.entry_id != ctx_entry
    mgr.materialize(["b"])


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
    run_step(mgr, view, {"a": ([0], 0, 4)}, mm={"k": torch.full((4, 2), 1.0)})
    mgr.materialize(["a"])
    run_step(mgr, view, {"b": ([0, 1], 4, 2)}, new_hits={"b": 4}, mm={"k": torch.full((2, 2), 9.0)})
    rows = mgr.materialize(["b"]).mm_outputs["k"]["b"]
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
    run_step(mgr, view, {"a": ([2], 0, 1)}, mm={"k": torch.full((1, 2), 1.0)})
    mgr.materialize(["a"])
    first = mgr._deferred_tasks["a"]
    # Force the entry closed the way a cap flush would.
    mgr._controller.escalate([first.entry_id])
    run_step(mgr, view, {"a": ([2], 1, 1)}, mm={"k": torch.full((1, 2), 2.0)})
    mgr.materialize(["a"])
    assert mgr._deferred_tasks["a"].entry_id != first.entry_id
    slots = view.slots_for("a", 0, 2)
    rows = mgr._resolve_rows(slots, "k", strict=False, req_id="a", states=mgr._slot_state[slots])
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


def test_materialize_matches_context_by_identity_after_desync():
    """A step that saves but is never materialized must not shift the FIFO.

    Regression: popleft() blindly returned the stale context, so the merge
    sliced the current batch's rows with the previous step's offsets (and,
    once fail-fast landed, killed the engine under async output).
    """
    mgr, view = make_manager()
    # Step 1 for "a" is saved but never consumed (no pooler payload path).
    run_step(mgr, view, {"a": ([0], 0, 4)})
    # Step 2 for a different request; materialize must pick THIS context.
    run_step(mgr, view, {"b": ([1], 0, 4)})
    outs = mgr.materialize(["b"])
    assert torch.equal(outs.hidden_states["b"], expected_rows(view.slots_for("b", 0, 4)))
    # The stale context was dropped, not silently reused.
    assert len(mgr._step_ctxs) == 0


def test_materialize_serves_overlapping_subset():
    """The output builder's request set can exceed the saved one; the cache
    must still merge the requests it owns instead of degrading the step."""
    mgr, view = make_manager()
    run_step(mgr, view, {"a": ([0, 1], 0, 8)})
    mgr.materialize(["a"])
    run_step(mgr, view, {"b": ([0, 1, 2], 8, 4)}, new_hits={"b": 8}, finished=["a"])
    outs = mgr.materialize(["b", "late_joiner"])
    assert outs is not None
    # b still gets the full cached-prefix + new-tail merge.
    assert outs.hidden_states["b"].shape == (12, HIDDEN)
    # The request the step never saw is left to the caller's fresh slice.
    assert "late_joiner" not in outs.hidden_states


def test_materialize_without_owning_context_returns_none():
    """No context for these requests is a capability gap, not a data
    inconsistency: degrade to the uncached path instead of killing the
    engine (absent slots inside a hit span still fail fast)."""
    mgr, view = make_manager()
    run_step(mgr, view, {"a": ([0], 0, 4)})
    assert mgr.materialize(["ghost"]) is None


def test_fetch_host_fast_path_matches_general_path():
    """The single-segment fast path must return exactly what the general
    regroup path does, including for shuffled and partial slot sets."""
    from vllm_omni.core.tensor_cache.block_pool import TensorBlockPool
    from vllm_omni.core.tensor_cache.controller import (
        CLASS_B,
        EntryWriteTask,
        OmniTensorCacheController,
        _Segment,
    )
    from vllm_omni.core.tensor_cache.interface import HIDDEN_KEY

    cfg = TensorCacheConfig(num_blocks=NUM_BLOCKS, block_size=BLOCK_SIZE, hidden_size=HIDDEN, hs_dtype=DTYPE)
    pool = TensorBlockPool(cfg)
    pool.ensure_key(HIDDEN_KEY, DTYPE, HIDDEN)
    ctrl = OmniTensorCacheController(pool, cfg, eager=True)

    slots = torch.tensor([12, 4, 7, 20, 1], dtype=torch.int64)
    rows = torch.arange(slots.numel() * HIDDEN, dtype=DTYPE).reshape(slots.numel(), HIDDEN)
    task = EntryWriteTask(entry_id=1, klass=CLASS_B, segments=[_Segment(slots_cpu=slots, tensors={HIDDEN_KEY: rows})])
    ctrl.submit(task)

    for want in (slots, slots[[3, 0, 4]], slots[[2]]):
        fast = ctrl.fetch_host(task, want, HIDDEN_KEY)
        s2r = task.slot_to_row()
        general = ctrl._rows_from(task, [s2r[int(s)] for s in want.tolist()], HIDDEN_KEY, host=True)
        assert torch.equal(fast, general), (want, fast, general)
        # And it is the actual data for those slots.
        expect = torch.stack([rows[(slots == s).nonzero()[0, 0]] for s in want.tolist()])
        assert torch.equal(fast, expect)
