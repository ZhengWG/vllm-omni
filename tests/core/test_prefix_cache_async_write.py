import pytest
import torch

from vllm_omni.core.prefix_cache import OmniTensorPrefixCache

# `cpu` is the lane selector, not a hardware claim: the only Buildkite steps that
# reach tests/ select `core_model and cpu`, and they run on GPU hardware (l4_1).
# Marking this `cuda` collects it nowhere. Real hardware need is the skipif below.
pytestmark = [
    pytest.mark.core_model,
    pytest.mark.cpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device"),
]

NUM_BLOCKS = 10
BLOCK_SIZE = 4
HIDDEN_SIZE = 2
DTYPE = torch.float32


def get_omni_pcache() -> OmniTensorPrefixCache:
    """Build an OmniTensorPrefixCache, but don't init mm tensors."""
    return OmniTensorPrefixCache(
        num_blocks=NUM_BLOCKS,
        block_size=BLOCK_SIZE,
        hidden_size=HIDDEN_SIZE,
        hs_dtype=DTYPE,
    )


def test_drain_ready_async_writes_is_a_noop_before_the_event_is_ready(monkeypatch):
    """drain_ready_async_writes() must not touch the cache while the
    scheduled D2H's event has not completed, and must consume it exactly
    once after it has.

    Real GPU timing (e.g. via torch.cuda._sleep to stall the copy stream)
    was tried and found unreliable on this box: cross-stream wait_stream/
    wait_event did not reliably delay event.query() from reporting ready
    immediately, likely a virtualization/driver quirk of this cloud GPU.
    Forcing the event's query() result directly tests the branching logic
    in drain_ready_async_writes/_consume_pending_write deterministically,
    independent of real CUDA scheduling behavior.
    """
    cache = get_omni_pcache()
    mm_key = "codes.audio"
    cache.maybe_init_missing_mm_cache_keys({mm_key: torch.zeros(4, HIDDEN_SIZE)}, seq_len=4)

    num_tokens = 4
    slot_mapping_gpu = torch.arange(8, 8 + num_tokens, dtype=torch.int64, device="cuda")
    mm_gpu = {
        mm_key: torch.arange(num_tokens * HIDDEN_SIZE, dtype=torch.float32, device="cuda").reshape(
            num_tokens, HIDDEN_SIZE
        )
    }

    cache.schedule_async_write(
        hidden_states_gpu=None,
        multimodal_outputs_gpu=mm_gpu,
        slot_mapping_gpu=slot_mapping_gpu,
        num_tokens_unpadded=num_tokens,
        num_tokens_padded=num_tokens,
    )

    pending_event = cache._pending_write.event
    rows = cache.mm_outputs_cache[mm_key].view(-1, HIDDEN_SIZE)

    monkeypatch.setattr(pending_event, "query", lambda: False)
    assert cache.drain_ready_async_writes() == 0
    assert torch.all(rows[8 : 8 + num_tokens] == 0)
    assert cache._pending_write is not None

    monkeypatch.setattr(pending_event, "query", lambda: True)
    assert cache.drain_ready_async_writes() == 1
    assert torch.equal(rows[8 : 8 + num_tokens], mm_gpu[mm_key].cpu())
    assert cache._pending_write is None

    # Nothing pending anymore: a second drain is a no-op, not a re-scatter.
    assert cache.drain_ready_async_writes() == 0


def test_schedule_async_write_consumes_previous_pending_write_first():
    """Scheduling step N+1's write must first consume (scatter) step N's
    pending write, even if the caller never explicitly called
    drain_ready_async_writes() in between -- otherwise step N's data would
    be silently dropped when overwritten by step N+1's pending write."""
    cache = get_omni_pcache()
    mm_key = "codes.audio"
    cache.maybe_init_missing_mm_cache_keys({mm_key: torch.zeros(4, HIDDEN_SIZE)}, seq_len=4)

    num_tokens = 4
    first_slots = torch.arange(8, 8 + num_tokens, dtype=torch.int64, device="cuda")
    first_gpu = {mm_key: torch.full((num_tokens, HIDDEN_SIZE), 1.0, device="cuda")}
    second_slots = torch.arange(12, 12 + num_tokens, dtype=torch.int64, device="cuda")
    second_gpu = {mm_key: torch.full((num_tokens, HIDDEN_SIZE), 2.0, device="cuda")}

    cache.schedule_async_write(
        hidden_states_gpu=None,
        multimodal_outputs_gpu=first_gpu,
        slot_mapping_gpu=first_slots,
        num_tokens_unpadded=num_tokens,
        num_tokens_padded=num_tokens,
    )
    torch.accelerator.synchronize()

    # This call's internal _consume_pending_write() must flush the FIRST
    # write before the second one is scheduled; the caller never drained.
    cache.schedule_async_write(
        hidden_states_gpu=None,
        multimodal_outputs_gpu=second_gpu,
        slot_mapping_gpu=second_slots,
        num_tokens_unpadded=num_tokens,
        num_tokens_padded=num_tokens,
    )
    torch.accelerator.synchronize()
    cache.drain_ready_async_writes()

    rows = cache.mm_outputs_cache[mm_key].view(-1, HIDDEN_SIZE)
    assert torch.all(rows[8:12] == 1.0)
    assert torch.all(rows[12:16] == 2.0)


def test_force_drain_lands_pending_write_for_same_step_hit():
    """Same-step optimistic-hash hit: the hit rows live in the pending async
    write scheduled this step. force_drain_pending_writes() must land them in
    the CPU mirror before the merge reads it; without it the mirror still
    holds zeros (the silent-stale bug this hotfix closes)."""
    cache = get_omni_pcache()
    num_tokens = 4
    slots = torch.arange(0, num_tokens, dtype=torch.int64, device="cuda")
    hidden_gpu = torch.arange(num_tokens * HIDDEN_SIZE, dtype=DTYPE, device="cuda").reshape(num_tokens, HIDDEN_SIZE)

    cache.schedule_async_write(
        hidden_states_gpu=hidden_gpu,
        multimodal_outputs_gpu=None,
        slot_mapping_gpu=slots,
        num_tokens_unpadded=num_tokens,
        num_tokens_padded=num_tokens,
    )
    flat = cache.hidden_states_cache.view(-1, HIDDEN_SIZE)

    cache.force_drain_pending_writes()
    assert torch.equal(flat[:num_tokens], hidden_gpu.cpu())
    # Idempotent when nothing is pending.
    cache.force_drain_pending_writes()
    assert torch.equal(flat[:num_tokens], hidden_gpu.cpu())


def _runner_with_batch(block_table, computed, req_ids, hit_ids, pending_write=True):
    """Minimal runner stub exercising _hit_blocks_written_this_step."""
    from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner

    class _Wrap:
        def __init__(self, t):
            self.cpu = t

    class _Group:
        def __init__(self, t):
            self.block_table = _Wrap(t)

    class _BT:
        def __init__(self, t):
            self._g = _Group(t)

        def __getitem__(self, i):
            return self._g

    class _IB:
        def __init__(self):
            self.req_ids = req_ids
            self.req_id_to_index = {r: i for i, r in enumerate(req_ids)}
            self.num_computed_tokens_cpu = torch.tensor(computed)
            self.block_table = _BT(block_table)

    runner = object.__new__(GPUARModelRunner)
    runner.input_batch = _IB()
    runner.cache_config = type("C", (), {"block_size": BLOCK_SIZE})()
    cache = get_omni_pcache()
    for r in hit_ids:
        cache.add_prefix_cached_new_req_id(r)
    if pending_write:
        # The overlap check is a no-op without an in-flight write.
        cache._pending_write = object()
    runner.omni_prefix_cache = cache
    return runner


def test_same_step_hit_on_freshly_written_blocks_requires_drain():
    """A hit on blocks another request is computing THIS step must drain."""
    # r_new hits blocks [3, 4]; r_prefill is writing block 3 this step.
    block_table = torch.tensor([[3, 4, 9], [3, 4, 7]])
    runner = _runner_with_batch(block_table, computed=[0, 8], req_ids=["r_prefill", "r_new"], hit_ids={"r_new"})
    assert runner._hit_blocks_written_this_step({"r_prefill": 8, "r_new": 4}) is True


def test_hit_on_committed_blocks_skips_drain():
    """A hit on blocks written in EARLIER steps must not block the step."""
    # r_new hits blocks [1, 2]; r_decode only writes into block 8 this step.
    block_table = torch.tensor([[8, 0, 0], [1, 2, 5]])
    runner = _runner_with_batch(block_table, computed=[0, 8], req_ids=["r_decode", "r_new"], hit_ids={"r_new"})
    assert runner._hit_blocks_written_this_step({"r_decode": 1, "r_new": 4}) is False


def test_no_new_hits_skips_drain():
    block_table = torch.tensor([[3, 4, 9]])
    runner = _runner_with_batch(block_table, computed=[0], req_ids=["r"], hit_ids=set())
    assert runner._hit_blocks_written_this_step({"r": 4}) is False


def test_partial_overlap_still_requires_drain():
    """One shared block out of many is enough to expose the stale window."""
    # r_new hits blocks [1, 2, 3]; r_prefill writes only block 3 this step.
    block_table = torch.tensor([[9, 3, 0], [1, 2, 3]])
    runner = _runner_with_batch(
        block_table, computed=[BLOCK_SIZE, 3 * BLOCK_SIZE], req_ids=["r_prefill", "r_new"], hit_ids={"r_new"}
    )
    assert runner._hit_blocks_written_this_step({"r_prefill": BLOCK_SIZE, "r_new": 4}) is True


def test_chunked_prefill_middle_chunk_is_covered():
    """A mid-prefill chunk writes blocks starting at num_computed, not 0."""
    # r_prefill already did 2 blocks and now writes block index 2 (id 7).
    block_table = torch.tensor([[5, 6, 7, 8], [7, 0, 0, 0]])
    runner = _runner_with_batch(
        block_table, computed=[2 * BLOCK_SIZE, BLOCK_SIZE], req_ids=["r_prefill", "r_new"], hit_ids={"r_new"}
    )
    # r_new hits block 7, which r_prefill writes in this very chunk.
    assert runner._hit_blocks_written_this_step({"r_prefill": BLOCK_SIZE, "r_new": 4}) is True
    # Its earlier blocks (5, 6) are already committed -> no drain needed.
    block_table2 = torch.tensor([[5, 6, 7, 8], [5, 0, 0, 0]])
    runner2 = _runner_with_batch(
        block_table2, computed=[2 * BLOCK_SIZE, BLOCK_SIZE], req_ids=["r_prefill", "r_new"], hit_ids={"r_new"}
    )
    assert runner2._hit_blocks_written_this_step({"r_prefill": BLOCK_SIZE, "r_new": 4}) is False


def test_hit_span_clamped_to_block_table_width():
    """num_computed can exceed the table width; the scan must not overrun."""
    block_table = torch.tensor([[4, 0], [1, 2]])
    runner = _runner_with_batch(block_table, computed=[0, 99 * BLOCK_SIZE], req_ids=["r_w", "r_new"], hit_ids={"r_new"})
    # Only blocks 1 and 2 are readable; r_w writes block 4 -> no overlap.
    assert runner._hit_blocks_written_this_step({"r_w": 1, "r_new": 4}) is False


def test_unscheduled_requests_write_nothing():
    """A request with 0 scheduled tokens contributes no written blocks."""
    block_table = torch.tensor([[3, 4], [3, 4]])
    runner = _runner_with_batch(
        block_table, computed=[0, 2 * BLOCK_SIZE], req_ids=["r_idle", "r_new"], hit_ids={"r_new"}
    )
    assert runner._hit_blocks_written_this_step({"r_idle": 0, "r_new": 4}) is False


def test_large_batch_all_hits_is_correct_and_bounded():
    """Batch-arrival case: many hitting requests, only one真 overlap."""
    n, width = 64, 8
    block_table = torch.arange(n * width, dtype=torch.int64).reshape(n, width)
    # r0 fills its block 0 (id 0) this step; r1..r63 hit disjoint blocks.
    req_ids = [f"r{i}" for i in range(n)]
    runner = _runner_with_batch(
        block_table, computed=[0] + [width * BLOCK_SIZE] * (n - 1), req_ids=req_ids, hit_ids=set(req_ids[1:])
    )
    # r0 must complete a block, or the cheap gate correctly skips the scan.
    sched = {r: (BLOCK_SIZE if r == "r0" else 4) for r in req_ids}
    assert runner._hit_blocks_written_this_step(sched) is False
    # Now make r1's first block collide with what r0 writes.
    block_table[1, 0] = 0
    runner2 = _runner_with_batch(
        block_table, computed=[0] + [width * BLOCK_SIZE] * (n - 1), req_ids=req_ids, hit_ids=set(req_ids[1:])
    )
    assert runner2._hit_blocks_written_this_step(sched) is True


def test_no_block_completed_this_step_skips_scan():
    """Only full blocks are hashable, so a step that completes none cannot
    expose the stale window — even when hit and written blocks overlap."""
    # r_decode writes inside block 3 (id 3) but does not fill it; r_new's hit
    # span covers that same block id, which would collide if it were readable.
    block_table = torch.tensor([[3, 4], [3, 4]])
    runner = _runner_with_batch(
        block_table,
        computed=[1, 2 * BLOCK_SIZE],
        req_ids=["r_decode", "r_new"],
        hit_ids={"r_new"},
    )
    assert runner._hit_blocks_written_this_step({"r_decode": 1, "r_new": 4}) is False
    # One more token completes the block -> the scan runs and finds the overlap.
    runner2 = _runner_with_batch(
        block_table,
        computed=[BLOCK_SIZE - 1, 2 * BLOCK_SIZE],
        req_ids=["r_decode", "r_new"],
        hit_ids={"r_new"},
    )
    assert runner2._hit_blocks_written_this_step({"r_decode": 1, "r_new": 4}) is True


def test_no_pending_write_skips_everything():
    """Nothing in flight means nothing to land before the merge."""
    block_table = torch.tensor([[3, 4], [3, 4]])
    runner = _runner_with_batch(
        block_table,
        computed=[0, 2 * BLOCK_SIZE],
        req_ids=["r_prefill", "r_new"],
        hit_ids={"r_new"},
        pending_write=False,
    )
    # No async write was scheduled.
    assert runner.omni_prefix_cache.has_pending_write() is False
    assert runner._hit_blocks_written_this_step({"r_prefill": 2 * BLOCK_SIZE, "r_new": 4}) is False
