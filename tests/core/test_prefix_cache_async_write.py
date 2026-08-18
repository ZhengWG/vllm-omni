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


def test_force_drain_lands_pending_write():
    """force_drain must scatter this step's pending rows before a same-step merge."""
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
    cache.force_drain_pending_writes()
    assert torch.equal(flat[:num_tokens], hidden_gpu.cpu())


def _overlap_runner(blocks, computed, sched, *, hits=True, pending=True):
    """Writer `w` + hitter `h` stub for _hit_blocks_written_this_step."""
    from types import SimpleNamespace

    from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner

    runner = object.__new__(GPUARModelRunner)
    runner.input_batch = SimpleNamespace(
        req_ids=["w", "h"],
        req_id_to_index={"w": 0, "h": 1},
        num_computed_tokens_cpu=torch.tensor(computed),
        block_table=[SimpleNamespace(block_table=SimpleNamespace(cpu=torch.tensor(blocks)))],
    )
    runner.cache_config = SimpleNamespace(block_size=BLOCK_SIZE)
    cache = get_omni_pcache()
    if hits:
        cache.add_prefix_cached_new_req_id("h")
    if pending:
        cache._pending_write = object()
    runner.omni_prefix_cache = cache
    return runner, {"w": sched[0], "h": sched[1]}


def test_same_step_overlap_requires_drain():
    runner, scheduled = _overlap_runner([[3, 4, 9], [3, 4, 7]], [0, 8], [8, 4])
    assert runner._hit_blocks_written_this_step(scheduled) is True


def test_hit_on_committed_prefix_skips_drain():
    runner, scheduled = _overlap_runner([[5, 6, 7, 8], [5, 6, 0, 0]], [8, 8], [4, 4])
    assert runner._hit_blocks_written_this_step(scheduled) is False
    runner, scheduled = _overlap_runner([[3, 4], [3, 4]], [0, 8], [8, 4], hits=False)
    assert runner._hit_blocks_written_this_step(scheduled) is False
