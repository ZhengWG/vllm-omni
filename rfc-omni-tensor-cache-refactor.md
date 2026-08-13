# [RFC] Refactor Omni Tensor Prefix Cache: async-by-default, decoupled, and extensible

## 1. Overview

vLLM-Omni's hidden-state prefix cache (`OmniTensorPrefixCache`, introduced in #2164) mirrors vLLM's KV-cache block/slot mapping to cache stage outputs (hidden states and per-token multimodal tensors) on CPU, so that prefix-cache hits skip both KV recomputation *and* inter-stage tensor recomputation.

Since the initial merge the implementation has grown organically (#4106 async writes, #3734 staging dedup, #3665 correctness fixes, deferred mm-chunk commits) and now has four structural problems:

1. **Performance**: the write pipeline is only half-async. The CPU scatter (`index_copy_`) still runs on the runner main thread every step, decode-side deferred mm chunks grow unboundedly on GPU and cause a synchronous commit spike at request finish, and the read/merge path is fully synchronous with a blocking `.cpu()` fallback.
2. **Correctness**: vLLM caches block hashes *optimistically at allocation* (`kv_cache_manager.py` commits `computed + num_new_tokens`), so a request scheduled later in the same step can hit blocks whose tokens are computed in that very step. Since #4106 the omni write lands in the CPU mirror one step later, so a same-step cross-request hit merges never-written rows (zeros/stale) and silently forwards them downstream — reproducible with concurrently arriving duplicate prompts. The pre-#4106 synchronous write did not have this window. Additionally, request-keyed deferred mm chunks are dropped when an aborted request is no longer in the input batch (`prefix_cache.py`) while its hashed blocks remain hittable from the free queue, leaving never-written mm rows reachable.
3. **Coupling**: the cache exposes ~12 methods plus raw attribute access, and the runner carries ~6 *implicit ordering contracts* (drain-before-merge, commit-before-batch-removal, consume-before-merge, ...) plus model-policy probing via `getattr`. This coupling is why prefix caching is currently **mutually exclusive** with async omni output (#4476 hard-disables it via a safety guard, `gpu_ar_model_runner.py`) and silently degrades under `async_chunk` (streaming continuation requests skip hit marking, `gpu_model_runner.py`).
4. **Extensibility**: `block_table[0]` (single KV-cache group) is hard-coded, blocking hybrid-attention models (e.g. Qwen3.5-style full+linear attention); and slot-indexed validity silently breaks under vLLM L2 KV offload / KV-connector restores.

This RFC proposes a refactor into an `OmniTensorCacheManager` with a 4-method surface, a background commit pipeline, and naming/structure aligned with vLLM's `v1/core` KV-cache design.

Related: #1184 (original prefix caching issue), #2164, #4106, #3734, #3665, #4442/#4476 (async omni output).

## 2. Scope & Objectives

### Goals

- **G1 — Zero main-thread cache cost.** All D2H copies and CPU scatters move to a background committer thread. Target: prefix-cache-related CPU time in `execute_model` drops from ms-level per step to ≈0 (bookkeeping only); no ITL spike at request finish.
- **G2 — Keep cache data consistent.** Cache reads/writes stay consistent with vLLM's KV state throughout the save/read process; on any detected divergence between omni-cached rows and vLLM KV data, the request falls back (salted re-run) instead of risking accuracy.
- **G3 — Narrow, contract-internalizing API.** Runner touches the cache at exactly 3 call sites (`new_step_starts` / `save_outputs` / `materialize`) plus a one-time `register_policy`. All ordering contracts except one become internal invariants.
- **G4 — Feature compatibility.** `async_scheduling + async_chunk + prefix caching` all-on passes e2e; the #4476 guard in `gpu_ar_model_runner.py` is deleted; streaming (`async_chunk`) requests get correct merge semantics via span-based hit tracking.
- **G5 — Extensibility seam for multi-KV-group.** A single `KVCacheGroupView` protocol isolates all vLLM block-table access; hybrid models plug in by providing a view implementation; no full-attention group ⇒ feature cleanly self-disables.

### Non-Goals

- **vLLM L2 KV offload interop.** Deferred to a follow-up stage. This RFC only adds (a) a startup guard making L2/offload connectors and omni tensor caching mutually exclusive per stage, and (b) per-block generation tags so any invalid read fails loudly instead of silently returning stale data. The long-term direction (omni rows riding the same offload connector payload so lifecycle is shared by construction) is out of scope here.
- **Content-addressed (block-hash-keyed) storage tier.** Not built until L2/multi-group demand is concrete.
- Speculative-decode interaction (orthogonal; existing guards unchanged).

## 3. Design

### Architecture

Naming and module layout mirror vLLM `v1/core` KV-cache design:

| This RFC | vLLM reference |
|---|---|
| `OmniTensorCacheManager` (facade) | `KVCacheManager` (`v1/core/kv_cache_manager.py`) |
| `new_step_starts(scheduler_output)` | `KVCacheManager.new_step_starts()` |
| `save_outputs(...)` | connector-side `save_kv_layer` verb family |
| `TensorBlockPool` (CPU block mirror + per-block generations) | `BlockPool` (`v1/core/block_pool.py`) |
| `KVCacheGroupView` / `FullAttentionGroupView` | `KVCacheGroupSpec` / `FullAttentionSpec` + `FullAttentionManager` |
| `TensorCacheConfig` | `KVCacheConfig` |

```
vllm_omni/core/tensor_cache/
├── interface.py      # TensorCacheConfig / ModelCachePolicy / StageCacheOutputs / InflightStageOutputs
├── manager.py        # OmniTensorCacheManager (with AsyncTensorCommitter on a background thread)
├── block_pool.py     # TensorBlockPool (+ per-block generation)
└── group_view.py     # KVCacheGroupView protocol / FullAttentionGroupView / factory
```

![Class diagram](rfc-omni-tensor-cache-assets/class-diagram.svg)

**Per-step flow (before):**

![image-20260810225427318](/Users/guanxiangtian/Library/Application Support/typora-user-images/image-20260810225427318.png)

**Per-step flow (after):**

![Per-step sequence](rfc-omni-tensor-cache-assets/per-step-sequence.svg)

### Multi-request interleaving — committer writes vs builder reads on the block pool

Three requests across three steps: **A** prefills at step N, **B** arrives at step N+1 and prefix-hits A's blocks, **C** decodes one token every step. The committer (writing the mirror) and the builder (reading it for a different request) run concurrently — the four numbered guarantees below make that race-free.

![image-20260810225639762](/Users/guanxiangtian/Library/Application Support/typora-user-images/image-20260810225639762.png)

### API & Interface Changes

**New public surface** (replaces ~12 methods + 3 raw attributes on `OmniTensorPrefixCache`):

```python
class OmniTensorCacheManager:
    def register_policy(self, policy: ModelCachePolicy) -> None: ...     # once, at load_model
    def new_step_starts(self, scheduler_output: SchedulerOutput) -> None: ...
    def save_outputs(self, hidden_states, mm_outputs, *,
                     num_tokens_unpadded: int, num_tokens_padded: int) -> InflightStageOutputs: ...
    def shutdown(self) -> None: ...

class InflightStageOutputs:
    """Refcounted handle over this step's not-yet-committed outputs.

    Kept separate from StageCacheOutputs on purpose: this object owns
    resources (frozen-entry refs, retire lifecycle, cap-K accounting) while
    StageCacheOutputs is a plain value; merging the two behind a state flag
    would couple resource lifetime to a data container.
    """
    def materialize(self, req_ids: list[str]) -> StageCacheOutputs: ...  # any thread

class StageCacheOutputs(NamedTuple):
    hidden_states: dict[str, torch.Tensor] | None   # req_id → full-prompt tensor (policy-gated)
    mm_outputs: dict[str, dict[str, Any]]           # req_id → per-request payload (req-major)

@dataclass(frozen=True)
class ModelCachePolicy:                              # replaces getattr probing on models
    needs_full_hidden_states: bool = True
    merge_consumed_by_postprocess: bool = False      # forces eager materialize
    deferred_keys: frozenset[str] = frozenset()      # strip-coalesced decode mm keys
    skip_keys: frozenset[str] = frozenset()
    default_placement: Placement = Placement.CPU     # GPU assembly reserved, not implemented

class KVCacheGroupView(Protocol):                    # sole vLLM-internals access path
    block_size: int
    num_blocks: int
    def slot_mapping_gpu(self, num_tokens: int) -> torch.Tensor: ...
    def slots_for(self, req_id: str, token_start: int, token_end: int) -> torch.Tensor: ...
    def block_generations(self, slots: torch.Tensor) -> torch.Tensor: ...
```

**Runner interaction, before → after:**

```python
# ── Before: 10 call sites across 2 files, ~6 implicit ordering contracts ──
# gpu_model_runner.py::_update_states
omni_prefix_cache.reset_prefix_cached_new_req_ids()
omni_prefix_cache.discard_deferred_mm_outputs(req_id)          # per finished req
omni_prefix_cache.add_prefix_cached_new_req_id(req_id)         # hit marking; skipped for streaming
# gpu_ar_model_runner.py::execute_model
omni_prefix_cache.drain_ready_async_writes()
omni_prefix_cache.commit_deferred_mm_outputs(finished, input_batch)   # before batch removal
# ...forward...
slot_mapping_gpu = input_batch.block_table[0].slot_mapping.gpu        # .gpu workaround at call site
omni_prefix_cache.schedule_async_write(hs, mm, slot_mapping_gpu, n, n_pad, skip_keys)
# sample_tokens / output build
self._stage_deferred_prefix_cache_mm_outputs(...)                     # per-request Python loops
combined_hs = omni_prefix_cache.get_merged_hidden_states(...)         # after consume, main thread
combined_mm = omni_prefix_cache.get_merged_multimodal_states(...)
# + runner-side policy probing: _model_needs_full_prefix_hidden_states(),
#   _deferred_prefix_cache_mm_keys(), payload gating, staging special cases

# ── After: 3 call sites + 1 registration ──
cache.register_policy(ModelCachePolicy.from_model(model))      # load_model, once
cache.new_step_starts(scheduler_output)                        # top of execute_model
inflight = cache.save_outputs(hidden, mm_flat,
                              num_tokens_unpadded=n, num_tokens_padded=n_pad)
outs = inflight.materialize(req_ids)                           # main thread eager or #4476 builder
```

`vllm_omni/core/prefix_cache.py` is superseded by `vllm_omni/core/tensor_cache/`. Models currently setting `requires_full_prefix_cached_hidden_states` / `deferred_prefix_cache_mm_keys` (qwen3-tts, higgs-v3, personaplex) keep working through a `ModelCachePolicy.from_model()` shim during a deprecation window.

**Invariants (internalized contracts):**

1. *Value freeze*: `save_outputs` performs one D2D copy on the compute stream into a preallocated slab; this doubles as the #4476 CUDA-graph-reuse snapshot (today's two copies become one). Entry content is immutable from that point; only its storage stage migrates (slab → ping-pong → mirror).
2. *Single-writer mirror + FIFO + inflight barrier*: the committer thread is the mirror's only writer (write-write races impossible) and scatters strictly in order, never past an unretired `InflightStageOutputs`; readers resolve freshest-stage-first, taking a queued entry over a half-published mirror row (per-entry atomic publication after scatter), so concurrent committer writes and builder reads — including across different requests — are race-free.
3. *Refcount + finalizer + cap*: entries are refcounted; `materialize` (or GC finalizer, with a warning) retires the inflight handle; outstanding handles are capped at K (excess falls back to eager materialize) so leaks surface as visible backpressure, not memory growth.
4. *Generation check*: every mirror row is stamped with its block generation at write; reads compare against the current generation and treat mismatch as a detected miss (bug-level log), never a silent stale read.
5. *Block-alignment dependency (external)*: merge math relies on vLLM guaranteeing block-aligned `num_computed_tokens` for prefix hits (full-hit rolls back a whole block; `vllm/v1/core/kv_cache_manager.py:249`). Guarded by an assertion.
6. *The single remaining runner contract*: `new_step_starts` must run before `_update_states` removes finished requests (their block tables are still needed). Documented at the call site.
7. *Consistency fallback (backs G2)*: hit-span validation (generation/presence) runs in `new_step_starts` — before the forward — and again inside `materialize`. On mismatch: the request is poisoned, its step output dropped, and it is resubmitted with a fresh `cache_salt` (all-miss → full recompute, bypassing poisoned blocks); repeated mismatches within a window escalate to a full reset (all generations bumped + vLLM `reset_prefix_cache()`) with an alert. Every trigger is a metric + bug-level log — never silent self-healing.

### Key Technical Decisions

| ID | Decision | Why |
|---|---|---|
| **D1** | Keep D2H on the copy stream; move `event` waits + `index_copy_` to a background thread (pinned ping-pong buffer, 2 slots) | On H20, D2H already overlaps the next forward; the main-thread cost is scatter (~4–10 ms / 96 MB), not PCIe |
| **D2** | `materialize` resolves segments from slab → ping-pong → mirror (reads never wait for scatter) | Drops the consume-before-merge contract, unblocks #4476 builder-thread materialize, and closes the same-step empty-read window |
| **D3** | Fixed-size GPU strips coalesce decode mm rows; subsumes per-request `deferred_prefix_cache_mm_keys` | O(1) GPU residency, no finish-time commit spike; discard/commit lifecycle disappears |
| **D4** | Facade-owned span registry + `delivered_upto` watermark from `scheduler_output` | Fixes `async_chunk` continuation skipping hit marking; does not re-deliver spans the downstream already accumulated |
| **D5** | Keep slot-indexed mirror; exclude L2/connector via startup guard + generation tags | One consistency domain; real L2 support should ride the connector payload (out of scope) |

Rejected alternatives: deeper ring with main-thread polling; GPU-resident cache; dual-cache cross-process sync; hash-addressed tier (deferred until demand is concrete).

## 4. Correctness & Testing Plans

**Definition:** for any request set and scheduling order, `StageCacheOutputs` must be element-wise identical to the no-prefix-cache path (same dtype/values — the cache is a transport optimization, never a numerical one), and all seven §3 invariants hold.

| Level | Gate |
|---|---|
| **L1** | Unit tests for pool / committer / hit-registry; port `tests/core/test_prefix_cache.py` to the new API with unchanged expectations (Phase 0) |
| **L2** | Same-step duplicate-prompt hit (merged HS == producer's rows); eager ≡ background ≡ cache-off on merged outputs; builder-thread throw → finalizer warning, no hang; generation-mismatch → salted-retry fallback exercised |
| **L3** | Existing `test_qwen3_omni` prefix cases stay green (Phase 0); after removing the #4476 guard, `async_scheduling + async_chunk + prefix caching` all-on with output parity (Phase 2) |
| **Smoke / Perf** | Second request `cache_hit_pct > 0` and output == miss path; #2164 bench + decode-ITL P99 at finish boundaries shows no spike vs cache-off |
