# [RFC] Refactor Omni Tensor Prefix Cache: async-by-default, decoupled, and extensible

The living implementation contract is
[docs/design/feature/omni_prefix_cache.md](docs/design/feature/omni_prefix_cache.md).
This RFC is the historical design record.

## 1. Overview

vLLM-Omni's hidden-state prefix cache (`OmniTensorPrefixCache`, #2164) mirrors vLLM KV-cache block/slot mapping and keeps stage outputs (hidden states and per-token multimodal tensors) on CPU, so a prefix hit skips both KV recompute and inter-stage tensor recompute. After #4106 / #3734 / #3665 it has four structural problems:

1. **Performance**: the write path is only half-async — CPU scatter still runs on the main thread every step; deferred mm chunks grow unboundedly on GPU and spike at request finish; the read path is fully synchronous.
2. **Correctness**: vLLM caches block hashes optimistically (commits `computed + num_new_tokens` at allocation). A same-step cross-request hit can read rows that land on CPU one step later (zeros / stale) and silently forward them. Aborted requests drop deferred chunks while their blocks remain hittable. Async D2H reads the live forward output with no `record_stream` (static-buffer overwrite and allocator reuse).
3. **Coupling**: ~12 methods + raw attributes + ~6 implicit runner ordering contracts + `getattr` policy probing. This is why prefix caching is mutually exclusive with async omni output (#4476 guard) and silently degrades under `async_chunk` (continuations skip hit marking).
4. **Extensibility**: `block_table[0]` hard-codes a single KV group. Slot indexes silently break under L2 offload / connector.

This RFC splits the design into two layers. `OmniPrefixCacheManager` owns `(slot, key)` state; `OmniPrefixCacheController` only moves data. Names follow vLLM `v1/core`.

Related: #1184, #2164, #4106, #3734, #3665, #4442 / #4476.

## 2. Scope & Objectives

### Goals

- **G1** — Scatter and most D2H stay off the engine critical path. Only these sync points remain; each is small and bounded per step:

  - D2D freeze: a small device-to-device kernel at `save_outputs`. Cost scales with batch size and is usually very cheap.
  - Whole-step D2H into `StagingBufferPool` is **launched at save** (copy stream + `host_event`), not submitted as a per-task committer copy. `materialize` / committer / `fetch_host` all wait that same event (record-once, wait-many).
  - Join of the previous step's `JOIN_NEXT_STEP` tasks (`host_ready` only) at the next save. Bounded; D2H is usually already done.
  - Rare: a prefix hit on **deferred** rows that are still GPU-staged needs a sync D2H. Almost never on the hot path.
  - Cap flush of the oldest deferred task when the GPU-byte budget is full. Degraded path; rare.

- **G2** — Omni PrefixCache stays consistent with vLLM KV. Only fatal errors are errors; miss is not an error:

  - **Fatal**: dump and exit. Bookkeeping on `(slot, key)` is wrong, `task_id` does not match, or a hit span is absent. On a prefix hit vLLM has already skipped those tokens; this step will not recompute them, so missing rows cannot be filled in. Do not silently degrade. Abort is the same rule: once a hash is in this step's batch, it must be written.
  - **Ordinary miss (not an error)**: the request is in this step's snapshot but has no hit span (new prompt / no hit). `materialize` uses only this step's forward slice. **No warning, no degradation counter.** This is the main path.
  - **`req_id` outside the snapshot**: `materialize` received an id that is not in this `step_id` snapshot — the runner passed the live batch. P0 forbids this (the async builder must pass the req list copied at save). Debug-assert only; no hot-path warning.
- **G3** — Narrow public API.
- **G4** — `async_scheduling + async_chunk + prefix caching` all on (**P2**).
- **G5** — `KVCacheGroupView` isolates block-table access so multi-KV-group can plug in later (**P2**).

### Non-Goals

- vLLM L2 KV offload interop (startup mutex only; long-term it should ride the connector payload).
- Content-addressed storage; speculative-decode interaction (orthogonal).

## 3. Design

### Architecture

| This RFC | vLLM counterpart |
|---|---|
| `OmniPrefixCacheManager` (facade, block/slot domain) | `KVCacheManager` |
| `OmniPrefixCacheController` (staging pool + committer) | none (closest: connector worker side) |
| `PrefixBlockPool` (pinned CPU block mirror) | `BlockPool` |
| `StagingBufferPool` (reusable step-sized D2H pages) | none |
| `KVCacheGroupView` / `FullAttentionGroupView` | `KVCacheGroupSpec` / `FullAttentionSpec` |
| `PrefixCacheConfig` | `KVCacheConfig` |

```
vllm_omni/core/prefix_cache/
├── interface.py      # PrefixCacheConfig / ModelCachePolicy / StageCacheOutputs
├── manager.py        # OmniPrefixCacheManager (slot state, span/hit, merge)
├── controller.py     # OmniPrefixCacheController + StagingBufferPool
├── block_pool.py     # PrefixBlockPool
└── group_view.py     # KVCacheGroupView / FullAttentionGroupView / factory
```

**Class relationships** (who owns whom, who is called on which path):

![Omni prefix cache class diagram](rfc-omni-tensor-cache-assets/class-diagram.svg)

Manager owns `(slot, key)` state; Controller only moves data and reports completion. `KVCacheGroupView` is the only way to touch the vLLM block table, and only from `new_step_starts` / `save_outputs`. A `WriteTask` is one write, identified by `task_id`. A `(slot, key)` stores the current `task_id`; it is not the slot itself.

**One-step sequence** (`execute_model` → `sample`; two requests in the same batch):

- **P**: already decoding. Previous JOIN_NEXT_STEP `t1` is still moving in the background; `deferred_keys` such as codes sit on JOIN_ON_FINISH until P finishes.
- **Q**: new this step, prefix-hit P's already-committed first 32 tokens. This-step forward computes only Q's suffix.

![Per-step sequence with requests P and Q](rfc-omni-tensor-cache-assets/per-step-sequence.svg)

Four public call sites: ① `new_step_starts` (register Q's hit, start prefix prefetch, drain completed tasks); ② `save_outputs` (join P's previous JOIN_NEXT_STEP off the lock, D2D-freeze, launch one whole-step D2H into the staging pool, submit one JOIN_NEXT_STEP task per request whose `seg.host` is a view of that page, return `sid=N`); ③ `materialize(N, [P,Q])` waits `host_event` and clones the views; ④ `discard_step(N)` when nobody will read this step (public API; Manager decides read vs discard from policy / whether this step has a hit — the runner does not expand that table). P is a miss and gets this-step slice only. Q's `[0,32)` is grouped by the **`tid` recorded at save**: committed rows come from the mirror; still-writing rows come from `fetch_host` (which waits that task's `host_event` on the staging path).

**What `key` is.** One slot usually has two independent tensors:
- `hidden_states`: written incrementally every step
- names in `ModelCachePolicy.deferred_keys` (e.g. codes): staged until the request finishes

`skip_keys` never enter the state machine. State must be keyed by `(slot, key)`, or hidden stays in-transit for the whole decode because of deferred mm, or unread mm is treated as already on disk.

**(slot, key) and WriteTask**

Each `(slot, key)` stores one `task_id`: who is writing it now.

`task_id = (req_id, key, seq)`
- `req_id`, `key`: which request, which tensor stream
- `seq`: how many times this pair has opened a write (JOIN_NEXT_STEP +1 every decode step; deferred usually stays; remount +1)

`task_id is None`: nobody is writing. Hidden and deferred have separate tables; the two keys on the same slot do not share state. `_req_tasks[req_id]` lists every task of a request.

A completion event whose `task_id` is no longer current on that `(slot, key)` is dropped.

After a write is registered, `(slot, key)` has two states:
- **in-transit**: write in progress, not on the mirror yet; readable from the GPU snapshot
- **committed**: already on the host mirror

**absent** means never registered. It is not a state you go back to after a write. Seeing it inside a hit span is an implementation bug: dump and exit.

**Reassigning a block = remount the task.** This happens only when P has not finished writing and the block is given to Q. Q's `save` remounts those `(slot, key)`s to Q's `task_id` and skips P's leftover writes on them. Slots Q did not take still belong to P and wait for finish / abort / cap; a later hit on them reads in-transit and does not wait for scatter. There is no separate “P was preempted” signal. Normal finish-then-reuse never reaches this path: the same `save` joins P first, then submits Q.

Reads group by the **`task_id` recorded at save**, not by “the whole request is in-transit”. One hit often spans several tasks: the old prefix is committed, this step's new rows are still in-transit. Treating “any in-transit ⇒ read the whole request from the snapshot” would touch already-freed staging or D2H committed rows again.

A `WriteTask` holds:
- **D2D freeze copy** (on the compute stream at save, with a freeze event) — the forward output may be a CUDA-graph static buffer, and today's D2H has no `record_stream`. The freeze closes both races and doubles as the #4476 snapshot.
- **Slot list fixed at save** — commit no longer looks at `input_batch`, so the current abort-drop-chunk bug disappears by construction.
- D2H event and completion report.

### Task synchronization

- **Sync (one step behind)**: `save_outputs` (after forward) is consume-then-schedule — join the previous step's JOIN_NEXT_STEP tasks (`host_ready` only), then launch this step's staging D2H and submit. Join is only a **bounded wait** that caps in-flight depth. Staging-page holders (`for_step` / `for_task` / `for_reader`) keep the reusable slot alive; the page is free only when none remain.
- **Write schedules** (key split, not token count): **JOIN_NEXT_STEP** — immediately-cached keys (hidden + non-deferred mm). D2H is already in flight at submit; the committer waits `host_event` then H2H-scatters into `PrefixBlockPool`. **JOIN_ON_FINISH** — deferred mm. Stays on the GPU freeze; the committer does that D2H, then scatters. Finish/abort or cap pressure escalates it.
- **Cap**: GPU staging has a byte budget. Over budget, block and flush the oldest task (its D2H is usually almost done).
- **Abort still writes; never roll back** — aborted full-block hashes remain hittable; rolling back would make a legal hit look unmatched. Rule: **once a hash is in this step's batch it must be saved, abort included**. When `new_step_starts` sees finished/abort, every already-registered task of that request is escalated and joined at the next save. There is **no “allocated then aborted before first save” exemption**: absent means a missed write or a missed registration — fail-fast.
- **Preemption reuse**: see “Reassigning a block = remount the task” above. Finish-then-reuse never goes there.

### Read path: materialize groups by save-time `task_id` of each `(slot, key)`

A hit span is grouped by the **`task_id` recorded at save** (one hit often spans tasks: old prefix committed, this-step rows in-transit). Each group is read from its current tier, then stitched back into caller order:

- **committed** → read the host mirror (`PrefixBlockPool`). Same bytes for every reader.
- **in-transit (JOIN_NEXT_STEP)** → `seg.host` is a staging view hung at save. Wait that step's `host_event`, then slice. No second D2H, no `publish_host`.
- **in-transit (JOIN_ON_FINISH)** → wait the freeze, then sync D2H from the GPU snapshot unless the committer has already written owned host tensors.
- **absent** → never registered. Inside a hit span this is bookkeeping corruption: dump and exit.

**`materialize` uses only the snapshot hung at `save` for this `step_id`. It does not touch live `input_batch`.** When the async builder runs one step late, finished requests are already removed and a preempted block table is already rewritten, so:

- **Inside the cache**: slots, hit blocks, and the save-time `task_id`s are stored on the step context at `new_step_starts` or `save_outputs`. `KVCacheGroupView` is used only at those two points, because it depends on this step's input.
- **Runner `materialize` args / return**: `req_ids` must be a subset of that `sid` snapshot. **The return is already stitched per request; the runner does not stitch again.** Snapshot hit → `cache[0, hit) + this-step slice` (when policy wants full hidden); snapshot miss → **this-step slice only** (not empty, not a full prompt assembled from cache); id outside the snapshot → debug assert; missing row in a hit span → dump and exit. `StageCacheOutputs.hidden_states is None` means the policy does not want hidden, not a miss.

This step's rows for next-stage handoff come from the staging clone after `materialize` waits `host_event` — not from a per-task `publish_host`. `d2h_claimed` is only the committer's single-claimer for the copy stage (wait the event, or do the deferred D2H). Eager mode copies into the staging page inline; async CUDA records `host_event` on the copy stream.

### API & Interface Changes

```python
class OmniPrefixCacheManager:
    def register_policy(self, policy: ModelCachePolicy) -> None: ...     # once, at load_model
    def new_step_starts(self, scheduler_output: SchedulerOutput) -> None: ...
    def save_outputs(self, hidden_states, mm_outputs, *,
                     num_tokens_unpadded: int, num_tokens_padded: int) -> int: ...
                                # returns step_id; the hung context is consumed exactly once
                                # by materialize or discard_step
    def materialize(self, step_id: int, req_ids: list[str]) -> StageCacheOutputs: ...  # any thread
    def discard_step(self, step_id: int) -> None: ...  # drop the snapshot if nobody will read;
                                                       # writes are unaffected. Manager decides
                                                       # from policy / this-step hit.
    def shutdown(self) -> None: ...

class StageCacheOutputs(NamedTuple):
    hidden_states: dict[str, torch.Tensor] | None   # req → already stitched: hit = cache+slice,
                                                    # miss = this-step only; None = policy skips hidden
    mm_outputs: dict[str, dict[str, Any]]           # likewise stitched per req;
                                                    # missing hit-span row exits the process

@dataclass(frozen=True)
class ModelCachePolicy:                              # replaces getattr probing
    needs_full_hidden_states: bool = True
    merge_consumed_by_postprocess: bool = False      # forces eager materialize
    deferred_keys: frozenset[str] = frozenset()      # JOIN_ON_FINISH deferred: GPU staged until finish
    skip_keys: frozenset[str] = frozenset()

class KVCacheGroupView(Protocol):                    # sole path into vLLM internals
    block_size: int
    num_blocks: int
    def step_slots_cpu(self, req_ids: list[str], num_scheduled: dict[str, int]) -> torch.Tensor: ...
    def slots_for(self, req_id: str, token_start: int, token_end: int) -> torch.Tensor: ...
    def cached_block_ids(self, req_id: str) -> torch.Tensor: ...
```

**Runner interaction (after: 4 call sites + 1 registration):**

```python
cache.register_policy(ModelCachePolicy.from_model(model))   # load_model

cache.new_step_starts(scheduler_output)   # very start of execute_model, before _update_states
#   registers hit spans (a span is a token range that prefix-hit) and handles
#   finish / abort lifecycle (escalate already-registered tasks)
# ...forward...
sid = cache.save_outputs(hidden, mm_flat, num_tokens_unpadded=n, num_tokens_padded=n_pad)
#   lock must not cover join / D2H / cap flush:
#     lock → list tids to join → unlock
#     join_host_ready(previous JOIN_NEXT_STEP + finished JOIN_ON_FINISH already escalated)
#     D2D freeze + stage_step_host (one whole-step D2H)          # off lock
#     lock → drain → submit JOIN_NEXT_STEP (seg.host = views) + stage deferred → hang ctx → unlock
#     return sid

outs = cache.materialize(sid, req_ids)    # sample_tokens or output build; main or builder thread
#   resolves this step by (slot, key); async_output / async_chunk share this API
#   missing row in a hit span → dump and exit. Nobody reading → discard_step(sid) (writes continue).
#   Manager decides read vs discard from policy / whether this step has a hit.
#   sid must reach the builder; step N+1 must not drop step N's still-waiting context.
```

The package is `vllm_omni/core/prefix_cache/` (it replaces the old single-file `prefix_cache.py`). Old models keep working through `ModelCachePolicy.from_model()` with no model-side change. On NPU, runner-side cache work moves into the Controller and runs eager.

**Invariants:**

1. *Value freeze*: one D2D freeze at save (freeze event; also the #4476 snapshot). Task contents are immutable; only the storage tier moves.
2. *Task atomicity + single writer + task_id check*: one task covers a set of `(slot, key)`s; it is **not** uniquely identified by one pair. The mirror has one committer at a time. After a new request takes a `(slot, key)`, the old task skips those pairs and they remount to the new `task_id`. Only an old task that is **not skipped and still overlaps the new write** must be waited on. Completions apply only if they match the **current `task_id`** on that `(slot, key)`.
3. *State-resolved read*: group by **save-time `tid`**; committed → mirror; in-transit JOIN_NEXT_STEP → wait `host_event` and slice the staging view; in-transit JOIN_ON_FINISH → freeze + committer/sync D2H; absent inside a hit span → dump and exit.
4. *Cap backpressure*: GPU staging byte budget; over budget, block and flush the oldest task. **Degraded path** — needs a budget and a metric; not counted as steady-state cost.
5. *Thread and snapshot contract*: one non-reentrant lock protects only the state tables. Both `save_outputs` and `materialize` are **lock → plan → unlock → join / D2H / cap flush → (if needed) lock again to submit or drain**, as in the snippet above. Do not `@locked` the whole `save_outputs`. Plans hold task refs only. `materialize` uses the step snapshot and never touches `KVCacheGroupView` / live `input_batch`.
6. *Runner and ordering contract*: `new_step_starts` must run before `_update_states` drops finished requests; `scheduler_output` is applied once, in execution order (warmup / dummy ignored). In the same `execute_model`, `save_outputs` must run before `materialize`. The async builder reads only the snapshot left at this step's save (looked up by `step_id`); `materialize` req_ids are a subset of that snapshot. Each step context is consumed exactly once by `materialize` or `discard_step`. Ids outside the snapshot are debug-asserted, not warned into a degraded path.

### Key Technical Decisions

| ID | Decision | Why |
|---|---|---|
| **D1** | At save, consume what is already done and schedule the rest in the background; no refcount / release ceremony | State resolution keeps correctness; resource management stays simple |
| **D2** | `materialize` groups `(slot, key)` by save-time `task_id` | A hit often spans task states; each group reads only that task's data |
| **D3** | WriteTask fixes the slots to materialize at save; finished requests go first; abort still finishes the write | Commit does not need `input_batch`; release is early and deterministic; no delayed free or finish spike |
| **D4** | Span registry + `delivered_upto` watermark (**P2**) | Fixes `async_chunk` continuation miss-mark / double-send. P0 skips this; continuations do not mark |
| **D5** | JOIN_NEXT_STEP vs JOIN_ON_FINISH; GPU-byte cap on deferred freeze | Immediate keys merge every step via staging; deferred trickles until finish / cap |
| **D6** | One whole-step D2H into `StagingBufferPool` at save; `seg.host` is a view | Committer does not D2H again on JOIN_NEXT_STEP; materialize / fetch_host / scatter share the page |

## 4. Correctness & Testing Plans

**Definition:** for any request set and scheduling order, `StageCacheOutputs` is element-wise identical to the no-prefix-cache path.

| Level | Gate |
|---|---|
| **L1 basics** | Unit: pool / controller state machine, cap, abort completion, hit registration. Port `tests/core/test_prefix_cache.py` to the new API with the same expected outputs. |
| **L2 details** | 1. A hit span that crosses task states (committed + in-transit) groups and reads correctly. 2. A same-step re-hit reads in-transit. 3. Different keys on the same slot resolve independently. 4. Eager, background write, and cache-off match. 5. Aborted hashed blocks remain valid hits. 6. After reuse, hits and the final mirror show the new request, regardless of who finishes first. 7. Injected illegal state fail-fasts. 8. **Missing row in a hit span dumps and exits (no silence, no degrade, no error return).** 9. Step context is consumed once by materialize / discard_step; it is not dropped while the builder is still waiting. 10. Cap backpressure numbers stay stable at the limit. |
| **L3 e2e (P2)** | Existing `test_qwen3_omni` prefix-cache cases: `async_scheduling + async_chunk + prefix caching` all match the no-cache output. |
| **Smoke / perf** | Repeated hits (`hit > 0`) match the all-miss path. No performance regression. |

## 5. Phasing

- **P0**: this RFC's main path — `(slot, key)` state machine, save-time `task_id` reads, unlock-then-join, abort still writes, hit-span absent fail-fast. Keep the #4476 guard. Continuations do not mark (no D4).
- **P1**: background scatter / D2H lands; G1's sync points are measurable.
- **P2**: G4 all-on (D4 `delivered_upto`), G5 multi-KV-group. L3 is accepted here.
