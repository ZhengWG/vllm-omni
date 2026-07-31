# Stage Transfer Refactor Plan

> Status: **Proposal** — derived from a deep review of inter-stage cache /
> payload / KV transfer paths. Incremental consolidation; **not** a rewrite of
> the Orchestrator / Connector / stage-processor layering.
>
> 中文版：[stage_transfer_refactor.zh.md](stage_transfer_refactor.zh.md)

## Table of Contents

1. [Goals and Non-Goals](#goals-and-non-goals)
2. [Current Architecture](#current-architecture)
3. [Problems Motivating Change](#problems-motivating-change)
4. [Target Architecture](#target-architecture)
5. [Phased Delivery](#phased-delivery)
6. [Stage Edge Contract](#stage-edge-contract)
7. [Compatibility and Migration](#compatibility-and-migration)
8. [Testing Strategy](#testing-strategy)
9. [Success Criteria](#success-criteria)
10. [Related Files](#related-files)

---

## Goals and Non-Goals

### Goals

1. **Correctness first**: eliminate known races, silent data loss, and
   process-killing error paths on the stage-transfer hot path.
2. **Lifecycle closure**: every `put` / prefetch / deferred buffer has a
   matching cleanup on finish, abort, and timeout.
3. **Explicit edge contract**: each stage edge declares *what* is transferred,
   *how* (sync / stream / KV), and *who* owns readiness — no role defaults that
   hang requests.
4. **Data-plane consolidation**: fold payload + chunk + KV ownership behind one
   Stage Transfer façade without changing Orchestrator topology.
5. **Keep model glue thin**: shared payload schema + helpers; model-specific
   `stage_input_processors` only encode true model differences.

### Non-Goals

- Cross-model reuse of KV / hidden / prefix-cache **contents** (architecturally
  impossible without identical weights + layouts).
- Unifying Diffusion step caches (Cache-DiT / TeaCache / …) with AR stage
  transfer — they remain per-pipeline accelerators.
- Big-bang rewrite of Orchestrator, StagePool, or Connector `put`/`get` API.
- Landing D2D transport in Phase 0–2 (tracked as Phase 3 capability work).

---

## Current Architecture

```text
                    ┌──────────────────────────────────────┐
                    │           Orchestrator               │
                    │  route / kv_ready / abort / PD / CFG │
                    └───────────────┬──────────────────────┘
                                    │ control plane
          ┌─────────────────────────┼─────────────────────────┐
          ▼                         ▼                         ▼
   Stage-0 Runner            Stage-1 Runner            Stage-N Runner
          │                         │                         │
          │  OmniConnectorModelRunnerMixin (~2.3k LOC)        │
          │    ├─ full_payload / async_chunk (+ bg threads)   │
          │    └─ delegates KV → OmniKVTransferManager (~1.9k)│
          │                         │                         │
          └──────────── OmniConnector put/get (D2H2D) ────────┘

  Side caches (stage-local):
    OmniTensorPrefixCache (AR, CPU, block/slot mirrored)
    Diffusion PromptEmbedCache / Cache-DiT / … (not stage-transfer)
```

**What stays:** control vs data plane split; Connector `put`/`get`; per-model
`stage_input_processors`; Orchestrator as request router.

**What changes:** ownership and contracts of the data plane; failure and abort
semantics; explicit edge configuration.

---

## Problems Motivating Change

Findings from code review (severity abbreviated). Full detail lives in the
review discussion that produced this plan.

| Sev | Area | Issue |
| --- | --- | --- |
| C | KV prefetch | H2D on `_bg_copy_stream` without mainstream `wait` → GPU race |
| C | Prefix cache | Async D2H of reused `slot_mapping.gpu` races with next `_update_states` |
| C | async_chunk | `_accumulate_payload` mutates dict outside `_lock` → silent chunk drop |
| C | Orchestrator | `process_engine_inputs` exception re-raised → kills whole loop |
| H | Connector | `cleanup(request_id)` never called from KV manager on finish/abort/timeout |
| H | CFG + KV | Prefetch-miss sync fallback drops `cfg_kv_collect_func`; kv_ready marks companions done before outputs exist |
| H | Prefix | Multi-kv-group only warns, still merges via `block_table[0]` |
| H | Abort | In-flight prefetch not cancelled; payload can be consumed then lost |
| M | Config | `stage_receives_chunks` defaults True when `role` unset → hung request |
| M | Debt | Mixin + KV manager overlap; legacy flat payload keys on hot path |

---

## Target Architecture

```text
                    ┌──────────────────────────────────────┐
                    │           Orchestrator               │
                    │  route + StageEdgePolicy only        │
                    │  (no transport / no cache internals) │
                    └───────────────┬──────────────────────┘
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          ▼                         ▼                         ▼
                 StageTransferFacade (per ModelRunner)
                    │
        ┌───────────┼────────────────┬────────────────┐
        ▼           ▼                ▼                ▼
   PayloadXfer  ChunkXfer        KvXfer           PrefixCache
   (full)       (async_chunk)    (wraps today's   (AR-only;
                                  KV manager)      optional)
        │           │                │
        └───────────┴───────┬────────┘
                            ▼
                   OmniConnector put/get
                   + RequestLifecycle hooks
                     (finish / abort / timeout → cleanup)
```

### Design rules

1. **One façade per runner**: ModelRunner talks only to `StageTransferFacade`
   for send/recv/cleanup/readiness (`OmniConnectorOutput` unchanged).
2. **KV manager is absorbed, not deleted on day one**: Phase 2 moves call sites
   behind the façade; internal modules may remain until file split is cheap.
3. **Orchestrator never touches tensors**: it only applies `StageEdgePolicy`
   (forward now vs wait for connector vs PD vs CFG barrier).
4. **Fail loud, isolate blast radius**: transfer/bridge failures mark the
   *request* failed; they must not tear down the orchestration loop.
5. **Diffusion caches stay out**: no attempt to share Cache-DiT / PromptEmbed
   across stages in this plan.

---

## Phased Delivery

Each phase is independently mergeable. Later phases must not depend on
unmerged earlier phases beyond listed prerequisites.

### Phase 0 — Correctness hotfix (no API change)

**Objective:** stop silent corruption and process death. Zero public API churn.

| Item | Change | Primary files |
| --- | --- | --- |
| P0-1 | Prefetch H2D: after bg copy, `Event.record` + consumer `wait_event` (or stream sync before apply) | `kv_transfer_manager.py` |
| P0-2 | Prefix async write: clone `slot_mapping[:n]` (or wait copy stream before next slot update) | `prefix_cache.py`, `gpu_ar_model_runner.py` |
| P0-3 | Hold `_lock` across `_accumulate_payload` + cache update in `_poll_single_request` | `omni_connector_model_runner_mixin.py` |
| P0-4 | Catch `process_engine_inputs` / `_forward_to_next_stage` errors per-request; emit error output + abort cleanup | `orchestrator.py` |
| P0-5 | Multi-kv-group: disable Omni prefix merge (treat as miss) instead of using group-0 only | `prefix_cache.py` |

**Exit criteria:** unit/integration tests for each item; Qwen3-Omni async_chunk
smoke + one KV-transfer model smoke green.

**Risk:** low. Behavior-preserving except P0-5 (safer degradation).

---

### Phase 1 — Lifecycle and failure semantics

**Objective:** close resource and readiness gaps; make misconfig fail fast.

| Item | Change |
| --- | --- |
| P1-1 | Introduce `StageTransferFacade.cleanup_request(req_id, reason=)` called from runner finished/abort paths; delegates to connector `cleanup`, prefetch cancel, payload/chunk state drop |
| P1-2 | `OmniKVTransferManager.cancel_prefetch(req_id)` + Orchestrator/runner abort wiring |
| P1-3 | Prefetch-miss sync path must pass through `cfg_kv_collect_func` |
| P1-4 | CFG: only mark companion complete on **finished** outputs; kv_ready must not call `on_companion_completed` |
| P1-5 | PD edge vs omni `kv_ready` edge: mutually exclusive policy — PD decode submit only from prefill **finish** path; `_handle_kv_ready_raw_outputs` skips `_pd_pair` edges |
| P1-6 | `stage_receives_chunks`: if connector role unset → `False` + config warning/error at startup (fail-fast for async_chunk edges missing role) |
| P1-7 | KV / CFG scatter: bounded timeout or poison-pill that unblocks followers; never infinite `recv` after failed send |

**Exit criteria:** abort under load leaves no SHM/RDMA growth in soak; CFG+KV
and PD configs covered by regression tests; miswired async_chunk fails at
startup.

**Risk:** medium (abort paths touch many call sites). Ship behind existing
tests + a new soak script for connector cleanup.

---

### Phase 2 — Data-plane consolidation + edge contract

**Objective:** make the architecture match the diagram; shrink mixin surface.

| Item | Change |
| --- | --- |
| P2-1 | Add `StageTransferFacade` with methods: `register_recv`, `poll_readiness`, `send_payload`, `send_chunk`, `send_kv`, `recv_*`, `cleanup_request` |
| P2-2 | Move mixin transport bodies into `vllm_omni/distributed/omni_connectors/stage_transfer/` modules (`payload.py`, `chunk.py`, `kv.py`, `facade.py`); mixin becomes thin delegator or disappears |
| P2-3 | Formalize `StageEdgeSpec` in stage/deploy config (see [Stage Edge Contract](#stage-edge-contract)); Orchestrator reads policy only |
| P2-4 | Shared `OmniPayload` nested schema enforced at send boundary; legacy flat keys → hard error after one release with warning |
| P2-5 | Extract common TTS/Omni handoff helpers from large `stage_input_processors/*` (concat/replace keys, codec span, language/speaker meta); models keep only model-specific transforms |

**Exit criteria:** mixin file &lt; ~400 LOC (or removed); all models use
`StageEdgeSpec`; no production path depends on legacy flat keys.

**Risk:** medium-high (touch every multi-stage model). Mitigate by keeping
Connector API stable and migrating one pipeline family per PR
(Qwen3-Omni → Qwen3-TTS → others).

---

### Phase 3 — Capability upgrades (optional / parallelizable)

Only start after Phase 1 exit; can overlap late Phase 2.

| Item | Change |
| --- | --- |
| P3-1 | D2D connector path (NCCL / UCX / IPC) for large KV/payload — roadmap item already noted in `disaggregated_inference.md` |
| P3-2 | Deeper async prefix-write pipeline (ring of pending writes) if profiling shows D2H bottleneck |
| P3-3 | Optional shared PromptEmbedCache across diffusion replicas (process-external store) — **only** if product needs it |
| P3-4 | Metrics: per-edge transfer latency, cleanup lag, prefetch hit/miss, race-detector counters in debug builds |

**Exit criteria:** each item has its own design addendum + bench numbers.

---

## Stage Edge Contract

Introduce a declarative edge spec (YAML + dataclass). Example:

```yaml
stage_args:
  - stage_id: 0
    edges:
      - to: 1
        mode: async_chunk          # full_payload | async_chunk | kv | control_only
        connector: shm_default
        role: sender               # required when mode uses connector
        payload_schema: omni_v1    # nested OmniPayload only
        on_failure: fail_request   # never fail_engine

  - stage_id: 1
    edges:
      - from: 0
        mode: async_chunk
        connector: shm_default
        role: receiver
      - to: 2
        mode: full_payload
        connector: shm_default
        role: sender
```

### Policy matrix (Orchestrator)

| Edge mode | Who feeds next stage | Forward trigger |
| --- | --- | --- |
| `control_only` | Orchestrator prompts / tokens | upstream finished (or streaming segment) |
| `full_payload` | Connector + Orchestrator submit | upstream finished |
| `async_chunk` | Connector chunks | prewarm + chunk readiness (`OmniConnectorOutput`) |
| `kv` | KV transfer + Orchestrator submit | `kv_ready` **or** finished (config chooses one; PD forces finished) |

Invalid combinations (e.g. PD pair + `kv_ready` forward, async_chunk without
roles) fail at config load.

---

## Compatibility and Migration

1. **Phase 0–1**: no deploy YAML changes required.
2. **Phase 2**: accept both legacy `input_connectors` /
   `output_connectors` + `async_chunk` flags **and** new `edges:` block;
   emit deprecation warnings; auto-derive `StageEdgeSpec` from legacy fields.
3. **One release later**: require `edges:` (or keep auto-derive indefinitely if
   cost is low — prefer derive + warn for one minor, then error in the next).
4. Model processors: no flag day; migrate helpers incrementally per model PR.

---

## Testing Strategy

| Layer | Coverage |
| --- | --- |
| Unit | Stream sync helpers; prefix slot clone; locked accumulate; edge-spec validation; CFG companion state machine; PD vs kv_ready policy |
| Integration | Abort during prefetch; connector cleanup after timeout; multi-kv-group prefix disabled; CFG scatter failure unblocks follower |
| E2E smoke | Qwen3-Omni async_chunk; one AR→Diffusion KV path; one PD pair |
| Soak | 30–60 min multi-request abort storm; assert `/dev/shm` + connector pool metrics do not monotonically grow |
| Perf gate | Phase 0/1 must not regress TTFP/E2E beyond noise on existing Qwen3-Omni async_chunk bench |

---

## Success Criteria

1. All Critical / High review items closed or explicitly waived with test proof.
2. Abort + timeout paths call connector cleanup; soak shows stable resource use.
3. Orchestrator bridge failures isolate to the request.
4. Stage transfer entrypoints reachable through one façade; mixin no longer owns
   transport policy.
5. New multi-stage model can wire an edge with `StageEdgeSpec` + one processor
   module without editing KV manager or mixin internals.
6. No attempt to share cache **contents** across different models; docs state
   this boundary clearly (framework reuse only).

---

## Related Files

| Area | Path |
| --- | --- |
| Orchestrator | `vllm_omni/engine/orchestrator.py` |
| Data-plane mixin | `vllm_omni/worker/omni_connector_model_runner_mixin.py` |
| KV transfer | `vllm_omni/distributed/omni_connectors/kv_transfer_manager.py` |
| Prefix cache | `vllm_omni/core/prefix_cache.py` |
| AR runner integration | `vllm_omni/worker/gpu_ar_model_runner.py` |
| Connector base | `vllm_omni/distributed/omni_connectors/connectors/base.py` |
| Edge/role helpers | `vllm_omni/distributed/omni_connectors/utils/config.py` |
| Stage processors | `vllm_omni/model_executor/stage_input_processors/` |
| Existing design | `docs/design/feature/disaggregated_inference.md`, `prefix_caching.md`, `async_chunk.md` |

---

## Suggested PR Sequence

1. `fix(stage-transfer): P0 correctness — prefetch sync, prefix slot, accumulate lock, orchestrator isolation`
2. `fix(stage-transfer): P0 multi-kv-group prefix disable`
3. `fix(stage-transfer): P1 cleanup/abort/prefetch cancel + connector.cleanup`
4. `fix(stage-transfer): P1 CFG/PD/kv_ready policy hardening`
5. `refactor(stage-transfer): P2 StageTransferFacade + module split`
6. `refactor(stage-transfer): P2 StageEdgeSpec + legacy derive`
7. Follow-ups: D2D / metrics / processor helper extraction as separate tracks
