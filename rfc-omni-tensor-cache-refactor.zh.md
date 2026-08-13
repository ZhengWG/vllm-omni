# [RFC] 重构 Omni Tensor Prefix Cache：默认异步、解耦、可扩展

> 本文为 [`rfc-omni-tensor-cache-refactor.md`](./rfc-omni-tensor-cache-refactor.md) 的中文译本。API / 类型名 / 路径保持英文原文，便于对照代码。

## 1. Overview

vLLM-Omni 的 hidden-state prefix cache（`OmniTensorPrefixCache`，引入于 #2164）镜像 vLLM KV-cache 的 block/slot 映射，把 stage 输出（hidden states 与 per-token 多模态张量）缓存在 CPU 上，使 prefix-cache hit 同时跳过 KV 重算**与**跨 stage 张量重算。

自首版合入后，实现又有机演进（#4106 异步写、#3734 staging 去重、#3665 正确性修复、deferred mm-chunk commit），目前存在四个结构性问题：

1. **性能**：写路径只做到半异步。CPU scatter（`index_copy_`）仍在 runner 主线程每步执行；decode 侧 deferred mm chunks 在 GPU 上无界增长，并在请求结束时造成同步 commit 尖峰；读/merge 路径完全同步，且带阻塞式 `.cpu()` 回退。
2. **正确性**：vLLM 在分配时**乐观地**缓存 block hash（`kv_cache_manager.py`：提交 `computed + num_new_tokens`），因此同一步中稍后调度的请求可能 hit 到本步才正在计算的 block。自 #4106 起，omni 写入要晚一步才落到 CPU mirror，同一步跨请求 hit 会 merge 到从未写入的行（全 0 / 陈旧数据）并无声转发给下游——并发到达的重复 prompt 可复现。#4106 之前的同步写没有这个窗口。此外，按 request 键控的 deferred mm chunks 在 abort 请求已不在 input batch 时会被丢弃（`prefix_cache.py`），但其 hashed blocks 仍可从 free queue 被 hit，导致从未写入的 mm 行仍可达。
3. **耦合**：cache 暴露约 12 个方法外加裸属性访问；runner 承载约 6 条*隐式顺序契约*（drain-before-merge、commit-before-batch-removal、consume-before-merge 等），并用 `getattr` 探测模型策略。正因如此，prefix caching 目前与 async omni output **互斥**（#4476 以安全 guard 硬关闭，`gpu_ar_model_runner.py`），并在 `async_chunk` 下静默降级（streaming continuation 跳过 hit marking，`gpu_model_runner.py`）。
4. **可扩展性**：`block_table[0]`（单一 KV-cache group）被写死，阻塞混合注意力模型（如 Qwen3.5 式 full+linear attention）；在 vLLM L2 KV offload / KV-connector restore 下，按 slot 索引的有效性会静默失效。

本 RFC 提议重构为 `OmniTensorCacheManager`：4 方法对外表面、后台 commit 流水线，命名与结构对齐 vLLM 的 `v1/core` KV-cache 设计。

相关：#1184（原始 prefix caching issue）、#2164、#4106、#3734、#3665、#4442/#4476（async omni output）。

## 2. Scope & Objectives

### Goals

- **G1 — 主线程零 cache 开销。** 所有 D2H 与 CPU scatter 迁到后台 committer 线程。目标：`execute_model` 中与 prefix-cache 相关的 CPU 时间从每步毫秒级降到 ≈0（仅 bookkeeping）；请求结束时无 ITL 尖峰。
- **G2 — 保持 cache 数据一致。** 在 save/read 全过程中，cache 读写与 vLLM 的 KV 状态保持一致；一旦发现 omni 缓存行与 vLLM KV 数据分叉，请求走 fallback（加 salt 重跑），而不是冒险损失精度。
- **G3 — 窄 API，契约内化。** Runner 常态只碰 3 个调用点（`new_step_starts` / `save_outputs` / `materialize`），外加一次性 `register_policy`。除一条外，所有顺序契约都变为内部不变量。
- **G4 — 特性兼容。** `async_scheduling + async_chunk + prefix caching` 全开可通过 e2e；删除 `gpu_ar_model_runner.py` 中 #4476 guard；streaming（`async_chunk`）请求通过 span-based hit tracking 获得正确 merge 语义。
- **G5 — 多 KV-group 扩展缝。** 用单一 `KVCacheGroupView` 协议隔离全部 vLLM block-table 访问；混合模型通过提供 view 实现接入；无 full-attention group ⇒ 特性干净自关闭。

### Non-Goals

- **vLLM L2 KV offload 互操作。** 留给后续阶段。本 RFC 仅增加：(a) 启动期 guard，使 L2/offload connector 与 omni tensor caching 在每个 stage 互斥；(b) per-block generation tags，使任何无效读大声失败，而非静默返回陈旧数据。长期方向（omni 行搭载同一 offload connector payload，生命周期天然共享）不在本次范围。
- **Content-addressed（以 block-hash 为键）存储层。** 等到 L2/multi-group 需求具体后再做。
- Speculative-decode 交互（正交；现有 guards 不变）。

## 3. Design

### Architecture

命名与模块布局镜像 vLLM `v1/core` KV-cache 设计：

| 本 RFC | vLLM 对照 |
|---|---|
| `OmniTensorCacheManager`（facade） | `KVCacheManager`（`v1/core/kv_cache_manager.py`） |
| `new_step_starts(scheduler_output)` | `KVCacheManager.new_step_starts()` |
| `save_outputs(...)` | connector 侧 `save_kv_layer` 一类动词 |
| `TensorBlockPool`（CPU block mirror + per-block generations） | `BlockPool`（`v1/core/block_pool.py`） |
| `KVCacheGroupView` / `FullAttentionGroupView` | `KVCacheGroupSpec` / `FullAttentionSpec` + `FullAttentionManager` |
| `TensorCacheConfig` | `KVCacheConfig` |

```
vllm_omni/core/tensor_cache/
├── interface.py      # TensorCacheConfig / ModelCachePolicy / StageCacheOutputs / InflightStageOutputs
├── manager.py        # OmniTensorCacheManager（含后台线程上的 AsyncTensorCommitter）
├── block_pool.py     # TensorBlockPool（+ per-block generation）
└── group_view.py     # KVCacheGroupView protocol / FullAttentionGroupView / factory
```

![Class diagram](rfc-omni-tensor-cache-assets/class-diagram.svg)

**逐步流程（重构前）：**

![image-20260810225427318](/Users/guanxiangtian/Library/Application Support/typora-user-images/image-20260810225427318.png)

**逐步流程（重构后）：**

![Per-step sequence](rfc-omni-tensor-cache-assets/per-step-sequence.svg)

### 多请求交错 — committer 写与 builder 读同时作用于 block pool

三个请求跨越三步：**A** 在 step N prefill，**B** 在 step N+1 到达并 prefix-hit A 的 blocks，**C** 每步 decode 一个 token。committer（写 mirror）与 builder（为另一个请求读 mirror）并发运行——下面四条编号保证使该竞争无数据竞争。

![image-20260810225639762](/Users/guanxiangtian/Library/Application Support/typora-user-images/image-20260810225639762.png)

### API & Interface Changes

**新的公开表面**（取代 `OmniTensorPrefixCache` 上约 12 个方法 + 3 个裸属性）：

```python
class OmniTensorCacheManager:
    def register_policy(self, policy: ModelCachePolicy) -> None: ...     # 一次，在 load_model
    def new_step_starts(self, scheduler_output: SchedulerOutput) -> None: ...
    def save_outputs(self, hidden_states, mm_outputs, *,
                     num_tokens_unpadded: int, num_tokens_padded: int) -> InflightStageOutputs: ...
    def shutdown(self) -> None: ...

class InflightStageOutputs:
    """对本步尚未 commit 的输出的引用计数 handle。

    有意与 StageCacheOutputs 分开：本对象拥有资源（frozen-entry 引用、
    retire 生命周期、cap-K 记账），而 StageCacheOutputs 是纯值；
    若用 state flag 把二者合并，会把资源寿命耦合到数据容器上。
    """
    def materialize(self, req_ids: list[str]) -> StageCacheOutputs: ...  # 任意线程

class StageCacheOutputs(NamedTuple):
    hidden_states: dict[str, torch.Tensor] | None   # req_id → full-prompt tensor（policy 门控）
    mm_outputs: dict[str, dict[str, Any]]           # req_id → per-request payload（req-major）

@dataclass(frozen=True)
class ModelCachePolicy:                              # 取代对模型的 getattr 探测
    needs_full_hidden_states: bool = True
    merge_consumed_by_postprocess: bool = False      # 强制 eager materialize
    deferred_keys: frozenset[str] = frozenset()      # strip 聚合的 decode mm keys
    skip_keys: frozenset[str] = frozenset()
    default_placement: Placement = Placement.CPU     # GPU assembly 预留，未实现

class KVCacheGroupView(Protocol):                    # 唯一访问 vLLM 内部的路径
    block_size: int
    num_blocks: int
    def slot_mapping_gpu(self, num_tokens: int) -> torch.Tensor: ...
    def slots_for(self, req_id: str, token_start: int, token_end: int) -> torch.Tensor: ...
    def block_generations(self, slots: torch.Tensor) -> torch.Tensor: ...
```

**Runner 交互，重构前 → 重构后：**

```python
# ── 重构前：2 个文件约 10 个调用点，约 6 条隐式顺序契约 ──
# gpu_model_runner.py::_update_states
omni_prefix_cache.reset_prefix_cached_new_req_ids()
omni_prefix_cache.discard_deferred_mm_outputs(req_id)          # 每个 finished req
omni_prefix_cache.add_prefix_cached_new_req_id(req_id)         # hit marking；streaming 会跳过
# gpu_ar_model_runner.py::execute_model
omni_prefix_cache.drain_ready_async_writes()
omni_prefix_cache.commit_deferred_mm_outputs(finished, input_batch)   # 必须在 batch 移除前
# ...forward...
slot_mapping_gpu = input_batch.block_table[0].slot_mapping.gpu        # 调用点上的 .gpu workaround
omni_prefix_cache.schedule_async_write(hs, mm, slot_mapping_gpu, n, n_pad, skip_keys)
# sample_tokens / output build
self._stage_deferred_prefix_cache_mm_outputs(...)                     # 按 request 的 Python 循环
combined_hs = omni_prefix_cache.get_merged_hidden_states(...)         # consume 之后，主线程
combined_mm = omni_prefix_cache.get_merged_multimodal_states(...)
# + runner 侧 policy 探测：_model_needs_full_prefix_hidden_states()、
#   _deferred_prefix_cache_mm_keys()、payload gating、staging 特例

# ── 重构后：3 个调用点 + 1 次注册 ──
cache.register_policy(ModelCachePolicy.from_model(model))      # load_model，一次
cache.new_step_starts(scheduler_output)                        # execute_model 顶部
inflight = cache.save_outputs(hidden, mm_flat,
                              num_tokens_unpadded=n, num_tokens_padded=n_pad)
outs = inflight.materialize(req_ids)                           # 主线程 eager 或 #4476 builder
```

`vllm_omni/core/prefix_cache.py` 由 `vllm_omni/core/tensor_cache/` 取代。当前设置 `requires_full_prefix_cached_hidden_states` / `deferred_prefix_cache_mm_keys` 的模型（qwen3-tts、higgs-v3、personaplex）在弃用窗口内继续通过 `ModelCachePolicy.from_model()` shim 工作。

**Invariants（内化后的契约）：**

1. *Value freeze*：`save_outputs` 在 compute stream 上对预分配 slab 做一次 D2D copy；同时兼任 #4476 CUDA-graph 复用快照（今天的两次拷贝变一次）。此后 entry 内容不可变，仅存储层级迁移（slab → ping-pong → mirror）。
2. *Single-writer mirror + FIFO + inflight barrier*：committer 线程是 mirror 的唯一 writer（写-写竞争不可能），并严格按序 scatter，且不越过尚未 retire 的 `InflightStageOutputs`；reader 按最新可达层解析，优先取 queued entry 而非半发布的 mirror 行（scatter 后按 entry 原子发布），因此并发的 committer 写与 builder 读——包括跨不同请求——无数据竞争。
3. *Refcount + finalizer + cap*：entry 有引用计数；由 `materialize`（或 GC finalizer，带 warning）retire inflight handle；未完成 handle 上限为 K（超出则 fallback 到 eager materialize），使泄漏表现为可见背压，而非内存无限增长。
4. *Generation check*：每个 mirror 行在写入时打上 block generation；读取时与当前 generation 比对，不匹配视为 detected miss（bug 级日志），绝不静默返回陈旧数据。
5. *Block-alignment dependency（外部）*：merge 数学依赖 vLLM 保证 prefix hit 的 `num_computed_tokens` 按 block 对齐（full-hit 整块回滚；`vllm/v1/core/kv_cache_manager.py:249`）。用 assertion 守住。
6. *唯一留给 runner 的契约*：`new_step_starts` 必须在 `_update_states` 移除 finished requests **之前**运行（仍需要它们的 block table）。在 call site 文档化。
7. *Consistency fallback（支撑 G2）*：hit-span 校验（generation/presence）在 `new_step_starts` 中、forward 之前运行，并在 `materialize` 内再跑一次。不匹配时：请求被 poison，本步输出丢弃，并以新的 `cache_salt` 重新提交（all-miss → 全量重算，绕开被 poison 的 blocks）；窗口内反复不匹配则升级为 full reset（所有 generation 递增 + vLLM `reset_prefix_cache()`）并告警。每次触发都有 metric + bug 级日志——绝不静默自愈。

### Key Technical Decisions

| ID | 决策 | 原因 |
|---|---|---|
| **D1** | D2H 仍走 copy stream；将 `event` wait + `index_copy_` 移到后台线程（pinned ping-pong buffer，2 槽） | H20 上 D2H 已与下一步 forward 重叠；主线程成本是 scatter（~4–10 ms / 96 MB），不是 PCIe |
| **D2** | `materialize` 按 slab → ping-pong → mirror 解析段（读不等待 scatter） | 去掉 consume-before-merge 契约，解锁 #4476 builder 线程 materialize，并堵住 same-step 空读窗口 |
| **D3** | 固定大小 GPU strip 聚合适配 decode mm 行；取代 per-request `deferred_prefix_cache_mm_keys` | O(1) GPU 驻留、无 finish-time commit 尖峰；discard/commit 生命周期消失 |
| **D4** | Facade 自建 span 注册表 + 来自 `scheduler_output` 的 `delivered_upto` 水位 | 修复 `async_chunk` continuation 跳过 hit marking；不重发下游已累积的 spans |
| **D5** | 保留 slot-indexed mirror；用启动期 guard + generation tags 排除 L2/connector | 单一一致性域；真正的 L2 支持应搭载 connector payload（本次不做） |

否决的备选：更深 ring 仍主线程 poll；GPU-resident cache；双 cache 跨进程同步；hash-addressed tier（需求未实再做）。

## 4. Correctness & Testing Plans

**定义：** 对任意请求集合与调度顺序，`StageCacheOutputs` 必须与 no-prefix-cache 路径逐元素一致（相同 dtype/数值——cache 是传输优化，绝不是数值优化），且 §3 七条 invariants 成立。

| Level | Gate |
|---|---|
| **L1** | pool / committer / hit-registry 单测；将 `tests/core/test_prefix_cache.py` 迁到新 API 且期望值不变（Phase 0） |
| **L2** | same-step 双重复 prompt hit（merged HS == producer 的行）；eager ≡ background ≡ cache-off（merged 输出）；builder 线程抛错 → finalizer warning、不挂死；generation-mismatch → 走 salted-retry fallback |
| **L3** | 现有 `test_qwen3_omni` prefix case 保持绿（Phase 0）；去掉 #4476 guard 后，`async_scheduling + async_chunk + prefix caching` 全开且输出对齐（Phase 2） |
| **Smoke / Perf** | 第二次请求 `cache_hit_pct > 0` 且输出 == miss 路径；#2164 bench + 请求结束边界的 decode-ITL P99 相对 cache-off 无尖峰 |
