# [RFC] 重构 Omni Tensor Prefix Cache：默认异步、解耦、可扩展

> 本文为 [`rfc-omni-tensor-cache-refactor.md`](./rfc-omni-tensor-cache-refactor.md) 的中文译本。API / 类型名 / 路径保持英文原文，便于对照代码。

## 1. Overview

vLLM-Omni 的 hidden-state prefix cache（`OmniTensorPrefixCache`，#2164）镜像 vLLM KV-cache 的 block/slot 映射，把 stage 输出（hidden states 与 per-token 多模态张量）缓存在 CPU 上，使 prefix hit 同时跳过 KV 重算与跨 stage 张量重算。历经 #4106/#3734/#3665 演进后存在四个结构性问题：

1. **性能**：写路径半异步——CPU scatter 仍在主线程每步执行；deferred mm chunks 在 GPU 无界增长并在请求结束时造成 commit 尖峰；读路径全同步。
2. **正确性**：vLLM 乐观缓存 block hash（分配时提交 `computed + num_new_tokens`），同 step 跨请求 hit 会读到晚一步才落 CPU 的未写行（全 0/陈旧）并静默下发；abort 请求的 deferred chunks 被丢弃但其 blocks 仍可命中；异步 D2H 直读 forward 输出且无 `record_stream`（静态 buffer 复写 / allocator 复用两个竞争面）。
3. **耦合**：~12 方法 + 裸属性 + runner 侧 ~6 条隐式顺序契约 + `getattr` 探测模型策略；导致与 async omni output 互斥（#4476 guard）、`async_chunk` 下静默降级（continuation 跳过 hit marking）。
4. **可扩展性**：`block_table[0]` 写死单 KV group；L2 offload/connector 下 slot 索引静默失效。

本 RFC 重构为**以 slot/entry 状态机为核心的两层结构**：`OmniTensorCacheManager`（facade，block/slot 语义域）+ `OmniTensorCacheController`（entry 搬运域），命名对齐 vLLM `v1/core`。

相关：#1184、#2164、#4106、#3734、#3665、#4442/#4476。

## 2. Scope & Objectives

### Goals

- **G1** — 尽可能降低主线程 cache 开销：D2H/scatter 移后台，主线程保留必要同步点 join（常态 ≈0）；请求结束降低 ITL 尖峰。
- **G2** — cache 与 vLLM KV 状态一致：entry 原子迁移保证同req下 slot 状态一致；unmatch（partial/absent）即 **fail-fast**（bug 级日志 + dump 后进程退出），不静默/自愈。
- **G3** — 窄 API：对外减少api。
- **G4** — `async_scheduling + async_chunk + prefix caching` 全开。
- **G5** — `KVCacheGroupView` 隔离 block-table 访问，多 KV-group 可扩展。

### Non-Goals

- vLLM L2 KV offload 互操作（仅加启动期互斥 guard；长期应搭 connector payload）。
- Content-addressed 存储层；speculative-decode 交互（正交）。

## 3. Design

### Architecture

| 本 RFC | vLLM 对照 |
|---|---|
| `OmniTensorCacheManager`（facade，block/slot 语义域） | `KVCacheManager` |
| `OmniTensorCacheController`（entry 搬运task控制） | 无（近似 connector worker 侧） |
| `TensorBlockPool`（pinned CPU block mirror） | `BlockPool` |
| `KVCacheGroupView` / `FullAttentionGroupView` | `KVCacheGroupSpec` / `FullAttentionSpec` |
| `TensorCacheConfig` | `KVCacheConfig` |

```
vllm_omni/core/tensor_cache/
├── interface.py      # TensorCacheConfig / ModelCachePolicy / StageCacheOutputs
├── manager.py        # OmniTensorCacheManager（slot 状态控制、span/hit、merge）
├── controller.py     # OmniTensorCacheController（EntryWriteTask 执行、队列、cap管理）
├── block_pool.py     # TensorBlockPool
└── group_view.py     # KVCacheGroupView / FullAttentionGroupView / factory
```

**Manager / Controller 边界**——req 层级一致性归 Manager：slot 状态唯一控制在 Manager，Controller 执行搬运并上报完成，Manager 在固定点（save 内 join / materialize / cap flush）drain 事件并做 entry 级原子批量迁移；跨 task 一致性不依赖锁。

| | Manager（block/slot 语义域） | Controller（entry/搬运域） |
|---|---|---|
| 职责 | 解析 slots（group view）；slot→entry 映射与状态迁移； | 两段队列：**copy queue**（D2H 发射，Class 优先级、分块传输、cap 管理）+ **scatter queue**（只装 copy 已完成任务，不保序）；req-end 提优；背压 flush |
| 引擎耦合 | 知道 req_id / scheduler_output | 只见 entry/slots/priority；|

Manager通过 `Controller.fetch_host(entry, rows)` 获取数据，分几种状态：

+ committed → mirror 零拷贝视图 已拷贝结束，直接返回tensor_cpu_view即可；

+ in-transit → read stream 切片 D2H + sync；

+ pre-staged → D2H完成，host buffer 直读。

*（class diagram / 时序图待按 v2 重绘）*

### 存储模型：slot 状态机 + entry 写任务

slot（= vLLM KV slot）三态：

- **absent**：从未登记写入。命中即 unmatch。
- **in-transit**：本步登记后（save 冻结 / deferred stage）即进入；数据在 GPU 暂存，搬运中。同 step 的命中读走这里，不算 absent。
- **committed**：当前 owner **entry 携带的全部 key** 已落 pinned mirror，GPU 暂存已释放（该 entry 未携带的 key 不受此状态约束）。

状态以 **entry**（一次 `save_outputs` 的全部 slots）为粒度**原子批量**迁移。**块重新分配是状态机的一条正式转移**：新 entry 登记时，slot 无论处于 in-transit 还是 committed 都回到 in-transit 且 owner 换新，旧 entry 的完成不再作用于该 slot（见「preemption 复用」）。

`EntryWriteTask` 持有：
- **D2D 冻结拷贝**（save 时在 compute stream 上，记录 freeze event）——forward 输出可能是 CUDA-graph 静态 buffer 且现网 D2H 无 `record_stream`，冻结同时关闭两个竞争面；兼任 #4476 快照（两拷贝合一）。
- **save 时确定的 slot 列表**——commit 不依赖 `input_batch`，现网 abort bug 按构造消失。
- D2H event 与完成上报。

### 同步与流控

- **同步（一步滞后）**：`save_outputs`（forward 后）内 consume-then-schedule——先 join 上一 step 的 Class B 任务，再提交本 step。正确性不依赖 join 位置（committed 行不再被写、未 committed 行不读 mirror）；join 只是**有界等待点**，控制在途任务深度（Class B 深度 1），不负责资源回收——GPU 暂存的引用在拷贝完成时即释放。
- **写入分层**：分两类，**Class A**（批量一次性：prefill hidden、deferred codes；cache 副本只被未来命中一次性重建消费）→ lazy 后台分批写入，req 结束可提优 + 下一次forward join 收尾（减少尖峰）；**Class B**（每步增量：decode hidden）→ 每步写、next-forward join。
- **cap 流控**：GPU 暂存设字节预算，超限阻塞 flush 最老 entry（D2H 大概率已完成）。
- **abort：仍然写，不回退**——abort 的满块 hash 仍可命中，回退会让合法命中误报 unmatch。
- **preemption 复用**：请求被抢占时，它的块会被 free 并分给新请求，但旧请求没 finish、在途写还没落盘——同一个 slot 短暂存在新旧两份数据。新 entry 提交时若发现 slot 上还挂着没写完的旧 entry，做三件事：① 让旧写**立刻开始搬**（否则 deferred 的旧写要等旧请求结束才动，新 entry 会被它拖住）；② 记一条「旧的先落盘」的顺序约束，保证 mirror 最终留下的是新数据；③ slot 归属当场改成新 entry——之后的读只会拿到新数据，旧写就算后来完成也改不动这个 slot 的状态。正常 finish 的复用走不到这里：finish 的写在同一个 save 里先被等完，新写才提交。

### 读路径：materialize 按 slot 状态解析

- 全部 **committed** → 读 mirror（merge 数学不变）。
- 含 **in-transit** → 等 freeze event → 读 GPU snapshot 切片 sync D2H → 拷贝线程已发布 host 副本时回退直读 host buffer。**同 step 跨 req 命中由此天然正确**——设计不变量，消除乐观 hash 空读窗口。deferred keys finish 前长期 in-transit，同路径覆盖。
- 含 **absent** → unmatch，fail-fast。

materialize 的全部输入（slot 列表、hit 块表、调度快照）在 `new_step_starts`/`save_outputs` 时物化，**不读 live input_batch**——builder 晚一步跑时 input_batch 可能已变，快照化使 materialize 与 runner 状态解耦。

大 entry 首次消费（build_payload）**直读 snapshot 不等 mirror**，D2H 产物回灌 Controller 作 pre-staged host copy（PCIe 1×，重建 #3734 去重收益）。eager 时 sync D2H 在主线程 sample/build 窗口（替换今天同位置的阻塞 `.cpu()`）；async output 时在 builder 线程。

### API & Interface Changes

```python
class OmniTensorCacheManager:
    def register_policy(self, policy: ModelCachePolicy) -> None: ...     # load_model，一次
    def new_step_starts(self, scheduler_output: SchedulerOutput) -> None: ...
    def save_outputs(self, hidden_states, mm_outputs, *,
                     num_tokens_unpadded: int, num_tokens_padded: int) -> None: ...
    def materialize(self, req_ids: list[str]) -> StageCacheOutputs: ...  # 任意线程
    def shutdown(self) -> None: ...

class StageCacheOutputs(NamedTuple):
    hidden_states: dict[str, torch.Tensor] | None   # req_id → full-prompt tensor（policy 门控）
    mm_outputs: dict[str, dict[str, Any]]           # req_id → per-request payload

@dataclass(frozen=True)
class ModelCachePolicy:                              # 取代 getattr 探测
    needs_full_hidden_states: bool = True
    merge_consumed_by_postprocess: bool = False      # 强制 eager materialize
    deferred_keys: frozenset[str] = frozenset()      # Class A-deferred：GPU staged 至 finish
    skip_keys: frozenset[str] = frozenset()

class KVCacheGroupView(Protocol):                    # 唯一访问 vLLM 内部的路径
    block_size: int
    num_blocks: int
    def slot_mapping_gpu(self, num_tokens: int) -> torch.Tensor: ...
    def slots_for(self, req_id: str, token_start: int, token_end: int) -> torch.Tensor: ...
    def cached_block_ids(self, req_id: str) -> torch.Tensor: ...
```

**Runner 交互（重构后，3 调用点 + 1 注册）：**

```python
cache.register_policy(ModelCachePolicy.from_model(model))   # load_model
cache.new_step_starts(scheduler_output)   # execute_model 顶部，_update_states 之前
#   内部：hit/span 注册、生命周期事件（finished/abort → 提优 flush）
# ...forward...
cache.save_outputs(hidden, mm_flat, num_tokens_unpadded=n, num_tokens_padded=n_pad)
#   内部：join { 上一 step Class B + 本步 new_step_starts 标记的 finished-req 提优任务 }
#         → D2D 冻结 + 物化 slots → 提交 EntryWriteTask
outs = cache.materialize(req_ids)         # sample_tokens/output build，主线程或 builder 线程
#   内部：按 slot 状态解析；async_output/async_chunk 的 slice 读同一路径
```

替代 `vllm_omni/core/prefix_cache.py`；旧模型旗标经 `ModelCachePolicy.from_model()` shim 过渡；NPU runners 迁到 Controller eager 实现。

**Invariants：**

1. *Value freeze*：save 时一次 D2D 冻结（记 freeze event，兼任 #4476 快照）；entry 内容不可变，仅存储层级迁移。
2. *Entry 原子性 + 单写者*：Manager 原子批量迁移 slot 状态；committer 线程是 mirror 唯一 writer。同一 slot 有新旧两笔写时：旧的先落盘、且立刻开始搬；slot 状态只认最新那笔写的完成（旧写完成不改状态）。Class B 与 finished-req 任务在下一次 save 里先等完，再提交新写。
3. *状态解析读*：committed→mirror；in-transit→snapshot 切片 D2H；absent→unmatch（fail-fast）。
4. *Cap 背压*：GPU 暂存字节预算，超限阻塞 flush 最老 entry。
5. *Block 对齐（外部依赖）*：vLLM 保证 prefix hit 的 `num_computed_tokens` 块对齐；assert 守住。
6. *唯一 runner 契约*：`new_step_starts` 在 `_update_states` 移除 finished 之前；scheduler_output **恰好一次、按序**（warmup/dummy 不喂）。
7. *Unmatch = fail-fast*：bug 级日志（dump span/slot/entry 状态）+ metric 后进程退出，与 vLLM 上游 fail-fast 习惯一致；无 salted-rerun/poison/reset 自愈阶梯——检测不完备时继续运行不诚实，恢复交给编排层重启；generation 校验降级 debug assert。

### Key Technical Decisions

| ID | 决策 | 原因 |
|---|---|---|
| **D1** | save 内 consume-then-schedule（一步滞后）+ 后台搬运，弃被动管理： free-committer + refcount/finalizer | 正确性由状态解析保证，join 只管资源；换大幅简单性 |
| **D2** | materialize 按 slot 状态解析 | 读不等 scatter；同 step 跨 req 命中天然正确；解锁 builder 线程 materialize |
| **D3** | entry save 时物化 slots；req-end 提优 + next-save join；abort 照常写完 | commit 不依赖 input_batch ⇒ abort bug 消失；释放确定性 ≤1 step 且无尖峰 |
| **D4** | span 注册表 + `delivered_upto` 水位 | 修 async_chunk 跳过 hit marking；不重发已下发 spans |
| **D5** | Class A/B 分层 + GPU 暂存字节 cap | 大 block 读者只有未来命中重建 ⇒ lazy；小 block 线性增长 ⇒ 每步 join；cap 硬兜底 |
| **D6** | 大 entry 首次消费直读 snapshot，D2H 产物回灌作 pre-staged | payload 反正要 D2H；PCIe 1×，重建 #3734 去重收益 |
| **D7** | 保留 slot-indexed mirror；启动期 guard 排除 L2/connector | 单一一致性域；L2 应搭 connector payload |

## 4. Correctness & Testing Plans

**定义：** 任意请求集合与调度顺序下，`StageCacheOutputs` 与 no-prefix-cache 路径逐元素一致，且七条 invariants 成立。

| Level | Gate |
|---|---|
| **L1** | pool / controller（状态机、cap、abort 写完）/ hit-registry 单测；`tests/core/test_prefix_cache.py` 迁新 API 期望值不变 |
| **L2** | same-step 重复 prompt hit（走 in-transit 读）；eager ≡ background ≡ cache-off；abort 满块命中数据有效；块被抢占复用后，不管新旧请求谁先结束，命中读到的和 mirror 最终留下的都是新请求的数据；人为 partial → fail-fast；cap 触顶背压数值不变；pre-staged 回灌 ≡ 直接 D2H |
| **L3** | 现有 `test_qwen3_omni` prefix case 通过；删 #4476 guard 后三特性全开输出对齐 |
| **Smoke / Perf** | 二次请求 hit>0 且输出 == miss 路径；性能无回退 |

## 5. 分期与退出标准

| Phase | 内容 | 退出标准 |
|---|---|---|
| **Phase 0**（可独立合入） | legacy 路径热修：命中步 merge 前强制落盘 pending 异步写 | 复现同 step 空读的单测通过；现有 prefix cache 测试不变；不依赖本重构 |
| **P0 重构** | tensor_cache 模块 + runner 接线（CUDA 走新路径，其他平台留 legacy） | 单测全绿；qwen3-omni e2e 文本/音频输出与 cache-off 一致；抢占压力下 unmatch=0、引擎存活 |
| **P1 性能** | Class A 滴灌调优、cap 默认值标定、主线程开销测量 | 命中步主线程 cache CPU 时间显著下降；请求结束边界 ITL P99 无尖峰；GPU 暂存稳态 ≤ cap |
| **P2 兼容** | 删 #4476 guard、D4 span/`delivered_upto`、NPU eager 迁移、多 KV group | 三特性全开 e2e 输出对齐；streaming continuation 命中不漏不重；NPU 回归通过 |

反馈窗口：Phase 0 与 P0 分别独立提 PR，各留一轮 review 后再进入下一期。
