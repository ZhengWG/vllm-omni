# [RFC] 重构 Omni Tensor Prefix Cache：默认异步、解耦、可扩展

现行实现契约见
[docs/design/feature/omni_prefix_cache.md](docs/design/feature/omni_prefix_cache.md)。
本文保留为历史 RFC。

## 1. Overview

vLLM-Omni 的 hidden-state prefix cache（`OmniTensorPrefixCache`，#2164）镜像 vLLM KV-cache 的 block/slot 映射，把 stage 输出（hidden states 与 per-token 多模态张量）缓存在 CPU 上，使 prefix hit 同时跳过 KV 重算与跨 stage 张量重算。历经 #4106/#3734/#3665 演进后存在四个结构性问题：

1. **性能**：写路径半异步——CPU scatter 仍在主线程每步执行；deferred mm chunks 在 GPU 无界增长并在请求结束时造成 commit 尖峰；读路径全同步。
2. **正确性**：vLLM 乐观缓存 block hash（分配时提交 `computed + num_new_tokens`），同 step 跨请求 hit 会读到晚一步才落 CPU 的未写行（全 0/陈旧）并静默下发；abort 请求的 deferred chunks 被丢弃但其 blocks 仍可命中；异步 D2H 直读 forward 输出且无 `record_stream`（静态 buffer 复写 / allocator 复用两个竞争面）。
3. **耦合**：~12 方法 + 裸属性 + runner 侧 ~6 条隐式顺序契约 + `getattr` 探测模型策略；导致与 async omni output 互斥（#4476 guard）、`async_chunk` 下静默降级（continuation 跳过 hit marking）。
4. **可扩展性**：`block_table[0]` 写死单 KV group；L2 offload/connector 下 slot 索引静默失效。

本 RFC 把结构改成两层。上层 `OmniPrefixCacheManager` 管 `(slot, key)` 状态；下层 `OmniPrefixCacheController` 只搬数据。名字和 vLLM `v1/core` 对齐。

相关：#1184、#2164、#4106、#3734、#3665、#4442/#4476。

## 2. Scope & Objectives

### Goals

- **G1** — scatter 和大部分 D2H 不挡引擎临界路径。只有如下同步点，每步开销小且可预估：

  - D2D 冻结：`save_outputs` 上的 device-to-device 小 kernel，随 batch 线性增加，通常很快；
  - 整步 D2H **在 save 时打进 `StagingBufferPool`**（copy stream + `host_event`），不再作为 per-task 交给 committer 再拷。`materialize` / committer / `fetch_host` 都等同一个 event（record-once, wait-many）；
  - 下一次 save 开头 join 上一拍 `JOIN_NEXT_STEP` 的 `host_ready`。有界等待，D2H 通常已经完成；
  - 很罕见：命中还在 GPU 上的 **deferred** 行，才需要同步 D2H；
  - GPU 字节预算打满、flush 最老的 deferred task（极端降级路径）。

- **G2** — Omni PrefixCache 始终和 vLLM KV 状态保持一致。真正的异常只有致命一类；miss 不是异常：

  - **严重错误**：直接 dump 并退出进程。例如 `(slot, key)` 记账错了、`task_id` 对不上，或命中 span 里发现 absent。命中时 vLLM 已经跳过这些 token，本步也不会重算，缺的数据补不回来，所以不能静默降级。abort 也一样：hash 一旦进了本步 batch，就必须按规则写入。

  - **普通 miss（不是错误）**：请求在本步快照里、但没有 hit span（新 prompt / 未命中）。`materialize` 只用本步 forward 切片，**不打 warning、不记降级**。这是主路径。
  - **快照外的 req_id**：`materialize` 如果遇到不属于当前 `step_id` 快照的 id，说明 runner 传进来了正在运行的 batch。这种情况本来就不允许（P0 约定 async builder 必须传 save 时的 req 列表），代码里用 debug assert 检查一下，不用打 warning。
- **G3** — 窄 API：对外减少 api。
- **G4** — `async_scheduling + async_chunk + prefix caching` 全开（**P2**）。
- **G5** — `KVCacheGroupView` 隔离 block-table 访问，多 KV-group 可扩展（**P2**）。

### Non-Goals

- vLLM L2 KV offload 互操作（仅加启动期互斥 guard；长期应搭 connector payload）。
- Content-addressed 存储层；speculative-decode 交互（正交）。

## 3. Design

### Architecture

| 本 RFC | vLLM 对照 |
|---|---|
| `OmniPrefixCacheManager`（facade，block/slot 语义域） | `KVCacheManager` |
| `OmniPrefixCacheController`（staging pool + committer） | 无（近似 connector worker 侧） |
| `PrefixBlockPool`（pinned CPU block mirror） | `BlockPool` |
| `StagingBufferPool`（可复用的 step 级 D2H 页） | 无 |
| `KVCacheGroupView` / `FullAttentionGroupView` | `KVCacheGroupSpec` / `FullAttentionSpec` |
| `PrefixCacheConfig` | `KVCacheConfig` |

```
vllm_omni/core/prefix_cache/
├── interface.py      # PrefixCacheConfig / ModelCachePolicy / StageCacheOutputs
├── manager.py        # OmniPrefixCacheManager（slot 状态控制、span/hit、merge）
├── controller.py     # OmniPrefixCacheController + StagingBufferPool
├── block_pool.py     # PrefixBlockPool
└── group_view.py     # KVCacheGroupView / FullAttentionGroupView / factory
```

**类关系**（谁拥有谁、谁在哪条路径上被调用）：

![Omni prefix cache class diagram](rfc-omni-tensor-cache-assets/class-diagram.svg)

Manager 管 `(slot, key)` 状态；Controller 只搬运并上报完成。`KVCacheGroupView` 是访问 vLLM block table 的唯一入口，且只在 `new_step_starts` / `save_outputs` 用。`WriteTask` 是一次写入，用 `task_id` 标识；`(slot, key)` 上记的是当前 `task_id`，不是 slot 本身。

**一步时序**（`execute_model` → `sample`；同 batch 两个请求）：

- **P**：已经在跑的 decode。上一拍 JOIN_NEXT_STEP `t1` 还在后台搬；codes 等 `deferred_keys` 挂在 JOIN_ON_FINISH，等 P 结束再写。
- **Q**：本步新进，prefix hit 了 P 早已 committed 的前 32 个 token。本步 forward 只算 Q 的后缀。

![Per-step sequence with requests P and Q](rfc-omni-tensor-cache-assets/per-step-sequence.svg)

四个对外调用点：① `new_step_starts`（登记 Q 的 hit，收掉已完成的 task）；② `save_outputs`（锁外 join P 上一拍的 JOIN_NEXT_STEP，冻结本步 hidden，按请求各提交一个 JOIN_NEXT_STEP task，返回 `sid=N`）；③ `materialize(N, [P,Q])`；④ 本步没人读则 `discard_step(N)`（公开 API；读还是丢由 Manager 按 policy / 有没有 hit 决定，runner 不用展开）。P 是 miss，只拿本步切片；Q 的 `[0,32)` 按 **save 时记下的 `task_id`** 分组：已落盘走 mirror，还在写走 `fetch_host`。

**key 对应的是哪一路张量** 同一 slot 通常有两路独立数据：
- `hidden_states`：每步增量写
- `ModelCachePolicy.deferred_keys` 里的那些 mm 名（如 codes）：挂到请求结束再写

`skip_keys` 不进状态机。状态必须按 `(slot, key)` 记，否则 hidden 会被 deferred mm 拖成整个 decode 都 in-transit，或把还没写的 mm 当成已落盘。

**(slot, key) 与 WriteTask**

每个 `(slot, key)` 记一个 `task_id`：当前是谁在写。

`task_id = (req_id, key, seq)`
- `req_id`、`key`：哪个请求、哪路张量
- `seq`：这对组合第几次开写（每步 decode +1；deferred 一路通常不变；被换走后再写 +1）

`task_id is None`：当前无人写入。hidden 和 deferred 各一张表，同 slot 两路互不影响。`_req_tasks[req_id]` 用来一次拿到某请求的全部 task。

完成事件对不上该 `(slot, key)` 上当前的 `task_id`，直接丢掉。

`(slot, key)` 写入后只有两态：
- **in-transit**：正在写，还没落盘；可以从 GPU 快照读
- **committed**：已经落到主机 mirror

**absent** 表示从未登记，不是写完再退回去的状态。命中 span 里读到它是实现错误，dump 退出。

**块分给别人 = 换 task。** 只发生在：P 还没写完，块分给了 Q。Q 的 `save` 把这些 `(slot, key)` 改挂成 Q 的 `task_id`，P 在其上的旧写 skip 掉。没被换走的仍挂 P，等 finish / abort / cap 写完；别人命中走 in-transit，不必等 scatter。不必单独探测「P 被抢占了」。正常 finish 后再复用走不到这里：同一个 `save` 里先 join 完 P，再提交 Q。

读的时候按 **save 时记下的 `task_id`** 分组，不要按请求整单判定。一次 hit 常常跨多个 task：旧前缀已 committed，本步新行还在 in-transit。按「整单只要有 in-transit 就全走 snapshot」会去读已经释放的暂存，或把已落盘的行再拷一遍。

`WriteTask` 带着：
- **D2D 冻结拷贝**（save 时在 compute stream 上，记 freeze event）——forward 输出可能是 CUDA-graph 静态 buffer，现网 D2H 又没 `record_stream`，冻结同时关掉这两处竞争；也当 #4476 快照用。
- **save 时定下的 slot 列表**——commit 不再看 `input_batch`，现网 abort 丢 chunk 按构造消失。
- D2H event 与完成上报。

### 任务同步

- **同步（一步滞后）**：`save_outputs`（forward 后）consume-then-schedule——先 join 上一拍 JOIN_NEXT_STEP（只等 `host_ready`），再启动本步 staging D2H 并提交。join 只是**有界等待**。staging 页靠 `for_step` / `for_task` / `for_reader` 撑住，三种 holder 都空了才回收。
- **写入调度**（按 key 分，不按 token 数）：**JOIN_NEXT_STEP** — 立即 cache 的 key（hidden + 非 deferred mm）。submit 时 D2H 已在飞；committer 等 `host_event` 再 H2H scatter 进 `PrefixBlockPool`。**JOIN_ON_FINISH** — deferred mm，留在 GPU freeze 上，由 committer 做 D2H 再 scatter；finish/abort 或 cap 时 escalate。
- **cap 流控**：GPU 暂存设字节预算，超限阻塞 flush 最老的 task（D2H 大概率快要完成）。
- **abort：仍然写，不回退**——abort 的满块 hash 仍可命中，回退会让合法命中误报 unmatch。写侧规则：**hash 一旦进入本步 batch 就必须 save，abort 依然写入**——`new_step_starts` 收到 finished/abort 时，该请求已登记的 task 一律 escalate 并在下一次 save join。因此**不存在「分配后、第一次 save 前 abort」这类豁免窄窗**：真读到 absent 就是漏写或漏登记，直接 fail-fast。
- **preemption 复用**：见上文「块分给别人 = 换 task」。正常 finish 的复用走不到那里。

### 读路径：materialize 按 (slot, key) 的 task_id 分组解析

命中 span 先按 **save 时记下的 `task_id`** 分组（一次 hit 常跨多个 task：旧前缀已 committed，本步新行还在 in-transit），每组按其所在层取数，再拼回调用方顺序：

- **committed** → 读 host mirror（`PrefixBlockPool`）。
- **in-transit（JOIN_NEXT_STEP）** → `seg.host` 是 save 时挂上的 staging view。等该步 `host_event` 再 slice。不再二次 D2H，也没有 `publish_host`。
- **in-transit（JOIN_ON_FINISH）** → 等 freeze，再从 GPU 快照 sync D2H；committer 已经写出 owned host 则直读。
- **absent** → 从未登记。命中 span 里读到它是记账错误，dump 后退出进程。

**`materialize` 只用该 `step_id` 在 `save` 时挂上的快照，不碰 live `input_batch`。** async builder 晚一步跑时，finished 请求已被 remove，被抢占的 block table 已改写，因此：

- **cache 内部**：slot、命中的 block、以及当时的 `task_id`，会在 `new_step_starts` 阶段或 `save_outputs` 阶段保存到 step 对应的上下文中。也就是说，`KVCacheGroupView` 只会在这两个阶段（即 `new_step_starts` 和 `save_outputs`）被使用，因为它依赖于当前 step 的输入。
- **runner 的 materialize 入参 / 返回值**：`req_ids` 只能是该 `sid` 快照的子集。**返回值已经按请求拼好，runner 不再二次拼接。** 快照内有 hit → `cache[0, hit) + 本步切片`（policy 要整段 hidden 时）；快照内 miss → **只用本步切片**（不是空、不是从 cache 凑整段 prompt）；快照外 id → debug assert；命中 span 缺行 → dump + 退出。`StageCacheOutputs.hidden_states` 为 `None` 仅表示 policy 不需要 hidden，不是 miss。

本步交给 next stage 的行来自 `materialize` 等完 `host_event` 后对 staging view 的 clone，不是 per-task `publish_host`。`d2h_claimed` 只表示 committer 独占 copy 阶段（等 event，或做 deferred D2H）。eager 当场拷进 staging 页；CUDA 上在 copy stream 记 `host_event`。

### API & Interface Changes

```python
class OmniPrefixCacheManager:
    def register_policy(self, policy: ModelCachePolicy) -> None: ...     # load_model，一次
    def new_step_starts(self, scheduler_output: SchedulerOutput) -> None: ...
    def save_outputs(self, hidden_states, mm_outputs, *,
                     num_tokens_unpadded: int, num_tokens_padded: int) -> int: ...
                                # 返回 step_id；挂上的上下文必须被下面两者之一恰好消费一次
    def materialize(self, step_id: int, req_ids: list[str]) -> StageCacheOutputs: ...  # 任意线程
    def discard_step(self, step_id: int) -> None: ...  # 无读侧消费者时丢快照；写不受影响。是否 discard 由 Manager 按 policy / 本步 hit 判断。
    def shutdown(self) -> None: ...

class StageCacheOutputs(NamedTuple):
    hidden_states: dict[str, torch.Tensor] | None   # req → 已拼好：hit 则 cache+本步，miss 则仅本步；None=policy 不要 hidden
    mm_outputs: dict[str, dict[str, Any]]           # 同样已按 req 拼好；命中 span 缺行直接进程退出

@dataclass(frozen=True)
class ModelCachePolicy:                              # 取代 getattr 探测
    needs_full_hidden_states: bool = True
    merge_consumed_by_postprocess: bool = False      # 强制 eager materialize
    deferred_keys: frozenset[str] = frozenset()      # JOIN_ON_FINISH deferred：GPU staged 至 finish
    skip_keys: frozenset[str] = frozenset()

class KVCacheGroupView(Protocol):                    # 唯一访问 vLLM 内部的路径
    block_size: int
    num_blocks: int
    def slot_mapping_gpu(self, num_tokens: int) -> torch.Tensor: ...
    def slots_for(self, req_id: str, token_start: int, token_end: int) -> torch.Tensor: ...
    def cached_block_ids(self, req_id: str) -> torch.Tensor: ...
```

**Runner 交互（重构后，4 调用点 + 1 注册）：**

```python
cache.register_policy(ModelCachePolicy.from_model(model))   # 在 load_model 阶段注册模型的缓存策略

cache.new_step_starts(scheduler_output)   # 在 execute_model 的最开始（在 _update_states 之前）调用，表示新一轮推理任务开始
#   内部会登记哪些缓存块被命中了（"命中块"或"spans"：span 指一段被缓存命中的 token 区间），并处理生命周期相关事件，比如优先清理已完成或中止的请求资源
# ...forward...
sid = cache.save_outputs(hidden, mm_flat, num_tokens_unpadded=n, num_tokens_padded=n_pad)
#   内部（锁不能包住 join / D2H / cap flush）：
#     lock → 列出要 join 的 tid → unlock
#     join_host_ready(上一拍 JOIN_NEXT_STEP + 已 escalate 的 JOIN_ON_FINISH)
#     D2D 冻结 + stage_step_host（一次整步 D2H）          # 锁外
#     lock → drain → 提交 JOIN_NEXT_STEP（seg.host = views）+ stage deferred → 挂 ctx → unlock
#     return sid

outs = cache.materialize(sid, req_ids)    # 在 sample_tokens 或 output build 阶段调用，主线程或 builder 线程都可以执行
#   内部：会根据 (slot, key) 对本 step 的缓存状态进行解析；async_output/async_chunk 等场景统一使用这个接口读取数据
#   命中 span 缺行 → dump 并退出。本步没人读则 discard_step(sid)（写照常）。
#   读还是丢由 Manager 根据 policy / 本步是否有 hit 决定。
#   sid 必须传到 builder；N+1 步不能丢掉 N 步还在等的上下文。
```

实现包是 `vllm_omni/core/prefix_cache/`（取代旧的单文件 `prefix_cache.py`）。旧模型通过 `ModelCachePolicy.from_model()` 适配，无需改模型侧。NPU 上 runner 侧缓存工作进 Controller，eager 执行。

**Invariants：**

1. *Value freeze*：save 时一次 D2D 冻结（记 freeze event，兼任 #4476 快照）；task 内容不可变，仅存储层级迁移。
2. *task 原子性 + 单写者 + task_id 校验*：每个 task 覆盖一组 `(slot, key)`，**不是**由一对 `(slot, key)` 唯一确定。mirror 每次只能有一个 committer 在写。同一 `(slot, key)` 被新请求占用后，旧 task 在其上直接 skip，改挂新 `task_id`。只有**没被 skip 且和新写入有区间重叠**的旧 task 才需要等。完成只认该 `(slot, key)` **当前的 `task_id`**。
3. *状态解析读*：按 **save 时的 `tid`** 分组；committed→mirror；in-transit JOIN_NEXT_STEP→等 `host_event` 再 slice staging view；in-transit JOIN_ON_FINISH→freeze + committer/sync D2H；absent 在命中 span 内 dump 退出。
4. *Cap 背压*：GPU 暂存字节预算，超限阻塞 flush 最老的 task。**这是降级路径**，需要预算与 metric，不计入常态开销。
5. *线程与快照契约*：一把非重入锁只护状态表。`save_outputs` / `materialize` 都是 **lock → 计划 → unlock → join/D2H/cap flush →（如需）再 lock 提交或 drain**，见上面伪代码。不能 `@locked` 包住整个 `save_outputs`。计划只存 task 引用。`materialize` 只用 step 快照，不碰 `KVCacheGroupView` / live `input_batch`。
6. *Runner 与时序契约*：`new_step_starts` 必须在 `_update_states` 清理完 finished 请求前调用；scheduler_output 只会调用一次，并且顺序和执行一致（warmup/dummy 不用管）。同一次 `execute_model`，`save_outputs` 必须在 `materialize` 之前运行。异步 builder 只读取本 step save 时留下的快照（用 `step_id` 找），`materialize` 的 req_ids 一定是快照内的子集。每个 step 的上下文要么被 `materialize`，要么被 `discard_step`，只能消费一次。快照没包含的 id 用 debug assert 抓出来，不能靠警告降级处理。

### Key Technical Decisions

| ID | 决策 | 原因 |
|---|---|---|
| **D1** | save 时先处理已消费部分并安排后台搬运，不再用复杂的引用计数和释放流程 | 用状态解析保证正确性，简化资源管理 |
| **D2** | materialize 时将 `(slot, key)` 按 save 时的 `task_id` 分组处理 | 命中经常跨多个 task 状态，分组后每次只读到该 task 的数据 |
| **D3** | WriteTask 在 save 时定下要写的 slots，结束请求优先处理，abort 也写完 | commit 不依赖 input_batch，释放提前且确定，没有释放延迟或突发高峰 |
| **D4** | span 注册表 + `delivered_upto` 水位（**P2**） | 修 async_chunk continuation 漏记 / 重复下发。P0 不做，continuation 不 marking |
| **D5** | JOIN_NEXT_STEP / JOIN_ON_FINISH；deferred freeze 有 GPU 字节上限 | 立即 cache 的 key 每步经 staging 合；deferred 拖到 finish / cap |
| **D6** | save 时一次整步 D2H 进 `StagingBufferPool`；`seg.host` 是 view | JOIN_NEXT_STEP 上 committer 不再 D2H；materialize / fetch_host / scatter 共用该页 |

## 4. Correctness & Testing Plans

**定义：** 任意请求集合与调度顺序下，`StageCacheOutputs` 与 no-prefix-cache 路径逐元素一致。

| 等级 | 检查点（Gate） |
|---|---|
| **L1 基础** | 单测：pool / controller 状态流转、容量控制、abort 完成、命中登记。`tests/core/test_prefix_cache.py` 改用新 API，测试输出不变。|
| **L2 细节** | 1. 命中区跨多种 task 状态（committed + in-transit）时能分组正确读取。2. 同步 step 内重复命中能正常从 in-transit 读到。3. 同 slot 不同 key 有不同状态时，各自能正常解析。4. eager、后台写、关 cache 下行为一致。5. abort 情况下命中块有效。6. 块被复用后，不管新旧请求谁先结束，命中和最终 mirror 中都是新请求数据。7. 人为制造非法状态时能立即报错（fail-fast）。8. **如果命中 span 有缺失，立即 dump 并退出进程（不静默、不降级、不返回错误码）。** 9. step 上下文只被 materialize/discard_step 消费一次，builder 等待期间不会提前清理。10. cap 达到上限时背压数值不变。|
| **L3 综合（P2）** | 现有 `test_qwen3_omni` 前缀缓存用例上，`async_scheduling + async_chunk + prefix caching` 三项对齐输出。|
| **冒烟/性能** | 多次请求命中（hit > 0）输出和完全 miss 路径一致。性能没有退步。|

## 5. Phasing

- **P0**：本 RFC 主路径——`(slot, key)` 状态机、按 save 时 `task_id` 读、unlock-then-join、abort 仍写、命中 absent fail-fast。#4476 guard 先留着。continuation 不 marking（D4 不做）。
- **P1**：后台 scatter / D2H 落地，G1 列出的同步点可测。
- **P2**：G4 三特性全开（D4 `delivered_upto`）、G5 多 KV-group。L3 在这一档验收。
