# Stage Transfer 重构方案（中文版）

> 状态：**提案** — 基于对 stage 间 cache / payload / KV 传输路径的深度 review。
> 采用增量收束，**不是**对 Orchestrator / Connector / stage-processor 分层的推倒重写。
>
> 英文版：[stage_transfer_refactor.md](stage_transfer_refactor.md)

## 目录

1. [目标与非目标](#目标与非目标)
2. [现状架构](#现状架构)
3. [重构动机（Review 问题摘要）](#重构动机review-问题摘要)
4. [目标架构](#目标架构)
5. [分期交付](#分期交付)
6. [Stage Edge 契约](#stage-edge-契约)
7. [兼容与迁移](#兼容与迁移)
8. [测试策略](#测试策略)
9. [成功标准](#成功标准)
10. [相关文件](#相关文件)

---

## 目标与非目标

### 目标

1. **正确性优先**：消灭 stage 传输热路径上已知的竞态、静默丢数、以及会打死整个进程的错误路径。
2. **生命周期闭环**：每一次 `put` / prefetch / deferred buffer，在 finish、abort、timeout 时都有对应 cleanup。
3. **显式边契约**：每条 stage edge 声明传什么、怎么传（同步 / 流式 / KV）、谁负责就绪；禁止会把请求挂死的默认 role。
4. **数据面收束**：在不改 Orchestrator 拓扑的前提下，把 payload + chunk + KV 所有权收敛到统一的 Stage Transfer 门面后。
5. **模型胶水保持轻薄**：共享 payload schema 与 helper；`stage_input_processors` 只保留真正的模型差异。

### 非目标

- 跨模型复用 KV / hidden / prefix-cache **内容**（权重与布局不同时架构上不可行）。
- 把 Diffusion step cache（Cache-DiT / TeaCache / …）并入 AR stage transfer —— 它们继续作为 pipeline 内加速器。
- 对 Orchestrator、StagePool 或 Connector `put`/`get` API 做大爆炸重写。
- 在 Phase 0–2 落地 D2D 传输（作为 Phase 3 能力项跟踪）。

---

## 现状架构

```text
                    ┌──────────────────────────────────────┐
                    │           Orchestrator               │
                    │  route / kv_ready / abort / PD / CFG │
                    └───────────────┬──────────────────────┘
                                    │ 控制面
          ┌─────────────────────────┼─────────────────────────┐
          ▼                         ▼                         ▼
   Stage-0 Runner            Stage-1 Runner            Stage-N Runner
          │                         │                         │
          │  OmniConnectorModelRunnerMixin (~2.3k LOC)        │
          │    ├─ full_payload / async_chunk（后台线程）      │
          │    └─ KV 委托 → OmniKVTransferManager (~1.9k)     │
          │                         │                         │
          └──────────── OmniConnector put/get (D2H2D) ────────┘

  旁路缓存（stage 本地）：
    OmniTensorPrefixCache（AR，CPU，复用 block/slot 映射）
    Diffusion PromptEmbedCache / Cache-DiT / …（不属于 stage transfer）
```

**保留：** 控制面 / 数据面分离；Connector `put`/`get`；按模型的
`stage_input_processors`；Orchestrator 作为请求路由器。

**改变：** 数据面的所有权与契约；失败与 abort 语义；显式 edge 配置。

---

## 重构动机（Review 问题摘要）

来自代码深度 review（严重度缩写）。完整细节见产出本方案的 review 讨论。

| 级别 | 区域 | 问题 |
| --- | --- | --- |
| C | KV prefetch | `_bg_copy_stream` 上 H2D 后主流未 `wait` → GPU 竞态 |
| C | Prefix cache | 复用 `slot_mapping.gpu` 的异步 D2H 与下一步 `_update_states` 竞态 |
| C | async_chunk | `_accumulate_payload` 在 `_lock` 外改 dict → chunk 静默丢失 |
| C | Orchestrator | `process_engine_inputs` 异常再抛 → 整条 orchestration loop 死亡 |
| H | Connector | finish/abort/timeout 从未从 KV manager 调用 `cleanup(request_id)` |
| H | CFG + KV | prefetch miss 的 sync 回退丢掉 `cfg_kv_collect_func`；kv_ready 在 companion 输出尚未 set 时就标记完成 |
| H | Prefix | 多 kv-group 只 warning，仍用 `block_table[0]` 做 merge |
| H | Abort | 进行中的 prefetch 未取消；payload 可能被消费后丢失 |
| M | Config | `role` 未设置时 `stage_receives_chunks` 默认 True → 请求挂起 |
| M | 债务 | Mixin 与 KV manager 职责重叠；热路径仍支持 legacy flat payload key |

---

## 目标架构

```text
                    ┌──────────────────────────────────────┐
                    │           Orchestrator               │
                    │  仅 route + StageEdgePolicy          │
                    │  （不碰传输 / 不碰 cache 内部）       │
                    └───────────────┬──────────────────────┘
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          ▼                         ▼                         ▼
                 StageTransferFacade（每个 ModelRunner 一份）
                    │
        ┌───────────┼────────────────┬────────────────┐
        ▼           ▼                ▼                ▼
   PayloadXfer  ChunkXfer        KvXfer           PrefixCache
   (full)       (async_chunk)    （包装现有       （仅 AR；
                                  KV manager）     可选）
        │           │                │
        └───────────┴───────┬────────┘
                            ▼
                   OmniConnector put/get
                   + RequestLifecycle hooks
                     （finish / abort / timeout → cleanup）
```

### 设计规则

1. **每个 runner 一个门面**：ModelRunner 只通过 `StageTransferFacade` 做
   send/recv/cleanup/readiness（`OmniConnectorOutput` 保持不变）。
2. **KV manager 先吸收、不一日删除**：Phase 2 把调用点迁到门面后；内部模块可
   等到拆分成本低时再落盘。
3. **Orchestrator 永不碰张量**：只应用 `StageEdgePolicy`
   （立刻转发 vs 等 connector vs PD vs CFG 栅栏）。
4. **失败要响、爆炸半径要小**：传输 / bridge 失败只标记**该请求**失败；不得
   拆掉 orchestration loop。
5. **Diffusion cache 置身事外**：本方案不尝试跨 stage 共享 Cache-DiT /
   PromptEmbed。

---

## 分期交付

每一期都可独立合入。后续阶段除列出的前置外，不得依赖尚未合入的更早阶段。

### Phase 0 — 正确性热修（不改 API）

**目标：** 止住静默损坏与进程死亡。零公开 API 变更。

| 条目 | 改动 | 主要文件 |
| --- | --- | --- |
| P0-1 | Prefetch H2D：bg copy 后 `Event.record`，消费侧 `wait_event`（或 apply 前 stream sync） | `kv_transfer_manager.py` |
| P0-2 | Prefix 异步写：clone `slot_mapping[:n]`（或下一步改 slot 前等待 copy stream） | `prefix_cache.py`，`gpu_ar_model_runner.py` |
| P0-3 | `_poll_single_request` 中对 `_accumulate_payload` + cache 更新全程持 `_lock` | `omni_connector_model_runner_mixin.py` |
| P0-4 | 按请求捕获 `process_engine_inputs` / `_forward_to_next_stage` 错误；发 error output + abort cleanup | `orchestrator.py` |
| P0-5 | 多 kv-group：禁用 Omni prefix merge（当作 miss），不再只用 group-0 | `prefix_cache.py` |

**退出标准：** 每条都有单测/集成测；Qwen3-Omni async_chunk smoke + 一条 KV
transfer 模型 smoke 通过。

**风险：** 低。除 P0-5（更安全的降级）外行为保持。

---

### Phase 1 — 生命周期与失败语义

**目标：** 补齐资源与就绪缺口；错误配置快速失败。

| 条目 | 改动 |
| --- | --- |
| P1-1 | 引入 `StageTransferFacade.cleanup_request(req_id, reason=)`，在 runner finish/abort 路径调用；委托 connector `cleanup`、prefetch cancel、payload/chunk 状态清理 |
| P1-2 | `OmniKVTransferManager.cancel_prefetch(req_id)` + Orchestrator/runner abort 接线 |
| P1-3 | Prefetch miss 的 sync 路径必须继续传入 `cfg_kv_collect_func` |
| P1-4 | CFG：仅在 **finished** 输出上标记 companion 完成；kv_ready 不得调用 `on_companion_completed` |
| P1-5 | PD edge 与 omni `kv_ready` edge 互斥 —— PD decode 只从 prefill **finish** 路径提交；`_handle_kv_ready_raw_outputs` 跳过 `_pd_pair` 边 |
| P1-6 | `stage_receives_chunks`：connector role 未设置 → `False` + 启动时 warning/error（async_chunk 缺 role 时 fail-fast） |
| P1-7 | KV / CFG scatter：有界超时或 poison-pill 解锁 follower；发送失败后禁止无限 `recv` |

**退出标准：** 负载下 abort 的 soak 无 SHM/RDMA 持续增长；CFG+KV 与 PD 配置有
回归测试；配错的 async_chunk 在启动失败。

**风险：** 中（abort 路径触及调用点多）。依赖现有测试 + 新增 connector cleanup
soak 脚本。

---

### Phase 2 — 数据面收束 + Edge 契约

**目标：** 让架构对齐目标图；压缩 mixin 表面积。

| 条目 | 改动 |
| --- | --- |
| P2-1 | 新增 `StageTransferFacade`，方法：`register_recv`、`poll_readiness`、`send_payload`、`send_chunk`、`send_kv`、`recv_*`、`cleanup_request` |
| P2-2 | 将 mixin 传输实现迁到 `vllm_omni/distributed/omni_connectors/stage_transfer/`（`payload.py`、`chunk.py`、`kv.py`、`facade.py`）；mixin 变薄委托或删除 |
| P2-3 | 在 stage/deploy 配置中正式化 `StageEdgeSpec`（见 [Stage Edge 契约](#stage-edge-契约)）；Orchestrator 只读策略 |
| P2-4 | 发送边界强制嵌套 `OmniPayload` schema；legacy flat key 经一版 warning 后改为硬错误 |
| P2-5 | 从大型 `stage_input_processors/*` 抽出 TTS/Omni 公共 handoff helper（concat/replace key、codec span、language/speaker meta）；模型只留差异变换 |

**退出标准：** mixin 文件 &lt; ~400 LOC（或已删除）；全部模型使用
`StageEdgeSpec`；生产路径不再依赖 legacy flat key。

**风险：** 中高（触及所有多 stage 模型）。缓解：保持 Connector API 稳定，按
pipeline 家族逐 PR 迁移（Qwen3-Omni → Qwen3-TTS → 其他）。

---

### Phase 3 — 能力升级（可选 / 可并行）

仅在 Phase 1 退出后开始；可与 Phase 2 后期重叠。

| 条目 | 改动 |
| --- | --- |
| P3-1 | 大 KV/payload 的 D2D connector 路径（NCCL / UCX / IPC）—— 已在 `disaggregated_inference.md` roadmap 中提及 |
| P3-2 | 若 profiling 证明 D2H 是瓶颈，加深 prefix 异步写流水线（pending write 环形缓冲） |
| P3-3 | 可选的跨 diffusion replica 共享 PromptEmbedCache（进程外存储）—— **仅**在有产品需求时做 |
| P3-4 | 指标：每边传输延迟、cleanup 滞后、prefetch hit/miss、debug 构建下的竞态探测计数 |

**退出标准：** 每项有独立设计附录 + bench 数据。

---

## Stage Edge 契约

引入声明式 edge spec（YAML + dataclass）。示例：

```yaml
stage_args:
  - stage_id: 0
    edges:
      - to: 1
        mode: async_chunk          # full_payload | async_chunk | kv | control_only
        connector: shm_default
        role: sender               # 使用 connector 的 mode 必填
        payload_schema: omni_v1    # 仅嵌套 OmniPayload
        on_failure: fail_request   # 禁止 fail_engine

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

### 策略矩阵（Orchestrator）

| Edge mode | 谁喂下游 | 转发触发 |
| --- | --- | --- |
| `control_only` | Orchestrator prompts / tokens | 上游 finished（或 streaming segment） |
| `full_payload` | Connector + Orchestrator submit | 上游 finished |
| `async_chunk` | Connector chunks | prewarm + chunk 就绪（`OmniConnectorOutput`） |
| `kv` | KV transfer + Orchestrator submit | `kv_ready` **或** finished（配置二选一；PD 强制 finished） |

非法组合（例如 PD pair + `kv_ready` 转发、async_chunk 缺 role）在配置加载时失败。

---

## 兼容与迁移

1. **Phase 0–1**：无需改 deploy YAML。
2. **Phase 2**：同时接受旧的 `input_connectors` /
   `output_connectors` + `async_chunk` 标志，以及新的 `edges:` 块；发出
   deprecation warning；从旧字段自动 derive `StageEdgeSpec`。
3. **再过一个 release**：要求 `edges:`（若 derive 成本很低也可长期保留 ——
   建议一个 minor：derive + warn，下一个 minor：error）。
4. 模型 processor：无 flag day；按模型 PR 渐进迁移 helper。

---

## 测试策略

| 层级 | 覆盖 |
| --- | --- |
| Unit | 流同步 helper；prefix slot clone；加锁 accumulate；edge-spec 校验；CFG companion 状态机；PD vs kv_ready 策略 |
| Integration | prefetch 中 abort；timeout 后 connector cleanup；多 kv-group prefix 禁用；CFG scatter 失败解锁 follower |
| E2E smoke | Qwen3-Omni async_chunk；一条 AR→Diffusion KV 路径；一条 PD pair |
| Soak | 30–60 分钟多请求 abort 风暴；断言 `/dev/shm` 与 connector 池指标不单调增长 |
| Perf gate | Phase 0/1 不得在现有 Qwen3-Omni async_chunk bench 上把 TTFP/E2E 回归到噪声带以外 |

---

## 成功标准

1. 所有 Critical / High review 项关闭，或有测试证明后显式豁免。
2. Abort + timeout 路径调用 connector cleanup；soak 显示资源使用稳定。
3. Orchestrator bridge 失败隔离到单个请求。
4. Stage transfer 入口统一经门面可达；mixin 不再拥有传输策略。
5. 新增多 stage 模型只需 `StageEdgeSpec` + 一个 processor 模块接线，无需改
   KV manager 或 mixin 内部。
6. 不尝试跨不同模型共享 cache **内容**；文档明确此边界（仅框架复用）。

---

## 相关文件

| 区域 | 路径 |
| --- | --- |
| Orchestrator | `vllm_omni/engine/orchestrator.py` |
| 数据面 mixin | `vllm_omni/worker/omni_connector_model_runner_mixin.py` |
| KV 传输 | `vllm_omni/distributed/omni_connectors/kv_transfer_manager.py` |
| Prefix cache | `vllm_omni/core/prefix_cache.py` |
| AR runner 集成 | `vllm_omni/worker/gpu_ar_model_runner.py` |
| Connector 基类 | `vllm_omni/distributed/omni_connectors/connectors/base.py` |
| Edge/role helper | `vllm_omni/distributed/omni_connectors/utils/config.py` |
| Stage processor | `vllm_omni/model_executor/stage_input_processors/` |
| 既有设计 | `docs/design/feature/disaggregated_inference.md`，`prefix_caching.md`，`async_chunk.md` |

---

## 建议 PR 顺序

1. `fix(stage-transfer): P0 correctness — prefetch sync, prefix slot, accumulate lock, orchestrator isolation`
2. `fix(stage-transfer): P0 multi-kv-group prefix disable`
3. `fix(stage-transfer): P1 cleanup/abort/prefetch cancel + connector.cleanup`
4. `fix(stage-transfer): P1 CFG/PD/kv_ready policy hardening`
5. `refactor(stage-transfer): P2 StageTransferFacade + module split`
6. `refactor(stage-transfer): P2 StageEdgeSpec + legacy derive`
7. Follow-ups：D2D / metrics / processor helper 抽取作为独立轨道

---

## 附录 A：AR 侧问题与待补全大特性

本附录把「Prefix / Stage Transfer 之外，AR 类模型还缺什么」收成可排期清单。
与正文 Phase 0–3 互补：正文偏传输正确性与收束；此处偏 AR 能力缺口。

### A.1 框架层问题（一类 AR 共性）

| 问题 | 现状 | 建议归属 |
| --- | --- | --- |
| Omni Prefix 仅单 kv-group | 多 group 只 warning，仍用 `block_table[0]` | Phase 0（先禁用 merge）→ 后续真支持另开特性 |
| Prefix × async output / speculative 互斥 | `_should_use_async_omni_output` 遇 prefix 或 speculative 直接关 | Prefix 安全 profile（附录 A.3-F2） |
| 传输 D2H2D | Connector 全路径 host 中转 | Phase 3 / A.3-F3 |
| async_chunk abort 状态脆 | scheduler realign/purge 已打补丁，adapter 与 coordinator 未统一 | A.3-F4 |
| Offline 结束早于传完 | `_free_request` 有 TODO | A.3-F4 |
| Generation 下游不吃 Omni Prefix | code2wav 等走 generation runner | 保持；勿强行统一 |
| 非 AR async_chunk 过滤假设 | 无音频码 chunk 可能被静默丢 | A.3-F4 |

### A.2 模型接入摩擦（接新 AR 时常踩）

1. **语义 opt-out**：只要 last-token 的 talker 必须关 full-hidden merge（Qwen3-TTS / Higgs 模式），否则 D2H 回归。
2. **deferred mm keys**：`codes.audio` 等需请求结束再写 prefix，否则挡 batching。
3. **默认关**：多数 deploy `enable_prefix_caching: false`；开 = 要单独测。
4. **胶水私有**：`stage_input_processors` 三套入口（token_only / full_payload / async_chunk），新模型成本高 → Phase 2 helper 抽取。
5. **能力参差**：部分路径 `batch=1`；MTP/CUDA graph 偏 talker 特化，非通用 AR 能力。

### A.3 待补全大特性（按建议优先级）

#### F1 — Stage Transfer 收束（进行中）

- **范围**：正文 Phase 0–2（正确性 → 生命周期 → Facade + EdgeSpec）。
- **依赖**：无。
- **为何第一**：不修完，后面 Prefix/D2D/PD 都会踩同一批竞态与 cleanup 坑。

#### F2 — Omni Prefix 安全 profile + 能力补全

- **范围**：
  - 文档化「何时可开」矩阵：单 kv-group / 无 speculative / 模型 opt-out 标志齐全。
  - multi-kv-group：短期硬禁用 merge；中期真支持或明确 unsupported。
  - 与 async scheduling / speculative 的共存策略（今日互斥可保留，但要配置期 fail-fast）。
  - 可选：更深 async write ring（profiling 证明瓶颈后再做）。
- **依赖**：F1 Phase 0（slot 竞态、multi-group 禁用）先合。
- **侵入性**：中（runner + prefix_cache + 配置校验）；不改 Orchestrator 拓扑。

#### F3 — D2D 传输

- **范围**：大 KV / hidden 的 NCCL、UCX 或 IPC connector；保留 `put`/`get` 外观。
- **依赖**：F1 Phase 1 cleanup 语义稳定后再接，避免 D2D buffer 泄漏难查。
- **侵入性**：中高（新 connector + 设备内存生命周期）；与正文 Phase 3-1 同一轨道。

#### F4 — async_chunk 通用化

- **范围**：
  - chunk adapter 与 input_coordinator 统一（scheduler TODO）。
  - abort / offline「传完再退出」闭环。
  - 非音频 AR 边：去掉「无 audio codes 就丢」的硬编码，改为 edge/payload schema 驱动。
  - 多副本与 prewarm 幂等（与正文 P1/P2 Orchestrator 项重叠）。
- **依赖**：F1 Phase 1–2（edge role、cleanup）。
- **侵入性**：中高（scheduler + mixin/facade + 多模型 processor）。

#### F5 — Omni 语义下的 AR Prefill–Decode 分离

- **范围**：PD 与 omni `kv_ready` / mm 输出交接互斥策略；多模态 PD 可用路径。
- **依赖**：F1 Phase 1（PD vs kv_ready 策略）+ 稳定 KV cleanup。
- **侵入性**：中（Orchestrator 策略 + 少量 runner）；不先做跨模型 cache 共享。

#### F6 — AR-Diffusion 会话 KV（实验 → 可产品化）

- **范围**：今日 `max_num_seqs=1`、单 session 驻留；后续 batch/step、多 session、与多 stage KV manager（#5244）对齐。
- **依赖**：与 F1 弱相关；可并行，但跨 stage KV 部分应等 F1 Facade 稳定。
- **侵入性**：高（独立 runner/能力面）；保持实验边界，勿并进 OmniTensorPrefixCache。

#### F7 — 跨 AR stage 的统一 cache 策略（产品级）

- **范围**：声明「哪些中间态可复用、何时失效」（KV / prefix tensor / payload），而不仅是每 stage 私有池 + connector 搬运。
- **依赖**：F1 EdgeSpec + F2 Prefix profile；否则策略无法表达。
- **侵入性**：高；属中长期，不阻塞 F1–F4。

### A.4 推荐排期（技术依赖序）

```text
F1 Phase0 ──► F1 Phase1 ──► F1 Phase2
                 │              │
                 ├─► F2 Prefix profile
                 ├─► F5 Omni PD
                 └─► F3 D2D（可与 F2 后期并行）
                        │
                        └─► F4 async_chunk 通用化（需 EdgeSpec）
                               │
                               └─► F7 统一 cache 策略（中长期）

F6 AR-Diffusion ── 与 F1 并行；跨 stage 部分挂 F1 之后
```

### A.5 明确不做（本附录边界）

- 跨不同模型复用 KV / hidden / prefix **内容**。
- 把 Diffusion step cache（Cache-DiT 等）并进 AR stage transfer。
- 为「看起来对称」强行让 code2wav 使用 OmniTensorPrefixCache。
