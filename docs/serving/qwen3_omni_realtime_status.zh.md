# Qwen3-Omni 视角看 vLLM-Omni 支持现状

> 关联阅读：[Realtime API 对比（中文）](realtime_api_comparison.zh.md) ·
> [Streaming Video Input API](video_stream_api.md) ·
> [`vllm_omni/experimental/fullduplex/DESIGN.md`](../../vllm_omni/experimental/fullduplex/DESIGN.md)

本文以 **Qwen3-Omni** 为主视角：先按 OpenAI Realtime 体系梳理关键概念，再对照展示 vLLM-Omni 当前能力与接口用法，最后基于当前代码与文档中的 TODO / 限制，给出能力补齐清单。

---

## 一、Realtime 整体架构（参照 OpenAI）

### 1.1 Realtime 是什么

OpenAI Realtime 是一条**有状态长连接**上的语音/多模态交互协议，核心对象：

| 对象 | 作用 |
| --- | --- |
| **Session** | 会话配置：模型、音色、音频格式、`turn_detection`（VAD）、instructions、tools、模态 |
| **Conversation** | 会话内累积的有序 items（user/assistant/system 消息、function call 等） |
| **Response** | 一次模型回合，产出文本/音频 item 写回 Conversation |

客户端发 **client events**（如 `session.update`、`input_audio_buffer.append`、`response.create`），服务端回 **server events**（如 `response.created`、`response.output_audio.delta`、`response.done`）。会话上限约 60 分钟；模型产出过音频后 `voice` 锁定。

### 1.2 三种传输：WebRTC / WebSocket / SIP

| | **WebRTC** | **WebSocket** | **SIP** |
| --- | --- | --- | --- |
| 定位 | 浏览器/移动端实时媒体 | 通用双向消息通道 | 电话网信令 + 媒体 |
| 音频路径 | 独立 media track，浏览器栈负责采集/播放/抖动/丢包 | 音频 Base64 进 JSON（或二进制帧），与控制同一连接 | RTP 等电信媒体 |
| 控制路径 | data channel（`oai-events`） | 同一 socket、有序 JSON | SIP + Realtime 控制 |
| 鉴权 | ephemeral token（`client_secrets`）或服务端 SDP 代理（`/v1/realtime/calls`） | 后端直接 `Authorization: Bearer` | 电话侧接入 |
| 适用 | 公网端侧语音、体验敏感 | 服务端管道、自建网关、demo | 呼叫中心 / PSTN |

**核心区别不在「能不能传音频」，而在谁负责媒体质量**：WebRTC 把抖动缓冲、重传、设备管理交给浏览器栈；WebSocket 需要应用自己做分片、缓冲、播放与重连；SIP 是电话形态。

**WebSocket 能否覆盖大部分场景？** 对自托管推理服务：**能覆盖大部分工程场景**（服务端↔服务端、网关后置、内网 demo、准实时批处理）。仍然更适合 WebRTC 的是「公网浏览器直连 + 低延迟口播体验」。常见架构是：浏览器（WebRTC/WS）→ 自建网关 → 推理服务（WebSocket）。

### 1.3 VAD（turn_detection）

VAD 决定「用户何时说完、模型何时接话」：

| 模式 | 行为 |
| --- | --- |
| `server_vad` / `semantic_vad`（默认开） | 服务端检测语音起止（`speech_started/stopped`），自动 commit 用户输入，通常自动创建 response |
| VAD 开 + 关自动回复 | `turn_detection.create_response=false` 等；仍收语音事件，客户端自行 `response.create` |
| `turn_detection: null` | 对讲机（push-to-talk）：客户端手动 `commit` + `response.create`（+ `clear`） |

原理上：对流式音频估计 speechness（能量阈值 / 学习型模型 / 语义级），静音超阈值判定回合结束，再联动自动提交与打断（barge-in）。

### 1.4 半双工 vs 全双工

| | 半双工 | 全双工 |
| --- | --- | --- |
| 交互形态 | 说完一轮 → 播一轮 | 边听边说；播放中持续收音 |
| 打断 | 无或弱 | 支持 barge-in（VAD interrupt 或模型策略） |
| 轮次决策 | 客户端 commit 或简单 VAD | 服务端 VAD 或模型自身 listen/speak 策略 |

**OpenAI 默认体验接近全双工**：VAD 常开、支持打断、说话即触发回合，用户无需显式提交。

### 1.5 对照补充：`/v1/video/chat/stream` 这类接口在 OpenAI 的位置

OpenAI Realtime 语音 API 支持图像 item（`conversation.item.create` + `input_image`），但**没有**「持续推视频帧建立会话上下文再提问」的独立视频流接口；视频理解通常走 Responses/Chat（整段视频或抽帧图片）。vLLM-Omni 的 `/v1/video/chat/stream` 是自定义补位接口，协议见第二章。

---

## 二、vLLM-Omni 当前能力

### 2.1 能力总览（对照第一章）

| 概念 | OpenAI | vLLM-Omni 现状 |
| --- | --- | --- |
| 传输 | WebRTC + WebSocket + SIP | **仅 WebSocket**；无 WebRTC/SIP/`client_secrets` |
| 会话模型 | Session/Conversation/Response 全量 | Qwen 路径：薄会话；MiniCPM duplex：较完整投影（实验） |
| VAD | 服务端 VAD / semantic VAD | **无服务端 VAD**。Qwen 靠客户端 commit；duplex 要求 `turn_detection: null`，由模型决策 listen/speak |
| 半/全双工 | 默认近全双工 | Qwen `/v1/realtime` 半双工；全双工在 **MiniCPM 实验栈**（`session_mode: duplex`），Qwen 无 |
| 事件命名 | GA：`response.output_audio.delta` 等 | 偏旧形态：`response.audio.delta`、`transcription.*` |
| Tools / 图像 | Realtime 一等能力 | Realtime 路径不支持（duplex 多为存储/回显） |
| 视频流输入理解 | 无独立接口 | `/v1/video/chat/stream`（Qwen3-Omni） |
| 流式视频生成 | 无对应 Realtime 面 | `/v1/realtime/video`（扩散 fMP4，路径重名而已） |

### 2.2 与 Qwen3-Omni 相关的 WebSocket 端点

```text
/v1/realtime              Qwen 半双工语音（本章主角）
/v1/video/chat/stream     Qwen 视频帧流理解
/v1/realtime?duplex=1     MiniCPM 全双工（对照，非 Qwen）
/v1/realtime/video        扩散视频输出（与对话 Realtime 无关）
```

### 2.3 `/v1/realtime`（Qwen 半双工语音）

复用上游 vLLM STT realtime 会话框架，omni 子类只改生成侧，把 Thinker→Talker 音频打成 `response.audio.delta`（实现见 `vllm_omni/entrypoints/openai/realtime_connection.py`）。

事件流（当前实现，无 VAD）：

```text
Client                                Server
  |── session.update {model} ──────────►|
  |── input_audio_buffer.commit(final=false)  # 打开生成
  |── input_audio_buffer.append × N ───►|      # base64 PCM16 16kHz mono
  |── input_audio_buffer.commit(final=true)   # 关闭输入
  |◄── transcription.delta / done ──────|
  |◄── response.audio.delta × M ────────|      # pcm16 + sample_rate_hz（常见 24kHz）
  |◄── response.audio.done ─────────────|
```

使用 demo：

```bash
# 服务端：该路径要求关闭 async_chunk
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091 \
  --no-async-chunk

# 客户端
python examples/online_serving/qwen3_omni/openai_realtime_client.py \
  --url ws://localhost:8091/v1/realtime \
  --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --input-wav input_16k_mono.wav \
  --output-wav realtime_output.wav
```

### 2.4 `/v1/video/chat/stream`（Qwen 视频流理解）

协议与语义（详见 [Streaming Video Input API](video_stream_api.md)）：

```text
Client                                Server
  |── session.config ─────────────────►|   # modalities/num_frames/EVS 等
  |── video.frame × N ────────────────►|   # base64 JPEG/PNG（可带 frame_id/pts_ms）
  |◄── video.frame.ack ────────────────|   # EVS 过滤结果、缓冲计数
  |── audio.chunk × M（可选）─────────►|   # base64 PCM16 16kHz mono
  |── video.query {text} ─────────────►|   # 手动触发一轮理解
  |◄── response.start ─────────────────|
  |◄── video.frames.consumed ──────────|   # 本轮实际选中的帧元数据
  |◄── response.text.delta*/done ──────|
  |◄── response.audio.delta*/done ─────|   # WAV base64（modalities 含 audio）
  |── video.done ─────────────────────►|
  |◄── session.done ───────────────────|
```

服务端机制要点：

- **EVS 近重复帧过滤** + 环形缓冲（`max_frames`，默认 50）；query 时按 `num_frames`（默认 4）均匀采样并保留最后一帧；
- 帧入缓冲后异步 PIL prewarm（md5 作 mm_cache uuid），降低 query 时延；
- 新 `video.query` 会 soft-interrupt 上一轮（必要时 `engine.abort`）；
- Qwen handler 的 `should_trigger_turn` 恒为 `False`：**只支持手动 query 触发**，不做自动轮次；
- 历史仅保留最近 2 条并文本化后拼入 prompt。

使用 demo：

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --deploy-config vllm_omni/deploy/qwen3_omni.yaml \
  --omni --port 8000 --trust-remote-code

python examples/online_serving/qwen3_omni/streaming_video_client.py \
  --host 127.0.0.1 --port 8000 \
  --video /path/to/video.mp4 \
  --query "Describe what is happening in the video."
```

### 2.5 全双工对照（MiniCPM 实验栈，非 Qwen）

`session_mode: duplex` 部署 + `?duplex=1` 路由后：浏览器约每 200ms 连续 `input_audio_buffer.append`（不跑 VAD、不 commit），MiniCPM Stage0 以约 1s 模型单元消费流式输入，模型自行决策 listen/speak，并投影出 OpenAI 形事件（`response.speak`/`response.listen` 为 omni 扩展），配合 `playback.ack` 推进历史提交、支持会话 resume/takeover。Qwen 目前**未接入**这条 runtime。

---

## 三、能力补齐

以下按「现状证据 → 需要补什么」组织。现状证据均来自当前仓库代码/文档。

### 3.1 `/v1/realtime`（Qwen 语音）待补齐

收敛为三项核心工作（半双工体验增强、打断等归入 3.2 全双工统一实现）：

| # | 现状证据 | 需要补齐 |
| --- | --- | --- |
| 1 | 客户端必须 `commit(final=true)` 才结束输入；无 `speech_started/stopped` | **服务端 VAD / `turn_detection`**（server_vad 起步，semantic 进阶），支持自动 commit + 自动 response |
| 2 | `recipes/Qwen/Qwen3-Omni.md`：`/v1/realtime` 在 `async_chunk` 开启时不可用，需 `--no-async-chunk` | **兼容 async_chunk** 或在服务端自动降级，而不是要求用户改部署 |
| 3 | 事件面只有 `transcription.*` + `response.audio.*`，且事件名为旧形态（`response.audio.delta`） | **协议对齐（一揽子）**：Conversation/Response 生命周期投影（`conversation.item.*`、`response.created/done`、content part 事件）+ GA 命名（`response.output_audio.delta`、`response.output_audio_transcript.delta`）+ 嵌套 `session.audio.input/output` 配置。事件面与事件名同属协议册内容，应作为一个协议版本一次性对齐，避免两次破坏性变更 |

低优先（当前不投入）：

- tools / 图像输入（Realtime 路径 function calling、`input_image` item）；
- WebRTC / SIP / `client_secrets` 传输与鉴权——如有浏览器需求，先以「浏览器 → 网关 → Omni WS」参考架构文档化替代。

### 3.2 全双工：栈本身的能力/缺口 + Qwen3-Omni 的 Gap

半双工路径上的「连续输入 runtime」「barge-in 打断」不再单独补齐——这两项本质是全双工能力，统一放到本节。

#### 3.2.1 全双工栈本身：已有能力与自身缺口

框架侧通用能力已就位（约 4:1 的通用/模型专属代码比）：WS actor 单信箱、`DuplexSession` 聚合根、Realtime 投影、resume/takeover、engine control plane + fence、`ServingRuntimeAdapter` 协议注入点。当前唯一 native 接入是 MiniCPM-o 4.5（`minicpmo45/`，约 3k 行适配器），另有独立的 JoyVL 栈。

栈自身缺口（DESIGN.md「不宣称」清单）：

| # | DESIGN.md 明确未完成 | 含义 |
| --- | --- | --- |
| 1 | scheduler-native KV append | 流式追加仍走 resumable request 方案，未进调度器原生路径 |
| 2 | 确定性 VAD 触发打断 | 打断依赖模型 listen/speak，无 VAD 驱动的确定性打断（硬 barge-in 缺口在此补） |
| 3 | 生产级多会话 admission/fairness/failure recovery | 仅验证 `max_sessions=2`（H20）；无容量预算与公平性 |
| 4 | 有界长会话 KV | 长会话 KV 无上界管理 |
| 5 | 视频输入 / A/V 同步 | duplex Realtime 无视频模态 |
| 6 | 版本化 plugin descriptor | 接入第二个 native duplex 模型前需先补：目前 MiniCPM 通过两个显式字段选择 engine extension + serving adapter，无开放握手 |
| 7 | 类型化 session/input-chunk 契约 | 扩展边界仍是 generic mapping，「especially before adding video」 |

#### 3.2.2 Qwen3-Omni 若要支持全双工：Gap 拆解

先分清两个命题——**原生全双工**与**全双工体验**不是一条路：

| 路线 | 决策来源 | Qwen3-Omni 可行性 |
| --- | --- | --- |
| **原生全双工**（MiniCPM 式） | 模型采样 `<|listen|>`/`<|speak|>`/`<|turn_eos|>` 控制 token | ❌ **第一性 Gap 在模型**：Qwen3-Omni 是 turn-based 训练，词表/训练目标里没有 listen/speak 时分复用机制，框架适配无法凭空造出该能力 |
| **VAD 驱动的全双工体验**（OpenAI gpt-realtime 式） | 服务端 VAD 判停 → 自动 commit/response → VAD interrupt 打断 | ✅ 现实路径，即 3.1 #1 做深 |

**路线 A（原生）** 的 Gap 链，按依赖顺序：

1. **模型能力（前提，非框架工作）**：需要 duplex 训练目标的 Qwen 变体——listen/speak 控制 token + unit 化流式输入训练。此项不存在，后续都无从谈起；
2. 框架前置债：3.2.1 #6/#7（plugin descriptor + 类型化契约），DESIGN.md 明确要求在第二个 native 模型接入前完成；
3. Qwen 版适配器（对照 `minicpmo45/` 约 3k 行）：帧率契约 policy、Stage0 流式 prefill（unit 拼装 + KV 连续 + terminator 重注入）、data plane 输出判定（listen/speak/turn_eos 识别）、engine runtime extension、输入规整；
4. 对等测试 + 实机 E2E 验证（参照 MiniCPM 的 H20 验证记录，成本不低）。

**路线 B（VAD 体验）** 的 Gap 链（不依赖模型改造）：

1. 服务端 VAD / `turn_detection`（3.1 #1）：语音起止检测 + 自动 commit + 自动 response；
2. 连续输入会话：播放期间持续收音（半双工协议改造，可借鉴 duplex 栈的流式 append，不必等 scheduler-native KV）；
3. VAD interrupt barge-in：检测到用户插话 → cancel 当前回合 + 截断音频输出；
4. 局限须明示：接话/打断时机是 VAD 规则而非模型语义决策，无法做到「听懂了再决定接不接话」。

**结论**：Qwen3-Omni 的全双工短期走 **路线 B**（纯框架工程，体验对齐 OpenAI 默认形态）；路线 A 取决于模型侧是否推出 duplex 训练版本，框架侧仅需提前还掉 3.2.1 #6/#7 的债。duplex 栈的直接适用对象是 MiniCPM 之外的下一个 native duplex 模型（如 Moshi/PersonaPlex 系，注意其「并行音频流」范式与 MiniCPM「1s unit」差异较大，适配器需按新范式重写）。

### 3.3 `/v1/video/chat/stream` 功能补齐确认

结论：**核心链路可用，但以下缺口未补齐**（来源：`docs/serving/video_stream_api.md` Known Limitations + 代码）：

| # | 现状证据 | 状态 | 需要补齐 |
| --- | --- | --- | --- |
| 1 | 文档：Session KV reuse / incremental prefill「not implemented in this PR」；每次 `video.query` 从缓冲重建 prompt | ❌ 未补 | 会话级 KV 复用 / 增量 prefill，把「推帧」变成真正的增量上下文而非重复编码 |
| 2 | `QwenOmniStreamingVideoHandler.should_trigger_turn` 恒 `False` | ❌ 未补 | 自动轮次触发（帧数/时间/语音触发），基类 hook 已预留 |
| 3 | `create_streaming_video_handler` 注释：「Additional pipelines can be selected here in follow-up PRs」 | ❌ 未补 | 多 pipeline 选择（当前只返回 Qwen handler） |
| 4 | 文档：连续短回合可能触发 engine 调度竞态，建议轮间 ≥200ms | ❌ 未修 | 修复调度竞态，移除客户端侧规避 |
| 5 | 音频缓冲溢出即报错并**清空**当前音频 | ⚠️ 行为粗糙 | 环形音频缓冲或背压，而不是整体丢弃 |
| 6 | 历史仅取最近 2 条且文本化 | ⚠️ 设计取舍 | 可配置历史深度 / 多模态历史（依赖 #1 的 KV 能力才划算） |
| 7 | 文档：接口面向 Qwen3-Omni，其他模型 mm processor 参数不保证 | ⚠️ 范围限制 | 模型能力探测或显式 capability 声明 |
| 8 | 输入只支持 JPEG/PNG 帧 + PCM16 音频 | ⚠️ 范围限制 | 如需真视频流（H.264/fMP4 输入、帧时间戳对齐）需扩协议 |

另外两处易混淆点确认：

- `session.prompt_update` 「not supported yet」属于 **`/v1/realtime/video`（视频生成输出）**，不是 `/v1/video/chat/stream`；
- `/v1/realtime/video` 当前仅 text-only 输入、`m4s` 单格式，image/reference 输入未纳入（README 明确）。

### 3.4 建议优先级（Qwen 视角）

1. **`/v1/realtime` 加服务端 VAD + async_chunk 兼容**（3.1 #1/#2）：不改架构即可显著接近 OpenAI 默认体验，也是 3.2.2 路线 B 的第一步；
2. **协议一揽子对齐**（3.1 #3）：事件面 + GA 命名 + session 配置结构一次性升级，让 OpenAI 生态客户端低成本迁移；
3. **VAD 双工体验闭环**（3.2.2 路线 B #2/#3）：连续输入会话 + VAD interrupt 打断，Qwen 全双工体验的落地路径；
4. **`/v1/video/chat/stream` 的 KV 复用与自动触发**（3.3 #1/#2）：这是「视频流理解」从 demo 到实用的关键；
5. **Duplex 栈插件化**（3.2.1 #6/#7）：为下一个 native duplex 模型接入还债；Qwen 原生全双工（路线 A）挂起，等模型侧 duplex 训练版本；
6. 低优先搁置：tools / 图像 / WebRTC / SIP / `client_secrets`（如有浏览器需求先文档化网关参考架构）。
