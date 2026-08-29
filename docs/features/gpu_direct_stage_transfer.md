# GPU-Direct Stage Transfer (TorchIpcConnector)

!!! warning "Experimental"

    `TorchIpcConnector` is an opt-in, per-edge transport. `SharedMemoryConnector`
    remains the default for all bundled deploy configs.

`TorchIpcConnector` moves inter-stage payload tensors that the downstream
stage consumes **on GPU** (model embeddings, hidden states) device-to-device
via torch CUDA IPC, instead of the default
serialize → D2H → `/dev/shm` → deserialize → H2D round-trip. Payloads without
CUDA tensors are plain shared-memory payloads, so the connector is
wire-compatible with `SharedMemoryConnector` in both directions and can be
enabled on a single edge.

Configuration reference: [Connector schema](../configuration/stage_configs.md#connector-schema).
Reference profile: `vllm_omni/deploy/qwen3_omni_moe_torch_ipc.yaml`.

## Requirements

| Requirement | Detail |
| --- | --- |
| Same host | Sender and receiver stages must share the machine. |
| Device topology | Edge endpoint devices either coincide (stages sharing one GPU) or have peer access (NVLink / PCIe P2P). The receiver raises an actionable error otherwise. |
| In-process worker | GPU placement disables itself (with a warning log) when the stage runs `world_size > 1` or a multi-process executor; the edge then behaves as plain SHM. |
| Allocator | torch CUDA IPC cannot export from `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` or `backend:cudaMallocAsync` allocations. Export failures degrade to the host-copy path with an error log. |

## What to expect

The mechanism removes the transport segment of the first-packet path for
GPU-consumed tensors. For Qwen3-Omni, the dominant payload is the
thinker→talker prefill handoff (`embed.prefill` + `hidden_states.output`,
≈ 2 × `seq_len` × `hidden` × 2 bytes — tens of MB at chat-sized prompts),
which sits directly on the TTFT/first-audio-packet path. Per-token decode
embeddings ride the same plane at negligible per-packet cost (handles only;
no host synchronization on either side).

Consequently the metrics to judge this feature by are **TTFT and
AUDIO_TTFP (mean and p99)**. End-to-end latency and throughput are
generation-dominated and expected to stay flat; treat them as regression
guardrails, not as the benefit signal.

## Full-stage TTFT A/B benchmark (Qwen3-Omni)

Both arms run the identical 3-stage topology on 2 GPUs; the only difference
is the thinker→talker transport.

Record for every run: GPU model and interconnect (`nvidia-smi topo -m`),
vLLM / vLLM-Omni versions, driver, and the exact deploy YAML.

**Server — arm A (SHM baseline):**

```bash
CUDA_VISIBLE_DEVICES=0,1 vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni \
    --deploy-config vllm_omni/deploy/qwen3_omni_moe.yaml --port 8090
```

**Server — arm B (GPU-direct thinker→talker edge):**

```bash
CUDA_VISIBLE_DEVICES=0,1 vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni \
    --deploy-config vllm_omni/deploy/qwen3_omni_moe_torch_ipc.yaml --port 8090
```

**Client — identical CLI for both arms** (fixed seed so both arms see the
same prompts; sweep input length and concurrency; ≥ 128 prompts per cell and
≥ 3 repetitions per cell — per-run TTFP variance at long inputs is large, so
never compare single small runs):

```bash
for IL in 4000 8000 16000; do
  for C in 1 4 16; do
    vllm bench serve --omni --host 127.0.0.1 --port 8090 \
      --backend openai-chat-omni --endpoint /v1/chat/completions \
      --dataset-name random --random-input-len "$IL" --random-output-len 128 \
      --num-prompts 128 --max-concurrency "$C" --num-warmups 4 --ignore-eos \
      --temperature 0 --seed 42 --extra-body '{}' \
      --percentile-metrics "ttft,e2el,audio_rtf,audio_ttfp" --metric-percentiles 99
  done
done
```

**Verify arm B actually uses the GPU plane** before recording numbers:

1. Startup log contains `TorchIpcConnector initialized`-class messages and
   **no** `gpu_tensor_keys configured but the stage runs out-of-process`
   warning.
2. No `TorchIpcConnector: GPU export failed` errors during the run (these
   indicate silent degradation to the host-copy path, e.g. an incompatible
   allocator config).
3. No `failed to open a CUDA IPC handle` errors (topology problem: missing
   P2P between the edge devices).

### Results template

| input | conc. | metric | SHM (A) | TorchIpc (B) | Δ |
| --- | --- | --- | --- | --- | --- |
| 4000 | 1 | TTFT mean / p99 (ms) | | | |
| 4000 | 1 | AUDIO_TTFP mean / p99 (ms) | | | |
| 4000 | 1 | E2EL mean (ms) *(guardrail)* | | | |
| 4000 | 4 | … | | | |
| 8000 | 4 | … | | | |
| 16000 | 16 | … | | | |

Mechanism-based expectations (pending hardware validation — replace with
measured data): the baseline transport segment (serialize + shm round-trip +
deserialize + H2D) costs roughly 25–50 ms at 4 k input and scales linearly
with input length, so TTFT/TTFP mean should improve by about that amount,
with larger p99 gains under concurrency (the removed serialization is CPU
work that queues on the send thread). E2EL and throughput should be flat;
any regression there is a bug, not a trade-off.

### Interpreting outcomes

- **TTFT/TTFP improve, E2E flat** — expected result; gains should grow with
  `--random-input-len`.
- **All metrics flat** — confirm the verification steps above; if the plane
  is active, the baseline transport segment on your hardware is smaller than
  the model-side TTFT share, and the feature is not worth enabling for this
  workload.
- **Anything regresses** — file a bug with the run logs; the design intends
  strict no-regression (payloads fall back to the SHM format whenever the
  GPU plane cannot be used).

Once measured numbers exist for a maintained hardware tier, wire a
`tests/dfx/perf/tests/test_qwen3_omni_torch_ipc.json` config (mirroring
`test_qwen3_omni_async_chunk.json`) with those numbers as the `baseline`
block so nightly perf CI tracks the delta.
