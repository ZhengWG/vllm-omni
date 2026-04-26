# Qwen3-Omni Benchmark

Performance benchmarks for the Qwen3-Omni multi-stage pipeline (Thinker -> Talker -> Code2Wav). The bench compares vLLM-Omni online serving (`/v1/chat/completions` with streaming) against HuggingFace Transformers offline inference and reports the same latency / throughput metrics used by the `Qwen3-TTS` benchmark.

## Prerequisites

```bash
pip install matplotlib aiohttp soundfile numpy tqdm

# Required only for the HF baseline:
pip install -U "git+https://github.com/huggingface/transformers" accelerate qwen-omni-utils
pip install -U flash-attn --no-build-isolation
```

The default deploy YAML (`vllm_omni/deploy/qwen3_omni_moe.yaml`) targets a 2-GPU layout: stage 0 (Thinker) on `cuda:0`, stages 1+2 (Talker + Code2Wav) on `cuda:1`. Adjust `GPU_DEVICES` / `--deploy-config` for other topologies.

## Quick Start

Run the full benchmark (vllm-omni + HF baseline) with a single command:

```bash
cd benchmarks/qwen3-omni
bash run_benchmark.sh
```

Results (per-config JSON + comparison PNG) are saved to `results/`.

### Common options

```bash
# Only vllm-omni (skip HF baseline)
bash run_benchmark.sh --vllm-only

# Only HF baseline (skip server)
bash run_benchmark.sh --hf-only

# Custom GPU layout, prompts, concurrency
GPU_DEVICES=0,1 NUM_PROMPTS=20 CONCURRENCY="1 4" bash run_benchmark.sh

# Multimodal input (audio understanding -> audio reply)
QUERY_TYPE=use_audio MODALITIES=text,audio bash run_benchmark.sh --vllm-only

# Use a specific TTS speaker (e.g. chelsie / ethan / aiden)
SPEAKER=chelsie bash run_benchmark.sh --vllm-only

# Override deploy config and/or stage budgets
DEPLOY_CONFIG=/path/to/custom.yaml \
STAGE_OVERRIDES='{"0":{"gpu_memory_utilization":0.8}}' \
    bash run_benchmark.sh --vllm-only
```

## Manual Steps

### 1) Start the vLLM-Omni server

```bash
CUDA_VISIBLE_DEVICES=0,1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
python -m vllm_omni.entrypoints.cli.main serve \
    "Qwen/Qwen3-Omni-30B-A3B-Instruct" \
    --omni --host 127.0.0.1 --port 8091 \
    --deploy-config vllm_omni/deploy/qwen3_omni_moe.yaml \
    --trust-remote-code
```

The bundled `qwen3_omni_moe.yaml` already enables `async_chunk: true`, so downstream stages start before the Thinker finishes.

### 2) Run the online serving benchmark

```bash
python benchmarks/qwen3-omni/vllm_omni/bench_omni_serve.py \
    --host 127.0.0.1 --port 8091 \
    --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --num-prompts 20 \
    --max-concurrency 1 4 10 \
    --query-type text \
    --modalities audio \
    --config-name async_chunk \
    --result-dir results/
```

The client uses streaming SSE on `/v1/chat/completions` and watches per-chunk `modality` to compute:

- **TTFT** (Time-to-First-Text): latency from request start to the first text delta
- **TTFP** (Time-to-First-Audio-Packet): latency from request start to the first audio delta (per stage-1/stage-2 emit)
- **E2E**: total wall time from request to the closing `[DONE]` event
- **RTF**: `e2e / total audio duration`
- **Audio throughput**: total generated audio seconds per wall-clock second

Supported `--query-type` values: `text`, `use_audio`, `use_image`, `use_video`. Multimodal queries reuse the same default asset URLs as the example clients in `examples/online_serving/qwen3_omni/`.

### 3) Run the HF transformers baseline

```bash
python benchmarks/qwen3-omni/transformers/bench_omni_hf.py \
    --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --num-prompts 5 \
    --num-warmups 1 \
    --gpu-device 0 \
    --query-type text \
    --modalities audio \
    --result-dir results/
```

The HF baseline runs prompts sequentially with `Qwen3OmniMoeForConditionalGeneration` (the only `batch_size` it supports for audio output is 1). When `--modalities` does not include `audio`, the talker is disabled (`model.disable_talker()`) and `return_audio=False` is passed to `generate()` for a much faster text-only run.

### 4) Generate comparison plots

The output JSON schema is identical to the qwen3-tts bench, so we reuse its plotter directly:

```bash
python benchmarks/qwen3-tts/plot_results.py \
    --results benchmarks/qwen3-omni/results/bench_async_chunk_*.json \
              benchmarks/qwen3-omni/results/bench_hf_transformers_*.json \
    --labels "vllm-omni" "hf_transformers" \
    --title "Qwen3-Omni" \
    --output benchmarks/qwen3-omni/results/qwen3_omni_benchmark.png
```

`plot_results.py` also prints a Markdown comparison table on stdout.

## Metrics

| Metric | Meaning |
| --- | --- |
| **TTFT** | Time from request to first **text** chunk (Thinker first token visible) |
| **TTFP** | Time from request to first **audio** packet (first Code2Wav output streamed back) |
| **E2E** | Total wall-clock latency to the last chunk |
| **RTF** | `E2E / audio_duration`; `< 1.0` means faster-than-real-time synthesis |
| **Audio throughput** | Total generated audio seconds per wall-clock second |
| **Request throughput** | Successful requests per wall-clock second |

For text-only modalities (`MODALITIES=text`) the audio-related fields stay at zero; for audio-only modalities (`MODALITIES=audio`) the TTFT/TTFP both fire on the first chunk for the active modality.

## Notes

- vLLM-Omni's `/v1/chat/completions` for Qwen3-Omni emits each delta with a `modality` field (`text` or `audio`); the bench separates them to compute TTFT/TTFP independently.
- Each audio delta is a base64-encoded WAV blob; the bench decodes it with `soundfile` to get the true audio duration. If decoding fails it falls back to a 24 kHz / 16-bit / mono PCM assumption.
- HF transformers offline currently only supports single-batch audio generation, so its `concurrency` is fixed to 1 in the result JSON.
- Concurrency settings of `1 4 10` mirror the design doc baselines in `docs/design/qwen3_omni_tts_performance_optimization.md`.
- The shared dataclasses, prompt fixtures, default sampling-params, stats aggregation, and summary printing all live in `benchmarks/qwen3-omni/_common.py`. The plotting code is reused from `benchmarks/qwen3-tts/plot_results.py` since the JSON schemas match.
