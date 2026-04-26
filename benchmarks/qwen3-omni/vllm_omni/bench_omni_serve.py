"""Benchmark client for Qwen3-Omni via /v1/chat/completions endpoint.

Measures TTFT (Time-to-First-Text), TTFP (Time-to-First-Audio-Packet),
E2E latency, and RTF (Real-Time Factor) across configurable concurrency
levels. Saves results as JSON for plotting.

Qwen3-Omni is a 3-stage multimodal pipeline (Thinker -> Talker -> Code2Wav)
that consumes text/audio/image/video inputs and emits text + audio in a
single streaming chat completions response. This script is the analogue of
``benchmarks/qwen3-tts/vllm_omni/bench_tts_serve.py`` for the Omni model.

Usage:
    python bench_omni_serve.py \
        --host 127.0.0.1 --port 8091 \
        --num-prompts 50 \
        --max-concurrency 1 4 10 \
        --query-type text \
        --modalities audio \
        --result-dir results/
"""

import argparse
import asyncio
import base64
import io
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import aiohttp
import numpy as np
import soundfile as sf
from tqdm.asyncio import tqdm

# ---------------------------------------------------------------------------
# Test prompts and multimodal asset URLs
# ---------------------------------------------------------------------------

PROMPTS = [
    "Explain the system architecture for a scalable audio generation pipeline. Answer in 15 words.",
    "Describe vLLM in one short paragraph.",
    "What are the benefits of multi-stage inference pipelines? Answer briefly.",
    "Summarize the importance of streaming in real-time AI applications.",
    "List three reasons why multimodal models matter.",
    "How does a token cache improve LLM serving throughput? Answer in 20 words.",
    "Explain real-time factor in TTS in two sentences.",
    "What is tensor parallelism? Keep the answer concise.",
    "Describe how chunked streaming reduces latency in audio generation.",
    "Summarize the main idea of Qwen3-Omni in one sentence.",
    "Why is GPU memory utilization important for LLM inference?",
    "What does end-to-end latency mean in audio synthesis pipelines?",
]

DEFAULT_AUDIO_URL = "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/mary_had_lamb.ogg"
DEFAULT_IMAGE_URL = "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/cherry_blossom.jpg"
DEFAULT_VIDEO_URL = "https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/sample_demo_1.mp4"

DEFAULT_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating "
    "text and speech."
)

# Default sampling params per stage. These mirror the bundled
# qwen3_omni_moe.yaml deploy file so the bench reproduces the production
# configuration. Only the ``stop_token_ids`` are included to keep
# response sizes bounded; everything else is also defined in the deploy
# yaml and would be applied by the server even if omitted.
DEFAULT_SAMPLING_PARAMS_LIST = [
    {
        "temperature": 0.4,
        "top_p": 0.9,
        "top_k": 1,
        "max_tokens": 2048,
        "seed": 42,
        "repetition_penalty": 1.05,
        "stop_token_ids": [151645],
    },
    {
        "temperature": 0.9,
        "top_k": 50,
        "max_tokens": 4096,
        "seed": 42,
        "detokenize": False,
        "repetition_penalty": 1.05,
        "stop_token_ids": [2150],
    },
    {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "max_tokens": 65536,
        "seed": 42,
        "detokenize": True,
        "repetition_penalty": 1.1,
    },
]

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RequestResult:
    success: bool = False
    ttft: float = 0.0  # Time to first text chunk (seconds)
    ttfp: float = 0.0  # Time to first audio packet (seconds)
    e2e: float = 0.0  # End-to-end latency (seconds)
    audio_bytes: int = 0
    audio_duration: float = 0.0  # Audio duration in seconds (decoded from WAV/PCM)
    rtf: float = 0.0  # e2e / audio_duration
    text_chars: int = 0
    prompt: str = ""
    error: str = ""


@dataclass
class BenchmarkResult:
    config_name: str = ""
    query_type: str = ""
    modalities: str = ""
    concurrency: int = 0
    num_prompts: int = 0
    completed: int = 0
    failed: int = 0
    duration_s: float = 0.0
    # TTFT stats (ms) -- text-only / first text token
    mean_ttft_ms: float = 0.0
    median_ttft_ms: float = 0.0
    p90_ttft_ms: float = 0.0
    p95_ttft_ms: float = 0.0
    p99_ttft_ms: float = 0.0
    # TTFP stats (ms) -- first audio packet
    mean_ttfp_ms: float = 0.0
    median_ttfp_ms: float = 0.0
    std_ttfp_ms: float = 0.0
    p90_ttfp_ms: float = 0.0
    p95_ttfp_ms: float = 0.0
    p99_ttfp_ms: float = 0.0
    # E2E stats (ms)
    mean_e2e_ms: float = 0.0
    median_e2e_ms: float = 0.0
    std_e2e_ms: float = 0.0
    p90_e2e_ms: float = 0.0
    p95_e2e_ms: float = 0.0
    p99_e2e_ms: float = 0.0
    # RTF stats
    mean_rtf: float = 0.0
    median_rtf: float = 0.0
    std_rtf: float = 0.0
    p99_rtf: float = 0.0
    # Audio stats
    mean_audio_duration_s: float = 0.0
    total_audio_duration_s: float = 0.0
    audio_throughput: float = 0.0
    request_throughput: float = 0.0
    # Per-request details
    per_request: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Payload / SSE parsing helpers
# ---------------------------------------------------------------------------


def build_user_content(query_type: str) -> tuple[list[dict], dict]:
    """Build the OpenAI ``content`` list and ``mm_processor_kwargs`` for a query."""
    mm_processor_kwargs: dict = {}
    if query_type == "text":
        content = [
            {
                "type": "text",
                "text": "{prompt}",
            }
        ]
    elif query_type == "use_audio":
        content = [
            {"type": "audio_url", "audio_url": {"url": DEFAULT_AUDIO_URL}},
            {"type": "text", "text": "{prompt}"},
        ]
    elif query_type == "use_image":
        content = [
            {"type": "image_url", "image_url": {"url": DEFAULT_IMAGE_URL}},
            {"type": "text", "text": "{prompt}"},
        ]
    elif query_type == "use_video":
        content = [
            {"type": "video_url", "video_url": {"url": DEFAULT_VIDEO_URL}},
            {"type": "text", "text": "{prompt}"},
        ]
    else:
        raise ValueError(f"Unsupported query_type: {query_type}")
    return content, mm_processor_kwargs


def materialize_content(template: list[dict], prompt: str) -> list[dict]:
    """Replace ``{prompt}`` placeholders inside the user content template."""
    materialized = []
    for part in template:
        new_part = dict(part)
        if new_part.get("type") == "text" and "{prompt}" in new_part.get("text", ""):
            new_part["text"] = new_part["text"].format(prompt=prompt)
        materialized.append(new_part)
    return materialized


def create_payload(
    model: str,
    prompt: str,
    query_type: str,
    modalities_list: list[str] | None,
    speaker: str | None,
    sampling_params_list: list[dict],
    stream: bool = True,
) -> dict:
    user_template, mm_processor_kwargs = build_user_content(query_type)
    user_content = materialize_content(user_template, prompt)

    payload: dict = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": DEFAULT_SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": user_content,
            },
        ],
        "stream": stream,
    }
    if modalities_list is not None:
        payload["modalities"] = modalities_list
    if speaker is not None:
        payload["speaker"] = speaker
    if sampling_params_list is not None:
        payload["sampling_params_list"] = sampling_params_list
    if mm_processor_kwargs:
        payload["mm_processor_kwargs"] = mm_processor_kwargs
    return payload


def decode_audio_bytes(audio_b64: str) -> tuple[int, float]:
    """Decode a base64-encoded audio chunk into (bytes, duration_s).

    The server returns a complete WAV blob (or a chunked sub-blob) for each
    audio delta. ``soundfile`` reads it from memory; if decoding fails (e.g.
    raw PCM stream), we fall back to assuming 24 kHz mono 16-bit samples.
    """
    raw = base64.b64decode(audio_b64)
    try:
        data, sr = sf.read(io.BytesIO(raw))
        n_samples = data.shape[0]
        return len(raw), n_samples / float(sr)
    except Exception:
        # Best-effort fallback for raw PCM streams.
        sample_rate = 24000
        sample_width = 2
        return len(raw), len(raw) / sample_width / sample_rate


# ---------------------------------------------------------------------------
# Single request
# ---------------------------------------------------------------------------


async def send_omni_request(
    session: aiohttp.ClientSession,
    api_url: str,
    model: str,
    prompt: str,
    query_type: str,
    modalities_list: list[str] | None,
    speaker: str | None,
    sampling_params_list: list[dict],
    stream: bool,
    pbar: tqdm | None = None,
) -> RequestResult:
    """Send a streaming chat completion and measure latency metrics."""
    payload = create_payload(
        model=model,
        prompt=prompt,
        query_type=query_type,
        modalities_list=modalities_list,
        speaker=speaker,
        sampling_params_list=sampling_params_list,
        stream=stream,
    )

    result = RequestResult(prompt=prompt)
    st = time.perf_counter()

    try:
        async with session.post(api_url, json=payload) as response:
            if response.status != 200:
                result.error = f"HTTP {response.status}: {await response.text()}"
                result.success = False
                return result

            if not stream:
                body = await response.json()
                result.e2e = time.perf_counter() - st
                # Best-effort metrics extraction for non-streaming mode.
                for choice in body.get("choices", []) or []:
                    msg = choice.get("message", {})
                    content = msg.get("content")
                    if content:
                        result.text_chars += len(content)
                    audio = msg.get("audio")
                    if audio and audio.get("data"):
                        nbytes, dur = decode_audio_bytes(audio["data"])
                        result.audio_bytes += nbytes
                        result.audio_duration += dur
                # No streaming -> TTFT/TTFP collapse to E2E.
                result.ttft = result.e2e
                if result.audio_duration > 0:
                    result.ttfp = result.e2e
                if result.audio_duration > 0:
                    result.rtf = result.e2e / result.audio_duration
                result.success = True
                return result

            async for raw_line in response.content:
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[len("data:") :].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                modality = chunk.get("modality", "text") or "text"
                choices = chunk.get("choices") or []
                for choice in choices:
                    delta = choice.get("delta") or {}
                    content = delta.get("content")
                    if not content:
                        continue
                    if modality == "audio":
                        if result.ttfp == 0.0:
                            result.ttfp = time.perf_counter() - st
                        nbytes, dur = decode_audio_bytes(content)
                        result.audio_bytes += nbytes
                        result.audio_duration += dur
                    else:
                        if result.ttft == 0.0:
                            result.ttft = time.perf_counter() - st
                        result.text_chars += len(content)

            result.e2e = time.perf_counter() - st
            if result.audio_duration > 0:
                result.rtf = result.e2e / result.audio_duration
            result.success = True

    except Exception as e:  # pylint: disable=broad-except
        result.error = str(e)
        result.success = False
        result.e2e = time.perf_counter() - st

    if pbar:
        pbar.update(1)
    return result


# ---------------------------------------------------------------------------
# Concurrency loop
# ---------------------------------------------------------------------------


async def run_benchmark(
    host: str,
    port: int,
    model: str,
    num_prompts: int,
    max_concurrency: int,
    query_type: str,
    modalities_list: list[str] | None,
    speaker: str | None,
    sampling_params_list: list[dict],
    num_warmups: int = 3,
    stream: bool = True,
) -> BenchmarkResult:
    api_url = f"http://{host}:{port}/v1/chat/completions"

    connector = aiohttp.TCPConnector(
        limit=max_concurrency,
        limit_per_host=max_concurrency,
        keepalive_timeout=60,
    )
    session = aiohttp.ClientSession(
        connector=connector,
        timeout=aiohttp.ClientTimeout(total=1800),
    )

    if num_warmups > 0:
        print(f"  Warming up with {num_warmups} requests...")
        warmup_tasks = [
            send_omni_request(
                session,
                api_url,
                model,
                PROMPTS[i % len(PROMPTS)],
                query_type,
                modalities_list,
                speaker,
                sampling_params_list,
                stream,
            )
            for i in range(num_warmups)
        ]
        await asyncio.gather(*warmup_tasks)
        print("  Warmup done.")

    request_prompts = [PROMPTS[i % len(PROMPTS)] for i in range(num_prompts)]

    print(f"  Running {num_prompts} requests with concurrency={max_concurrency}...")
    semaphore = asyncio.Semaphore(max_concurrency)
    pbar = tqdm(total=num_prompts, desc=f"  concurrency={max_concurrency}")

    async def limited(prompt: str) -> RequestResult:
        async with semaphore:
            return await send_omni_request(
                session,
                api_url,
                model,
                prompt,
                query_type,
                modalities_list,
                speaker,
                sampling_params_list,
                stream,
                pbar,
            )

    start_time = time.perf_counter()
    tasks = [asyncio.create_task(limited(p)) for p in request_prompts]
    results: list[RequestResult] = await asyncio.gather(*tasks)
    duration = time.perf_counter() - start_time
    pbar.close()

    await session.close()

    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    bench = BenchmarkResult(
        query_type=query_type,
        modalities=",".join(modalities_list) if modalities_list else "default",
        concurrency=max_concurrency,
        num_prompts=num_prompts,
        completed=len(successful),
        failed=len(failed),
        duration_s=duration,
    )

    if successful:
        ttfts = [r.ttft * 1000 for r in successful if r.ttft > 0]
        ttfps = [r.ttfp * 1000 for r in successful if r.ttfp > 0]
        e2es = [r.e2e * 1000 for r in successful]
        rtfs = [r.rtf for r in successful if r.rtf > 0]
        audio_durs = [r.audio_duration for r in successful]

        if ttfts:
            bench.mean_ttft_ms = float(np.mean(ttfts))
            bench.median_ttft_ms = float(np.median(ttfts))
            bench.p90_ttft_ms = float(np.percentile(ttfts, 90))
            bench.p95_ttft_ms = float(np.percentile(ttfts, 95))
            bench.p99_ttft_ms = float(np.percentile(ttfts, 99))
        if ttfps:
            bench.mean_ttfp_ms = float(np.mean(ttfps))
            bench.median_ttfp_ms = float(np.median(ttfps))
            bench.std_ttfp_ms = float(np.std(ttfps))
            bench.p90_ttfp_ms = float(np.percentile(ttfps, 90))
            bench.p95_ttfp_ms = float(np.percentile(ttfps, 95))
            bench.p99_ttfp_ms = float(np.percentile(ttfps, 99))

        bench.mean_e2e_ms = float(np.mean(e2es))
        bench.median_e2e_ms = float(np.median(e2es))
        bench.std_e2e_ms = float(np.std(e2es))
        bench.p90_e2e_ms = float(np.percentile(e2es, 90))
        bench.p95_e2e_ms = float(np.percentile(e2es, 95))
        bench.p99_e2e_ms = float(np.percentile(e2es, 99))

        if rtfs:
            bench.mean_rtf = float(np.mean(rtfs))
            bench.median_rtf = float(np.median(rtfs))
            bench.std_rtf = float(np.std(rtfs))
            bench.p99_rtf = float(np.percentile(rtfs, 99))

        if audio_durs:
            bench.mean_audio_duration_s = float(np.mean(audio_durs))
            bench.total_audio_duration_s = float(np.sum(audio_durs))
            bench.audio_throughput = bench.total_audio_duration_s / duration
        bench.request_throughput = len(successful) / duration

        bench.per_request = [
            {
                "ttft_ms": r.ttft * 1000,
                "ttfp_ms": r.ttfp * 1000,
                "e2e_ms": r.e2e * 1000,
                "rtf": r.rtf,
                "audio_duration_s": r.audio_duration,
                "text_chars": r.text_chars,
                "prompt": r.prompt,
            }
            for r in successful
        ]

    # Print summary in standardized performance template.
    W = 50
    print("")
    print(f"{'=' * W}")
    print(f"{'Serving Benchmark Result':^{W}}")
    print(f"{'=' * W}")
    print(f"{'Query type:':<40}{query_type:<10}")
    print(f"{'Modalities:':<40}{bench.modalities:<10}")
    print(f"{'Successful requests:':<40}{bench.completed:<10}")
    print(f"{'Failed requests:':<40}{bench.failed:<10}")
    print(f"{'Maximum request concurrency:':<40}{max_concurrency:<10}")
    print(f"{'Benchmark duration (s):':<40}{duration:<10.2f}")
    print(f"{'Request throughput (req/s):':<40}{bench.request_throughput:<10.2f}")
    print(f"{'-' * W}")
    print(f"{'End-to-end Latency':^{W}}")
    print(f"{'-' * W}")
    print(f"{'Mean E2EL (ms):':<40}{bench.mean_e2e_ms:<10.2f}")
    print(f"{'Median E2EL (ms):':<40}{bench.median_e2e_ms:<10.2f}")
    print(f"{'P99 E2EL (ms):':<40}{bench.p99_e2e_ms:<10.2f}")
    print(f"{'=' * W}")
    print(f"{'Audio Result':^{W}}")
    print(f"{'=' * W}")
    print(f"{'Total audio duration generated (s):':<40}{bench.total_audio_duration_s:<10.2f}")
    print(f"{'Audio throughput (audio duration/s):':<40}{bench.audio_throughput:<10.2f}")
    print(f"{'-' * W}")
    print(f"{'Time to First Text Chunk':^{W}}")
    print(f"{'-' * W}")
    print(f"{'Mean TEXT_TTFT (ms):':<40}{bench.mean_ttft_ms:<10.2f}")
    print(f"{'Median TEXT_TTFT (ms):':<40}{bench.median_ttft_ms:<10.2f}")
    print(f"{'P99 TEXT_TTFT (ms):':<40}{bench.p99_ttft_ms:<10.2f}")
    print(f"{'-' * W}")
    print(f"{'Time to First Audio Packet':^{W}}")
    print(f"{'-' * W}")
    print(f"{'Mean AUDIO_TTFP (ms):':<40}{bench.mean_ttfp_ms:<10.2f}")
    print(f"{'Median AUDIO_TTFP (ms):':<40}{bench.median_ttfp_ms:<10.2f}")
    print(f"{'P99 AUDIO_TTFP (ms):':<40}{bench.p99_ttfp_ms:<10.2f}")
    print(f"{'-' * W}")
    print(f"{'Real Time Factor':^{W}}")
    print(f"{'-' * W}")
    print(f"{'Mean AUDIO_RTF:':<40}{bench.mean_rtf:<10.3f}")
    print(f"{'Median AUDIO_RTF:':<40}{bench.median_rtf:<10.3f}")
    print(f"{'P99 AUDIO_RTF:':<40}{bench.p99_rtf:<10.3f}")
    print(f"{'=' * W}")
    print("")

    if failed:
        for r in failed[:3]:
            print(f"  [ERROR] {r.error[:200]}")

    return bench


async def main(args):
    modalities_list = None
    if args.modalities:
        modalities_list = [m.strip() for m in args.modalities.split(",") if m.strip()]

    sampling_params_list = DEFAULT_SAMPLING_PARAMS_LIST
    if args.no_sampling_params:
        sampling_params_list = None

    all_results = []
    for concurrency in args.max_concurrency:
        result = await run_benchmark(
            host=args.host,
            port=args.port,
            model=args.model,
            num_prompts=args.num_prompts,
            max_concurrency=concurrency,
            query_type=args.query_type,
            modalities_list=modalities_list,
            speaker=args.speaker,
            sampling_params_list=sampling_params_list,
            num_warmups=args.num_warmups,
            stream=not args.no_stream,
        )
        result.config_name = args.config_name
        all_results.append(asdict(result))

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = result_dir / f"bench_{args.config_name}_{timestamp}.json"

    with open(result_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {result_file}")
    return all_results


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3-Omni Benchmark Client")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        help="Model name registered with the server.",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=50,
        help="Number of prompts per concurrency level.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        nargs="+",
        default=[1, 4, 10],
        help="Concurrency levels to test.",
    )
    parser.add_argument("--num-warmups", type=int, default=3)
    parser.add_argument(
        "--query-type",
        type=str,
        default="text",
        choices=["text", "use_audio", "use_image", "use_video"],
        help="Type of multimodal input to send.",
    )
    parser.add_argument(
        "--modalities",
        type=str,
        default="audio",
        help=(
            "Comma-separated output modalities filter "
            "(e.g. 'text', 'audio', 'text,audio'). "
            "Default: 'audio' (matches the typical Qwen3-Omni TTS workflow)."
        ),
    )
    parser.add_argument(
        "--speaker",
        type=str,
        default=None,
        help="Optional TTS speaker (e.g. 'chelsie'). When omitted, the server default is used.",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="async_chunk",
        help="Label for this config (used in result filenames).",
    )
    parser.add_argument("--result-dir", type=str, default="results")
    parser.add_argument(
        "--no-sampling-params",
        action="store_true",
        help="Skip sending the bundled per-stage sampling_params_list (use server defaults).",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable SSE streaming (TTFT/TTFP will collapse to E2E latency).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
