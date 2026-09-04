"""Shared dataclasses, stats helpers, and prompt fixtures for the Qwen3-Omni
benchmark scripts.

The serving (``vllm_omni/bench_omni_serve.py``) and HF transformers offline
(``transformers/bench_omni_hf.py``) runners share most of their plumbing:
the same prompt list, the same per-stage default sampling params, and the
same JSON output schema (which is also compatible with the qwen3-tts
plotter at ``benchmarks/qwen3-tts/plot_results.py``). Putting that here
keeps the two runner files focused on their actual differences (SSE
streaming + multimodal inputs vs. ``Qwen3OmniMoeForConditionalGeneration``
offline generation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Prompt fixtures
# ---------------------------------------------------------------------------

PROMPTS: list[str] = [
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

# Per-stage default sampling params; mirror ``vllm_omni/deploy/qwen3_omni_moe.yaml``
# so the bench reproduces the production configuration.
DEFAULT_SAMPLING_PARAMS_LIST: list[dict[str, Any]] = [
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
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RequestResult:
    """Per-request timing record used by both serving and HF runners.

    For the offline HF baseline ``ttft`` and ``ttfp`` are set to ``e2e``
    (no streaming), which keeps the JSON schema identical to the serving
    runner and lets the existing qwen3-tts plotter consume both.
    """

    success: bool = False
    ttft: float = 0.0  # seconds
    ttfp: float = 0.0  # seconds
    e2e: float = 0.0
    audio_bytes: int = 0
    audio_duration: float = 0.0
    rtf: float = 0.0
    text_chars: int = 0
    prompt: str = ""
    error: str = ""


@dataclass
class BenchmarkResult:
    """Aggregated benchmark result; JSON schema matches qwen3-tts."""

    config_name: str = ""
    query_type: str = ""
    modalities: str = ""
    concurrency: int = 0
    num_prompts: int = 0
    completed: int = 0
    failed: int = 0
    duration_s: float = 0.0
    # TTFT (first text chunk) -- not produced by qwen3-tts but consumed by
    # the omni plotter / left at 0.0 for offline runs.
    mean_ttft_ms: float = 0.0
    median_ttft_ms: float = 0.0
    p90_ttft_ms: float = 0.0
    p95_ttft_ms: float = 0.0
    p99_ttft_ms: float = 0.0
    # TTFP (first audio packet)
    mean_ttfp_ms: float = 0.0
    median_ttfp_ms: float = 0.0
    std_ttfp_ms: float = 0.0
    p90_ttfp_ms: float = 0.0
    p95_ttfp_ms: float = 0.0
    p99_ttfp_ms: float = 0.0
    # E2E latency
    mean_e2e_ms: float = 0.0
    median_e2e_ms: float = 0.0
    std_e2e_ms: float = 0.0
    p90_e2e_ms: float = 0.0
    p95_e2e_ms: float = 0.0
    p99_e2e_ms: float = 0.0
    # RTF
    mean_rtf: float = 0.0
    median_rtf: float = 0.0
    std_rtf: float = 0.0
    p99_rtf: float = 0.0
    # Audio stats
    mean_audio_duration_s: float = 0.0
    total_audio_duration_s: float = 0.0
    audio_throughput: float = 0.0
    request_throughput: float = 0.0
    per_request: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _percentiles_ms(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def aggregate_results(
    results: list[RequestResult],
    concurrency: int,
    num_prompts: int,
    duration: float,
    query_type: str = "",
    modalities: str = "",
    config_name: str = "",
) -> BenchmarkResult:
    """Aggregate per-request timings into a ``BenchmarkResult``."""
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    bench = BenchmarkResult(
        config_name=config_name,
        query_type=query_type,
        modalities=modalities,
        concurrency=concurrency,
        num_prompts=num_prompts,
        completed=len(successful),
        failed=len(failed),
        duration_s=duration,
    )

    if not successful:
        return bench

    ttfts_ms = [r.ttft * 1000 for r in successful if r.ttft > 0]
    ttfps_ms = [r.ttfp * 1000 for r in successful if r.ttfp > 0]
    e2es_ms = [r.e2e * 1000 for r in successful]
    rtfs = [r.rtf for r in successful if r.rtf > 0]
    audio_durs = [r.audio_duration for r in successful]

    if s := _percentiles_ms(ttfts_ms):
        bench.mean_ttft_ms = s["mean"]
        bench.median_ttft_ms = s["median"]
        bench.p90_ttft_ms = s["p90"]
        bench.p95_ttft_ms = s["p95"]
        bench.p99_ttft_ms = s["p99"]
    if s := _percentiles_ms(ttfps_ms):
        bench.mean_ttfp_ms = s["mean"]
        bench.median_ttfp_ms = s["median"]
        bench.std_ttfp_ms = s["std"]
        bench.p90_ttfp_ms = s["p90"]
        bench.p95_ttfp_ms = s["p95"]
        bench.p99_ttfp_ms = s["p99"]
    if s := _percentiles_ms(e2es_ms):
        bench.mean_e2e_ms = s["mean"]
        bench.median_e2e_ms = s["median"]
        bench.std_e2e_ms = s["std"]
        bench.p90_e2e_ms = s["p90"]
        bench.p95_e2e_ms = s["p95"]
        bench.p99_e2e_ms = s["p99"]

    if rtfs:
        rtf_arr = np.asarray(rtfs, dtype=float)
        bench.mean_rtf = float(np.mean(rtf_arr))
        bench.median_rtf = float(np.median(rtf_arr))
        bench.std_rtf = float(np.std(rtf_arr))
        bench.p99_rtf = float(np.percentile(rtf_arr, 99))

    if audio_durs:
        bench.mean_audio_duration_s = float(np.mean(audio_durs))
        bench.total_audio_duration_s = float(np.sum(audio_durs))
        if duration > 0:
            bench.audio_throughput = bench.total_audio_duration_s / duration
    if duration > 0:
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
    return bench


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------


def print_summary(bench: BenchmarkResult, header: str = "Benchmark Result") -> None:
    """Print a standardized performance summary box for ``bench``."""
    W = 50
    print("")
    print(f"{'=' * W}")
    print(f"{header:^{W}}")
    print(f"{'=' * W}")
    print(f"{'Query type:':<40}{bench.query_type:<10}")
    print(f"{'Modalities:':<40}{bench.modalities:<10}")
    print(f"{'Successful requests:':<40}{bench.completed:<10}")
    print(f"{'Failed requests:':<40}{bench.failed:<10}")
    print(f"{'Maximum request concurrency:':<40}{bench.concurrency:<10}")
    print(f"{'Benchmark duration (s):':<40}{bench.duration_s:<10.2f}")
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


# ---------------------------------------------------------------------------
# Multimodal user content builders (shared between vllm-omni and HF runners)
# ---------------------------------------------------------------------------


def build_user_content_vllm(query_type: str, prompt: str) -> tuple[list[dict], dict]:
    """Build the OpenAI-format ``content`` list for the vllm-omni serving runner.

    Returns ``(content, mm_processor_kwargs)``.
    """
    mm_processor_kwargs: dict = {}
    if query_type == "text":
        content = [{"type": "text", "text": prompt}]
    elif query_type == "use_audio":
        content = [
            {"type": "audio_url", "audio_url": {"url": DEFAULT_AUDIO_URL}},
            {"type": "text", "text": prompt},
        ]
    elif query_type == "use_image":
        content = [
            {"type": "image_url", "image_url": {"url": DEFAULT_IMAGE_URL}},
            {"type": "text", "text": prompt},
        ]
    elif query_type == "use_video":
        content = [
            {"type": "video_url", "video_url": {"url": DEFAULT_VIDEO_URL}},
            {"type": "text", "text": prompt},
        ]
    else:
        raise ValueError(f"Unsupported query_type: {query_type}")
    return content, mm_processor_kwargs


def build_user_content_hf(query_type: str, prompt: str) -> list[dict]:
    """Build the HF ``conversation[*].content`` list for the HF runner."""
    user_content: list[dict] = []
    if query_type == "use_audio":
        user_content.append({"type": "audio", "audio": DEFAULT_AUDIO_URL})
    elif query_type == "use_image":
        user_content.append({"type": "image", "image": DEFAULT_IMAGE_URL})
    elif query_type == "use_video":
        user_content.append({"type": "video", "video": DEFAULT_VIDEO_URL})
    elif query_type != "text":
        raise ValueError(f"Unsupported query_type: {query_type}")
    user_content.append({"type": "text", "text": prompt})
    return user_content
