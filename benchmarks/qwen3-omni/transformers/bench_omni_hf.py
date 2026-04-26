"""Benchmark Qwen3-Omni using HuggingFace transformers (offline).

Measures E2E latency, RTF, and audio duration for offline (non-serving)
inference with ``Qwen3OmniMoeForConditionalGeneration``. Results are saved
in the same JSON format as ``bench_omni_serve.py`` for unified plotting.

Notes:
    * HF transformers offline inference for Qwen3-Omni currently only
      supports ``batch_size=1`` when audio output is enabled, so this
      script always runs prompts sequentially (concurrency = 1).
    * For text-only requests we call ``model.disable_talker()`` and pass
      ``return_audio=False`` to ``generate()``.
    * For text+audio we leave the talker enabled and ask ``generate()`` to
      return both text tokens and waveform audio.

Usage:
    python bench_omni_hf.py \
        --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
        --query-type text \
        --modalities audio \
        --num-prompts 10 \
        --num-warmups 1 \
        --gpu-device 0 \
        --result-dir results/
"""

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

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


@dataclass
class BenchmarkResult:
    config_name: str = ""
    query_type: str = ""
    modalities: str = ""
    concurrency: int = 1  # HF baseline is always offline / sequential.
    num_prompts: int = 0
    completed: int = 0
    failed: int = 0
    duration_s: float = 0.0
    # TTFT/TTFP collapse to E2E for HF offline (no streaming).
    mean_ttft_ms: float = 0.0
    median_ttft_ms: float = 0.0
    p90_ttft_ms: float = 0.0
    p95_ttft_ms: float = 0.0
    p99_ttft_ms: float = 0.0
    mean_ttfp_ms: float = 0.0
    median_ttfp_ms: float = 0.0
    std_ttfp_ms: float = 0.0
    p90_ttfp_ms: float = 0.0
    p95_ttfp_ms: float = 0.0
    p99_ttfp_ms: float = 0.0
    mean_e2e_ms: float = 0.0
    median_e2e_ms: float = 0.0
    std_e2e_ms: float = 0.0
    p90_e2e_ms: float = 0.0
    p95_e2e_ms: float = 0.0
    p99_e2e_ms: float = 0.0
    mean_rtf: float = 0.0
    median_rtf: float = 0.0
    std_rtf: float = 0.0
    p99_rtf: float = 0.0
    mean_audio_duration_s: float = 0.0
    total_audio_duration_s: float = 0.0
    audio_throughput: float = 0.0
    request_throughput: float = 0.0
    per_request: list = field(default_factory=list)


def build_conversation(prompt: str, query_type: str) -> list[dict]:
    """Build a Qwen3-Omni conversation in the HF chat-template format."""
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

    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": DEFAULT_SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]


def run_one(
    model,
    processor,
    prompt: str,
    query_type: str,
    return_audio: bool,
    speaker: str | None,
    use_audio_in_video: bool,
    max_new_tokens: int,
):
    from qwen_omni_utils import process_mm_info  # local import to keep CLI parsing fast

    conversation = build_conversation(prompt, query_type)
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_audio_in_video)

    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=use_audio_in_video,
    )
    inputs = inputs.to(model.device).to(model.dtype)

    gen_kwargs: dict = {
        "thinker_return_dict_in_generate": True,
        "use_audio_in_video": use_audio_in_video,
        "max_new_tokens": max_new_tokens,
    }
    if return_audio and speaker is not None:
        gen_kwargs["speaker"] = speaker
    if not return_audio:
        gen_kwargs["return_audio"] = False

    text_ids, audio = model.generate(**inputs, **gen_kwargs)

    text_out_list = processor.batch_decode(
        text_ids.sequences[:, inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    text_out = text_out_list[0] if text_out_list else ""

    audio_np: np.ndarray | None = None
    if audio is not None:
        audio_np = audio.reshape(-1).detach().cpu().numpy()
    return text_out, audio_np


def run_benchmark(args):
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

    device = f"cuda:{args.gpu_device}"
    print(f"Loading model: {args.model} on {device}")

    return_audio = args.modalities and "audio" in [m.strip() for m in args.modalities.split(",") if m.strip()]

    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        args.model,
        dtype="auto",
        device_map=device,
        attn_implementation="flash_attention_2" if args.flash_attn else "eager",
    )
    if not return_audio:
        # Disable the talker to free up ~10GB and speed up text-only runs.
        model.disable_talker()
    processor = Qwen3OmniMoeProcessor.from_pretrained(args.model)
    print("Model loaded.")

    sample_rate = args.audio_sample_rate
    use_audio_in_video = args.use_audio_in_video

    audio_dir: Path | None = None
    if args.save_audio and return_audio:
        audio_dir = Path(args.result_dir) / "audio_hf"
        audio_dir.mkdir(parents=True, exist_ok=True)

    if args.num_warmups > 0:
        print(f"Warming up with {args.num_warmups} requests...")
        for i in range(args.num_warmups):
            p = PROMPTS[i % len(PROMPTS)]
            try:
                run_one(
                    model,
                    processor,
                    p,
                    args.query_type,
                    return_audio=return_audio,
                    speaker=args.speaker,
                    use_audio_in_video=use_audio_in_video,
                    max_new_tokens=args.max_new_tokens,
                )
            except Exception as e:  # pylint: disable=broad-except
                print(f"  warmup {i} failed: {e}")
        torch.cuda.synchronize(device)
        print("Warmup done.")

    print(f"Running {args.num_prompts} requests sequentially...")
    e2e_times: list[float] = []
    rtfs: list[float] = []
    audio_durations: list[float] = []
    per_request: list[dict] = []
    failed = 0

    total_start = time.perf_counter()
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(args.num_prompts)]
    for i, prompt in enumerate(prompts):
        try:
            torch.cuda.synchronize(device)
            st = time.perf_counter()

            _text_out, audio_np = run_one(
                model,
                processor,
                prompt,
                args.query_type,
                return_audio=return_audio,
                speaker=args.speaker,
                use_audio_in_video=use_audio_in_video,
                max_new_tokens=args.max_new_tokens,
            )

            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - st

            audio_dur = float(len(audio_np) / sample_rate) if audio_np is not None else 0.0
            rtf = elapsed / audio_dur if audio_dur > 0 else 0.0

            e2e_times.append(elapsed)
            rtfs.append(rtf)
            audio_durations.append(audio_dur)
            per_request.append(
                {
                    "ttft_ms": elapsed * 1000,
                    "ttfp_ms": elapsed * 1000,
                    "e2e_ms": elapsed * 1000,
                    "rtf": rtf,
                    "audio_duration_s": audio_dur,
                    "prompt": prompt,
                }
            )

            if audio_dir is not None and audio_np is not None:
                sf.write(str(audio_dir / f"output_{i:04d}.wav"), audio_np, sample_rate)

            if (i + 1) % 5 == 0 or i == 0:
                print(
                    f"  [{i + 1}/{args.num_prompts}] e2e={elapsed * 1000:.0f}ms  rtf={rtf:.3f}  audio={audio_dur:.2f}s"
                )
        except Exception as e:  # pylint: disable=broad-except
            print(f"  [{i + 1}/{args.num_prompts}] FAILED: {e}")
            failed += 1

    total_duration = time.perf_counter() - total_start
    completed = len(e2e_times)

    result = BenchmarkResult(
        config_name=args.config_name,
        query_type=args.query_type,
        modalities=args.modalities or "default",
        concurrency=1,
        num_prompts=args.num_prompts,
        completed=completed,
        failed=failed,
        duration_s=total_duration,
    )

    if e2e_times:
        e2e_ms = [t * 1000 for t in e2e_times]
        result.mean_e2e_ms = float(np.mean(e2e_ms))
        result.median_e2e_ms = float(np.median(e2e_ms))
        result.std_e2e_ms = float(np.std(e2e_ms))
        result.p90_e2e_ms = float(np.percentile(e2e_ms, 90))
        result.p95_e2e_ms = float(np.percentile(e2e_ms, 95))
        result.p99_e2e_ms = float(np.percentile(e2e_ms, 99))

        # No streaming -> TTFT/TTFP collapse to E2E.
        result.mean_ttft_ms = result.mean_e2e_ms
        result.median_ttft_ms = result.median_e2e_ms
        result.p90_ttft_ms = result.p90_e2e_ms
        result.p95_ttft_ms = result.p95_e2e_ms
        result.p99_ttft_ms = result.p99_e2e_ms
        result.mean_ttfp_ms = result.mean_e2e_ms
        result.median_ttfp_ms = result.median_e2e_ms
        result.std_ttfp_ms = result.std_e2e_ms
        result.p90_ttfp_ms = result.p90_e2e_ms
        result.p95_ttfp_ms = result.p95_e2e_ms
        result.p99_ttfp_ms = result.p99_e2e_ms

        if rtfs:
            result.mean_rtf = float(np.mean(rtfs))
            result.median_rtf = float(np.median(rtfs))
            result.std_rtf = float(np.std(rtfs))
            result.p99_rtf = float(np.percentile(rtfs, 99))

        if audio_durations:
            result.mean_audio_duration_s = float(np.mean(audio_durations))
            result.total_audio_duration_s = float(np.sum(audio_durations))
            result.audio_throughput = result.total_audio_duration_s / total_duration
        result.request_throughput = completed / total_duration
        result.per_request = per_request

    W = 50
    print("")
    print(f"{'=' * W}")
    print(f"{'HF Transformers Benchmark Result':^{W}}")
    print(f"{'=' * W}")
    print(f"{'Query type:':<40}{args.query_type:<10}")
    print(f"{'Modalities:':<40}{result.modalities:<10}")
    print(f"{'Successful requests:':<40}{completed:<10}")
    print(f"{'Failed requests:':<40}{failed:<10}")
    print(f"{'Concurrency:':<40}{1:<10}")
    print(f"{'Benchmark duration (s):':<40}{total_duration:<10.2f}")
    print(f"{'Request throughput (req/s):':<40}{result.request_throughput:<10.2f}")
    print(f"{'-' * W}")
    print(f"{'Mean E2EL (ms):':<40}{result.mean_e2e_ms:<10.2f}")
    print(f"{'Mean AUDIO_TTFP (ms):':<40}{result.mean_ttfp_ms:<10.2f}")
    print(f"{'Mean AUDIO_RTF:':<40}{result.mean_rtf:<10.3f}")
    print(f"{'Audio throughput (audio-s/wall-s):':<40}{result.audio_throughput:<10.2f}")
    print(f"{'=' * W}")
    print("")

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = result_dir / f"bench_{args.config_name}_{timestamp}.json"

    with open(result_file, "w") as f:
        json.dump([asdict(result)], f, indent=2)
    print(f"Results saved to {result_file}")
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3-Omni HF transformers offline benchmark")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        help="HuggingFace model id or local path.",
    )
    parser.add_argument("--num-prompts", type=int, default=10)
    parser.add_argument("--num-warmups", type=int, default=1)
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument(
        "--query-type",
        type=str,
        default="text",
        choices=["text", "use_audio", "use_image", "use_video"],
    )
    parser.add_argument(
        "--modalities",
        type=str,
        default="audio",
        help=(
            "Comma-separated output modalities. If 'audio' is included, "
            "the talker is enabled and ``return_audio=True`` is passed to "
            "``generate()``. Default: 'audio'."
        ),
    )
    parser.add_argument("--speaker", type=str, default="Ethan")
    parser.add_argument(
        "--use-audio-in-video",
        action="store_true",
        help="Whether to feed video soundtrack to the model.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Cap on generated tokens for the thinker stage.",
    )
    parser.add_argument(
        "--audio-sample-rate",
        type=int,
        default=24000,
        help="Sample rate to assume when computing audio duration / writing WAVs.",
    )
    parser.add_argument(
        "--flash-attn",
        action="store_true",
        default=True,
        help="Enable flash_attention_2 (recommended). Pass --no-flash-attn to disable.",
    )
    parser.add_argument("--no-flash-attn", dest="flash_attn", action="store_false")
    parser.add_argument(
        "--config-name",
        type=str,
        default="hf_transformers",
        help="Label for this config (used in result filenames).",
    )
    parser.add_argument("--result-dir", type=str, default="results")
    parser.add_argument("--save-audio", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    # Force eager / V0 vLLM path is irrelevant here; we only need transformers,
    # but we still set spawn for compatibility with multiprocessing libs.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    run_benchmark(args)
