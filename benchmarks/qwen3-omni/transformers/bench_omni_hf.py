"""Benchmark Qwen3-Omni using HuggingFace transformers (offline).

Measures E2E latency, RTF, and audio duration for offline (non-serving)
inference with ``Qwen3OmniMoeForConditionalGeneration``. Results are saved
in the same JSON schema as ``bench_omni_serve.py`` (and as the qwen3-tts
serving runner) so the qwen3-tts plotter can compare them.

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
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from _common import (  # noqa: E402
    DEFAULT_SYSTEM_PROMPT,
    PROMPTS,
    RequestResult,
    aggregate_results,
    build_user_content_hf,
    print_summary,
)


def _build_conversation(prompt: str, query_type: str) -> list[dict]:
    return [
        {"role": "system", "content": [{"type": "text", "text": DEFAULT_SYSTEM_PROMPT}]},
        {"role": "user", "content": build_user_content_hf(query_type, prompt)},
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

    conversation = _build_conversation(prompt, query_type)
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

    return_audio = bool(args.modalities) and "audio" in [m.strip() for m in args.modalities.split(",") if m.strip()]

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
            try:
                run_one(
                    model,
                    processor,
                    PROMPTS[i % len(PROMPTS)],
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
    results: list[RequestResult] = []

    total_start = time.perf_counter()
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(args.num_prompts)]
    for i, prompt in enumerate(prompts):
        rr = RequestResult(prompt=prompt)
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
            # No streaming -> TTFT/TTFP collapse to E2E.
            rr.success = True
            rr.e2e = elapsed
            rr.ttft = elapsed
            rr.ttfp = elapsed if audio_dur > 0 else 0.0
            rr.audio_duration = audio_dur
            rr.rtf = elapsed / audio_dur if audio_dur > 0 else 0.0

            if audio_dir is not None and audio_np is not None:
                sf.write(str(audio_dir / f"output_{i:04d}.wav"), audio_np, sample_rate)

            if (i + 1) % 5 == 0 or i == 0:
                print(
                    f"  [{i + 1}/{args.num_prompts}] e2e={elapsed * 1000:.0f}ms  "
                    f"rtf={rr.rtf:.3f}  audio={audio_dur:.2f}s"
                )
        except Exception as e:  # pylint: disable=broad-except
            rr.success = False
            rr.error = str(e)
            print(f"  [{i + 1}/{args.num_prompts}] FAILED: {e}")
        results.append(rr)

    total_duration = time.perf_counter() - total_start

    bench = aggregate_results(
        results,
        concurrency=1,
        num_prompts=args.num_prompts,
        duration=total_duration,
        query_type=args.query_type,
        modalities=args.modalities or "default",
        config_name=args.config_name,
    )
    print_summary(bench, header="HF Transformers Benchmark Result")

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = result_dir / f"bench_{args.config_name}_{timestamp}.json"

    with open(result_file, "w") as f:
        json.dump([asdict(bench)], f, indent=2)
    print(f"Results saved to {result_file}")
    return bench


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3-Omni HF transformers offline benchmark")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
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
    parser.add_argument("--use-audio-in-video", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--audio-sample-rate", type=int, default=24000)
    parser.add_argument(
        "--flash-attn",
        action="store_true",
        default=True,
        help="Enable flash_attention_2 (recommended). Pass --no-flash-attn to disable.",
    )
    parser.add_argument("--no-flash-attn", dest="flash_attn", action="store_false")
    parser.add_argument("--config-name", type=str, default="hf_transformers")
    parser.add_argument("--result-dir", type=str, default="results")
    parser.add_argument("--save-audio", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    run_benchmark(args)
