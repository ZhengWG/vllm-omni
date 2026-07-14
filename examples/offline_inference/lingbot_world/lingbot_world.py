# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LingBot-World interactive world-model demo: image + prompt + keyboard actions -> video.

The LingBot-World checkpoint is licensed CC BY-NC-SA (non-commercial); this
integration code stays Apache-2.0.

Example:
    python lingbot_world.py \
        --model robbyant/lingbot-world-fast-diffusers \
        --image scene.png \
        --prompt "A first-person walk through a sunlit forest" \
        --actions "w-36,lw-24,w-21" \
        --output lingbot_world.mp4

Actions are per video frame (interpolated to the latent grid internally):
w/a/s/d move, i/k pitch (2°), j/l yaw (2°), none = idle; "w-36" holds w for
36 frames. 81 output frames = 21 latent frames = 7 chunks.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from diffusers.utils import export_to_video

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="robbyant/lingbot-world-fast-diffusers")
    parser.add_argument("--image", required=True, help="first-frame image path")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--actions", required=True, help="e.g. 'w-36,lw-24,w-21' (one key set per video frame)")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=81, help="(n-1) %% 4 == 0 and latent frames %% 3 == 0")
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--output", default="lingbot_world.mp4")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    omni = Omni(
        model=args.model,
        model_class_name="LingBotWorldCausalDMDPipeline",
        tensor_parallel_size=args.tensor_parallel_size,
    )
    prompt_dict = {
        "prompt": args.prompt,
        "multi_modal_data": {"image": args.image},
    }
    sampling_params = OmniDiffusionSamplingParams(
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        generator=torch.Generator("cuda").manual_seed(args.seed),
        extra_args={"camera_actions": args.actions},
    )

    start = time.perf_counter()
    output = omni.generate(prompt_dict, sampling_params)
    print(f"Generated in {time.perf_counter() - start:.1f}s")

    if isinstance(output, list):
        output = output[0]
    assert isinstance(output, OmniRequestOutput), f"unexpected output type {type(output)}"
    video = output.images[0]["video"]  # np.ndarray [B, F, H, W, C] in [0, 1]
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(list(video[0]), str(out_path), fps=args.fps)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
