# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in E2E accuracy test for LingBot-World (requires GPU + checkpoint).

Run on a CUDA box:

    # 1) sanity + write golden candidate
    LINGBOT_E2E=1 pytest tests/e2e/offline_inference/test_lingbot_world.py -s

    # 2) compare against a reference (.npy array or a video file such as .mp4,
    #    e.g. a previous golden run or official generate.py output on the same
    #    first frame / trajectory / prompt)
    LINGBOT_E2E=1 LINGBOT_REFERENCE=/path/to/reference.mp4 \
        pytest tests/e2e/offline_inference/test_lingbot_world.py -s

Notes on alignment: RNG sequences differ between frameworks, so bit-exact
parity with the official repo is not expected — the acceptance bar is
metric-level (PSNR / per-frame MAE) on the same seed-fixed inputs. Camera
geometry and the DMD sigma schedule are already covered by CPU parity tests
against the official sources.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

MODEL = os.environ.get("LINGBOT_MODEL", "robbyant/lingbot-world-fast-diffusers")
OUTPUT_DIR = Path(os.environ.get("LINGBOT_OUTPUT_DIR", "/tmp/lingbot_e2e"))
# 21 frames = 6 latent frames = 2 chunks: exercises the causal cache handoff
# (chunk 1 sees chunk 0 as history) at minimal cost.
NUM_FRAMES = 21
PSNR_THRESHOLD = float(os.environ.get("LINGBOT_PSNR_THRESHOLD", "20.0"))


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(1.0 / mse)


@pytest.mark.skipif(os.environ.get("LINGBOT_E2E") != "1", reason="opt-in GPU E2E (set LINGBOT_E2E=1)")
def test_lingbot_world_short_generation():
    import torch

    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    first_frame = os.environ.get("LINGBOT_IMAGE")
    assert first_frame, "set LINGBOT_IMAGE to a first-frame image path"

    omni = Omni(model=MODEL, model_class_name="LingBotWorldCausalDMDPipeline")
    prompt_dict = {
        "prompt": os.environ.get("LINGBOT_PROMPT", "A first-person walk forward through the scene."),
        "multi_modal_data": {"image": first_frame},
    }
    action_dir = os.environ.get("LINGBOT_ACTION_DIR")
    if action_dir:
        prompt_dict["multi_modal_data"]["camera"] = action_dir
        extra_args = {}
    else:
        extra_args = {"camera_actions": "w-21"}
    output = omni.generate(
        prompt_dict,
        OmniDiffusionSamplingParams(
            height=480,
            width=832,
            num_frames=NUM_FRAMES,
            generator=torch.Generator("cuda").manual_seed(42),
            extra_args=extra_args,
        ),
    )
    if isinstance(output, list):
        output = output[0]
    video = output.images[0]
    if isinstance(video, dict):  # post-process may wrap as {"video": ...}
        video = video.get("video") or video.get("frames")
    video = np.asarray(video)
    if video.ndim == 5:  # [B, F, H, W, C] -> [F, H, W, C]
        video = video[0]

    # structural sanity
    assert video.shape == (NUM_FRAMES, 480, 832, 3), video.shape
    assert np.isfinite(video).all()
    assert 0.02 < float(video.std()) < 0.6, f"degenerate output, std={video.std():.4f}"
    # temporal continuity: adjacent frames should be similar but not frozen
    frame_deltas = np.abs(np.diff(video, axis=0)).mean(axis=(1, 2, 3))
    assert float(frame_deltas.max()) < 0.25, "temporal discontinuity across chunks"
    assert float(frame_deltas.mean()) > 1e-4, "video is frozen"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / "lingbot_e2e_video.npy", video)
    print(f"\nsaved golden candidate: {OUTPUT_DIR / 'lingbot_e2e_video.npy'}")

    reference = os.environ.get("LINGBOT_REFERENCE")
    if reference:
        if reference.endswith(".npy"):
            ref = np.load(reference)
        else:  # video file (mp4/...): decode to [F, H, W, C] in [0, 1]
            import imageio.v3 as iio

            ref = iio.imread(reference).astype(np.float32) / 255.0
        assert ref.shape == video.shape, f"reference shape {ref.shape} != {video.shape}"
        psnr = _psnr(video, ref)
        mae = float(np.abs(video.astype(np.float64) - ref.astype(np.float64)).mean())
        print(f"vs reference: PSNR={psnr:.2f} dB, MAE={mae:.4f}")
        assert psnr >= PSNR_THRESHOLD, f"PSNR {psnr:.2f} below threshold {PSNR_THRESHOLD}"
