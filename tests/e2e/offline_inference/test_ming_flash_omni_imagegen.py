# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline E2E test for Ming-flash-omni-2.0 image generation.

Validates the end-to-end thinker -> diffusion (``MingImagePipeline``) path
runs under the standard ``Omni`` entrypoint and produces a PIL image at
the requested resolution. Mirrors the test layout used by Bagel's
``test_bagel_text2img.py`` so CI/maintenance touches stay localised.
"""

from __future__ import annotations

import os

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_TEST_CLEAN_GPU_MEMORY", "0")

from pathlib import Path

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner

# The L4 stage matrix targets H100 because the thinker stage is an
# 80-GB-class model. Image generation requires the four-card thinker
# (TP=4) plus an additional diffusion-stage allocation.
HARDWARE_TEST = hardware_test(res={"cuda": "H100"}, num_cards=4)

MODEL = "Jonathan1909/Ming-flash-omni-2.0"

IMAGEGEN_CI_CONFIG = str(
    Path(__file__).parent.parent / "stage_configs" / "bailingmm_moe_v2_lite_imagegen_ci.yaml"
)

POSITIVE_PROMPT = "Please draw a cute cat sitting on a windowsill, soft afternoon light."
NEGATIVE_PROMPT = "blurry, low quality, distorted, watermark"


def _build_imagegen_sampling_params(omni) -> list:
    """Configure sampling params: short denoise + CFG companion via negative prompt.

    ``omni.default_sampling_params_list`` returns one params object per
    stage. The diffusion stage (index 1) accepts ``num_inference_steps``
    and ``extra_args`` for ``cfg_text_scale`` / ``negative_prompt`` on
    Ming. We override only what's needed to keep the test runtime small.
    """
    params_list = list(omni.default_sampling_params_list)
    if len(params_list) >= 2:
        params_list[1].num_inference_steps = 2  # type: ignore[attr-defined]
        params_list[1].extra_args = {  # type: ignore[attr-defined]
            "cfg_text_scale": 4.0,
            "negative_prompt": NEGATIVE_PROMPT,
        }
    return params_list


def _extract_first_image(omni_outputs):
    """Pull the first generated PIL image out of an ``omni.generate`` iterator."""
    for req_output in omni_outputs:
        if images := getattr(req_output, "images", None):
            return images[0]
        request_output = getattr(req_output, "request_output", None)
        if request_output is not None and getattr(request_output, "images", None):
            return request_output.images[0]
    return None


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.omni
@HARDWARE_TEST
def test_ming_text2img(run_level) -> None:
    """End-to-end Ming text-to-image generation.

    Drives the dual-stage Ming pipeline through the public ``Omni``
    entrypoint: AR thinker -> ``thinker2imagegen`` slice ->
    ``MingImagePipeline`` denoise + VAE. Validates that the diffusion
    stage emits an image at the requested resolution.

    At ``run_level=advanced_model``/``full_model`` the CI fixture loads
    real weights; at lower run levels the stage YAML keeps
    ``load_format: dummy`` so the test exercises wiring only.
    """
    with OmniRunner(
        MODEL,
        stage_configs_path=IMAGEGEN_CI_CONFIG,
    ) as runner:
        omni = runner.omni
        params_list = _build_imagegen_sampling_params(omni)

        prompts = [
            {
                "prompt": POSITIVE_PROMPT,
                "modalities": ["image"],
            }
        ]

        omni_outputs = list(omni.generate(prompts=prompts, sampling_params_list=params_list))

        image = _extract_first_image(omni_outputs)
        assert image is not None, "MingImagePipeline did not emit any images"
        # Default Ming img_gen sampling params target 1024x1024; allow either
        # 1024 (real weights) or any positive size when load_format=dummy is
        # in effect.  We assert only invariants that hold across both modes.
        assert image.size[0] > 0 and image.size[1] > 0
        if run_level in {"advanced_model", "full_model"}:
            # Real weights → request matches the model's default native
            # resolution of 1024×1024 (see MingImageGenConfig).
            assert image.size == (1024, 1024), f"Expected 1024x1024, got {image.size}"
