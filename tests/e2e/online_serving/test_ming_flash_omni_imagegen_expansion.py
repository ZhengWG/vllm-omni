# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""L4 functionality tests for Ming-flash-omni-2.0 image generation.

The image-generation flow is a two-stage pipeline:

* Stage 0 (thinker, AR): tokenises the user prompt, expands it with the
  ``<image><imagePatch>*256</image>`` query-token block, and exports the
  thinker's final hidden states as DiT conditioning.
* Stage 1 (imagegen, diffusion): ``MingImagePipeline`` (Qwen2 connector +
  ZImage DiT + VAE) runs the FlowMatchEuler denoise + VAE decode and
  returns a PIL image via the OpenAI ``/v1/chat/completions`` response.

These tests serve as the *L4 functionality* tests required by
``docs/contributing/model/adding_diffusion_model.md``: each test case
exercises a different runtime configuration so that the diffusion-stage
adapter, the cross-stage hidden-state plumbing
(``thinker2imagegen``), and the CFG companion expansion
(``expand_cfg_prompts``) all stay covered as upstream evolves.

Validation is delegated to :func:`assert_diffusion_response`, which
verifies that the response carries a properly shaped image payload at
the requested resolution.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_TEST_CLEAN_GPU_MEMORY", "0")

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import (
    OmniServer,
    OmniServerParams,
    OpenAIClientHandler,
    dummy_messages_from_mix_data,
)
from tests.helpers.media import generate_synthetic_image

pytestmark = [pytest.mark.full_model, pytest.mark.diffusion, pytest.mark.omni]

MODEL = "Jonathan1909/Ming-flash-omni-2.0"

# A two-stage Ming imagegen run is too heavy for L4-class GPUs, so the L4
# functionality matrix here targets H100 (the hardware Ming was developed and
# validated on).  Image gen requires the four-card thinker (TP=4) on cards
# 0..3 plus the diffusion stage on card 3 / overlapping the thinker, so we
# reserve four cards across the matrix.
H100_FOUR_CARDS = hardware_marks(res={"cuda": "H100"}, num_cards=4)

# CI stage config: dummy weights by default; full weights when the test is
# run at run_level={advanced_model, full_model}.  See
# ``tests/helpers/fixtures/runtime.py`` for how the dummy load_format is
# stripped for higher run levels.
IMAGEGEN_CI_CONFIG = str(
    Path(__file__).parent.parent / "stage_configs" / "bailingmm_moe_v2_lite_imagegen_ci.yaml"
)

POSITIVE_PROMPT = "Please draw a cute cat sitting on a windowsill, soft afternoon light."
NEGATIVE_PROMPT = "blurry, low quality, distorted, watermark"


def _get_imagegen_feature_cases() -> list[pytest.param]:
    """L4 feature matrix for Ming-flash-omni-2.0 image generation.

    Each case wires a distinct runtime config so the matrix exercises the
    end-to-end thinker → diffusion path under different feature toggles.
    The list is deliberately small (Ming is a *normal priority* model in
    the L4 matrix; see ``docs/contributing/ci/test_examples/l4_functionality_tests.inc.md``).
    """
    return [
        pytest.param(
            OmniServerParams(
                model=MODEL,
                stage_config_path=IMAGEGEN_CI_CONFIG,
                use_stage_cli=False,
                server_args=[
                    "--trust-remote-code",
                ],
            ),
            id="ming_imagegen_default",
            marks=H100_FOUR_CARDS,
        ),
        pytest.param(
            OmniServerParams(
                model=MODEL,
                stage_config_path=IMAGEGEN_CI_CONFIG,
                use_stage_cli=False,
                server_args=[
                    "--trust-remote-code",
                    "--cache-backend",
                    "cache_dit",
                ],
            ),
            id="ming_imagegen_cache_dit",
            marks=H100_FOUR_CARDS,
        ),
        pytest.param(
            OmniServerParams(
                model=MODEL,
                stage_config_path=IMAGEGEN_CI_CONFIG,
                use_stage_cli=False,
                server_args=[
                    "--trust-remote-code",
                    "--cfg-parallel-size",
                    "2",
                ],
            ),
            id="ming_imagegen_cfg_parallel_2",
            marks=H100_FOUR_CARDS,
        ),
    ]


@pytest.mark.parametrize(
    "omni_server",
    _get_imagegen_feature_cases(),
    indirect=True,
)
def test_ming_imagegen_text2img(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
) -> None:
    """Ming-flash-omni-2.0 text-to-image via OpenAI ``/v1/chat/completions``.

    The diffusion stage runs at a low step count to keep CI cost bounded.
    The default request includes a ``negative_prompt`` so the CFG
    companion expansion (``expand_cfg_prompts``) is also exercised
    end-to-end.
    """
    messages = dummy_messages_from_mix_data(content_text=POSITIVE_PROMPT)

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "modalities": ["image"],
        "extra_body": {
            "height": 512,
            "width": 512,
            "num_inference_steps": 2,
            "negative_prompt": NEGATIVE_PROMPT,
            "cfg_text_scale": 4.0,
            "seed": 42,
        },
    }

    openai_client.send_diffusion_request(request_config)


@pytest.mark.parametrize(
    "omni_server",
    [
        pytest.param(
            OmniServerParams(
                model=MODEL,
                stage_config_path=IMAGEGEN_CI_CONFIG,
                use_stage_cli=False,
                server_args=["--trust-remote-code"],
            ),
            id="ming_imagegen_img2img_default",
            marks=H100_FOUR_CARDS,
        )
    ],
    indirect=True,
)
def test_ming_imagegen_img2img(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
) -> None:
    """Ming-flash-omni-2.0 image edit (img2img) via the chat API.

    Sends a single reference image with an edit instruction; the chat
    endpoint should detect the reference image, prepend the ``<IMAGE>``
    placeholder so the thinker's ref-image prompt replacement still
    fires when the multimodal cache is warm, and route the request
    through ``MingImagePipeline`` in img2img mode.
    """
    image_data_url = (
        f"data:image/jpeg;base64,{generate_synthetic_image(512, 512)['base64']}"
    )

    messages = dummy_messages_from_mix_data(
        image_data_url=image_data_url,
        content_text="Turn this into a watercolour painting.",
    )

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "modalities": ["image"],
        "extra_body": {
            "height": 512,
            "width": 512,
            "num_inference_steps": 2,
            "seed": 42,
        },
    }

    openai_client.send_diffusion_request(request_config)
