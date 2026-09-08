# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.io_support import get_diffusion_output_type
from vllm_omni.entrypoints.openai.utils import (
    image_generation_stage_index,
    is_image_generation_pipeline,
    is_video_generation_pipeline,
    pipeline_supports_image_generations,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_video_pipeline_requires_declared_final_video_stage():
    assert is_video_generation_pipeline(
        [
            SimpleNamespace(
                stage_type="llm",
                final_output=False,
                final_output_type=None,
            ),
            SimpleNamespace(
                stage_type="diffusion",
                final_output=True,
                final_output_type="video",
            ),
        ]
    )


@pytest.mark.parametrize(
    "stage_configs",
    [
        [SimpleNamespace(stage_type="diffusion")],
        [
            SimpleNamespace(
                stage_type="diffusion",
                final_output=True,
                final_output_type="image",
            )
        ],
        [
            {
                "stage_type": "diffusion",
                "final_output": False,
                "final_output_type": "video",
            }
        ],
    ],
)
def test_video_pipeline_rejects_non_video_final_outputs(stage_configs):
    assert not is_video_generation_pipeline(stage_configs)


@pytest.mark.parametrize(
    "model_class_name",
    [
        "LTX2TwoStagePipeline",
        "LTX2DistilledOneStagePipeline",
        "LTX2DistilledTwoStagePipeline",
        "WanDMDPipeline",
        "LingBotWorldCausalDMDPipeline",
        "LongCatVideoAvatarPipeline",
    ],
)
def test_registered_video_aliases_declare_video_output(model_class_name):
    assert get_diffusion_output_type(model_class_name) == "video"


@pytest.mark.parametrize(
    "model_class_name",
    [
        "SanaVideoPipeline",
        "SanaImageToVideoPipeline",
    ],
)
def test_sana_video_pipelines_declare_video_output(model_class_name):
    assert get_diffusion_output_type(model_class_name) == "video"


def test_image_pipeline_accepts_llm_generation_final_image_stage() -> None:
    # MammothModa2 DiT / Dynin token2image: LLM stage, image output.
    stage_configs = [
        SimpleNamespace(stage_type="llm", final_output=False, final_output_type=None),
        SimpleNamespace(stage_type="llm", final_output=True, final_output_type="image"),
    ]
    assert is_image_generation_pipeline(stage_configs)
    assert pipeline_supports_image_generations(stage_configs)
    assert image_generation_stage_index(stage_configs) == 1


def test_image_pipeline_still_accepts_diffusion_typed_stage() -> None:
    stage_configs = [SimpleNamespace(stage_type="diffusion")]
    assert pipeline_supports_image_generations(stage_configs)
    assert image_generation_stage_index(stage_configs) == 0
    assert not is_image_generation_pipeline(stage_configs)


def test_image_pipeline_rejects_text_only_llm_stages() -> None:
    stage_configs = [
        SimpleNamespace(stage_type="llm", final_output=True, final_output_type="text"),
    ]
    assert not is_image_generation_pipeline(stage_configs)
    assert not pipeline_supports_image_generations(stage_configs)
    assert image_generation_stage_index(stage_configs) is None
