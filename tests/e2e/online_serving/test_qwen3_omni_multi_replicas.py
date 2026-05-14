"""
Core-model CI guard for Qwen3-Omni multi-replica stage-pool routing on 4 GPUs.
"""

import os

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.media import generate_synthetic_audio, generate_synthetic_image, generate_synthetic_video
from tests.helpers.runtime import OmniResponse, OmniServerParams, dummy_messages_from_mix_data
from tests.helpers.stage_config import get_deploy_config_path

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
MULTI_REPLICA_DEPLOY = get_deploy_config_path("ci/qwen3_omni_moe_multi_replicas_4gpu.yaml")
ROUTE_STRESS_REQUESTS = 6
MIXED_MODAL_REQUESTS = 4
# Regression guard for the conc>=24 crash from the post-mortem: under
# multi-replica the talker stops being the bottleneck so the receiver wakes
# the talker scheduler on a partial embed.prefill payload (root cause #2),
# and the per-batch concurrent prefill count crosses the logits_indices
# OOB threshold (root cause #1). 24 concurrent prefills was the smallest
# bench point that crashed; we use it here as the smoke threshold.
HIGH_CONCURRENCY_REQUESTS = 24

test_params = [
    OmniServerParams(
        model=MODEL,
        stage_config_path=MULTI_REPLICA_DEPLOY,
        server_args=["--disable-log-stats"],
    )
]


def _system_prompt() -> dict[str, object]:
    return {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": (
                    "You are Qwen, a virtual human developed by the Qwen Team, "
                    "Alibaba Group, capable of perceiving auditory and visual inputs, "
                    "as well as generating text and speech."
                ),
            }
        ],
    }


def _text_messages() -> list[dict[str, object]]:
    return dummy_messages_from_mix_data(
        system_prompt=_system_prompt(),
        content_text="What is the capital of China? Answer in one short sentence.",
    )


def _mixed_messages() -> list[dict[str, object]]:
    video_data_url = f"data:video/mp4;base64,{generate_synthetic_video(224, 224, 300)['base64']}"
    image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(224, 224)['base64']}"
    audio_data_url = f"data:audio/wav;base64,{generate_synthetic_audio(5, 1)['base64']}"
    return dummy_messages_from_mix_data(
        system_prompt=_system_prompt(),
        video_data_url=video_data_url,
        image_data_url=image_data_url,
        audio_data_url=audio_data_url,
        content_text="What is recited in the audio? What is in this image? Describe the video briefly.",
    )


def _assert_batch_size(responses: list[OmniResponse], expected: int) -> None:
    assert len(responses) == expected, f"Expected {expected} responses, got {len(responses)}"
    assert all(resp.success for resp in responses), "At least one request failed"
    assert all(resp.e2e_latency is not None and resp.e2e_latency > 0 for resp in responses), "Missing request latency"


def _assert_text_outputs(responses: list[OmniResponse]) -> None:
    assert all(resp.text_content is not None for resp in responses), "Missing text output"
    assert all(resp.audio_bytes is None for resp in responses), "Text-only request unexpectedly produced audio"


def _assert_audio_outputs(responses: list[OmniResponse], *, expect_text: bool) -> None:
    assert all(resp.audio_bytes is not None and len(resp.audio_bytes) > 128 for resp in responses), (
        "Missing audio output"
    )
    if expect_text:
        assert all(resp.text_content is not None for resp in responses), "Missing text output"
    else:
        assert all(not (resp.text_content or "").strip() for resp in responses), (
            "Audio-only request unexpectedly produced text"
        )


@pytest.mark.full_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=4)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_text_only_batch_uses_multi_replica_talker(omni_server, openai_client) -> None:
    request_config = {
        "model": omni_server.model,
        "messages": _text_messages(),
        "stream": False,
        "modalities": ["text"],
    }

    responses = openai_client.send_omni_request(request_config, request_num=ROUTE_STRESS_REQUESTS)
    _assert_batch_size(responses, ROUTE_STRESS_REQUESTS)
    _assert_text_outputs(responses)


@pytest.mark.full_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=4)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_text_to_audio_stream_batch_uses_multi_replica_vocoder(omni_server, openai_client) -> None:
    request_config = {
        "model": omni_server.model,
        "messages": _text_messages(),
        "stream": True,
        "modalities": ["audio"],
    }

    responses = openai_client.send_omni_request(request_config, request_num=ROUTE_STRESS_REQUESTS)
    _assert_batch_size(responses, ROUTE_STRESS_REQUESTS)
    _assert_audio_outputs(responses, expect_text=False)


@pytest.mark.full_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=4)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_mixed_modal_stream_batch_generates_text_and_audio(omni_server, openai_client) -> None:
    request_config = {
        "model": omni_server.model,
        "messages": _mixed_messages(),
        "stream": True,
    }

    responses = openai_client.send_omni_request(request_config, request_num=MIXED_MODAL_REQUESTS)
    _assert_batch_size(responses, MIXED_MODAL_REQUESTS)
    _assert_audio_outputs(responses, expect_text=True)


@pytest.mark.full_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=4)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_high_concurrency_text_to_audio_stream_does_not_crash(omni_server, openai_client) -> None:
    """Regression guard for the multi-replica conc>=24 crash.

    Pre-fix symptoms (see commits ec8d3b0a / 70de2098 / 665c9c37):
      * ``RuntimeError: tensor a (6) vs b (9)`` from
        ``_get_talker_assistant_parts`` slicing an empty assistant segment
        out of a partial ``embed.prefill`` (root cause #2).
      * ``device-side assert`` from ``hidden_states[logits_indices]`` once
        the per-batch ``num_prefill * 21`` system-segment skip drift exceeded
        the talker view of hidden_states (root cause #1).

    Both surface only under multi-replica because single-replica's "talker is
    bottleneck" timing accidentally serialized the offending paths. This test
    drives 24 concurrent text→audio streaming prefills against the
    multi-replica deploy so any future regression that re-opens either gap
    crashes the engine and trips the assertion below.
    """
    request_config = {
        "model": omni_server.model,
        "messages": _text_messages(),
        "stream": True,
        "modalities": ["audio"],
    }

    responses = openai_client.send_omni_request(request_config, request_num=HIGH_CONCURRENCY_REQUESTS)
    _assert_batch_size(responses, HIGH_CONCURRENCY_REQUESTS)
    _assert_audio_outputs(responses, expect_text=False)
