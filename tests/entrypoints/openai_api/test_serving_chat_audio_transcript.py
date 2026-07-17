# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ChatCompletionAudio.transcript population."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def serving_chat(monkeypatch):
    chat = object.__new__(OmniOpenAIServingChat)
    chat.create_audio = MagicMock(return_value=SimpleNamespace(audio_data="YmFzZTY0YXVkaW8=", media_type="audio/wav"))
    return chat


def _audio_omni_output(text_in_mm: str | None = None):
    mm_output = {"audio": torch.zeros(8, dtype=torch.float32), "sr": 24000}
    if text_in_mm is not None:
        mm_output["transcript"] = text_in_mm
    completion = SimpleNamespace(
        index=0,
        multimodal_output=mm_output,
        finish_reason="stop",
        stop_reason=None,
        token_ids=[],
    )
    request_output = SimpleNamespace(outputs=[completion])
    return SimpleNamespace(request_output=request_output, final_output_type="audio")


class _FakeReasoningParser:
    def extract_reasoning(self, model_output: str, request=None):
        # Mimic <think>...</think>content split used by many parsers.
        marker = "</think>"
        if marker in model_output:
            reasoning, content = model_output.split(marker, 1)
            return reasoning, content
        return model_output, ""


def test_resolve_audio_transcript_prefers_index_map():
    assert (
        OmniOpenAIServingChat._resolve_audio_transcript({0: "hello from thinker"}, 0, {"transcript": "mm"})
        == "hello from thinker"
    )


def test_resolve_audio_transcript_falls_back_to_mm_output():
    assert OmniOpenAIServingChat._resolve_audio_transcript({}, 0, {"transcript": "from mm"}) == "from mm"
    assert OmniOpenAIServingChat._resolve_audio_transcript(None, 0, {"text": "from text"}) == "from text"
    assert OmniOpenAIServingChat._resolve_audio_transcript(None, 0, None) == ""


def test_visible_content_for_transcript_strips_reasoning():
    request = SimpleNamespace()
    raw = "<think>hidden plan</think>Hello world"
    assert OmniOpenAIServingChat._visible_content_for_transcript(raw, request, _FakeReasoningParser()) == "Hello world"
    assert OmniOpenAIServingChat._visible_content_for_transcript(raw, request, None) == raw


def test_transcript_stream_delta_is_incremental():
    assert OmniOpenAIServingChat._transcript_stream_delta("Hello", "") == "Hello"
    assert OmniOpenAIServingChat._transcript_stream_delta("Hello world", "Hello") == " world"
    assert OmniOpenAIServingChat._transcript_stream_delta("Hello world", "Hello world") == ""
    # Non-prefix rewrite sends full text for client replace.
    assert OmniOpenAIServingChat._transcript_stream_delta("Hi", "Hello") == "Hi"


def test_create_audio_choice_fills_transcript(serving_chat):
    request = SimpleNamespace(return_token_ids=False)
    choices = OmniOpenAIServingChat._create_audio_choice(
        serving_chat,
        _audio_omni_output(),
        role="assistant",
        request=request,
        stream=False,
        transcripts={0: "Spoken reply from thinker."},
    )
    assert len(choices) == 1
    assert choices[0].message.audio is not None
    assert choices[0].message.audio.transcript == "Spoken reply from thinker."
    assert choices[0].message.audio.data == "YmFzZTY0YXVkaW8="


def test_create_audio_choice_stream_stable_id_and_incremental_transcript(serving_chat):
    request = SimpleNamespace(return_token_ids=False)
    stream_state: dict = {"ids": {}, "expires_at": {}, "transcripts_sent": {}}

    first = OmniOpenAIServingChat._create_audio_choice(
        serving_chat,
        _audio_omni_output(),
        role="assistant",
        request=request,
        stream=True,
        transcripts={0: "Hello"},
        stream_state=stream_state,
    )
    second = OmniOpenAIServingChat._create_audio_choice(
        serving_chat,
        _audio_omni_output(),
        role="assistant",
        request=request,
        stream=True,
        transcripts={0: "Hello world"},
        stream_state=stream_state,
    )
    third = OmniOpenAIServingChat._create_audio_choice(
        serving_chat,
        _audio_omni_output(),
        role="assistant",
        request=request,
        stream=True,
        transcripts={0: "Hello world"},
        stream_state=stream_state,
    )

    audio1 = getattr(first[0].delta, "audio", None)
    audio2 = getattr(second[0].delta, "audio", None)
    audio3 = getattr(third[0].delta, "audio", None)
    assert isinstance(audio1, dict) and isinstance(audio2, dict) and isinstance(audio3, dict)

    # Stable id / expires_at across chunks.
    assert audio1["id"] == audio2["id"] == audio3["id"]
    assert audio1["expires_at"] == audio2["expires_at"] == audio3["expires_at"]

    # Incremental transcript deltas (clients concatenate).
    assert audio1["transcript"] == "Hello"
    assert audio2["transcript"] == " world"
    assert audio3["transcript"] == ""

    # Legacy content path unchanged.
    assert first[0].delta.content == "YmFzZTY0YXVkaW8="


def test_create_audio_choice_uses_mm_transcript_fallback(serving_chat):
    request = SimpleNamespace(return_token_ids=False)
    choices = OmniOpenAIServingChat._create_audio_choice(
        serving_chat,
        _audio_omni_output(text_in_mm="mm transcript"),
        role="assistant",
        request=request,
        stream=False,
        transcripts=None,
    )
    assert choices[0].message.audio.transcript == "mm transcript"


def test_resolve_prefers_mm_over_empty_prompt_map():
    # Diffusion callers must not inject prompt via transcripts map; mm wins.
    assert (
        OmniOpenAIServingChat._resolve_audio_transcript(
            None,
            0,
            {"transcript": "model transcript"},
        )
        == "model transcript"
    )
