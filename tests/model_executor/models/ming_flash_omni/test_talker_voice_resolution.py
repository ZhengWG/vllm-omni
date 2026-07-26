# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ming-flash-omni talker voice resolution and its missing-preset diagnostics."""

from __future__ import annotations

import logging

import pytest

torch = pytest.importorskip("torch")

from vllm_omni.model_executor.models.ming_flash_omni.ming_flash_omni_talker import (  # noqa: E402
    MingFlashOmniTalkerForConditionalGeneration,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _StubPresets:
    def __init__(self, registered: dict | None = None) -> None:
        self.registered = registered or {}

    def __contains__(self, voice_name: str) -> bool:
        return voice_name in self.registered

    def get(self, voice_name: str):
        return self.registered.get(voice_name)


class _StubTalker:
    """Minimal surface `_resolve_voice` touches."""

    _resolve_voice = MingFlashOmniTalkerForConditionalGeneration._resolve_voice

    def __init__(self, presets: _StubPresets) -> None:
        self.voice_presets = presets


def _preset(**overrides):
    base = {
        "prompt_wav_lat": torch.zeros(1, 4, 8),
        "prompt_wav_emb": torch.zeros(1, 4, 8),
        "spk_emb": [torch.zeros(1, 8)],
        "prompt_text": "reference line",
    }
    base.update(overrides)
    return base


def test_registered_preset_is_resolved_and_marked_projected():
    talker = _StubTalker(_StubPresets({"DB30": _preset()}))

    voice = talker._resolve_voice({"voice_name": "DB30"})

    assert voice.already_projected is True
    assert voice.spk_emb is not None
    assert voice.prompt_text == "reference line"


def test_caller_prompt_text_wins_over_the_preset():
    talker = _StubTalker(_StubPresets({"DB30": _preset()}))

    voice = talker._resolve_voice({"voice_name": "DB30", "prompt_text": "caller line"})

    assert voice.prompt_text == "caller line"


def test_unknown_preset_is_reported(caplog: pytest.LogCaptureFixture):
    talker = _StubTalker(_StubPresets({"DB30": _preset()}))

    with caplog.at_level(logging.WARNING):
        voice = talker._resolve_voice({"voice_name": "nope"})

    assert voice.spk_emb is None
    assert voice.already_projected is False
    assert any("nope" in record.getMessage() for record in caplog.records)


def test_empty_registry_is_reported(caplog: pytest.LogCaptureFixture):
    """The manifest is loaded best-effort, so an empty registry is plausible."""
    talker = _StubTalker(_StubPresets({}))

    with caplog.at_level(logging.WARNING):
        talker._resolve_voice({"voice_name": "DB30"})

    assert any("DB30" in record.getMessage() for record in caplog.records)


def test_explicit_speaker_embedding_is_not_reported(caplog: pytest.LogCaptureFixture):
    talker = _StubTalker(_StubPresets({}))
    supplied = torch.zeros(1, 8)

    with caplog.at_level(logging.WARNING):
        voice = talker._resolve_voice({"voice_name": "nope", "spk_emb": supplied})

    assert voice.spk_emb is supplied
    assert voice.already_projected is False
    assert not caplog.records


def test_explicit_prompt_wav_is_not_reported(caplog: pytest.LogCaptureFixture):
    """Zero-shot cloning conditions the voice without a preset."""
    talker = _StubTalker(_StubPresets({}))

    with caplog.at_level(logging.WARNING):
        voice = talker._resolve_voice(
            {"voice_name": "nope", "prompt_wav_emb": torch.zeros(1, 4, 8), "prompt_text": "ref"}
        )

    assert voice.prompt_wav_emb is not None
    assert not caplog.records


def test_no_voice_name_is_not_reported(caplog: pytest.LogCaptureFixture):
    talker = _StubTalker(_StubPresets({}))

    with caplog.at_level(logging.WARNING):
        voice = talker._resolve_voice({})

    assert voice.spk_emb is None
    assert not caplog.records
