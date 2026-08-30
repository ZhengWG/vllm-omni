# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Voice-preset prompt features must be encoded in the caller's dtype."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("soundfile")

from vllm_omni.model_executor.models.ming_flash_omni.voice_presets import (  # noqa: E402
    VoicePresetRegistry,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_PATCH_SIZE = 4
_LATENT_DIM = 8
_HOP = 2


class _StubEncoder:
    hop_size = _HOP
    patch_size = 1


class _StubAudioVAE:
    def __init__(self) -> None:
        self.encoder = _StubEncoder()
        self.seen_dtype: torch.dtype | None = None

    def encode_latent(self, speech: torch.Tensor, lengths: torch.Tensor):
        del lengths
        self.seen_dtype = speech.dtype
        frames = _PATCH_SIZE
        return torch.zeros((1, frames, _LATENT_DIM), dtype=speech.dtype), None


def _make_registry(audio_vae: _StubAudioVAE) -> VoicePresetRegistry:
    return VoicePresetRegistry(
        talker_dir="/nonexistent",
        model_path="/nonexistent",
        download_dir=None,
        audio_vae=audio_vae,
        aggregator=lambda latents: latents,
        spk_head=lambda emb: emb,
        patch_size=_PATCH_SIZE,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_prompt_features_are_encoded_in_the_requested_dtype(dtype: torch.dtype):
    audio_vae = _StubAudioVAE()
    registry = _make_registry(audio_vae)
    speech = torch.zeros((1, _HOP * _PATCH_SIZE * 3), dtype=torch.float32)

    prompt_wav_lat, prompt_wav_emb = registry._build_wav_embeddings(
        "probe", speech, device=torch.device("cpu"), dtype=dtype
    )

    assert audio_vae.seen_dtype == dtype
    assert prompt_wav_lat.dtype == dtype
    assert prompt_wav_emb.dtype == dtype


def test_no_audio_vae_yields_no_prompt_features():
    registry = VoicePresetRegistry(
        talker_dir="/nonexistent",
        model_path="/nonexistent",
        download_dir=None,
        audio_vae=None,
        aggregator=lambda latents: latents,
        spk_head=lambda emb: emb,
        patch_size=_PATCH_SIZE,
    )

    assert registry._build_wav_embeddings(
        "probe", torch.zeros((1, 8)), device=torch.device("cpu"), dtype=torch.float32
    ) == (None, None)
