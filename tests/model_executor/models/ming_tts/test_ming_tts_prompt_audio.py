# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ming-TTS reference-audio coercion: channel layout and duration accounting."""

from __future__ import annotations

import pytest
import torch

from vllm_omni.model_executor.models.ming_tts.audio_prep import (
    coerce_prompt_waveform,
    count_prompt_waveform_patches,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.tts]


def test_mono_row_vector_is_unchanged():
    waveform = torch.arange(8, dtype=torch.float32).reshape(1, 8)

    out = coerce_prompt_waveform(waveform)

    assert out.shape == (1, 8)
    torch.testing.assert_close(out, waveform)


def test_1d_waveform_gains_a_channel_dim():
    out = coerce_prompt_waveform(torch.arange(8, dtype=torch.float32))

    assert out.shape == (1, 8)


def test_mono_column_vector_is_transposed_not_truncated():
    """A (samples, 1) column vector is mono; it must keep all its samples."""
    waveform = torch.arange(8, dtype=torch.float32).reshape(8, 1)

    out = coerce_prompt_waveform(waveform)

    assert out.shape == (1, 8)
    torch.testing.assert_close(out, torch.arange(8, dtype=torch.float32).reshape(1, 8))


def test_stereo_keeps_one_channel_instead_of_splicing():
    """Flattening (2, N) would concatenate L and R into one 2N-sample stream."""
    left = torch.zeros(8, dtype=torch.float32)
    right = torch.ones(8, dtype=torch.float32)
    waveform = torch.stack([left, right], dim=0)

    out = coerce_prompt_waveform(waveform)

    assert out.shape == (1, 8)
    torch.testing.assert_close(out, left.reshape(1, 8))


def test_samples_first_stereo_is_recognized_by_shape():
    """soundfile.read yields (samples, channels); it must not be truncated to
    channel-count samples by assuming a channels-first layout."""
    left = torch.arange(100, dtype=torch.float32)
    right = -left
    waveform = torch.stack([left, right], dim=1)  # (100, 2)

    out = coerce_prompt_waveform(waveform)

    assert out.shape == (1, 100)
    torch.testing.assert_close(out, left.reshape(1, 100))


def test_multi_channel_duration_is_not_inflated():
    """Patch accounting is derived from sample count, so channel splicing doubles it."""
    samples = 44100
    mono = torch.zeros((1, samples), dtype=torch.float32)
    stereo = torch.zeros((2, samples), dtype=torch.float32)

    assert count_prompt_waveform_patches(stereo) == count_prompt_waveform_patches(mono)


def test_dtype_is_normalized_to_float32():
    out = coerce_prompt_waveform(torch.zeros((2, 8), dtype=torch.float64))

    assert out.dtype == torch.float32


def test_rank_3_input_is_rejected():
    with pytest.raises(ValueError, match="Unsupported Ming prompt waveform rank"):
        coerce_prompt_waveform(torch.zeros((1, 2, 8)))


def test_list_of_clips_is_concatenated_along_time():
    out = coerce_prompt_waveform([torch.zeros((1, 4)), torch.ones((1, 6))])

    assert out.shape == (1, 10)
