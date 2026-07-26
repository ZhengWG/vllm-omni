# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Ming-TTS adapter's max_new_tokens floor must match Stage-0's decode window.

Pure-Python validation logic; no model or GPU resources are loaded.
"""

import pytest

from vllm_omni.entrypoints.openai.tts_adapters.ming_tts import MingTTSAdapter
from vllm_omni.model_executor.models.ming_tts.config_ming_tts import KEY_MAX_DECODE_STEPS
from vllm_omni.model_executor.models.ming_tts.constants import STOP_HEAD_MIN_STEPS
from vllm_omni.model_executor.models.ming_tts.patch_emission import _validate_ming_decode_window

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.tts]


def _validate_engine_side(max_decode_steps: int) -> None:
    _validate_ming_decode_window(
        [{KEY_MAX_DECODE_STEPS: max_decode_steps}],
        min_stop_step=STOP_HEAD_MIN_STEPS,
        default_max_decode_steps=200,
    )


def test_adapter_floor_is_accepted_by_stage_0():
    _validate_engine_side(MingTTSAdapter.max_new_tokens_min)


def test_one_below_the_adapter_floor_is_rejected_by_stage_0():
    """If this passes, the adapter floor is higher than it needs to be."""
    with pytest.raises(ValueError, match="min_required_decode_steps"):
        _validate_engine_side(MingTTSAdapter.max_new_tokens_min - 1)


def test_adapter_floor_is_above_the_generic_default():
    """The base adapter allows 1, which Ming's stop-head warm-up cannot serve."""
    assert MingTTSAdapter.max_new_tokens_min > 1


def test_adapter_window_is_not_empty():
    assert MingTTSAdapter.max_new_tokens_min <= MingTTSAdapter.max_new_tokens_max
