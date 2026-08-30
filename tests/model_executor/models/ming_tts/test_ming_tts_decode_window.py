# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ming-TTS decode-window resolution from the prompt `Duration:` hint."""

from __future__ import annotations

import pytest

from vllm_omni.model_executor.models.ming_tts.config_ming_tts import (
    KEY_MAX_DECODE_STEPS,
    KEY_MIN_DECODE_STEPS,
)
from vllm_omni.model_executor.models.ming_tts.constants import STOP_HEAD_MIN_STEPS
from vllm_omni.model_executor.models.ming_tts.patch_emission import _validate_ming_decode_window
from vllm_omni.model_executor.models.ming_tts.prompt_assembly import (
    estimate_decode_step_window_for_duration,
    parse_duration_seconds,
    resolve_effective_runtime_controls,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.tts]

DURATION_TEXT = "A calm piano loop. Duration: 30s"


def _assert_window_is_servable(controls: dict) -> None:
    """The resolved window must survive Stage-0's own validation."""
    _validate_ming_decode_window(
        [controls],
        min_stop_step=STOP_HEAD_MIN_STEPS,
        default_max_decode_steps=200,
    )


def test_duration_is_parsed_case_insensitively():
    assert parse_duration_seconds("Duration: 30s") == 30.0
    assert parse_duration_seconds("duration:2.5 s") == 2.5
    assert parse_duration_seconds("no hint here") is None


@pytest.mark.parametrize("duration_s", [0.05, 0.3, 0.6, 1.0])
def test_short_duration_hints_still_produce_a_servable_window(duration_s: float):
    """A sub-second Duration hint must generate short audio, not a ValueError."""
    controls = resolve_effective_runtime_controls(
        text=f"Duration: {duration_s}s",
        runtime_controls=None,
    )

    assert controls[KEY_MIN_DECODE_STEPS] <= controls[KEY_MAX_DECODE_STEPS]
    _assert_window_is_servable(controls)


def test_both_bounds_derived_when_caller_supplies_neither():
    expected_min, expected_max = estimate_decode_step_window_for_duration(30.0)

    controls = resolve_effective_runtime_controls(text=DURATION_TEXT, runtime_controls=None)

    assert controls[KEY_MIN_DECODE_STEPS] == expected_min
    assert controls[KEY_MAX_DECODE_STEPS] == expected_max
    _assert_window_is_servable(controls)


def test_explicit_max_still_gets_a_duration_derived_min():
    """max_new_tokens must not silently discard the prompt's duration hint."""
    expected_min, _ = estimate_decode_step_window_for_duration(30.0)

    controls = resolve_effective_runtime_controls(
        text=DURATION_TEXT,
        runtime_controls={KEY_MAX_DECODE_STEPS: 400},
    )

    assert controls[KEY_MAX_DECODE_STEPS] == 400
    assert controls[KEY_MIN_DECODE_STEPS] == expected_min
    _assert_window_is_servable(controls)


def test_explicit_min_still_gets_a_duration_derived_max():
    _, expected_max = estimate_decode_step_window_for_duration(30.0)

    controls = resolve_effective_runtime_controls(
        text=DURATION_TEXT,
        runtime_controls={KEY_MIN_DECODE_STEPS: 7},
    )

    assert controls[KEY_MIN_DECODE_STEPS] == 7
    assert controls[KEY_MAX_DECODE_STEPS] == expected_max
    _assert_window_is_servable(controls)


def test_derived_min_is_clamped_to_a_smaller_explicit_max():
    """A short max_new_tokens must not produce min > max, which Stage-0 rejects."""
    controls = resolve_effective_runtime_controls(
        text=DURATION_TEXT,
        runtime_controls={KEY_MAX_DECODE_STEPS: 8},
    )

    assert controls[KEY_MAX_DECODE_STEPS] == 8
    assert controls[KEY_MIN_DECODE_STEPS] <= 8
    _assert_window_is_servable(controls)


def test_derived_max_is_raised_to_a_larger_explicit_min():
    controls = resolve_effective_runtime_controls(
        text="Duration: 1s",
        runtime_controls={KEY_MIN_DECODE_STEPS: 120},
    )

    assert controls[KEY_MIN_DECODE_STEPS] == 120
    assert controls[KEY_MAX_DECODE_STEPS] >= 120
    _assert_window_is_servable(controls)


def test_both_explicit_bounds_are_left_alone():
    controls = resolve_effective_runtime_controls(
        text=DURATION_TEXT,
        runtime_controls={KEY_MIN_DECODE_STEPS: 11, KEY_MAX_DECODE_STEPS: 22},
    )

    assert controls == {KEY_MIN_DECODE_STEPS: 11, KEY_MAX_DECODE_STEPS: 22}


def test_no_duration_hint_leaves_controls_untouched():
    controls = resolve_effective_runtime_controls(
        text="just some text",
        runtime_controls={KEY_MAX_DECODE_STEPS: 64},
    )

    assert controls == {KEY_MAX_DECODE_STEPS: 64}


def test_caller_mapping_is_not_mutated():
    supplied = {KEY_MAX_DECODE_STEPS: 400}

    resolve_effective_runtime_controls(text=DURATION_TEXT, runtime_controls=supplied)

    assert supplied == {KEY_MAX_DECODE_STEPS: 400}
