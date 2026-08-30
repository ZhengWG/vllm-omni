# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM 0.28 load_model requires SupportsPP models to expose
``make_empty_intermediate_tensors``. MiMo Audio's inner LLM omitted the bind,
so the stage wrapper's copy raised AttributeError during engine-core startup.

Source-level checks: constructing the real classes needs a full VllmConfig
and registered Qwen2 backbone.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_MIMO_DIR = Path(__file__).resolve().parents[4] / "vllm_omni" / "model_executor" / "models" / "mimo_audio"


def test_mimo_audio_llm_binds_empty_intermediate_tensors_from_backbone():
    src = (_MIMO_DIR / "mimo_audio_llm.py").read_text()
    assert "self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors" in src


def test_mimo_audio_wrapper_copies_hook_from_fused_thinker_talker():
    src = (_MIMO_DIR / "mimo_audio.py").read_text()
    assert "self.fused_thinker_talker.make_empty_intermediate_tensors" in src


def test_mimo_audio_code2wav_binds_empty_intermediate_tensors_hook():
    src = (_MIMO_DIR / "mimo_audio_code2wav.py").read_text()
    assert "self.make_empty_intermediate_tensors" in src
