# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ming-TTS Stage-0 decode guards: NaN detection without per-step device syncs."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.ming_tts.ming_tts_llm import MingLLMModel
from vllm_omni.model_executor.models.ming_tts.patch_emission import (
    MING_STOP_REASON_CONTINUE,
    MING_TTS_DEBUG_CHECKS_ENV,
    _resolve_ming_stop_decision,
    ming_tts_debug_checks_enabled,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.tts]

PATCH_SIZE = 2
LATENT_DIM = 4
HISTORY_PATCH_SIZE = 8
HIDDEN_SIZE = 6

STOP_DECISION_KWARGS = {
    "step": 5,
    "stop_threshold": 0.5,
    "min_stop_step": 3,
    "min_decode_steps": 0,
    "max_decode_steps": 200,
    "audio_dummy_token_id": 11,
    "text_eos_token_id": 22,
}


class _StubFlowLoss:
    """Stand-in for ``FlowLoss`` that returns a caller-supplied latent patch."""

    def __init__(self, latent: torch.Tensor) -> None:
        self.latent = latent
        self.calls = 0

    def sample(self, **_: object) -> torch.Tensor:
        self.calls += 1
        return self.latent


class _StubDecoder:
    """Minimal ``MingLLMModel`` surface needed by ``_decode_one_step``."""

    _decode_one_step = MingLLMModel._decode_one_step

    def __init__(self, latent: torch.Tensor, stop_logits: torch.Tensor) -> None:
        self.fm_dtype = torch.float32
        self.ming_config = SimpleNamespace(patch_size=PATCH_SIZE, latent_dim=LATENT_DIM)
        self.flowloss = _StubFlowLoss(latent)
        self.linear_proj_audio = lambda patch: patch.new_zeros((patch.shape[0], 1, HIDDEN_SIZE))
        self.stop_head = lambda _hidden: stop_logits

    def _maybe_build_cfm_graph(self, _z_diff_cond: torch.Tensor) -> None:
        return None


def _run_decode_step(latent: torch.Tensor, stop_logits: torch.Tensor | None = None):
    if stop_logits is None:
        stop_logits = torch.tensor([[0.0, 0.0]])
    decoder = _StubDecoder(latent, stop_logits)
    return decoder._decode_one_step(
        hidden_states=torch.zeros((1, HIDDEN_SIZE)),
        latent_history=torch.zeros((1, HISTORY_PATCH_SIZE, LATENT_DIM)),
        cfg_scale=2.0,
        sigma=0.25,
        temperature=0.0,
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, False), ("", False), ("0", False), ("off", False), ("1", True), ("true", True), ("ON", True)],
)
def test_debug_checks_env_parsing(monkeypatch: pytest.MonkeyPatch, value: str | None, expected: bool):
    if value is None:
        monkeypatch.delenv(MING_TTS_DEBUG_CHECKS_ENV, raising=False)
    else:
        monkeypatch.setenv(MING_TTS_DEBUG_CHECKS_ENV, value)

    assert ming_tts_debug_checks_enabled() is expected


def test_decode_step_skips_finite_guards_by_default(monkeypatch: pytest.MonkeyPatch):
    """Serving default must not pay a device sync per guard on every decode step."""
    monkeypatch.delenv(MING_TTS_DEBUG_CHECKS_ENV, raising=False)
    isfinite_calls = 0
    real_isfinite = torch.isfinite

    def counting_isfinite(tensor: torch.Tensor) -> torch.Tensor:
        nonlocal isfinite_calls
        isfinite_calls += 1
        return real_isfinite(tensor)

    monkeypatch.setattr(torch, "isfinite", counting_isfinite)
    _run_decode_step(torch.zeros((1, PATCH_SIZE, LATENT_DIM)))

    assert isfinite_calls == 0


def test_decode_step_tolerates_non_finite_latent_by_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(MING_TTS_DEBUG_CHECKS_ENV, raising=False)
    latent = torch.full((1, PATCH_SIZE, LATENT_DIM), float("nan"))

    sampled, _, new_history, _ = _run_decode_step(latent)

    assert torch.isnan(sampled).all()
    assert torch.isnan(new_history[:, -PATCH_SIZE:, :]).all()


def test_decode_step_rejects_non_finite_latent_when_debugging(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(MING_TTS_DEBUG_CHECKS_ENV, "1")
    latent = torch.full((1, PATCH_SIZE, LATENT_DIM), float("nan"))

    with pytest.raises(RuntimeError, match="Non-finite sampled_token_latent"):
        _run_decode_step(latent)


def test_decode_step_rejects_non_finite_conditioning_when_debugging(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(MING_TTS_DEBUG_CHECKS_ENV, "1")
    decoder = _StubDecoder(torch.zeros((1, PATCH_SIZE, LATENT_DIM)), torch.tensor([[0.0, 0.0]]))

    with pytest.raises(RuntimeError, match="Non-finite z_diff_cond"):
        decoder._decode_one_step(
            hidden_states=torch.full((1, HIDDEN_SIZE), float("inf")),
            latent_history=torch.zeros((1, HISTORY_PATCH_SIZE, LATENT_DIM)),
            cfg_scale=2.0,
            sigma=0.25,
            temperature=0.0,
        )


def test_stop_decision_rejects_non_finite_stop_prob():
    """NaN detection survives with the guards off, via the host-side stop scalar."""
    for stop_prob in (float("nan"), float("inf")):
        with pytest.raises(RuntimeError, match="Non-finite stop_probs"):
            _resolve_ming_stop_decision(stop_prob=stop_prob, **STOP_DECISION_KWARGS)


def test_stop_decision_accepts_finite_stop_prob():
    stop_reason, should_stop, _, _, next_token_id = _resolve_ming_stop_decision(stop_prob=0.1, **STOP_DECISION_KWARGS)

    assert stop_reason == MING_STOP_REASON_CONTINUE
    assert should_stop is False
    assert next_token_id == STOP_DECISION_KWARGS["audio_dummy_token_id"]
