# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The CFM timestep schedule is constant, so the AR loop may cache it."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_omni.model_executor.models.common.ming.fm import apply_sway_sampling, integrate_cfm_steps
from vllm_omni.model_executor.models.ming_flash_omni.talker_module import (
    MingAudioGenerator,
    get_epss_timesteps,
)

torch = pytest.importorskip("torch")

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_STEPS = 10
_PATCH_SIZE = 4
_LATENT_DIM = 8


def _make_generator(steps: int = _STEPS) -> MingAudioGenerator:
    return MingAudioGenerator(
        config=SimpleNamespace(steps=steps, patch_size=_PATCH_SIZE),
        llm_config=SimpleNamespace(),
        model=None,
        cfm=None,
        aggregator=None,
        stop_head=None,
        audio_vae=None,
        patch_size=_PATCH_SIZE,
        his_patch_size=_PATCH_SIZE * 2,
        latent_dim=_LATENT_DIM,
        cfg_strength=2.0,
        use_cuda_graphs=False,
    )


def test_epss_timesteps_are_deterministic():
    device = torch.device("cpu")
    first = get_epss_timesteps(_STEPS, device=device, dtype=torch.float32)
    second = get_epss_timesteps(_STEPS, device=device, dtype=torch.float32)

    assert torch.equal(first, second)


def test_generator_reuses_one_schedule_tensor():
    generator = _make_generator()
    device = torch.device("cpu")

    first = generator._epss_timesteps(device, torch.float32)
    second = generator._epss_timesteps(device, torch.float32)

    assert first is second
    assert torch.equal(first, get_epss_timesteps(_STEPS, device=device, dtype=torch.float32))


def test_schedule_cache_is_keyed_by_dtype():
    generator = _make_generator()
    device = torch.device("cpu")

    as_f32 = generator._epss_timesteps(device, torch.float32)
    as_f64 = generator._epss_timesteps(device, torch.float64)

    assert as_f32 is not as_f64
    assert as_f32.dtype == torch.float32
    assert as_f64.dtype == torch.float64


def test_sway_sampling_does_not_mutate_the_schedule():
    """Caching (and reusing a graph placeholder) is only safe if t is read-only."""
    t = get_epss_timesteps(_STEPS, device=torch.device("cpu"), dtype=torch.float32)
    snapshot = t.clone()

    adjusted = apply_sway_sampling(t, -1.0)

    assert adjusted is not t
    assert torch.equal(t, snapshot)


def test_cfm_integration_does_not_mutate_the_schedule():
    t = get_epss_timesteps(_STEPS, device=torch.device("cpu"), dtype=torch.float32)
    snapshot = t.clone()
    y0 = torch.zeros((1, _PATCH_SIZE, _LATENT_DIM))
    sde_args = torch.tensor([2.0, 0.25, 0.0])
    sde_rnd = torch.zeros((_STEPS, 1, _PATCH_SIZE, _LATENT_DIM))

    integrate_cfm_steps(lambda _t, x: torch.zeros_like(x), y0, t, sde_args, sde_rnd, _STEPS)

    assert torch.equal(t, snapshot)
