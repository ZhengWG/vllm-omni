# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit test for FA3 scheduler_metadata shape mismatch in Qwen3-Omni talker.

Root cause: get_scheduler_metadata(num_splits=X) produces metadata with a
shape that depends on X. The FA3 fwd kernel internally recomputes the
expected metadata_size using its own num_splits heuristic (when num_splits=0).
If the metadata was built with a different num_splits than what fwd computes,
the shapes mismatch and the kernel crashes with:
    RuntimeError: scheduler_metadata must have shape (metadata_size)

The fix (_nullify_talker_scheduler_metadata) sets scheduler_metadata to None
when talker prefill exceeds max_cudagraph_capture_size, making FA3 compute
metadata internally so it always matches.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import vllm_omni.model_executor.models.qwen3_omni.qwen3_omni as qwen3_omni_mod

pytestmark = [pytest.mark.core_model]


def _make_model(max_cudagraph_capture_size: int):
    from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni import (
        Qwen3OmniMoeForConditionalGeneration,
    )

    model = object.__new__(Qwen3OmniMoeForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            max_cudagraph_capture_size=max_cudagraph_capture_size,
        ),
    )
    return model


def _make_attn_metadata(num_actual_tokens: int, num_layers: int = 2):
    meta = {}
    for i in range(num_layers):
        meta[f"talker.language_model.model.layers.{i}.self_attn.attn"] = SimpleNamespace(
            num_actual_tokens=num_actual_tokens,
            scheduler_metadata=torch.zeros(13, dtype=torch.int32),
        )
    return meta


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fa3_metadata_shape_mismatch_exists():
    """Verify the FA3 metadata shape mismatch bug exists at C kernel level.

    get_scheduler_metadata with different num_splits produces different
    metadata shapes for the same batch, which crashes fwd when mismatched.
    """
    try:
        from vllm.vllm_flash_attn.flash_attn_interface import get_scheduler_metadata
    except ImportError:
        pytest.skip("FA3 not available")

    device = "cuda"
    bs, max_q, max_kv = 2, 460, 6547
    cu_q = torch.tensor([0, 460, 461], dtype=torch.int32, device=device)
    kv = torch.tensor([460, 6547], dtype=torch.int32, device=device)

    sm_ns0 = get_scheduler_metadata(
        batch_size=bs, max_seqlen_q=max_q, max_seqlen_k=max_kv,
        num_heads_q=16, num_heads_kv=2, headdim=128,
        cache_seqlens=kv, cu_seqlens_q=cu_q,
        page_size=16, causal=True, num_splits=0,
    )
    sm_ns1 = get_scheduler_metadata(
        batch_size=bs, max_seqlen_q=max_q, max_seqlen_k=max_kv,
        num_heads_q=16, num_heads_kv=2, headdim=128,
        cache_seqlens=kv, cu_seqlens_q=cu_q,
        page_size=16, causal=True, num_splits=1,
    )
    # Different num_splits can produce different metadata shapes
    assert sm_ns0.shape != sm_ns1.shape, (
        f"Expected different shapes for num_splits=0 vs 1, "
        f"got {sm_ns0.shape} and {sm_ns1.shape}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fa3_fwd_crashes_with_wrong_metadata():
    """Verify FA3 fwd crashes when given metadata from a different num_splits."""
    try:
        from vllm.vllm_flash_attn.flash_attn_interface import (
            flash_attn_varlen_func,
            get_scheduler_metadata,
        )
    except ImportError:
        pytest.skip("FA3 not available")

    device = "cuda"
    dtype = torch.bfloat16
    bs, max_q, max_kv, ps = 2, 460, 6547, 16
    nh_q, nh_kv, hd = 16, 2, 128
    nb = (max_kv + ps - 1) // ps

    cu_q = torch.tensor([0, 460, 461], dtype=torch.int32, device=device)
    kv = torch.tensor([460, 6547], dtype=torch.int32, device=device)
    q = torch.randn(461, nh_q, hd, dtype=dtype, device=device)
    k_cache = torch.randn(nb * bs, ps, nh_kv, hd, dtype=dtype, device=device)
    v_cache = torch.randn(nb * bs, ps, nh_kv, hd, dtype=dtype, device=device)
    bt = torch.arange(nb * bs, dtype=torch.int32, device=device).reshape(bs, -1)

    # Build metadata with num_splits=1 but call fwd with num_splits=0
    sm_wrong = get_scheduler_metadata(
        batch_size=bs, max_seqlen_q=max_q, max_seqlen_k=max_kv,
        num_heads_q=nh_q, num_heads_kv=nh_kv, headdim=hd,
        cache_seqlens=kv, cu_seqlens_q=cu_q,
        page_size=ps, causal=True, num_splits=1,
    )
    with pytest.raises(RuntimeError, match="scheduler_metadata must have shape"):
        flash_attn_varlen_func(
            q=q, k=k_cache, v=v_cache, cu_seqlens_q=cu_q,
            max_seqlen_q=max_q, seqused_k=kv, max_seqlen_k=max_kv,
            causal=True, block_table=bt,
            scheduler_metadata=sm_wrong, num_splits=0, fa_version=3,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fa3_fwd_ok_with_none_metadata():
    """Verify FA3 fwd works when scheduler_metadata=None (our fix path)."""
    try:
        from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func
    except ImportError:
        pytest.skip("FA3 not available")

    device = "cuda"
    dtype = torch.bfloat16
    bs, max_q, max_kv, ps = 2, 460, 6547, 16
    nh_q, nh_kv, hd = 16, 2, 128
    nb = (max_kv + ps - 1) // ps

    cu_q = torch.tensor([0, 460, 461], dtype=torch.int32, device=device)
    kv = torch.tensor([460, 6547], dtype=torch.int32, device=device)
    q = torch.randn(461, nh_q, hd, dtype=dtype, device=device)
    k_cache = torch.randn(nb * bs, ps, nh_kv, hd, dtype=dtype, device=device)
    v_cache = torch.randn(nb * bs, ps, nh_kv, hd, dtype=dtype, device=device)
    bt = torch.arange(nb * bs, dtype=torch.int32, device=device).reshape(bs, -1)

    # None metadata → FA3 computes internally → always consistent
    flash_attn_varlen_func(
        q=q, k=k_cache, v=v_cache, cu_seqlens_q=cu_q,
        max_seqlen_q=max_q, seqused_k=kv, max_seqlen_k=max_kv,
        causal=True, block_table=bt,
        scheduler_metadata=None, num_splits=0, fa_version=3,
    )


def test_nullify_when_over_threshold():
    """Fix should nullify scheduler_metadata when tokens > threshold."""
    model = _make_model(max_cudagraph_capture_size=128)
    attn_metadata = _make_attn_metadata(num_actual_tokens=460)
    ctx = SimpleNamespace(attn_metadata=attn_metadata)

    with patch.object(qwen3_omni_mod, "get_forward_context", return_value=ctx):
        model._nullify_talker_scheduler_metadata()

    for meta in attn_metadata.values():
        assert meta.scheduler_metadata is None


def test_no_nullify_when_under_threshold():
    """Fix should preserve scheduler_metadata when tokens <= threshold."""
    model = _make_model(max_cudagraph_capture_size=128)
    attn_metadata = _make_attn_metadata(num_actual_tokens=64)
    originals = [m.scheduler_metadata for m in attn_metadata.values()]
    ctx = SimpleNamespace(attn_metadata=attn_metadata)

    with patch.object(qwen3_omni_mod, "get_forward_context", return_value=ctx):
        model._nullify_talker_scheduler_metadata()

    for meta, orig in zip(attn_metadata.values(), originals):
        assert meta.scheduler_metadata is orig


def test_skip_when_eager_mode():
    """Fix should be a no-op in eager mode (max_cudagraph_capture_size=0)."""
    model = _make_model(max_cudagraph_capture_size=0)
    attn_metadata = _make_attn_metadata(num_actual_tokens=460)
    ctx = SimpleNamespace(attn_metadata=attn_metadata)

    with patch.object(qwen3_omni_mod, "get_forward_context", return_value=ctx):
        model._nullify_talker_scheduler_metadata()

    for meta in attn_metadata.values():
        assert meta.scheduler_metadata is not None
