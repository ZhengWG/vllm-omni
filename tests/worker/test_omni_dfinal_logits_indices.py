# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for D-final: worker-side logits_indices recompute.

Background
==========
qwen3-omni's ``talker_preprocess_prefill`` deliberately drops the ChatML
``system`` segment, so the actual talker hidden_states tensor is shorter than
what the scheduler thinks (``num_scheduled_tokens`` is computed against the
thinker view). The worker indexes the talker hidden_states with
``logits_indices = cumsum(num_scheduled_tokens) - 1``; the resulting indices
exceed ``hidden_states.shape[0]`` once enough concurrent prefills accumulate
the per-request 21-token gap (``num_prefill × 21``). Under multi-replica
benchmark this fires as a ``device-side assert`` at ``hidden_states[logits_indices]``.

D-final fixes this by:
  1. ``OmniGPUModelRunner._preprocess`` collects per-request ``seg_len = min(span_len, req_embeds.shape[0])``
     into ``self._omni_talker_seg_lens`` (the value is already computed, just
     not previously published).
  2. The AR runner (CUDA + NPU) recomputes a *local* ``sampling_logits_indices``
     from ``cumsum(seg_lens) - 1`` immediately before
     ``sample_hidden_states = hidden_states[logits_indices]`` and uses it
     ONLY for that gather. The original ``logits_indices`` (used by attention
     metadata / spec-decode) is left untouched.

The pure-arithmetic recompute helper exercised here lives in
``vllm_omni.worker.dfinal_utils``.
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

import numpy as np
import pytest
import torch

# Import the helper directly from the file to avoid triggering vllm_omni
# package init (which pulls in vllm + transformers and is unnecessary for
# this pure-arithmetic helper).
_DFINAL_UTILS_PATH = Path(__file__).resolve().parents[2] / "vllm_omni" / "worker" / "dfinal_utils.py"
_spec = importlib.util.spec_from_file_location("vllm_omni_dfinal_utils_under_test", _DFINAL_UTILS_PATH)
_dfinal_utils = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_dfinal_utils)
recompute_sampling_logits_indices = _dfinal_utils.recompute_sampling_logits_indices

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _scheduler_logits_indices(num_scheduled_tokens: list[int]) -> torch.Tensor:
    """Mirror gpu_model_runner.py:938 ``cumsum(num_scheduled_tokens) - 1``."""
    return torch.as_tensor(np.cumsum(num_scheduled_tokens) - 1, dtype=torch.long)


# ------------------------------------------------------------------ #
#  Pure-arithmetic recompute helper
# ------------------------------------------------------------------ #


class TestRecomputeArithmetic:
    """Exercise the standalone helper directly (no runner mock required)."""

    def test_returns_none_when_seg_lens_is_none(self):
        scheduler_li = _scheduler_logits_indices([1038, 1038, 1, 1])
        result = recompute_sampling_logits_indices(
            seg_lens=None,
            num_reqs=4,
            hidden_states_rows=4154,
            scheduler_logits_indices=scheduler_li,
        )
        assert result is None, "no seg_lens → caller must keep scheduler view"

    def test_returns_none_when_num_reqs_mismatch(self):
        # Mismatch indicates the runner is in a state where the per-req
        # talker preprocess wasn't run (e.g., non-omni model). Be safe and
        # do not recompute.
        scheduler_li = _scheduler_logits_indices([1038, 1038])
        result = recompute_sampling_logits_indices(
            seg_lens=np.array([1017], dtype=np.int32),
            num_reqs=2,
            hidden_states_rows=2034,
            scheduler_logits_indices=scheduler_li,
        )
        assert result is None

    def test_returns_none_when_seg_lens_sum_mismatches_hidden_states(self, caplog: pytest.LogCaptureFixture):
        """Belt-and-suspenders fallback: if seg_lens.sum() doesn't match
        hidden_states.shape[0], skip recompute and let the original
        scheduler-view indexing happen (RF-3 clamp can still catch it).
        """
        scheduler_li = _scheduler_logits_indices([1038, 1038, 1, 1])
        seg_lens = np.array([1017, 1017, 1, 1], dtype=np.int32)  # sum = 2036
        with caplog.at_level(logging.WARNING, logger=_dfinal_utils.logger.name):
            result = recompute_sampling_logits_indices(
                seg_lens=seg_lens,
                num_reqs=4,
                hidden_states_rows=9999,  # mismatch on purpose
                scheduler_logits_indices=scheduler_li,
            )
        assert result is None
        assert any("sum=2036" in rec.message and "9999" in rec.message for rec in caplog.records), (
            "expected sum-mismatch warning"
        )

    def test_recomputes_indices_for_qwen3_omni_doc_dump125(self):
        """Reproduce DUMP-125 from the post-mortem and validate the fix output.

        From the doc:
          n_hs = 6104     (talker actual)
          li.numel = 8
          li.max = 6229   (scheduler view, OOB)
          total_sched = 6230
          li.list = [1037, 2075, 3113, 4151, 5189, 6227, 6228, 6229]
          gap = 126 = 6 prefill × 21 system-segment tokens
          per-prefill talker seg_len = 1017 = 1038 - 21
        """
        num_scheduled_tokens = [1038] * 6 + [1, 1]  # 6 prefill + 2 decode
        scheduler_li = _scheduler_logits_indices(num_scheduled_tokens)
        assert int(scheduler_li.max().item()) == 6229
        assert scheduler_li.tolist() == [1037, 2075, 3113, 4151, 5189, 6227, 6228, 6229]

        seg_lens = np.array([1017] * 6 + [1, 1], dtype=np.int32)
        assert int(seg_lens.sum()) == 6104

        recomputed = recompute_sampling_logits_indices(
            seg_lens=seg_lens,
            num_reqs=8,
            hidden_states_rows=6104,
            scheduler_logits_indices=scheduler_li,
        )

        assert recomputed is not None
        assert recomputed.dtype == scheduler_li.dtype
        assert recomputed.device == scheduler_li.device
        # cumsum([1017]*6 + [1,1]) - 1 = [1016, 2033, 3050, 4067, 5084, 6101, 6102, 6103]
        expected = [1016, 2033, 3050, 4067, 5084, 6101, 6102, 6103]
        assert recomputed.tolist() == expected
        # Critical: max stays in-bounds for hidden_states[6104]
        assert int(recomputed.max().item()) < 6104

    def test_no_op_when_seg_lens_match_scheduler_view(self):
        """Other models (or omni without system-segment) get seg_len == span_len.

        In that case cumsum(seg_lens) == cumsum(num_scheduled_tokens), so the
        recomputed indices must equal the scheduler view exactly.
        """
        num_scheduled_tokens = [512, 512, 1, 1]
        scheduler_li = _scheduler_logits_indices(num_scheduled_tokens)
        seg_lens = np.array(num_scheduled_tokens, dtype=np.int32)
        recomputed = recompute_sampling_logits_indices(
            seg_lens=seg_lens,
            num_reqs=4,
            hidden_states_rows=int(seg_lens.sum()),
            scheduler_logits_indices=scheduler_li,
        )
        assert recomputed is not None
        assert torch.equal(recomputed, scheduler_li)

    def test_decode_only_batch_no_op(self):
        num_scheduled_tokens = [1, 1, 1, 1, 1, 1, 1, 1]
        scheduler_li = _scheduler_logits_indices(num_scheduled_tokens)
        seg_lens = np.array(num_scheduled_tokens, dtype=np.int32)
        recomputed = recompute_sampling_logits_indices(
            seg_lens=seg_lens,
            num_reqs=8,
            hidden_states_rows=8,
            scheduler_logits_indices=scheduler_li,
        )
        assert recomputed is not None
        assert torch.equal(recomputed, scheduler_li)

    def test_preserves_dtype_and_device(self):
        scheduler_li = torch.as_tensor([0, 1, 2], dtype=torch.int64)
        seg_lens = np.array([1, 1, 1], dtype=np.int32)
        recomputed = recompute_sampling_logits_indices(
            seg_lens=seg_lens,
            num_reqs=3,
            hidden_states_rows=3,
            scheduler_logits_indices=scheduler_li,
        )
        assert recomputed is not None
        assert recomputed.dtype == torch.int64
        assert recomputed.device == scheduler_li.device

    def test_zero_reqs_returns_none(self):
        """Empty batch: nothing to recompute."""
        scheduler_li = torch.as_tensor([], dtype=torch.long)
        result = recompute_sampling_logits_indices(
            seg_lens=np.array([], dtype=np.int32),
            num_reqs=0,
            hidden_states_rows=0,
            scheduler_logits_indices=scheduler_li,
        )
        assert result is None

    def test_seg_lens_with_zero_entry_skips(self):
        """A degenerate per-req seg_len of 0 would emit a -1 index; skip."""
        scheduler_li = _scheduler_logits_indices([1, 1])
        seg_lens = np.array([0, 1], dtype=np.int32)
        result = recompute_sampling_logits_indices(
            seg_lens=seg_lens,
            num_reqs=2,
            hidden_states_rows=1,
            scheduler_logits_indices=scheduler_li,
        )
        # Negative index would be invalid for gather → fall back.
        assert result is None
