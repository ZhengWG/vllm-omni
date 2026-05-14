# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""D-final: worker-side helper to recompute sampling ``logits_indices`` from
per-request talker segment lengths.

Background
----------
The Omni AR worker computes the sampler gather indices as
``cumsum(scheduler_output.num_scheduled_tokens) - 1`` (see
``vllm.v1.worker.gpu_model_runner``). For models like qwen3-omni whose talker
preprocess deliberately drops the ChatML ``system`` segment, the talker output
is shorter than the scheduler view by ~21 rows per prefill. The mismatch
accumulates linearly with the number of concurrent prefills in a batch and
eventually makes ``hidden_states[logits_indices]`` index out of bounds → a
``device-side assert`` at conc≥24 in multi-replica nightly perf runs.

D-final closes the loop by recomputing the gather indices from the
*actual* per-request talker segment lengths (collected during
``OmniGPUModelRunner._preprocess``) right before the sampler gather. The
function in this module is a deliberately tiny pure-arithmetic helper so it
can be unit-tested without spinning up a runner.

Design constraints
------------------
* This recompute is used **only** for ``sample_hidden_states = hidden_states[…]``;
  it must NOT replace the ``logits_indices`` passed to attention metadata or
  spec-decode (those still need the scheduler view).
* The helper is conservative: any inconsistency (None seg_lens, length
  mismatch, sum mismatch with ``hidden_states.shape[0]``, degenerate zero
  entries) returns ``None`` and lets the caller fall back to the scheduler
  view. RF-3 (clamp) downstream remains a defense-in-depth net.
* For non-omni models or omni models whose talker does not skip any segment,
  ``seg_len == span_len`` for every request → cumsum is identical to the
  scheduler view → caller gets a tensor numerically equal to the original.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def recompute_sampling_logits_indices(
    seg_lens: np.ndarray | None,
    num_reqs: int,
    hidden_states_rows: int,
    scheduler_logits_indices: torch.Tensor,
) -> torch.Tensor | None:
    """Return per-request gather indices computed from talker seg_lens.

    Args:
        seg_lens: Per-request talker segment lengths (1D, dtype int).
            ``None`` when the runner did not collect them (e.g., non-omni
            model or the ``has_preprocess`` path was skipped).
        num_reqs: Current input batch size, for cross-checking ``len(seg_lens)``.
        hidden_states_rows: ``hidden_states.shape[0]`` from the model forward
            output. Must equal ``seg_lens.sum()`` for the recompute to be safe.
        scheduler_logits_indices: The original gather tensor; only consulted
            for dtype/device. Returned unchanged on fallback.

    Returns:
        A new ``torch.Tensor`` of shape ``[num_reqs]`` with the corrected
        sampler gather indices, or ``None`` if the recompute is not safe and
        the caller should keep the scheduler view.
    """
    if seg_lens is None:
        return None
    if num_reqs <= 0:
        return None
    if len(seg_lens) != num_reqs:
        logger.debug(
            "D-final: seg_lens length=%d != num_reqs=%d; skipping recompute",
            len(seg_lens),
            num_reqs,
        )
        return None
    if (seg_lens <= 0).any():
        # A zero-length segment would emit a -1 gather index; bail out and
        # let the original scheduler-view path (with RF-3 clamp) handle it.
        logger.debug(
            "D-final: seg_lens contains non-positive entries (%s); skipping recompute",
            seg_lens.tolist(),
        )
        return None

    expected_total = int(seg_lens.sum())
    if expected_total != int(hidden_states_rows):
        logger.warning(
            "D-final: talker seg_lens sum=%d != hidden_states.shape[0]=%d; "
            "falling back to scheduler-view logits_indices (RF-3 clamp may apply).",
            expected_total,
            int(hidden_states_rows),
        )
        return None

    new_li = np.cumsum(seg_lens, dtype=np.int64) - 1
    return torch.as_tensor(
        new_li,
        dtype=scheduler_logits_indices.dtype,
        device=scheduler_logits_indices.device,
    )
