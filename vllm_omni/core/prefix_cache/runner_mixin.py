# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Runner-facing access layer for the omni prefix cache.

One copy of the integration rules shared by the GPU and NPU runners:
one-shot manager construction, the model cache-policy snapshot, and the
per-step lifecycle entry. The manager/controller state machines and the
platform output-building flows stay in their owners.

Not exported from the package ``__init__``; runners import this module
directly. No module-level vLLM import (``get_pp_group`` is resolved at
call time) so ``tests/core`` keeps loading the package without vLLM.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from vllm_omni.core.prefix_cache.adapter import (
    PrefixCacheSchedulerAdapter,
    PrefixCacheStep,
    PrefixCacheWrite,
    PrefixCacheWriteLayout,
)
from vllm_omni.core.prefix_cache.group_view import get_prefix_cache_group_view
from vllm_omni.core.prefix_cache.interface import (
    ModelCachePolicy,
    OmniPrefixCacheUnmatchError,
    PrefixCacheConfig,
)
from vllm_omni.core.prefix_cache.manager import OmniPrefixCacheManager, _is_step_token_tensor
from vllm_omni.data_entry_keys import flatten_payload

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput


class _DecodeRowCoalescer:
    """Buffer pure-decode rows until a capacity or lifecycle boundary."""

    def __init__(self, config: PrefixCacheConfig) -> None:
        self._config = config
        self._refs: dict[str, list[tuple[int, dict[str, torch.Tensor], int]]] = {}
        self._keys: frozenset[str] = frozenset()
        self._num_rows = self._num_steps = self._num_bytes = 0

    def can_append(self, n: int, rows: dict[str, torch.Tensor]) -> bool:
        nbytes = sum(t.numel() * t.element_size() for t in rows.values())
        return (
            (not self._refs or frozenset(rows) == self._keys)
            and self._num_rows + n <= self._config.staging_capacity_tokens
            and self._num_bytes + nbytes <= self._config.gpu_staging_bytes
        )

    def park(self, req_ids: list[str], slots: torch.Tensor, rows: dict[str, torch.Tensor]) -> bool:
        """Snapshot a step after ``can_append``; return whether a flush is due."""
        # The model may reuse its output storage on the next forward.
        step = {key: val.clone() for key, val in rows.items()}
        self._keys = frozenset(step)
        for i, (req_id, slot) in enumerate(zip(req_ids, slots.tolist())):
            self._refs.setdefault(req_id, []).append((slot, step, i))
        self._num_rows += len(req_ids)
        self._num_steps += 1
        self._num_bytes += sum(t.numel() * t.element_size() for t in step.values())
        return (
            self._num_steps >= self._config.block_size
            or self._num_rows >= self._config.staging_capacity_tokens
            or self._num_bytes >= self._config.gpu_staging_bytes
        )

    def take(self) -> tuple[PrefixCacheWriteLayout, dict[str, torch.Tensor]] | None:
        """Pack all buffered rows, including partial blocks, for an ordinary save."""
        if not self._refs:
            return None
        writes = []
        refs: list[tuple[int, dict[str, torch.Tensor], int]] = []
        for req_id, request_refs in self._refs.items():
            start = len(refs)
            refs.extend(request_refs)
            writes.append(PrefixCacheWrite(req_id, start, len(refs)))
        slots = torch.tensor([slot for slot, _, _ in refs], dtype=torch.long)
        rows = {key: torch.cat([step[key][i : i + 1] for _, step, i in refs]) for key in self._keys}
        self._refs.clear()
        self._num_rows = self._num_steps = self._num_bytes = 0
        return PrefixCacheWriteLayout(tuple(writes), len(refs), slots), rows


class PrefixCacheRunnerMixin:
    """Prefix-cache integration for a model runner (no state machine here).

    Host-class contract — the runner must provide:
        ``input_batch``        live batch with the group-0 block table
        ``kv_cache_config``    for ``kv_cache_groups`` at construction
        ``omni_prefix_cache``  set here; ``None`` until first-step build
        ``_omni_prefix_cache_cfg``  staged by the runner's kv-cache init
                               (``stage_prefix_cache_config``), consumed here

    No cooperative ``__init__``: the class attributes below are safe
    defaults (``ModelCachePolicy`` is frozen); runners keep initializing
    their instance fields as before.
    """

    # Host-class fields (annotation only; the runner initializes them).
    input_batch: Any
    kv_cache_config: Any
    is_pooling_model: bool

    omni_prefix_cache: OmniPrefixCacheManager | None = None
    _omni_prefix_cache_cfg: PrefixCacheConfig | None = None
    # Snapshotted once at load_model; the hot path must not re-probe model
    # attributes every step. The same object is registered on the manager.
    _omni_cache_policy: ModelCachePolicy = ModelCachePolicy()
    _prefix_cache_adapter: PrefixCacheSchedulerAdapter | None = None
    _prefix_cache_group_view: Any = None
    _prefix_cache_step: PrefixCacheStep | None = None
    _prefix_cache_coalesce_decode = False
    _prefix_cache_coalescer: _DecodeRowCoalescer | None = None
    _prefix_cache_pure_decode = False

    def _snapshot_prefix_cache_model_policy(self, model) -> None:
        """Freeze the model's cache policy at load_model."""
        self._omni_cache_policy = ModelCachePolicy.from_model(model)

    def _model_needs_full_prefix_hidden_states(self) -> bool:
        """Opt-out hook for models whose postprocess only consumes the tail.

        When False, we skip both the per-step hidden-state write into the
        omni prefix cache and the merged-tensor reconstruction on hits;
        postprocess receives the normal scheduled-token slice instead. Models
        that need the full cached_prefix + new_tail span (default) are not
        affected.
        """
        return self._omni_cache_policy.needs_full_hidden_states

    def _model_mm_outputs_written_in_sample(self) -> bool:
        """True when the step snapshot cannot hold the model's mm outputs.

        Such a model appends them in ``sample()``, after
        ``_prefix_cache_save_step``; the output path discards the step
        context and builds the payload from the live outputs.
        """
        return self._omni_cache_policy.mm_outputs_written_in_sample

    def _ensure_omni_prefix_cache(self) -> None:
        """One-shot construction (caller gates on the staged config).

        Only the last PP rank writes and materializes the cache. Other ranks
        would register hits they can never serve (save_outputs is last-rank
        only), so skip construction there.
        """
        from vllm.distributed.parallel_state import get_pp_group

        cfg = self._omni_prefix_cache_cfg
        self._omni_prefix_cache_cfg = None
        if cfg is None or not get_pp_group().is_last_rank:
            return
        view = get_prefix_cache_group_view(
            self.input_batch,
            cfg.block_size,
            kv_cache_groups=getattr(self.kv_cache_config, "kv_cache_groups", None),
        )
        if view is None:
            raise OmniPrefixCacheUnmatchError(
                "omni prefix caching requires a block table on the input batch; "
                "disable enable_prefix_caching for this model"
            )
        manager = OmniPrefixCacheManager(cfg)
        manager.register_policy(self._omni_cache_policy)
        self.omni_prefix_cache = manager
        self._prefix_cache_adapter = PrefixCacheSchedulerAdapter()
        self._prefix_cache_group_view = view
        policy = self._omni_cache_policy
        if (
            self._prefix_cache_coalesce_decode
            and policy.needs_full_hidden_states
            and not policy.deferred_keys
            and not policy.mm_outputs_written_in_sample
        ):
            self._prefix_cache_coalescer = _DecodeRowCoalescer(cfg)

    def _prefix_cache_step_begin(self, scheduler_output: SchedulerOutput) -> None:
        """Per-step lifecycle entry; must run before ``_update_states``
        removes finished requests (finished/abort escalation happens inside
        ``new_step_starts``). Builds the manager on the first real step."""
        if self.omni_prefix_cache is None and self._omni_prefix_cache_cfg is not None:
            self._ensure_omni_prefix_cache()
        if self.omni_prefix_cache is not None:
            if self._prefix_cache_adapter is None or self._prefix_cache_group_view is None:
                raise RuntimeError("prefix-cache adapter and view were not initialized together")
            step = self._prefix_cache_adapter.translate_step(scheduler_output)
            self._prefix_cache_step = step
            self._prefix_cache_pure_decode = (
                bool(step.scheduled_tokens)
                and all(count == 1 for _, count in step.scheduled_tokens)
                and not step.events
                and not getattr(scheduler_output, "scheduled_new_reqs", ())
                and not getattr(scheduler_output, "preempted_req_ids", ())
            )
            if self._prefix_cache_coalescer is not None and self._prefix_cache_pure_decode:
                return
            # Flush before hit planning or slot reuse. A streaming extension
            # retains partial blocks, just like the original per-step writes.
            self._prefix_cache_flush_decode()
            self.omni_prefix_cache.new_step_starts(step)

    def _prefix_cache_save_step(
        self,
        hidden_states: torch.Tensor,
        multimodal_outputs: dict | None,
        *,
        num_tokens_unpadded: int,
        num_tokens_padded: int,
    ) -> int | None:
        """Write this step's outputs into the cache.

        None when this stage/rank does not write (pooling, cache off, or not
        the last PP rank), or when a pure decode step was buffered.
        The returned step id must be consumed exactly once
        — materialize or discard_step — by the output path.
        """
        from vllm.distributed.parallel_state import get_pp_group

        if self.is_pooling_model or self.omni_prefix_cache is None or not get_pp_group().is_last_rank:
            return None
        if self._prefix_cache_adapter is None or self._prefix_cache_group_view is None:
            raise RuntimeError("prefix-cache adapter was not initialized")
        if self._prefix_cache_step is None:
            raise RuntimeError("prefix-cache step snapshot was not initialized")
        layout = self._prefix_cache_adapter.build_write_layout(
            self._prefix_cache_group_view,
            num_scheduled_tokens=dict(self._prefix_cache_step.scheduled_tokens),
        )
        mm = flatten_payload(multimodal_outputs) if multimodal_outputs else {}
        coalescer = self._prefix_cache_coalescer
        n = num_tokens_unpadded
        hidden_key = self._omni_cache_policy.hidden_key
        if (
            coalescer is not None
            and self._prefix_cache_pure_decode
            and hidden_key is not None
            and _is_step_token_tensor(hidden_states, n, num_tokens_padded)
            and len(layout.writes) == n
            and all(write.row_end - write.row_start == 1 for write in layout.writes)
            and layout.slots_cpu is not None
            and layout.slots_cpu.numel() == n
        ):
            rows = {hidden_key: hidden_states[:n]}
            rows.update(
                (key, val[:n])
                for key, val in mm.items()
                if _is_step_token_tensor(val, n, num_tokens_padded)
                and not self._omni_cache_policy.skip_immediate_mm(key)
            )
            if not coalescer.can_append(n, rows):
                self._prefix_cache_flush_decode()
            if coalescer.can_append(n, rows):
                if coalescer.park([write.req_id for write in layout.writes], layout.slots_cpu, rows):
                    self._prefix_cache_flush_decode()
                return None
        # Unsupported rows or a single step exceeding the buffer's byte limit
        # use the ordinary save. Flush first, before it claims a staging slot.
        self._prefix_cache_flush_decode()
        if coalescer is not None and self._prefix_cache_pure_decode:
            # The pure-decode entry skipped the manager, but this step fell back.
            self.omni_prefix_cache.new_step_starts(self._prefix_cache_step)
        return self.omni_prefix_cache.save_outputs(
            hidden_states,
            mm,
            num_tokens_unpadded=num_tokens_unpadded,
            num_tokens_padded=num_tokens_padded,
            write_layout=layout,
        )

    def _prefix_cache_flush_decode(self) -> None:
        """Write buffered decode rows through the manager's ordinary save path."""
        coalescer = self._prefix_cache_coalescer
        if coalescer is None or (parked := coalescer.take()) is None:
            return
        layout, rows = parked
        hidden_key = self._omni_cache_policy.hidden_key
        assert hidden_key is not None
        hidden = rows.pop(hidden_key)
        cache = self.omni_prefix_cache
        assert cache is not None
        step_id = cache.save_outputs(
            hidden,
            rows,
            num_tokens_unpadded=layout.total_rows,
            num_tokens_padded=layout.total_rows,
            write_layout=layout,
        )
        cache.discard_step(step_id)

    def _shutdown_omni_prefix_cache(self) -> None:
        """Stop the manager and release the runner-owned cache unit."""
        cache = self.omni_prefix_cache
        try:
            if cache is not None:
                cache.shutdown()
        finally:
            self.omni_prefix_cache = None
            self._prefix_cache_adapter = None
            self._prefix_cache_group_view = None
            self._prefix_cache_step = None
            self._omni_prefix_cache_cfg = None
            self._prefix_cache_coalescer = None
            self._prefix_cache_pure_decode = False

    def _prefix_cache_materialize(
        self, step_id: int | None, req_ids: list[str]
    ) -> tuple[dict[str, torch.Tensor] | None, dict | None]:
        """Per-request merged outputs for a saved step.

        ``req_ids`` must be the save-time snapshot, never the live
        ``input_batch`` (under async output this runs a step late).
        ``step_id`` None means save did not run — nothing to consume.
        ``mm_outputs`` is all-or-nothing; empty only when the step had no mm.
        """
        if step_id is None or self.omni_prefix_cache is None:
            return None, None
        outs = self.omni_prefix_cache.materialize(step_id, list(req_ids))
        return outs.hidden_states, (outs.mm_outputs or None)
