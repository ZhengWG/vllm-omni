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

from vllm_omni.core.prefix_cache.adapter import (
    PrefixCacheEventKind,
    PrefixCacheSchedulerAdapter,
    PrefixCacheStep,
    PrefixCacheWrite,
    PrefixCacheWriteLayout,
)
from vllm_omni.core.prefix_cache.coalesce import DecodeRowCoalescer, ParkedRows
from vllm_omni.core.prefix_cache.group_view import get_prefix_cache_group_view
from vllm_omni.core.prefix_cache.interface import (
    ModelCachePolicy,
    OmniPrefixCacheUnmatchError,
    PrefixCacheConfig,
)
from vllm_omni.core.prefix_cache.manager import OmniPrefixCacheManager, _is_step_token_tensor
from vllm_omni.data_entry_keys import flatten_payload

if TYPE_CHECKING:
    import torch
    from vllm.v1.core.sched.output import SchedulerOutput


class PrefixCacheRunnerMixin:
    """Prefix-cache integration for a model runner (no state machine here).

    Host-class contract — the runner must provide:
        ``input_batch``        live batch with the group-0 block table
        ``kv_cache_config``    for ``kv_cache_groups`` at construction
        ``omni_prefix_cache``  set here; ``None`` until first-step build
        ``_omni_prefix_cache_cfg``  staged by the runner's kv-cache init
                               (``stage_prefix_cache_config``), consumed here
        ``scheduler_config``   ``max_num_seqs`` sizes the decode rings when
                               ``_prefix_cache_coalesce_decode`` is set

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
    # Runners whose group view has ``decode_slots_cpu`` opt in to parking
    # decode rows until their block fills (see coalesce.py).
    _prefix_cache_coalesce_decode = False
    _prefix_cache_coalescer: DecodeRowCoalescer | None = None
    _prefix_cache_parking = False

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
        self._prefix_cache_coalescer = self._build_decode_coalescer(cfg, view)

    def _build_decode_coalescer(self, cfg: PrefixCacheConfig, view: Any) -> DecodeRowCoalescer | None:
        """Decode rows are parked only for the plain policy: full hidden,
        no deferred keys and the step snapshot holds every mm output."""
        policy = self._omni_cache_policy
        if (
            not self._prefix_cache_coalesce_decode
            or not policy.needs_full_hidden_states
            or policy.deferred_keys
            or policy.mm_outputs_written_in_sample
            or not hasattr(view, "decode_slots_cpu")
        ):
            return None
        capacity = int(getattr(getattr(self, "scheduler_config", None), "max_num_seqs", 0) or 0)
        return DecodeRowCoalescer(cfg.block_size, capacity) if capacity > 0 else None

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
            if self._prefix_cache_coalescer is not None and self._prefix_cache_begin_parking(scheduler_output, step):
                return
            self.omni_prefix_cache.new_step_starts(step)

    def _prefix_cache_begin_parking(self, scheduler_output: SchedulerOutput, step: PrefixCacheStep) -> bool:
        """True when this step parks its decode rows instead of saving.

        Only a pure decode step parks: no arrival, terminal or resume
        event, one token per scheduled request. It has nothing for
        ``new_step_starts`` to do, so that is skipped too. Any other step
        first writes every parked row, so its save and its hits see the
        blocks those rows complete.
        """
        coalescer = self._prefix_cache_coalescer
        assert coalescer is not None
        new_req_ids = [
            PrefixCacheSchedulerAdapter._req_id(data)
            for data in getattr(scheduler_output, "scheduled_new_reqs", ()) or ()
        ]
        gone = [event.req_id for event in step.events if event.kind is not PrefixCacheEventKind.STARTED]
        gone += [str(req_id) for req_id in getattr(scheduler_output, "preempted_req_ids", ()) or ()]
        coalescer.drop(gone + new_req_ids)
        req_ids = [req_id for req_id, _ in step.scheduled_tokens]
        self._prefix_cache_parking = (
            not step.events
            and not new_req_ids
            and bool(req_ids)
            and all(count == 1 for _, count in step.scheduled_tokens)
            and coalescer.has_room(req_ids)
        )
        if not self._prefix_cache_parking:
            self._prefix_cache_write_parked(coalescer.take_all())
        return self._prefix_cache_parking

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
        the last PP rank). The returned step id must be consumed exactly once
        — materialize or discard_step — by the output path.
        """
        from vllm.distributed.parallel_state import get_pp_group

        if self.is_pooling_model or self.omni_prefix_cache is None or not get_pp_group().is_last_rank:
            return None
        if self._prefix_cache_adapter is None or self._prefix_cache_group_view is None:
            raise RuntimeError("prefix-cache adapter was not initialized")
        if self._prefix_cache_step is None:
            raise RuntimeError("prefix-cache step snapshot was not initialized")
        if self._prefix_cache_parking:
            self._prefix_cache_parking = False
            if self._prefix_cache_park_step(hidden_states, multimodal_outputs, num_tokens_unpadded, num_tokens_padded):
                return None
            # The batch does not look like the pure decode step it was
            # scheduled as: run this step through the per-step save.
            assert self._prefix_cache_coalescer is not None
            self._prefix_cache_write_parked(self._prefix_cache_coalescer.take_all())
            self.omni_prefix_cache.new_step_starts(self._prefix_cache_step)
        layout = self._prefix_cache_adapter.build_write_layout(
            self._prefix_cache_group_view,
            num_scheduled_tokens=dict(self._prefix_cache_step.scheduled_tokens),
        )
        return self.omni_prefix_cache.save_outputs(
            hidden_states,
            flatten_payload(multimodal_outputs) if multimodal_outputs else {},
            num_tokens_unpadded=num_tokens_unpadded,
            num_tokens_padded=num_tokens_padded,
            write_layout=layout,
        )

    def _prefix_cache_park_step(
        self,
        hidden_states: torch.Tensor | None,
        multimodal_outputs: dict | None,
        num_tokens_unpadded: int,
        num_tokens_padded: int,
    ) -> bool:
        """Park this pure decode step's rows; write the blocks it fills."""
        coalescer = self._prefix_cache_coalescer
        hidden_key = self._omni_cache_policy.hidden_key
        view = self._prefix_cache_group_view
        n = num_tokens_unpadded
        req_ids = list(view.batch_req_ids())
        if (
            coalescer is None
            or hidden_key is None
            or hidden_states is None
            or len(req_ids) != n
            or not _is_step_token_tensor(hidden_states, n, num_tokens_padded)
        ):
            return False
        slots = view.decode_slots_cpu(req_ids)
        if slots is None:
            return False
        rows = {hidden_key: hidden_states[:n]}
        for key, val in (flatten_payload(multimodal_outputs) if multimodal_outputs else {}).items():
            if _is_step_token_tensor(val, n, num_tokens_padded) and not self._omni_cache_policy.skip_immediate_mm(key):
                rows[key] = val[:n]
        if set(rows) != coalescer.keys():
            # The ring layout follows the step's keys; parked rows of the old
            # layout are written before it changes.
            self._prefix_cache_write_parked(coalescer.take_all())
            coalescer.reset_keys()
        filled = coalescer.park(req_ids, slots, rows)
        if filled:
            self._prefix_cache_write_parked(coalescer.take(filled))
        return True

    def _prefix_cache_write_parked(self, parked: ParkedRows | None) -> None:
        """Write rows taken from the decode rings; nothing materializes them."""
        if parked is None:
            return
        writes: list[PrefixCacheWrite] = []
        cursor = 0
        for req_id, count in zip(parked.req_ids, parked.counts):
            writes.append(PrefixCacheWrite(req_id, cursor, cursor + count))
            cursor += count
        rows = dict(parked.rows)
        hidden = rows.pop(self._omni_cache_policy.hidden_key, None)  # type: ignore[arg-type]
        cache = self.omni_prefix_cache
        assert cache is not None
        step_id = cache.save_outputs(
            hidden,
            rows,
            num_tokens_unpadded=cursor,
            num_tokens_padded=cursor,
            write_layout=PrefixCacheWriteLayout(tuple(writes), cursor, parked.slots_cpu),
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
            self._prefix_cache_parking = False

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
