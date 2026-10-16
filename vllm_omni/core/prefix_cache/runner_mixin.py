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
    # Planned at step begin, consumed by the save.
    _prefix_cache_park_req_ids: tuple[str, ...] = ()
    _prefix_cache_hit_bases: frozenset[int] = frozenset()
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
        return DecodeRowCoalescer(cfg.block_size)

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
            if self._prefix_cache_coalescer is not None and self._prefix_cache_plan_parking(scheduler_output, step):
                return
            self.omni_prefix_cache.new_step_starts(step)

    def _prefix_cache_plan_parking(self, scheduler_output: SchedulerOutput, step: PrefixCacheStep) -> bool:
        """Pick the requests whose row this step parks.

        Every one-token request that is not arriving parks. Returns True for
        a pure decode step (all requests park, no event), which skips
        ``new_step_starts`` as it has nothing to do. A departed request
        writes its full parked blocks (vLLM still caches them) and drops
        the partial one; parked blocks an arrival hits are written here, so
        the hit's prefetch can plan them.
        """
        coalescer = self._prefix_cache_coalescer
        assert coalescer is not None
        new_req_ids = {
            PrefixCacheSchedulerAdapter._req_id(data)
            for data in getattr(scheduler_output, "scheduled_new_reqs", ()) or ()
        }
        gone = [event.req_id for event in step.events if event.kind is not PrefixCacheEventKind.STARTED]
        gone += [str(req_id) for req_id in getattr(scheduler_output, "preempted_req_ids", ()) or ()]
        gone += list(new_req_ids)
        if gone:
            self._prefix_cache_write_parked(coalescer.take(gone, full_only=True))
            coalescer.drop(gone)
        bs = coalescer.block_size
        self._prefix_cache_hit_bases = frozenset(
            block * bs
            for event in step.events
            if event.kind is PrefixCacheEventKind.STARTED and event.hit_end > 0 and event.block_ids
            for block in event.block_ids[0][: event.hit_end // bs]
        )
        if self._prefix_cache_hit_bases:
            self._prefix_cache_write_parked(coalescer.take(bases=self._prefix_cache_hit_bases))
        self._prefix_cache_park_req_ids = tuple(
            req_id for req_id, count in step.scheduled_tokens if count == 1 and req_id not in new_req_ids
        )
        self._prefix_cache_pure_decode = (
            not step.events
            and not new_req_ids
            and len(self._prefix_cache_park_req_ids) == len(step.scheduled_tokens) > 0
        )
        return self._prefix_cache_pure_decode

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
        the last PP rank), or when every row of the step was parked. The
        returned step id must be consumed exactly once — materialize or
        discard_step — by the output path.
        """
        from vllm.distributed.parallel_state import get_pp_group

        if self.is_pooling_model or self.omni_prefix_cache is None or not get_pp_group().is_last_rank:
            return None
        if self._prefix_cache_adapter is None or self._prefix_cache_group_view is None:
            raise RuntimeError("prefix-cache adapter was not initialized")
        if self._prefix_cache_step is None:
            raise RuntimeError("prefix-cache step snapshot was not initialized")
        mm = flatten_payload(multimodal_outputs) if multimodal_outputs else {}
        coalescer = self._prefix_cache_coalescer
        parked: list[str] = []
        flush = False
        if coalescer is not None:
            pure = self._prefix_cache_pure_decode
            parked, flush = self._prefix_cache_park_step(hidden_states, mm, num_tokens_unpadded, num_tokens_padded)
            if pure and len(parked) == num_tokens_unpadded > 0:
                if flush:
                    self._prefix_cache_write_parked(coalescer.take(full_only=True))
                return None
            if pure:
                # Scheduled as a pure decode step but not parked whole:
                # new_step_starts was skipped, run it before the save.
                self.omni_prefix_cache.new_step_starts(self._prefix_cache_step)
            # A request saved this step continues from its parked rows, and a
            # block an arrival hits must be complete before this save plans
            # the hit.
            parked_set = set(parked)
            unparked = [r for r, _ in self._prefix_cache_step.scheduled_tokens if r not in parked_set]
            self._prefix_cache_write_parked(coalescer.take(unparked))
            if self._prefix_cache_hit_bases:
                self._prefix_cache_write_parked(coalescer.take(bases=self._prefix_cache_hit_bases))
        layout = self._prefix_cache_adapter.build_write_layout(
            self._prefix_cache_group_view,
            num_scheduled_tokens=dict(self._prefix_cache_step.scheduled_tokens),
        )
        skip = {"skip_write_req_ids": frozenset(parked)} if parked else {}
        step_id = self.omni_prefix_cache.save_outputs(
            hidden_states,
            mm,
            num_tokens_unpadded=num_tokens_unpadded,
            num_tokens_padded=num_tokens_padded,
            write_layout=layout,
            **skip,
        )
        if flush and coalescer is not None:
            self._prefix_cache_write_parked(coalescer.take(full_only=True))
        return step_id

    def _prefix_cache_park_step(
        self,
        hidden_states: torch.Tensor | None,
        mm: dict[str, Any],
        num_tokens_unpadded: int,
        num_tokens_padded: int,
    ) -> tuple[list[str], bool]:
        """Park the planned one-token requests' rows. Returns the parked
        requests (empty when the step cannot be parked) and whether full
        blocks are due for a write."""
        coalescer = self._prefix_cache_coalescer
        hidden_key = self._omni_cache_policy.hidden_key
        view = self._prefix_cache_group_view
        step = self._prefix_cache_step
        want = list(self._prefix_cache_park_req_ids)
        n = num_tokens_unpadded
        if (
            coalescer is None
            or step is None
            or not want
            or hidden_key is None
            or hidden_states is None
            or not _is_step_token_tensor(hidden_states, n, num_tokens_padded)
        ):
            return [], False
        rows = {hidden_key: hidden_states[:n]}
        for key, val in mm.items():
            if _is_step_token_tensor(val, n, num_tokens_padded) and not self._omni_cache_policy.skip_immediate_mm(key):
                rows[key] = val[:n]
        if coalescer.has_parked() and frozenset(rows) != coalescer.keys():
            # Parked rows follow one key set; write them before it changes.
            self._prefix_cache_write_parked(coalescer.take())
        scheduled = dict(step.scheduled_tokens)
        offset: dict[str, int] = {}
        cursor = 0
        for req_id in view.batch_req_ids():
            offset[req_id] = cursor
            cursor += scheduled.get(req_id, 0)
        if cursor != n or any(req_id not in offset for req_id in want):
            return [], False
        slots = view.decode_slots_cpu(want)
        if slots is None:
            return [], False
        flush = coalescer.park(want, slots, rows, [offset[req_id] for req_id in want])
        return want, flush

    def _prefix_cache_write_parked(self, parked: ParkedRows | None) -> None:
        """Write rows taken from the coalescer; nothing materializes them."""
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
            write_only=True,
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
