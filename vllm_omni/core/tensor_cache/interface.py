"""Interface types for the omni tensor cache.

Naming aligns with vLLM's v1/core KV-cache design; see
rfc-omni-tensor-cache-refactor.md for the full contract.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, NamedTuple

import torch

# Reserved pool key for hidden states (mm keys are flat dotted names).
HIDDEN_KEY = "__hidden_states__"


class WriteSchedule(Enum):
    """Write scheduling policy for one WriteTask."""

    # Copied every step; completion joined at the NEXT step's save.
    JOIN_NEXT_STEP = "join_next_step"
    # Background trickle; escalated & joined when the request finishes
    # (cap pressure may spill it earlier).
    JOIN_ON_FINISH = "join_on_finish"


@dataclass(frozen=True)
class TensorCacheConfig:
    """Sizing and flow-control knobs (mirrors KVCacheConfig)."""

    num_blocks: int
    block_size: int
    hidden_size: int
    hs_dtype: torch.dtype
    # GPU staging byte budget; exceeding it force-flushes oldest entries.
    gpu_staging_bytes: int = 512 * 1024 * 1024
    # Tasks covering more than this many tokens use JOIN_ON_FINISH;
    # smaller ones use JOIN_NEXT_STEP.
    join_on_finish_min_tokens: int = 256
    # Step host ring: Temporarily buffers D2H (device-to-host) data for the entire step phase.
    # Each slot holds data for one step (not per request).
    # It is recommended to use from_vllm_config to automatically set ring_capacity_tokens
    # to match max_num_batched_tokens.
    ring_depth: int = 4
    ring_capacity_tokens: int = 1024
    # D2H chunk size for the JOIN_ON_FINISH trickle.
    copy_chunk_bytes: int = 16 * 1024 * 1024
    # vLLM kv-cache groups (KVCacheGroupSpec list); the group-view factory
    # selects by spec type. Excluded from equality (config identity is the
    # sizing knobs).
    kv_cache_groups: Any = field(default=None, compare=False)

    @classmethod
    def from_vllm_config(
        cls,
        *,
        num_blocks: int,
        block_size: int,
        hidden_size: int,
        hs_dtype: torch.dtype,
        scheduler_config: Any = None,
        model_config: Any = None,
        kv_cache_groups: Any = None,
    ) -> "TensorCacheConfig":
        """Size the step-host ring from the running scheduler.

        A slot holds one *step* (the whole batch), not one request:
        ``ring_capacity_tokens`` is ``max_num_batched_tokens`` (falls back
        to ``max_model_len``); ``ring_depth`` is the in-flight step bound,
        not ``max_num_seqs``.
        """
        batched = getattr(scheduler_config, "max_num_batched_tokens", None)
        model_len = getattr(scheduler_config, "max_model_len", None)
        if not model_len and model_config is not None:
            model_len = getattr(model_config, "max_model_len", None)
        try:
            capacity = int(batched or model_len or 1024)
        except (TypeError, ValueError):
            capacity = 1024
        return cls(
            num_blocks=num_blocks,
            block_size=block_size,
            hidden_size=hidden_size,
            hs_dtype=hs_dtype,
            ring_depth=4,
            ring_capacity_tokens=max(1, capacity),
            kv_cache_groups=kv_cache_groups,
        )


class StageCacheOutputs(NamedTuple):
    """Plain value object: per-request merged stage outputs."""

    # req_id -> full-prompt hidden states (None when policy skips them)
    hidden_states: dict[str, torch.Tensor] | None
    # mm_key -> req_id -> payload element (req-major)
    mm_outputs: dict[str, dict[str, Any]]


class OmniTensorCacheUnmatchError(RuntimeError):
    """A hit span resolved to absent slots: omni cache diverged from vLLM KV
    state. Fail-fast — this must never fire in a correct system."""


@dataclass(frozen=True)
class ModelCachePolicy:
    """Replaces getattr probing on models for cache behavior decisions."""

    needs_full_hidden_states: bool = True
    # Forces eager materialize on the main thread.
    merge_consumed_by_postprocess: bool = False
    deferred_keys: frozenset[str] = frozenset()
    skip_keys: frozenset[str] = frozenset()

    @classmethod
    def from_model(cls, model: Any) -> "ModelCachePolicy":
        """Shim over legacy per-model attributes (deprecation window).

        Legacy runners pass the deferred keys as the write-path skip keys,
        so both policy fields map from the same attribute.
        """
        deferred = frozenset(getattr(model, "deferred_prefix_cache_mm_keys", ()) or ())
        return cls(
            needs_full_hidden_states=bool(getattr(model, "requires_full_prefix_cached_hidden_states", True)),
            deferred_keys=deferred,
            skip_keys=deferred,
        )
