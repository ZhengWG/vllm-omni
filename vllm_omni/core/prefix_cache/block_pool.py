"""PrefixBlockPool: pinned CPU block mirror (mirrors vLLM BlockPool).

Storage is (num_blocks, block_size, feat) per key, viewed flat as
(num_slots, feat) so vLLM slot ids index rows directly. Write access is
single-writer (the controller committer thread); readers take row views.
"""

import logging

import torch

from vllm_omni.core.prefix_cache.interface import PrefixCacheConfig

logger = logging.getLogger(__name__)


class PrefixBlockPool:
    def __init__(self, config: PrefixCacheConfig):
        self._config = config
        self.num_slots = config.num_blocks * config.block_size
        self._caches: dict[str, torch.Tensor] = {}

    def _alloc(self, dtype: torch.dtype, feat: int) -> torch.Tensor:
        return torch.zeros(
            (self._config.num_blocks, self._config.block_size, feat),
            dtype=dtype,
            device="cpu",
            # Pinning enables true async D2H and fast scatter; unsupported on
            # CPU-only builds where the async pipeline is off anyway.
            pin_memory=torch.cuda.is_available(),
        )

    def ensure_key(self, key: str, dtype: torch.dtype, feat: int) -> None:
        if key in self._caches:
            return
        self._caches[key] = self._alloc(dtype, feat)
        logger.info("prefix_cache: initialized mirror %s for key %s", list(self._caches[key].shape), key)

    def has_key(self, key: str) -> bool:
        return key in self._caches

    def keys(self) -> set[str]:
        return set(self._caches.keys())

    def _flat(self, key: str) -> torch.Tensor:
        cache = self._caches[key]
        return cache.view(-1, cache.shape[-1])

    def rows(self, key: str, slots: torch.Tensor) -> torch.Tensor:
        """Gather rows for non-contiguous slots (returns a copy)."""
        return self._flat(key).index_select(0, slots)

    def write(self, key: str, slots: torch.Tensor, src_cpu: torch.Tensor) -> None:
        """Row scatter; caller (committer thread) is the single writer."""
        if slots.dtype != torch.int64:
            slots = slots.to(torch.int64)
        # index_copy_ dispatches to a faster single-dim CPU path than
        # advanced-indexing assignment (aten::index_put_).
        self._flat(key).index_copy_(0, slots, src_cpu)
