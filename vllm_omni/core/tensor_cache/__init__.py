"""Omni tensor cache (successor of vllm_omni/core/prefix_cache.py).

See rfc-omni-tensor-cache-refactor.md. Structure mirrors vllm/v1/core.
"""

from vllm_omni.core.tensor_cache.block_pool import TensorBlockPool
from vllm_omni.core.tensor_cache.controller import OmniTensorCacheController
from vllm_omni.core.tensor_cache.group_view import (
    FullAttentionGroupView,
    KVCacheGroupView,
    get_tensor_cache_group_view,
)
from vllm_omni.core.tensor_cache.interface import (
    HIDDEN_KEY,
    ModelCachePolicy,
    OmniTensorCacheUnmatchError,
    Placement,
    StageCacheOutputs,
    TensorCacheConfig,
)
from vllm_omni.core.tensor_cache.manager import OmniTensorCacheManager

__all__ = [
    "HIDDEN_KEY",
    "FullAttentionGroupView",
    "KVCacheGroupView",
    "ModelCachePolicy",
    "OmniTensorCacheController",
    "OmniTensorCacheManager",
    "OmniTensorCacheUnmatchError",
    "Placement",
    "StageCacheOutputs",
    "TensorBlockPool",
    "TensorCacheConfig",
    "get_tensor_cache_group_view",
]
