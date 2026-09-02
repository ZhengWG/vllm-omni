# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Omni prefix cache.

Structure mirrors vllm/v1/core.
"""

from vllm_omni.core.prefix_cache.block_pool import PrefixBlockPool
from vllm_omni.core.prefix_cache.controller import OmniPrefixCacheController
from vllm_omni.core.prefix_cache.group_view import (
    FullAttentionGroupView,
    check_prefix_cache_kv_groups,
    get_prefix_cache_group_view,
)
from vllm_omni.core.prefix_cache.interface import (
    HIDDEN_KEY,
    ModelCachePolicy,
    OmniPrefixCacheUnmatchError,
    PrefixCacheConfig,
    StageCacheOutputs,
    WriteSchedule,
)
from vllm_omni.core.prefix_cache.manager import OmniPrefixCacheManager

__all__ = [
    "HIDDEN_KEY",
    "FullAttentionGroupView",
    "ModelCachePolicy",
    "OmniPrefixCacheController",
    "OmniPrefixCacheManager",
    "OmniPrefixCacheUnmatchError",
    "StageCacheOutputs",
    "PrefixBlockPool",
    "PrefixCacheConfig",
    "WriteSchedule",
    "check_prefix_cache_kv_groups",
    "get_prefix_cache_group_view",
]
