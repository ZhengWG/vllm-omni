"""KV-cache group access seam for the omni tensor cache.

The sole path through which the tensor cache touches vLLM scheduler
internals (block tables, slot mappings). Hybrid-attention models plug in
by providing another view implementation; no usable group => the factory
returns None and the feature self-disables.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_input_batch import InputBatch

logger = logging.getLogger(__name__)


@runtime_checkable
class KVCacheGroupView(Protocol):
    block_size: int
    num_blocks: int

    def slot_mapping_gpu(self, num_tokens: int) -> torch.Tensor: ...

    def slots_for(self, req_id: str, token_start: int, token_end: int) -> torch.Tensor: ...

    def cached_block_ids(self, req_id: str) -> torch.Tensor: ...

    def batch_req_ids(self) -> list[str]: ...


class FullAttentionGroupView:
    """View over the first (full-attention) KV-cache group.

    Behavior-preserving port of the block-table math previously inlined in
    the legacy prefix cache (_get_slot_ids_for_token_range /
    _get_cached_block_ids) plus the runner-side slot_mapping access.
    """

    def __init__(self, input_batch: InputBatch, block_size: int, num_blocks: int):
        self._input_batch = input_batch
        self.block_size = block_size
        self.num_blocks = num_blocks

    def _block_table_cpu(self) -> torch.Tensor:
        return self._input_batch.block_table[0].block_table.cpu

    def slot_mapping_gpu(self, num_tokens: int) -> torch.Tensor:
        return self._input_batch.block_table[0].slot_mapping.gpu[:num_tokens]

    def batch_req_ids(self) -> list[str]:
        return list(self._input_batch.req_ids)

    def step_slots_cpu(self, req_ids: list[str], num_scheduled: dict[str, int]) -> torch.Tensor:
        """This step's slot mapping, computed on CPU from the block table.

        The device slot_mapping would need a stream sync to read back, which
        stalls the whole forward; the CPU block table carries the same
        information (positions are num_computed .. +num_scheduled per request).
        """
        block_table = self._block_table_cpu()
        bs = self.block_size
        max_blocks = int(block_table.shape[1])
        computed = self._input_batch.num_computed_tokens_cpu
        parts: list[torch.Tensor] = []
        for req_id in req_ids:
            n = int(num_scheduled.get(req_id, 0))
            if n <= 0:
                continue
            req_idx = self._input_batch.req_id_to_index[req_id]
            start = int(computed[req_idx])
            pos = torch.arange(start, start + n, dtype=torch.long)
            offs = pos // bs
            if int(offs[-1]) >= max_blocks:
                keep = offs < max_blocks
                pos, offs = pos[keep], offs[keep]
                if pos.numel() == 0:
                    continue
            parts.append(block_table[req_idx, offs].to(torch.long) * bs + (pos % bs))
        return torch.cat(parts) if parts else torch.empty((0,), dtype=torch.long)

    def slots_for(self, req_id: str, token_start: int, token_end: int) -> torch.Tensor:
        """Flat cache-row indices for a logical token range of a request.

        Positions past the request's block table are dropped (legacy
        clamping behavior for over-long deferred chunks).
        """
        if token_end <= token_start:
            return torch.empty((0,), dtype=torch.long)

        req_idx = self._input_batch.req_id_to_index[req_id]
        block_table = self._block_table_cpu()
        token_positions = torch.arange(token_start, token_end, dtype=torch.long)
        block_offsets = token_positions // self.block_size
        max_blocks = int(block_table.shape[1])
        valid = block_offsets < max_blocks
        if not bool(valid.all()):
            token_positions = token_positions[valid]
            block_offsets = block_offsets[valid]
        if token_positions.numel() == 0:
            return torch.empty((0,), dtype=torch.long)

        block_ids = block_table[req_idx, block_offsets].to(torch.long)
        return block_ids * self.block_size + (token_positions % self.block_size)

    def cached_block_ids(self, req_id: str) -> torch.Tensor:
        """Block ids covering a request's prefix-cache hit.

        Relies on vLLM guaranteeing block-aligned num_computed_tokens for
        prefix hits (full-hit rolls back a whole block).
        """
        req_idx = self._input_batch.req_id_to_index[req_id]
        num_computed = self._input_batch.num_computed_tokens_cpu[req_idx]
        num_cached_blocks = num_computed // self.block_size
        return self._block_table_cpu()[req_idx, :num_cached_blocks]


def get_tensor_cache_group_view(
    input_batch: InputBatch,
    block_size: int,
    num_blocks: int,
    kv_cache_groups: object = None,
) -> KVCacheGroupView | None:
    """Build the group view for a runner's input batch; None means no
    usable group (the caller fails fast rather than serving hits without
    cached tensors).

    Selection is by kv-cache group SPEC, not by counting block tables: a
    hybrid model's group 0 is not necessarily full attention, and a
    narrower per-group table would make step_slots_cpu silently clamp.
    """
    block_tables = getattr(input_batch.block_table, "block_tables", None)
    if not block_tables:
        return None
    groups = list(kv_cache_groups or ())
    if len(groups) != 1:
        logger.warning("Omni prefix caching needs exactly one kv-cache group, found %d.", len(groups))
        return None
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    spec = getattr(groups[0], "kv_cache_spec", None)
    if not isinstance(spec, FullAttentionSpec):
        logger.warning("Omni prefix caching needs a full-attention group 0, found %s.", type(spec).__name__)
        return None
    return FullAttentionGroupView(input_batch, block_size, num_blocks)
