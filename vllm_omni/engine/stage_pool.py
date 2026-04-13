"""StagePool: per-logical-stage replica container.

Groups the {client, output_processor, vllm_config} triple of each replica
under a single logical stage and centralizes replica selection (round-robin
+ per-request affinity).  The Orchestrator still owns flat lists as a
compatibility view; StagePool is the canonical lookup going forward.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm_omni.engine.orchestrator import OrchestratorRequestState


@dataclass
class StageReplica:
    """One replica of a logical stage.

    flat_index is the index into Orchestrator's flat stage_clients list; it
    is the value cached in OrchestratorRequestState.chosen_client_index so
    existing call sites that resolve a flat client keep working unchanged.
    """

    logical_stage_id: int
    replica_index: int
    flat_index: int
    client: Any
    output_processor: Any
    vllm_config: Any


class StagePool:
    """Replicas of one logical stage with RR + affinity selection."""

    def __init__(
        self,
        logical_stage_id: int,
        stage_type: str | None,
        replicas: list[StageReplica],
    ) -> None:
        if not replicas:
            raise ValueError(f"StagePool for logical stage {logical_stage_id} has no replicas")
        self.logical_stage_id = logical_stage_id
        self.stage_type = stage_type
        self.replicas: list[StageReplica] = replicas
        self._rr_cursor = 0
        self._by_flat_index: dict[int, StageReplica] = {r.flat_index: r for r in replicas}

    @property
    def num_replicas(self) -> int:
        return len(self.replicas)

    def get_replica_by_flat_index(self, flat_index: int) -> StageReplica:
        return self._by_flat_index[flat_index]

    def select_replica(
        self,
        req_state: OrchestratorRequestState,
        *,
        affinity_from: StageReplica | None = None,
    ) -> StageReplica:
        """Pick a replica for *req_state* and cache the choice.

        Resolution order:
          1. Existing choice recorded on req_state (per-request affinity).
          2. affinity_from (explicit cross-request binding, e.g. CFG companion
             inheriting its parent's replica at stage 0).
          3. Round-robin across replicas.
        """
        cached = req_state.chosen_client_index.get(self.logical_stage_id)
        if cached is not None:
            return self._by_flat_index[cached]

        if affinity_from is not None:
            if affinity_from.logical_stage_id != self.logical_stage_id:
                raise ValueError(
                    f"affinity_from is for logical stage {affinity_from.logical_stage_id}, "
                    f"cannot be used to select in stage {self.logical_stage_id}"
                )
            chosen = affinity_from
        elif self.num_replicas == 1:
            chosen = self.replicas[0]
        else:
            chosen = self.replicas[self._rr_cursor % self.num_replicas]
            self._rr_cursor += 1

        req_state.chosen_client_index[self.logical_stage_id] = chosen.flat_index
        return chosen


def build_stage_pools(
    stage_clients: list[Any],
    output_processors: list[Any],
    stage_vllm_configs: list[Any],
    logical_stage_to_clients: list[list[int]],
) -> list[StagePool]:
    """Assemble StagePool list from the flat-list view owned by the engine."""
    pools: list[StagePool] = []
    for logical_id, client_indices in enumerate(logical_stage_to_clients):
        replicas = [
            StageReplica(
                logical_stage_id=logical_id,
                replica_index=ri,
                flat_index=ci,
                client=stage_clients[ci],
                output_processor=output_processors[ci],
                vllm_config=stage_vllm_configs[ci],
            )
            for ri, ci in enumerate(client_indices)
        ]
        stage_type = getattr(stage_clients[client_indices[0]], "stage_type", None)
        pools.append(StagePool(logical_id, stage_type, replicas))
    return pools
