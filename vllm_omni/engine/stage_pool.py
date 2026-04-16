"""StagePool: per-logical-stage replica container.

Groups the {client, output_processor, vllm_config} triple of each replica
under a single logical stage and centralizes replica selection (round-robin
+ per-request affinity).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm_omni.engine.orchestrator import OrchestratorRequestState

logger = init_logger(__name__)


@dataclass(eq=False)
class StageReplica:
    """One replica of a logical stage.

    ``eq=False`` keeps identity-based equality/hash so StageReplica instances
    can be used as dict keys (Orchestrator caches them on req_state and in
    per-replica metrics accumulators).
    """

    logical_stage_id: int
    replica_index: int
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

    # ---- Construction helpers ----

    @classmethod
    def from_attach_results(
        cls,
        logical_stage_id: int,
        clients: Sequence[Any],
        output_processors: Sequence[Any],
        vllm_configs: Sequence[Any],
    ) -> StagePool:
        """Build a pool from parallel lists returned by _attach_llm_stage.

        Each positional index corresponds to one replica of the same logical
        stage.  The first replica's ``client.stage_type`` is used as the
        pool-level stage_type.
        """
        replicas = [
            StageReplica(
                logical_stage_id=logical_stage_id,
                replica_index=ri,
                client=clients[ri],
                output_processor=output_processors[ri],
                vllm_config=vllm_configs[ri],
            )
            for ri in range(len(clients))
        ]
        stage_type = getattr(clients[0], "stage_type", None) if clients else None
        return cls(logical_stage_id, stage_type, replicas)

    @classmethod
    def from_diffusion_client(
        cls,
        logical_stage_id: int,
        client: Any,
    ) -> StagePool:
        """Build a single-replica pool for a diffusion stage.

        Diffusion stages have no output_processor or vllm_config on the
        orchestrator side.
        """
        replica = StageReplica(
            logical_stage_id=logical_stage_id,
            replica_index=0,
            client=client,
            output_processor=None,
            vllm_config=None,
        )
        return cls(logical_stage_id, "diffusion", [replica])

    # ---- Selection / admission ----

    @property
    def num_replicas(self) -> int:
        return len(self.replicas)

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
        cached = req_state.chosen_replica.get(self.logical_stage_id)
        if cached is not None:
            return cached

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

        req_state.chosen_replica[self.logical_stage_id] = chosen
        return chosen

    def admit(
        self,
        req_state: OrchestratorRequestState,
        request: Any,
        prompt_text: Any,
        *,
        affinity_from: StageReplica | None = None,
    ) -> StageReplica:
        """Select a replica and register *request* on its output_processor.

        Atomically couples replica selection with output_processor registration
        so that "which replica will serve this request" and "which processor
        knows about this request" are the same by construction.
        """
        stage_replica = self.select_replica(req_state, affinity_from=affinity_from)
        stage_replica.output_processor.add_request(
            request=request,
            prompt=prompt_text,
            parent_req=None,
            request_index=0,
            queue=None,
        )
        return stage_replica


def compute_replica_layout(
    stage_configs: Sequence[Any],
) -> tuple[list[int], dict[int, list[str]], int]:
    """Compute per-stage replica counts and device assignments.

    Returns:
        replicas_per_stage: num_replicas per logical stage.
        replica_devices_map: stage_idx -> per-replica device strings
            (only for stages with num_replicas > 1).
        total_llm_replicas: total LLM replica count across all stages.
    """
    from vllm_omni.engine.stage_init_utils import get_stage_tp_size, split_devices_for_replicas

    replicas_per_stage: list[int] = []
    for stage_cfg in stage_configs:
        runtime_cfg = getattr(stage_cfg, "runtime", {})
        num_replicas = int(
            runtime_cfg.get("num_replicas", 1)
            if hasattr(runtime_cfg, "get")
            else getattr(runtime_cfg, "num_replicas", 1)
        )
        replicas_per_stage.append(max(1, num_replicas))

    replica_devices_map: dict[int, list[str]] = {}
    for logical_id, stage_cfg in enumerate(stage_configs):
        num_replicas = replicas_per_stage[logical_id]
        if num_replicas <= 1:
            continue
        runtime_cfg = getattr(stage_cfg, "runtime", {})
        devices_str = (
            runtime_cfg.get("devices") if hasattr(runtime_cfg, "get") else getattr(runtime_cfg, "devices", None)
        )
        tp_size = get_stage_tp_size(stage_cfg)
        replica_devices_map[logical_id] = split_devices_for_replicas(
            devices_str,
            num_replicas,
            tp_size,
            logical_id,
        )
        logger.info(
            "[StagePool] Stage %s: %d replicas, tp=%d, devices split: %s",
            logical_id,
            num_replicas,
            tp_size,
            replica_devices_map[logical_id],
        )

    total_llm_replicas = sum(
        replicas_per_stage[i] for i, cfg in enumerate(stage_configs) if getattr(cfg, "stage_type", "llm") != "diffusion"
    )
    return replicas_per_stage, replica_devices_map, total_llm_replicas
