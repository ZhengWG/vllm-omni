"""Unified stage-local runtime abstraction for vLLM-Omni."""

from __future__ import annotations

import asyncio
import time as _time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger
from vllm.v1.engine import EngineCoreOutputs

from vllm_omni.metrics.stats import StageRequestStats as StageRequestMetrics
from vllm_omni.metrics.stats import StageStats
from vllm_omni.metrics.utils import count_tokens_from_outputs

if TYPE_CHECKING:
    from vllm_omni.engine.orchestrator import OrchestratorRequestState

logger = init_logger(__name__)


@dataclass(eq=False)
class StageReplica:
    """One physical route of a logical stage."""

    replica_index: int
    client: Any


@dataclass
class StagePollResult:
    """Stage-local poll result returned to the orchestrator."""

    outputs: list[Any] = field(default_factory=list)
    raw_outputs: EngineCoreOutputs | None = None


@dataclass
class _ReplicaMetrics:
    """Per-replica metrics accumulators owned by a stage pool."""

    batch_seq: int = 0
    agg_total_tokens: int = 0
    agg_total_gen_time_ms: float = 0.0


class StagePool:
    """Replicas of one logical stage with RR + affinity selection."""

    def __init__(
        self,
        stage_id: int,
        replicas: list[StageReplica],
        *,
        output_processor: Any = None,
        stage_vllm_config: Any = None,
    ) -> None:
        if not replicas:
            raise ValueError(f"StagePool for stage {stage_id} has no replicas")
        self.stage_id = stage_id
        self.replicas: list[StageReplica] = replicas
        self._output_processor = output_processor
        self._stage_vllm_config = stage_vllm_config
        self._next_replica_idx = 0
        self._request_bindings: dict[str, StageReplica] = {}
        self._replica_metrics: dict[StageReplica, _ReplicaMetrics] = {sr: _ReplicaMetrics() for sr in self.replicas}

    # ---- Construction helpers ----

    @classmethod
    def build_from_replicas(
        cls,
        stage_id: int,
        clients: Sequence[Any],
        output_processor: Any,
        stage_vllm_config: Any,
    ) -> StagePool:
        """Build a pool from client replicas plus stage-level shared state."""
        replicas = [
            StageReplica(
                replica_index=ri,
                client=clients[ri],
            )
            for ri in range(len(clients))
        ]
        return cls(
            stage_id,
            replicas,
            output_processor=output_processor,
            stage_vllm_config=stage_vllm_config,
        )

    @classmethod
    def build_from_diffusion_client(
        cls,
        stage_id: int,
        client: Any,
    ) -> StagePool:
        """Build a single-replica pool for a diffusion stage."""
        replica = StageReplica(
            replica_index=0,
            client=client,
        )
        return cls(stage_id, [replica], output_processor=None, stage_vllm_config=None)

    # ---- Stage-level properties ----

    @property
    def num_replicas(self) -> int:
        return len(self.replicas)

    @property
    def stage_type(self) -> str | None:
        return getattr(self.stage_client, "stage_type", None)

    @property
    def final_output(self) -> bool:
        return bool(getattr(self.replicas[0].client, "final_output", False))

    @property
    def stage_client(self) -> Any:
        return self.replicas[0].client

    @property
    def stage_vllm_config(self) -> Any:
        return self._stage_vllm_config

    @property
    def output_processor(self) -> Any:
        return self._output_processor

    # ---- Route binding lifecycle ----

    def get_bound_replica(self, request_id: str) -> StageReplica | None:
        """Return the currently bound replica for *request_id* if present."""
        return self._request_bindings.get(request_id)

    def release_binding(self, request_id: str) -> None:
        """Drop the route binding for *request_id* in this stage."""
        self._request_bindings.pop(request_id, None)

    def select_replica(
        self,
        request_id: str,
        *,
        affinity_request_id: str | None = None,
    ) -> StageReplica:
        """Pick a replica for *request_id* and cache the choice."""
        cached = self._request_bindings.get(request_id)
        if cached is not None:
            return cached

        chosen = self.get_bound_replica(affinity_request_id) if affinity_request_id is not None else None
        if chosen is not None:
            pass
        elif self.num_replicas == 1:
            chosen = self.replicas[0]
        else:
            chosen = self.replicas[self._next_replica_idx]
            self._next_replica_idx = (self._next_replica_idx + 1) % self.num_replicas

        self._request_bindings[request_id] = chosen
        return chosen

    # ---- Metrics ----

    def build_stage_metrics(
        self,
        request_id: str,
        request_outputs: list[Any],
        *,
        submit_ts: float,
    ) -> StageRequestMetrics:
        """Build stage metrics using the bound route for *request_id*."""
        stage_replica = self.get_bound_replica(request_id)
        if stage_replica is None:
            stage_replica = self.replicas[0]

        now = _time.time()
        stage_gen_time_ms = (now - submit_ts) * 1000.0

        num_tokens_out = count_tokens_from_outputs(request_outputs)
        num_tokens_in = 0
        if self.stage_id == 0:
            for ro in request_outputs:
                ptids = getattr(ro, "prompt_token_ids", None)
                if ptids is not None:
                    num_tokens_in += len(ptids)

        metrics = self._replica_metrics[stage_replica]
        metrics.batch_seq += 1
        batch_id = metrics.batch_seq
        metrics.agg_total_tokens += num_tokens_out
        metrics.agg_total_gen_time_ms += stage_gen_time_ms

        return StageRequestMetrics(
            num_tokens_in=num_tokens_in,
            num_tokens_out=num_tokens_out,
            stage_gen_time_ms=stage_gen_time_ms,
            batch_id=batch_id,
            batch_size=1,
            rx_decode_time_ms=0.0,
            rx_transfer_bytes=0,
            rx_in_flight_time_ms=0.0,
            stage_stats=StageStats(
                total_token=metrics.agg_total_tokens,
                total_gen_time_ms=metrics.agg_total_gen_time_ms,
            ),
        )

    # ---- Stage-local admission ----

    async def submit_initial(
        self,
        request_id: str,
        req_state: OrchestratorRequestState,
        request: Any,
        *,
        prompt_text: Any = None,
        affinity_request_id: str | None = None,
        submit_kwargs: dict[str, Any] | None = None,
    ) -> StageReplica:
        """Submit a stage-entry request into this pool."""
        params = req_state.sampling_params_list[self.stage_id]
        submit_kwargs = dict(submit_kwargs or {})
        if self.stage_type == "diffusion":
            stage_replica = self.select_replica(
                request_id,
                affinity_request_id=affinity_request_id,
            )
            if isinstance(request, list):
                await stage_replica.client.add_batch_request_async(request_id, request, params, **submit_kwargs)
            else:
                await stage_replica.client.add_request_async(request_id, request, params, **submit_kwargs)
            return stage_replica

        stage_replica = self.select_replica(
            request_id,
            affinity_request_id=affinity_request_id,
        )
        self.output_processor.add_request(
            request=request,
            prompt=prompt_text,
            parent_req=None,
            request_index=0,
            queue=None,
        )
        await stage_replica.client.add_request_async(request, **submit_kwargs)
        return stage_replica

    async def submit_update(
        self,
        request_id: str,
        req_state: OrchestratorRequestState,
        request: Any,
    ) -> StageReplica:
        """Submit a streaming update to an already admitted request."""
        params = req_state.sampling_params_list[self.stage_id]
        stage_replica = self.get_bound_replica(request_id)
        if stage_replica is None:
            stage_replica = self.select_replica(request_id)

        if self.stage_type == "diffusion":
            await stage_replica.client.add_request_async(request_id, request, params)
        else:
            await stage_replica.client.add_request_async(request)
        return stage_replica

    # ---- Stage-local polling ----

    async def _poll_stage_raw(self, stage_replica: StageReplica) -> EngineCoreOutputs | None:
        """Pull raw EngineCoreOutputs from a stage replica without processing."""
        outputs = await stage_replica.client.get_output_async()
        if not outputs.outputs:
            return None
        return outputs

    async def _process_stage_outputs(
        self,
        stage_replica: StageReplica,
        raw_outputs: EngineCoreOutputs,
    ) -> list[Any]:
        """Run the output processor on raw outputs, returning processed outputs."""
        processor = self.output_processor
        processed = processor.process_outputs(
            raw_outputs.outputs,
            raw_outputs.timestamp,
            None,
        )

        if processed.reqs_to_abort:
            await stage_replica.client.abort_requests_async(processed.reqs_to_abort)

        if raw_outputs.scheduler_stats is not None:
            processor.update_scheduler_stats(raw_outputs.scheduler_stats)

        return processed.request_outputs

    async def poll_ready_outputs(self, *, timeout_s: float = 0.001) -> list[StagePollResult]:
        """Poll this stage pool once and return all ready outputs."""
        poll_results: list[StagePollResult] = []

        for stage_replica in self.replicas:
            if stage_replica.client.stage_type == "diffusion":
                output = stage_replica.client.get_diffusion_output_nowait()
                if output is not None:
                    poll_results.append(StagePollResult(outputs=[output]))
                continue

            try:
                raw_outputs = await asyncio.wait_for(
                    self._poll_stage_raw(stage_replica),
                    timeout=timeout_s,
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "[StagePool] _poll_stage_raw failed for stage-%s replica-%s",
                    self.stage_id,
                    stage_replica.replica_index,
                )
                raise

            if raw_outputs is None:
                continue

            request_outputs = await self._process_stage_outputs(stage_replica, raw_outputs)
            poll_results.append(
                StagePollResult(
                    outputs=request_outputs,
                    raw_outputs=raw_outputs,
                )
            )

        return poll_results

    # ---- Stage-local control plane ----

    async def abort_requests(self, request_ids: list[str]) -> None:
        """Abort the given requests across all physical routes in this pool."""
        for stage_replica in self.replicas:
            await stage_replica.client.abort_requests_async(request_ids)

    async def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[Any]:
        """Dispatch a stage-scoped control-plane RPC across this pool."""
        kwargs = dict(kwargs or {})
        results: list[Any] = []

        for stage_replica in self.replicas:
            try:
                if hasattr(stage_replica.client, "collective_rpc_async"):
                    stage_result = await stage_replica.client.collective_rpc_async(
                        method=method,
                        timeout=timeout,
                        args=args,
                        kwargs=kwargs,
                    )
                else:
                    stage_result = {
                        "supported": False,
                        "todo": True,
                        "reason": (
                            f"{stage_replica.client.__class__.__name__}.collective_rpc_async is not implemented yet"
                        ),
                    }
            except Exception as exc:
                logger.exception(
                    "[StagePool] collective_rpc failed: stage=%s replica=%s method=%s",
                    self.stage_id,
                    stage_replica.replica_index,
                    method,
                )
                stage_result = {
                    "supported": False,
                    "error": str(exc),
                }

            results.append(stage_result)

        return results

    def shutdown(self) -> None:
        """Shutdown all backend handles in this stage pool."""
        for stage_replica in self.replicas:
            try:
                stage_replica.client.shutdown()
                logger.info(
                    "[StagePool] Stage %d replica %d shut down",
                    self.stage_id,
                    stage_replica.replica_index,
                )
            except Exception as e:
                logger.warning(
                    "[StagePool] Failed to shutdown stage %d replica %d: %s",
                    self.stage_id,
                    stage_replica.replica_index,
                    e,
                )
