"""Unified stage-local runtime abstraction for vLLM-Omni."""

from __future__ import annotations

import asyncio
import copy
import time as _time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreOutputs

from vllm_omni.distributed.omni_connectors.adapter import compute_talker_prompt_ids_length
from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.engine.serialization import serialize_additional_information
from vllm_omni.metrics.stats import StageRequestStats as StageRequestMetrics
from vllm_omni.metrics.stats import StageStats
from vllm_omni.metrics.utils import count_tokens_from_outputs

if TYPE_CHECKING:
    from vllm_omni.engine.orchestrator import OrchestratorRequestState

logger = init_logger(__name__)


def build_engine_core_request_from_tokens(
    request_id: str,
    prompt: dict[str, Any],
    params: SamplingParams | PoolingParams,
    arrival_time: float | None = None,
    model_config: ModelConfig | None = None,
) -> OmniEngineCoreRequest:
    """Build an OmniEngineCoreRequest directly from an OmniTokensPrompt."""
    if arrival_time is None:
        arrival_time = _time.time()

    prompt_token_ids = prompt["prompt_token_ids"]

    sampling_params = None
    pooling_params = None
    if isinstance(params, SamplingParams):
        sampling_params = params.clone()
        if sampling_params.max_tokens is None and model_config is not None:
            sampling_params.max_tokens = model_config.max_model_len - len(prompt_token_ids)
    else:
        pooling_params = params.clone()

    prompt_embeds: torch.Tensor | None = prompt.get("prompt_embeds")
    additional_info_payload = serialize_additional_information(
        prompt.get("additional_information"),
        log_prefix=f"build_engine_core_request_from_tokens req={request_id}",
    )

    return OmniEngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        mm_features=None,
        sampling_params=sampling_params,
        pooling_params=pooling_params,
        arrival_time=arrival_time,
        lora_request=getattr(params, "lora_request", None),
        cache_salt=None,
        data_parallel_rank=None,
        prompt_embeds=prompt_embeds,
        additional_information=additional_info_payload,
    )


@dataclass(eq=False)
class StageReplica:
    """One physical route of a logical stage."""

    stage_id: int
    replica_index: int
    client: Any
    output_processor: Any
    vllm_config: Any


@dataclass
class StagePollResult:
    """Stage-local poll result returned to the orchestrator."""

    stage_id: int
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
        stage_type: str | None,
        replicas: list[StageReplica],
    ) -> None:
        if not replicas:
            raise ValueError(f"StagePool for stage {stage_id} has no replicas")
        self.stage_id = stage_id
        self.stage_type = stage_type
        self.replicas: list[StageReplica] = replicas
        self._next_replica_idx = 0
        self._request_bindings: dict[str, StageReplica] = {}
        self._replica_metrics: dict[StageReplica, _ReplicaMetrics] = {
            sr: _ReplicaMetrics() for sr in self.replicas
        }

    # ---- Construction helpers ----

    @classmethod
    def build_from_replicas(
        cls,
        stage_id: int,
        clients: Sequence[Any],
        output_processors: Sequence[Any],
        vllm_configs: Sequence[Any],
    ) -> StagePool:
        """Build a pool from parallel replica lists."""
        replicas = [
            StageReplica(
                stage_id=stage_id,
                replica_index=ri,
                client=clients[ri],
                output_processor=output_processors[ri],
                vllm_config=vllm_configs[ri],
            )
            for ri in range(len(clients))
        ]
        stage_type = getattr(clients[0], "stage_type", None) if clients else None
        return cls(stage_id, stage_type, replicas)

    @classmethod
    def build_from_diffusion_client(
        cls,
        stage_id: int,
        client: Any,
    ) -> StagePool:
        """Build a single-replica pool for a diffusion stage."""
        replica = StageReplica(
            stage_id=stage_id,
            replica_index=0,
            client=client,
            output_processor=None,
            vllm_config=None,
        )
        return cls(stage_id, "diffusion", [replica])

    # ---- Stage-level properties ----

    @property
    def num_replicas(self) -> int:
        return len(self.replicas)

    @property
    def final_output(self) -> bool:
        return bool(getattr(self.replicas[0].client, "final_output", False))

    @property
    def stage_client(self) -> Any:
        return self.replicas[0].client

    @property
    def stage_vllm_config(self) -> Any:
        return self.replicas[0].vllm_config

    @property
    def output_processor(self) -> Any:
        return self.replicas[0].output_processor

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
        affinity_from: StageReplica | None = None,
        affinity_request_id: str | None = None,
    ) -> StageReplica:
        """Pick a replica for *request_id* and cache the choice."""
        cached = self._request_bindings.get(request_id)
        if cached is not None:
            return cached

        if affinity_from is None and affinity_request_id is not None:
            affinity_from = self.get_bound_replica(affinity_request_id)

        if affinity_from is not None:
            if affinity_from.stage_id != self.stage_id:
                raise ValueError(
                    f"affinity_from is for stage {affinity_from.stage_id}, "
                    f"cannot be used to select in stage {self.stage_id}"
                )
            chosen = affinity_from
        elif self.num_replicas == 1:
            chosen = self.replicas[0]
        else:
            chosen = self.replicas[self._next_replica_idx]
            self._next_replica_idx = (self._next_replica_idx + 1) % self.num_replicas

        self._request_bindings[request_id] = chosen
        return chosen

    def set_bound_engine_outputs(self, request_id: str, outputs: Any) -> StageReplica:
        """Set engine outputs on the currently bound physical route."""
        stage_replica = self.get_bound_replica(request_id)
        if stage_replica is None:
            raise KeyError(f"No bound replica for req={request_id} in stage {self.stage_id}")
        stage_replica.client.set_engine_outputs(outputs)
        return stage_replica

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

    # ---- Stage-local helper views ----

    @staticmethod
    def first_replica_clients(stage_pools: Sequence[StagePool]) -> list[Any]:
        """Return one representative client per logical stage."""
        return [pool.replicas[0].client for pool in stage_pools]

    @staticmethod
    def bound_stage_clients(
        stage_pools: Sequence[StagePool],
        request_id: str,
    ) -> list[Any]:
        """Return stage clients, preferring the request-bound route when present."""
        clients: list[Any] = []
        for pool in stage_pools:
            bound = pool.get_bound_replica(request_id)
            clients.append(bound.client if bound is not None else pool.stage_client)
        return clients

    @staticmethod
    def build_kv_sender_info(
        stage_pools: Sequence[StagePool],
        sender_stage_ids: list[int],
        *,
        request_id: str | None = None,
    ) -> dict[int, dict[str, Any]] | None:
        """Build per-request sender info for diffusion KV-transfer receivers."""
        sender_infos: dict[int, dict[str, Any]] = {}
        for sender_stage_id in dict.fromkeys(sender_stage_ids):
            if sender_stage_id < 0 or sender_stage_id >= len(stage_pools):
                continue

            sender_pool = stage_pools[sender_stage_id]
            sender_replica = sender_pool.get_bound_replica(request_id) if request_id is not None else None
            sender_stage = sender_replica.client if sender_replica is not None else sender_pool.stage_client
            get_sender_info = getattr(sender_stage, "get_kv_sender_info", None)
            if not callable(get_sender_info):
                continue

            sender_info = get_sender_info()
            if not sender_info:
                logger.warning(
                    "[StagePool] Stage-%s has no KV sender info available",
                    sender_stage_id,
                )
                continue

            sender_infos[sender_stage_id] = sender_info

        return sender_infos or None

    # ---- Stage-local admission ----

    def _admit_llm_request(
        self,
        request_id: str,
        request: Any,
        prompt_text: Any,
        *,
        affinity_from: StageReplica | None = None,
        affinity_request_id: str | None = None,
    ) -> StageReplica:
        """Select a route and register the request on its output processor."""
        stage_replica = self.select_replica(
            request_id,
            affinity_from=affinity_from,
            affinity_request_id=affinity_request_id,
        )
        stage_replica.output_processor.add_request(
            request=request,
            prompt=prompt_text,
            parent_req=None,
            request_index=0,
            queue=None,
        )
        return stage_replica

    async def submit_initial(
        self,
        request_id: str,
        req_state: OrchestratorRequestState,
        request: Any,
        *,
        prompt_text: Any = None,
        affinity_from: StageReplica | None = None,
        affinity_request_id: str | None = None,
    ) -> StageReplica:
        """Submit a stage-entry request into this pool."""
        params = req_state.sampling_params_list[self.stage_id]
        if self.stage_type == "diffusion":
            stage_replica = self.select_replica(
                request_id,
                affinity_from=affinity_from,
                affinity_request_id=affinity_request_id,
            )
            if isinstance(request, list):
                await stage_replica.client.add_batch_request_async(request_id, request, params)
            else:
                await stage_replica.client.add_request_async(request_id, request, params)
            return stage_replica

        stage_replica = self._admit_llm_request(
            request_id,
            request,
            prompt_text,
            affinity_from=affinity_from,
            affinity_request_id=affinity_request_id,
        )
        await stage_replica.client.add_request_async(request)
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

    async def submit_from_upstream(
        self,
        request_id: str,
        req_state: OrchestratorRequestState,
        output: Any,
        *,
        source_pool: StagePool,
        stage_pools: Sequence[StagePool],
        companion_request_ids: dict[str, str] | None = None,
    ) -> StageReplica:
        """Submit a downstream request using an upstream stage output."""
        params = req_state.sampling_params_list[self.stage_id]
        stage_replica = self.select_replica(request_id)
        source_replica = source_pool.get_bound_replica(request_id)
        if source_replica is None:
            raise KeyError(f"No bound source replica for req={request_id} in stage {source_pool.stage_id}")

        if self.stage_type == "diffusion":
            source_pool.set_bound_engine_outputs(request_id, [output])
            if stage_replica.client.custom_process_input_func is not None:
                stage_list = self.bound_stage_clients(stage_pools, request_id)
                diffusion_prompt = stage_replica.client.custom_process_input_func(
                    stage_list,
                    stage_replica.client.engine_input_source,
                    req_state.prompt,
                    getattr(stage_replica.client, "requires_multimodal_data", False),
                )
            else:
                diffusion_prompt = req_state.prompt

            if companion_request_ids:
                from vllm_omni.inputs.data import OmniDiffusionSamplingParams

                if isinstance(params, OmniDiffusionSamplingParams):
                    params = copy.deepcopy(params)
                    params.cfg_kv_request_ids = companion_request_ids
                    logger.info(
                        "[StagePool] Attaching cfg_kv_request_ids=%s to req %s",
                        companion_request_ids,
                        request_id,
                    )

            source_stage_ids = list(getattr(stage_replica.client, "engine_input_source", None) or [source_pool.stage_id])
            kv_sender_info = self.build_kv_sender_info(stage_pools, source_stage_ids, request_id=request_id)
            if isinstance(diffusion_prompt, list):
                await stage_replica.client.add_batch_request_async(
                    request_id,
                    diffusion_prompt,
                    params,
                    kv_sender_info=kv_sender_info,
                )
            else:
                await stage_replica.client.add_request_async(
                    request_id,
                    diffusion_prompt,
                    params,
                    kv_sender_info=kv_sender_info,
                )
            return stage_replica

        source_pool.set_bound_engine_outputs(request_id, [output])
        stage_list = self.bound_stage_clients(stage_pools, request_id)
        try:
            next_inputs = stage_replica.client.process_engine_inputs(
                stage_list=stage_list,
                prompt=req_state.prompt,
                source_client=source_replica.client,
            )
        except Exception:
            logger.exception(
                "[StagePool] req=%s process_engine_inputs FAILED for stage-%s replica-%s",
                request_id,
                self.stage_id,
                stage_replica.replica_index,
            )
            raise

        for next_input in next_inputs:
            request = build_engine_core_request_from_tokens(
                request_id=request_id,
                prompt=next_input,
                params=params,
                model_config=stage_replica.vllm_config.model_config,
            )
            request.external_req_id = request.request_id
            stage_replica.output_processor.add_request(
                request=request,
                prompt=None,
                parent_req=None,
                request_index=0,
                queue=None,
            )
            await stage_replica.client.add_request_async(request)
        return stage_replica

    async def prewarm(
        self,
        request_id: str,
        stage0_request: Any,
        req_state: OrchestratorRequestState,
        *,
        stage_pools: Sequence[StagePool],
    ) -> StageReplica:
        """Pre-submit this stage for async-chunk mode."""
        prompt_token_ids = getattr(stage0_request, "prompt_token_ids", None)
        if prompt_token_ids is None:
            raise ValueError(f"async_chunk prewarm missing prompt_token_ids for req={request_id}")

        params = req_state.sampling_params_list[self.stage_id]
        stage_replica = self.select_replica(request_id)

        if self.stage_type == "diffusion":
            source_stage_ids = list(getattr(stage_replica.client, "engine_input_source", None) or [self.stage_id - 1])
            kv_sender_info = self.build_kv_sender_info(stage_pools, source_stage_ids, request_id=request_id)
            await stage_replica.client.add_request_async(
                request_id,
                req_state.prompt,
                params,
                kv_sender_info=kv_sender_info,
            )
            return stage_replica

        try:
            next_prompt_len = max(1, compute_talker_prompt_ids_length(prompt_token_ids))
        except Exception:
            next_prompt_len = max(1, len(prompt_token_ids))

        original_prompt = req_state.prompt
        if isinstance(original_prompt, dict):
            base_input = copy.deepcopy(original_prompt)
        else:
            base_input = {}
        base_input["prompt_token_ids"] = [0] * next_prompt_len
        base_input["multi_modal_data"] = None
        base_input["mm_processor_kwargs"] = None

        request = build_engine_core_request_from_tokens(
            request_id=request_id,
            prompt=base_input,
            params=params,
            model_config=stage_replica.vllm_config.model_config,
        )
        request.external_req_id = request.request_id
        stage_replica.output_processor.add_request(
            request=request,
            prompt=None,
            parent_req=None,
            request_index=0,
            queue=None,
        )
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
        processor = stage_replica.output_processor
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
                    poll_results.append(StagePollResult(stage_id=self.stage_id, outputs=[output]))
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
                    stage_id=self.stage_id,
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
                    stage_replica.stage_id,
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
                    stage_replica.stage_id,
                    stage_replica.replica_index,
                )
            except Exception as e:
                logger.warning(
                    "[StagePool] Failed to shutdown stage %d replica %d: %s",
                    stage_replica.stage_id,
                    stage_replica.replica_index,
                    e,
                )
