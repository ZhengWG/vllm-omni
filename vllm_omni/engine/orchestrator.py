"""
Orchestrator for vLLM-Omni multi-stage runtime.

Runs inside a background thread with its own asyncio event loop.
Owns logical request progression across stage pools and handles
stage-to-stage transfer logic.
"""

from __future__ import annotations

import asyncio
import time as _time
from dataclasses import dataclass, field
from typing import Any

import janus
from vllm.logger import init_logger
from vllm.v1.engine import EngineCoreOutputs

from vllm_omni.engine.cfg_companion_tracker import CfgCompanionTracker
from vllm_omni.engine.stage_pool import StagePool

logger = init_logger(__name__)


@dataclass
class OrchestratorRequestState:
    """Per-request bookkeeping inside the Orchestrator."""

    request_id: str
    prompt: Any = None
    sampling_params_list: list[Any] = field(default_factory=list)
    final_stage_id: int = -1

    # Metrics: timestamp when request was submitted to each stage.
    stage_submit_ts: dict[int, float] = field(default_factory=dict)


class Orchestrator:
    """Runs inside a background thread's asyncio event loop."""

    def __init__(
        self,
        request_async_queue: janus.AsyncQueue[dict[str, Any]],
        output_async_queue: janus.AsyncQueue[dict[str, Any]],
        rpc_async_queue: janus.AsyncQueue[dict[str, Any]],
        stage_pools: list[StagePool],
        *,
        async_chunk: bool = False,
    ) -> None:
        self.request_async_queue = request_async_queue
        self.output_async_queue = output_async_queue
        self.rpc_async_queue = rpc_async_queue

        self.async_chunk = bool(async_chunk)
        self.num_stages = len(stage_pools)
        self.stage_pools: list[StagePool] = stage_pools

        self.request_states: dict[str, OrchestratorRequestState] = {}
        self._cfg_tracker = CfgCompanionTracker()

        self._shutdown_event = asyncio.Event()
        self._stages_shutdown = False

    async def run(self) -> None:
        """Main entry point for the Orchestrator event loop."""
        logger.info("[Orchestrator] Starting event loop")

        request_task = asyncio.create_task(self._request_handler(), name="orchestrator-request-handler")
        output_task = asyncio.create_task(
            self._orchestration_output_handler(),
            name="orchestrator-stage-output-handler",
        )

        try:
            await asyncio.gather(request_task, output_task)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("[Orchestrator] Fatal error in orchestrator tasks")
            raise
        finally:
            self._shutdown_event.set()
            for task in (request_task, output_task):
                if not task.done():
                    task.cancel()
            try:
                await asyncio.gather(request_task, output_task, return_exceptions=True)
            except Exception:
                pass

            self._shutdown_stages()

            loop = asyncio.get_running_loop()
            pending = [t for t in asyncio.all_tasks(loop) if t is not asyncio.current_task() and not t.done()]
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

    async def _request_handler(self) -> None:
        """Read messages from the main thread via request_async_queue."""
        while True:
            msg = await self.request_async_queue.get()
            msg_type = msg.get("type")

            if msg_type == "add_request":
                await self._handle_add_request(msg)
            elif msg_type == "streaming_update":
                await self._handle_streaming_update(msg)
            elif msg_type == "add_companion_request":
                await self._handle_add_companion(msg)
            elif msg_type == "abort":
                await self._handle_abort(msg)
            elif msg_type == "collective_rpc":
                await self._handle_collective_rpc(msg)
            elif msg_type == "shutdown":
                logger.info("[Orchestrator] Received shutdown signal")
                self._shutdown_event.set()
                self._shutdown_stages()
                break
            else:
                logger.warning("[Orchestrator] Unknown message type: %s", msg_type)

    async def _orchestration_output_handler(self) -> None:
        """Poll all stages, handle transfers, send final outputs to main."""
        try:
            await self._orchestration_loop()
        except asyncio.CancelledError:
            logger.debug("[Orchestrator] _orchestration_output_handler cancelled")
            return

    async def _orchestration_loop(self) -> None:
        """Poll stage pools and route logical outputs."""
        while not self._shutdown_event.is_set():
            idle = True
            for pool in self.stage_pools:
                if self._shutdown_event.is_set():
                    return

                poll_results = await pool.poll_ready_outputs(timeout_s=0.001)
                if not poll_results:
                    continue

                idle = False
                for poll_result in poll_results:
                    if self._shutdown_event.is_set():
                        return

                    stage_id = poll_result.stage_id
                    if poll_result.raw_outputs is not None:
                        await self._handle_kv_ready_raw_outputs(pool, poll_result.raw_outputs)

                    for output in poll_result.outputs:
                        req_state = self.request_states.get(output.request_id)
                        if req_state is None:
                            logger.warning(
                                "[Orchestrator] Dropping output for unknown req %s at stage-%s (known reqs: %s)",
                                output.request_id,
                                stage_id,
                                list(self.request_states.keys()),
                            )
                            continue

                        if getattr(output, "error", None) is not None:
                            await self._handle_stage_error(pool, output)
                            continue

                        stage_metrics = None
                        if output.finished:
                            stage_metrics = pool.build_stage_metrics(
                                output.request_id,
                                [output],
                                submit_ts=req_state.stage_submit_ts.get(stage_id, _time.time()),
                            )

                        await self._route_output(pool, output, req_state, stage_metrics)

            if idle:
                await asyncio.sleep(0.001)
            else:
                await asyncio.sleep(0)

    async def _handle_stage_error(self, source_pool: StagePool, output: Any) -> None:
        """Emit a frontend-visible error and clean up request state."""
        parent_id = self._get_cfg_parent_id(output.request_id)
        await self.output_async_queue.put(
            {
                "type": "error",
                "request_id": parent_id,
                "stage_id": source_pool.stage_id,
                "error": output.error,
            }
        )
        all_request_ids = [parent_id, *self._cfg_tracker.cleanup_parent(parent_id)]
        await self._abort_request_ids(all_request_ids)
        self._release_request_bindings(all_request_ids)
        for rid in all_request_ids:
            self.request_states.pop(rid, None)

    async def _route_output(
        self,
        source_pool: StagePool,
        output: Any,
        req_state: OrchestratorRequestState,
        stage_metrics: Any,
    ) -> None:
        """Route a processed output: send to frontend and/or forward."""
        stage_id = source_pool.stage_id
        req_id = output.request_id
        finished = output.finished
        submit_ts = req_state.stage_submit_ts.get(stage_id)

        if finished and self._cfg_tracker.is_companion(req_id):
            await self._handle_cfg_companion_ready(req_id)
            self._release_request_bindings([req_id])
            self.request_states.pop(req_id, None)
            return

        if source_pool.final_output:
            await self.output_async_queue.put(
                {
                    "type": "output",
                    "request_id": req_id,
                    "stage_id": stage_id,
                    "engine_outputs": output,
                    "metrics": stage_metrics,
                    "finished": finished and stage_id == req_state.final_stage_id,
                    "stage_submit_ts": submit_ts,
                }
            )
        elif stage_metrics is not None:
            await self.output_async_queue.put(
                {
                    "type": "stage_metrics",
                    "request_id": req_id,
                    "stage_id": stage_id,
                    "metrics": stage_metrics,
                    "stage_submit_ts": submit_ts,
                }
            )

        if (
            finished
            and stage_id < req_state.final_stage_id
            and not self.async_chunk
            and not self._next_stage_already_submitted(stage_id, req_state)
        ):
            if self._cfg_tracker.has_companions(req_id) and not self._cfg_tracker.all_companions_done(req_id):
                self._cfg_tracker.defer_parent(req_id, output, stage_id)
            else:
                await self._forward_to_next_stage(req_id, source_pool, output, req_state)

        if finished and stage_id == req_state.final_stage_id:
            companion_ids = self._cfg_tracker.cleanup_parent(req_id)
            self._release_request_bindings([req_id, *companion_ids])
            self.request_states.pop(req_id, None)

    def _next_stage_already_submitted(self, stage_id: int, req_state: OrchestratorRequestState) -> bool:
        return (stage_id + 1) in req_state.stage_submit_ts

    def _get_cfg_parent_id(self, request_id: str) -> str:
        """Resolve a request ID to the parent request for CFG companions."""
        if self._cfg_tracker.is_companion(request_id):
            return self._cfg_tracker.get_parent_id(request_id) or request_id
        return request_id

    async def _handle_cfg_companion_ready(self, req_id: str) -> None:
        """Mark a CFG companion as done; if all companions are done, flush deferred parent."""
        parent_id = self._cfg_tracker.on_companion_completed(req_id)
        if parent_id is None:
            return

        deferred = self._cfg_tracker.pop_pending_parent(parent_id)
        if deferred is None:
            return

        parent_state = self.request_states.get(parent_id)
        if parent_state is None:
            return

        stage_id = deferred["stage_id"]
        if self._next_stage_already_submitted(stage_id, parent_state):
            return

        await self._forward_to_next_stage(
            parent_id,
            self.stage_pools[stage_id],
            deferred["engine_outputs"],
            parent_state,
        )

    async def _handle_kv_ready_raw_outputs(
        self,
        source_pool: StagePool,
        raw_outputs: EngineCoreOutputs,
    ) -> None:
        """Forward split requests once stage-0 KV is ready."""
        if self.async_chunk:
            return

        stage_id = source_pool.stage_id
        for raw_output in raw_outputs.outputs:
            kv_params = getattr(raw_output, "kv_transfer_params", None)
            if not (isinstance(kv_params, dict) and kv_params.get("kv_ready")):
                continue

            req_id = raw_output.request_id
            req_state = self.request_states.get(req_id)
            if req_state is None:
                continue
            if self._cfg_tracker.is_companion(req_id):
                await self._handle_cfg_companion_ready(req_id)
                continue
            if stage_id >= req_state.final_stage_id:
                continue
            if self._next_stage_already_submitted(stage_id, req_state):
                continue

            if self._cfg_tracker.has_companions(req_id) and not self._cfg_tracker.all_companions_done(req_id):
                self._cfg_tracker.defer_parent(req_id, raw_output, stage_id)
            else:
                await self._forward_to_next_stage(req_id, source_pool, raw_output, req_state)

    async def _forward_to_next_stage(
        self,
        req_id: str,
        source_pool: StagePool,
        output: Any,
        req_state: OrchestratorRequestState,
    ) -> None:
        """Forward output from the current logical stage to the next one."""
        next_logical = source_pool.stage_id + 1
        next_pool = self.stage_pools[next_logical]
        await next_pool.submit_from_upstream(
            request_id=req_id,
            req_state=req_state,
            output=output,
            source_pool=source_pool,
            stage_pools=self.stage_pools,
            companion_request_ids=self._cfg_tracker.get_companion_request_ids(req_id),
        )
        req_state.stage_submit_ts[next_logical] = _time.time()

    async def _handle_add_request(self, msg: dict[str, Any]) -> None:
        """Handle an add_request message from the main thread."""
        stage_id = 0
        request_id = msg["request_id"]
        prompt = msg["prompt"]
        original_prompt = msg.get("original_prompt", prompt)
        sampling_params_list = msg["sampling_params_list"]
        if not sampling_params_list:
            raise ValueError(f"Missing sampling params for stage 0. Got {len(sampling_params_list)} stage params.")
        final_stage_id = msg["final_stage_id"]

        logger.info(
            "[Orchestrator] _handle_add_request: stage=%s req=%s "
            "prompt_type=%s original_prompt_type=%s final_stage=%s "
            "num_sampling_params=%d",
            stage_id,
            request_id,
            type(prompt).__name__,
            type(original_prompt).__name__,
            final_stage_id,
            len(sampling_params_list),
        )

        req_state = OrchestratorRequestState(
            request_id=request_id,
            prompt=original_prompt,
            sampling_params_list=sampling_params_list,
            final_stage_id=final_stage_id,
        )
        self.request_states[request_id] = req_state
        req_state.stage_submit_ts[stage_id] = _time.time()

        await self.stage_pools[stage_id].submit_initial(
            request_id,
            req_state,
            prompt,
            prompt_text=msg.get("output_prompt_text"),
        )

        if self.async_chunk and stage_id == 0 and final_stage_id > 0:
            await self._prewarm_async_chunk_stages(request_id, prompt, req_state)

    async def _handle_streaming_update(self, msg: dict[str, Any]) -> None:
        """Handle a streaming_update message for an existing request."""
        stage_id = 0
        request_id = msg["request_id"]
        request = msg["prompt"]

        req_state = self.request_states.get(request_id)
        if req_state is None:
            logger.warning(
                "[Orchestrator] streaming_update for unknown req=%s, falling back to add_request",
                request_id,
            )
            fallback_msg = dict(msg)
            fallback_msg["type"] = "add_request"
            await self._handle_add_request(fallback_msg)
            return

        if "sampling_params_list" in msg and msg["sampling_params_list"]:
            req_state.sampling_params_list = msg["sampling_params_list"]

        req_state.stage_submit_ts[stage_id] = _time.time()
        await self.stage_pools[stage_id].submit_update(request_id, req_state, request)

    async def _prewarm_async_chunk_stages(
        self,
        request_id: str,
        stage0_request: Any,
        req_state: OrchestratorRequestState,
    ) -> None:
        """Pre-submit downstream stages for async-chunk mode."""
        if req_state.final_stage_id <= 0:
            return

        prompt_token_ids = getattr(stage0_request, "prompt_token_ids", None)
        if prompt_token_ids is None:
            logger.warning(
                "[Orchestrator] async_chunk prewarm skipped for req=%s: stage0 prompt_token_ids missing",
                request_id,
            )
            return

        for next_stage_id in range(1, req_state.final_stage_id + 1):
            await self.stage_pools[next_stage_id].prewarm(
                request_id,
                stage0_request,
                req_state,
                stage_pools=self.stage_pools,
            )
            req_state.stage_submit_ts[next_stage_id] = _time.time()

    async def _handle_add_companion(self, msg: dict[str, Any]) -> None:
        """Handle an add_companion_request message: submit companion to stage 0."""
        companion_id = msg["companion_id"]
        parent_id = msg["parent_id"]
        role = msg["role"]
        companion_prompt = msg["prompt"]
        sampling_params_list = msg["sampling_params_list"]

        parent_state = self.request_states.get(parent_id)
        if parent_state is None:
            logger.info(
                "[Orchestrator] Dropping CFG companion %s (role=%s): parent %s is no longer active",
                companion_id,
                role,
                parent_id,
            )
            return

        self._cfg_tracker.register_companion(parent_id, role, companion_id)

        companion_state = OrchestratorRequestState(
            request_id=companion_id,
            prompt=companion_prompt,
            sampling_params_list=sampling_params_list,
            final_stage_id=0,
        )
        self.request_states[companion_id] = companion_state
        companion_state.stage_submit_ts[0] = _time.time()

        stage0_pool = self.stage_pools[0]
        companion_replica = await stage0_pool.submit_initial(
            companion_id,
            companion_state,
            companion_prompt,
            prompt_text=msg.get("companion_prompt_text"),
            affinity_request_id=parent_id,
        )

        logger.info(
            "[Orchestrator] CFG companion submitted: %s (role=%s, parent=%s, stage-0 replica-%s)",
            companion_id,
            role,
            parent_id,
            companion_replica.replica_index,
        )

    async def _handle_abort(self, msg: dict[str, Any]) -> None:
        """Handle an abort message from the main thread."""
        request_ids = msg["request_ids"]
        all_ids_to_abort = self._cfg_tracker.abort_parents(request_ids)
        await self._abort_request_ids(all_ids_to_abort)
        self._release_request_bindings(all_ids_to_abort)
        for req_id in all_ids_to_abort:
            self.request_states.pop(req_id, None)
        logger.info("[Orchestrator] Aborted request(s) %s", request_ids)

    async def _abort_request_ids(self, request_ids: list[str]) -> None:
        """Broadcast abort requests to all stage pools."""
        if not request_ids:
            return
        for pool in self.stage_pools:
            await pool.abort_requests(request_ids)

    def _release_request_bindings(self, request_ids: list[str]) -> None:
        """Release all stage-local route bindings for the given request ids."""
        if not request_ids:
            return
        for pool in self.stage_pools:
            for req_id in request_ids:
                pool.release_binding(req_id)

    async def _handle_collective_rpc(self, msg: dict[str, Any]) -> None:
        """Handle a control-plane RPC request from the main thread."""
        rpc_id = msg["rpc_id"]
        method = msg["method"]
        timeout = msg.get("timeout")
        args = tuple(msg.get("args", ()))
        kwargs = dict(msg.get("kwargs") or {})
        requested_stage_ids = msg.get("stage_ids")

        target_pools: list[StagePool] = []
        if requested_stage_ids is None:
            target_pools.extend(self.stage_pools)
        else:
            for lid in requested_stage_ids:
                if 0 <= lid < self.num_stages:
                    target_pools.append(self.stage_pools[lid])

        results: list[Any] = []
        stage_ids: list[int] = []
        for pool in target_pools:
            stage_results = await pool.collective_rpc(
                method=method,
                timeout=timeout,
                args=args,
                kwargs=kwargs,
            )
            stage_ids.extend([pool.stage_id] * len(stage_results))
            results.extend(stage_results)

        await self.rpc_async_queue.put(
            {
                "type": "collective_rpc_result",
                "rpc_id": rpc_id,
                "method": method,
                "stage_ids": stage_ids,
                "results": results,
            }
        )

    def _shutdown_stages(self) -> None:
        """Shutdown all stage pools."""
        if self._stages_shutdown:
            return

        self._stages_shutdown = True
        total = sum(pool.num_replicas for pool in self.stage_pools)
        logger.info("[Orchestrator] Shutting down all %d client(s)", total)
        for pool in self.stage_pools:
            pool.shutdown()
