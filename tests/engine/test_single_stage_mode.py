"""Unit tests for AsyncOmniEngine single-stage mode and OmniMasterServer."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest
from pytest_mock import MockerFixture
from vllm.v1.engine.utils import EngineZmqAddresses

from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
from vllm_omni.engine.stage_engine_core_client import StageEngineCoreClientBase
from vllm_omni.engine.stage_engine_startup import (
    OmniMasterServer,
    StageAllocation,
    StageCoordinatorAddresses,
    connect_remote_engine_cores,
    launch_omni_core_engines,
)
from vllm_omni.engine.stage_init_utils import LogicalStageInitPlan, ReplicaInitPlan

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_stage_cfg(stage_id: int, stage_type: str = "llm"):
    """Return a lightweight stage config mock."""
    return SimpleNamespace(
        stage_id=stage_id,
        stage_type=stage_type,
        runtime=SimpleNamespace(devices="0"),
        engine_args=SimpleNamespace(
            async_chunk=False,
            model_stage=None,
            engine_output_type=None,
        ),
    )


def _make_llm_plan(
    stage_idx: int,
    *,
    configured_stage_id: int,
    launch_mode: str,
    vllm_config: Any | None = None,
) -> LogicalStageInitPlan:
    stage_cfg = _make_stage_cfg(configured_stage_id)
    metadata = SimpleNamespace(
        stage_id=configured_stage_id,
        stage_type="llm",
        runtime_cfg={"devices": "0"},
        prompt_expand_func=None,
        final_output=False,
        final_output_type=None,
        default_sampling_params=SimpleNamespace(),
        custom_process_input_func=None,
        engine_input_source=[],
        engine_output_type="token_ids",
        replica_id=0,
    )
    return LogicalStageInitPlan(
        stage_idx=stage_idx,
        configured_stage_id=configured_stage_id,
        replicas=[
            ReplicaInitPlan(
                replica_id=0,
                num_replicas=1,
                launch_mode=launch_mode,
                stage_cfg=stage_cfg,
                metadata=metadata,
                stage_connector_spec={},
                omni_kv_connector=(None, None, None),
                stage_vllm_config=vllm_config
                or SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_size_local=1)),
                executor_class=object,
            )
        ],
    )


def _make_diffusion_plan(
    stage_idx: int,
    *,
    configured_stage_id: int,
    launch_mode: str,
) -> LogicalStageInitPlan:
    stage_cfg = _make_stage_cfg(configured_stage_id, stage_type="diffusion")
    metadata = SimpleNamespace(
        stage_id=configured_stage_id,
        stage_type="diffusion",
        runtime_cfg={"devices": "0"},
        prompt_expand_func=None,
        final_output=True,
        final_output_type="image",
        default_sampling_params=SimpleNamespace(),
        custom_process_input_func=None,
        engine_input_source=[],
        cfg_kv_collect_func=None,
        replica_id=0,
    )
    return LogicalStageInitPlan(
        stage_idx=stage_idx,
        configured_stage_id=configured_stage_id,
        replicas=[
            ReplicaInitPlan(
                replica_id=0,
                num_replicas=1,
                launch_mode=launch_mode,
                stage_cfg=stage_cfg,
                metadata=metadata,
                stage_connector_spec={},
                omni_kv_connector=(None, None, None),
            )
        ],
    )


# ---------------------------------------------------------------------------
# OmniMasterServer address pre-allocation
# ---------------------------------------------------------------------------


class TestOmniMasterServerAllocation:
    def test_public_address_and_port_properties_expose_registration_endpoint(self):
        server = OmniMasterServer(master_address="127.0.0.1", master_port=15000, stage_ids=[0])
        assert server.address == "127.0.0.1"
        assert server.port == 15000

    def test_allocations_created_for_each_stage_id(self):
        server = OmniMasterServer(master_address="127.0.0.1", master_port=15000, stage_ids=[0, 1, 2])
        assert set(server._allocations.keys()) == {0, 1, 2}

    def test_each_allocation_is_stage_allocation(self):
        server = OmniMasterServer(master_address="127.0.0.1", master_port=15000, stage_ids=[0, 1])
        for sid in (0, 1):
            assert isinstance(server._allocations[sid], StageAllocation)

    def test_get_zmq_addresses_returns_bind_addresses(self):
        server = OmniMasterServer(master_address="127.0.0.1", master_port=15002, stage_ids=[0])
        alloc = server._allocations[0]
        zmq_addrs = server.get_zmq_addresses(0)
        assert zmq_addrs.inputs == [alloc.input_bind_address]
        assert zmq_addrs.outputs == [alloc.output_bind_address]


# ---------------------------------------------------------------------------
# OmniMasterServer registration flow
# ---------------------------------------------------------------------------


class TestOmniMasterServerRegistration:
    def test_registration_reply_contains_handshake_address(self):
        import msgspec
        import zmq
        from vllm.utils.network_utils import get_open_port

        master_port = get_open_port()
        server = OmniMasterServer(master_address="127.0.0.1", master_port=master_port, stage_ids=[0])
        server.start()
        expected_hs = server._allocations[0].handshake_connect_address

        ctx = zmq.Context()
        try:
            sock = ctx.socket(zmq.DEALER)
            sock.connect(f"tcp://127.0.0.1:{master_port}")
            sock.send(msgspec.msgpack.encode({"stage_id": 0}))
            assert sock.poll(timeout=5_000)
            reply = msgspec.msgpack.decode(sock.recv())
            assert reply["handshake_address"] == expected_hs
        finally:
            sock.close(linger=0)
            ctx.term()
            server.stop()

    def test_registration_stores_stage_config(self):
        import msgspec
        import zmq
        from vllm.utils.network_utils import get_open_port

        master_port = get_open_port()
        server = OmniMasterServer(master_address="127.0.0.1", master_port=master_port, stage_ids=[0])
        server.start()

        stage_config = {"stage_id": 0, "stage_type": "llm"}
        ctx = zmq.Context()
        try:
            sock = ctx.socket(zmq.DEALER)
            sock.connect(f"tcp://127.0.0.1:{master_port}")
            sock.send(msgspec.msgpack.encode({"stage_id": 0, "stage_config": stage_config}))
            assert sock.poll(timeout=5_000)
            sock.recv()
            assert server.get_stage_config(0) == stage_config
        finally:
            sock.close(linger=0)
            ctx.term()
            server.stop()


# ---------------------------------------------------------------------------
# AsyncOmniEngine single_stage_mode detection in __init__
# ---------------------------------------------------------------------------


class TestSingleStageModeDetection:
    def _make_engine_no_thread(
        self,
        mocker: MockerFixture,
        *,
        stage_cfgs: list[Any] | None = None,
        **kwargs: Any,
    ) -> AsyncOmniEngine:
        mock_stage_configs = stage_cfgs or [_make_stage_cfg(0)]

        mocker.patch.object(
            AsyncOmniEngine,
            "_resolve_stage_configs",
            return_value=("/fake/path", mock_stage_configs),
        )
        mocker.patch.object(AsyncOmniEngine, "_bootstrap_orchestrator")
        mock_thread_cls = mocker.patch("threading.Thread")
        mock_future_cls = mocker.patch("concurrent.futures.Future")

        mock_future = mocker.Mock()
        mock_future.result.return_value = mocker.Mock()
        mock_future_cls.return_value = mock_future

        mock_thread = mocker.Mock()
        mock_thread.is_alive.return_value = False
        mock_thread_cls.return_value = mock_thread

        return AsyncOmniEngine(model="fake-model", **kwargs)

    def test_explicit_single_stage_mode_true(self, mocker: MockerFixture):
        engine = self._make_engine_no_thread(
            mocker,
            single_stage_mode=True,
            omni_master_address="127.0.0.1",
            omni_master_port=20000,
        )
        assert engine.single_stage_mode is True

    def test_stage_id_kwarg_promotes_to_single_stage_mode(self, mocker: MockerFixture):
        engine = self._make_engine_no_thread(
            mocker,
            stage_id=0,
            omni_master_address="127.0.0.1",
            omni_master_port=20001,
        )
        assert engine.single_stage_mode is True

    def test_stage_id_kwarg_sets_filter(self, mocker: MockerFixture):
        engine = self._make_engine_no_thread(
            mocker,
            stage_id=1,
            omni_master_address="127.0.0.1",
            omni_master_port=20002,
        )
        assert engine._single_stage_id_filter == 1

    def test_no_stage_id_no_single_stage_mode(self, mocker: MockerFixture):
        engine = self._make_engine_no_thread(mocker)
        assert engine.single_stage_mode is False
        assert engine._single_stage_id_filter is None


# ---------------------------------------------------------------------------
# AsyncOmniEngine single-stage initialization paths
# ---------------------------------------------------------------------------


class TestSingleStageInitialization:
    def _build_engine(self, stage_cfgs: list[Any], *, single_stage_mode: bool, stage_id_filter: int | None) -> AsyncOmniEngine:
        engine = object.__new__(AsyncOmniEngine)
        engine.model = "fake-model"
        engine.config_path = "/fake/stages.yaml"
        engine.num_stages = len(stage_cfgs)
        engine.stage_configs = stage_cfgs
        engine.single_stage_mode = single_stage_mode
        engine._single_stage_id_filter = stage_id_filter
        engine._omni_master_address = "127.0.0.1"
        engine._omni_master_port = 26000
        engine._omni_master_server = None
        engine.async_chunk = False
        engine.diffusion_batch_size = 2
        return engine

    def test_build_logical_stage_init_plans_marks_non_matching_stage_remote(self, mocker: MockerFixture):
        import vllm_omni.engine.async_omni_engine as engine_mod

        stage_cfgs = [_make_stage_cfg(7), _make_stage_cfg(11)]
        engine = self._build_engine(stage_cfgs, single_stage_mode=True, stage_id_filter=7)

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(engine_mod, "extract_stage_metadata", lambda cfg: SimpleNamespace(
            stage_id=cfg.stage_id,
            stage_type=getattr(cfg, "stage_type", "llm"),
            prompt_expand_func=None,
            runtime_cfg={},
        ))
        monkeypatch.setattr(engine_mod, "get_stage_connector_spec", lambda **_: {})
        monkeypatch.setattr(engine_mod, "resolve_omni_kv_config_for_stage", lambda *_: (None, None, None))
        monkeypatch.setattr(engine_mod, "build_engine_args_dict", lambda *_, **__: {})
        monkeypatch.setattr(engine_mod, "build_vllm_config", lambda *_, **__: (SimpleNamespace(), object))
        try:
            stage_plans, _ = engine._build_logical_stage_init_plans(None, [1, 1], {})
        finally:
            monkeypatch.undo()

        assert [plan.replicas[0].launch_mode for plan in stage_plans] == ["local", "remote"]

    def test_start_omni_master_server_uses_configured_stage_ids(self, mocker: MockerFixture):
        import vllm_omni.engine.async_omni_engine as engine_mod

        engine = self._build_engine([], single_stage_mode=True, stage_id_filter=7)
        mock_oms = mocker.Mock(spec=OmniMasterServer)
        mocker.patch.object(engine_mod, "OmniMasterServer", return_value=mock_oms)

        stage_plans = [
            _make_llm_plan(0, configured_stage_id=7, launch_mode="local"),
            _make_diffusion_plan(1, configured_stage_id=11, launch_mode="remote"),
        ]

        engine._start_omni_master_server(stage_plans)

        engine_mod.OmniMasterServer.assert_called_once_with(
            master_address="127.0.0.1",
            master_port=26000,
            stage_ids=[7, 11],
        )
        mock_oms.start.assert_called_once()

    def test_start_omni_master_server_duplicate_stage_ids_raise(self):
        engine = self._build_engine([], single_stage_mode=True, stage_id_filter=7)
        stage_plans = [
            _make_llm_plan(0, configured_stage_id=7, launch_mode="local"),
            _make_llm_plan(1, configured_stage_id=7, launch_mode="remote"),
        ]

        with pytest.raises(ValueError, match="Duplicate stage_id"):
            engine._start_omni_master_server(stage_plans)

    def test_initialize_stages_calls_master_server_only_in_single_stage_mode(self, mocker: MockerFixture):
        import vllm_omni.engine.async_omni_engine as engine_mod

        stage_cfgs = [_make_stage_cfg(0)]
        engine = self._build_engine(stage_cfgs, single_stage_mode=True, stage_id_filter=0)
        stage_plan = _make_llm_plan(0, configured_stage_id=0, launch_mode="local")
        client = SimpleNamespace(
            stage_type="llm",
            is_comprehension=False,
            final_output=True,
            final_output_type=None,
            default_sampling_params=SimpleNamespace(),
        )

        mocker.patch.object(engine_mod, "prepare_engine_environment")
        mocker.patch.object(engine_mod, "load_omni_transfer_config_for_model", return_value=None)
        mocker.patch.object(engine_mod, "compute_replica_layout", return_value=([1], {}))
        mocker.patch.object(engine, "_build_logical_stage_init_plans", return_value=([stage_plan], None))
        mock_start = mocker.patch.object(engine, "_start_omni_master_server")
        mocker.patch.object(engine, "_initialize_stage_replicas", return_value={0: [client]})
        mocker.patch.object(engine_mod, "build_stage0_input_processor", return_value=object())
        mocker.patch.object(engine_mod, "build_llm_stage_output_processor", return_value=object())

        engine._initialize_stages(stage_init_timeout=60)
        mock_start.assert_called_once()

        engine = self._build_engine(stage_cfgs, single_stage_mode=False, stage_id_filter=None)
        mocker.patch.object(engine_mod, "prepare_engine_environment")
        mocker.patch.object(engine_mod, "load_omni_transfer_config_for_model", return_value=None)
        mocker.patch.object(engine_mod, "compute_replica_layout", return_value=([1], {}))
        mocker.patch.object(engine, "_build_logical_stage_init_plans", return_value=([stage_plan], None))
        mock_start = mocker.patch.object(engine, "_start_omni_master_server")
        mocker.patch.object(engine, "_initialize_stage_replicas", return_value={0: [client]})
        mocker.patch.object(engine_mod, "build_stage0_input_processor", return_value=object())
        mocker.patch.object(engine_mod, "build_llm_stage_output_processor", return_value=object())

        engine._initialize_stages(stage_init_timeout=60)
        mock_start.assert_not_called()


class TestSingleStageReplicaInitialization:
    def test_initialize_llm_replica_remote_uses_connect_remote_engine_cores(self, mocker: MockerFixture):
        import vllm_omni.engine.async_omni_engine as engine_mod

        engine = object.__new__(AsyncOmniEngine)
        engine.model = "fake-model"
        engine.single_stage_mode = True
        engine._omni_master_server = mocker.Mock(spec=OmniMasterServer)
        engine._omni_master_server.get_stage_config.return_value = {"stage_id": 7, "stage_type": "llm"}

        fake_vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_size_local=1))
        fake_addresses = SimpleNamespace(inputs=["tcp://in"], outputs=["tcp://out"], frontend_stats_publish_address=None)
        fake_manager = mocker.Mock()
        fake_coordinator = mocker.Mock()

        @contextmanager
        def _fake_connect(**kwargs):
            yield fake_manager, fake_coordinator, fake_addresses

        plan = _make_llm_plan(0, configured_stage_id=7, launch_mode="remote", vllm_config=fake_vllm_config).replicas[0]
        sentinel_client = SimpleNamespace()

        mock_connect = mocker.patch.object(engine_mod, "connect_remote_engine_cores", side_effect=_fake_connect)
        mocker.patch.object(
            StageEngineCoreClientBase,
            "make_async_mp_client",
            side_effect=lambda **_: sentinel_client,
        )

        result = engine._initialize_llm_replica(plan, stage_init_timeout=60, llm_stage_launch_lock=threading.Lock())

        assert result is sentinel_client
        engine._omni_master_server.get_stage_config.assert_called_once_with(7, timeout_s=60)
        assert fake_vllm_config.parallel_config.data_parallel_size_local == 0
        assert mock_connect.call_args.kwargs["stage_id"] == 7

    def test_initialize_llm_replica_single_stage_local_uses_launch_omni_core_engines(self, mocker: MockerFixture):
        import vllm_omni.engine.async_omni_engine as engine_mod
        from vllm_omni.platforms import current_omni_platform

        engine = object.__new__(AsyncOmniEngine)
        engine.model = "fake-model"
        engine.single_stage_mode = True
        engine._omni_master_server = mocker.Mock(spec=OmniMasterServer)
        engine.stage_configs = []

        fake_vllm_config = SimpleNamespace(parallel_config=SimpleNamespace())
        fake_addresses = SimpleNamespace(inputs=["tcp://in"], outputs=["tcp://out"], frontend_stats_publish_address=None)

        @contextmanager
        def _fake_launch(**kwargs):
            yield mocker.Mock(), None, fake_addresses

        plan = _make_llm_plan(0, configured_stage_id=3, launch_mode="local", vllm_config=fake_vllm_config).replicas[0]
        sentinel_client = SimpleNamespace()

        device_env_var = current_omni_platform.device_control_env_var
        prev_device_env = os.environ.get(device_env_var)
        os.environ[device_env_var] = "0"

        mocker.patch.object(engine_mod, "setup_stage_devices")
        mocker.patch.object(engine_mod, "build_engine_args_dict", return_value={})
        mocker.patch.object(engine_mod, "acquire_device_locks", return_value=[])
        mocker.patch.object(engine_mod, "release_device_locks")
        mock_launch = mocker.patch.object(engine_mod, "launch_omni_core_engines", side_effect=_fake_launch)
        mocker.patch.object(
            StageEngineCoreClientBase,
            "make_async_mp_client",
            side_effect=lambda **_: sentinel_client,
        )
        try:
            result = engine._initialize_llm_replica(plan, stage_init_timeout=60, llm_stage_launch_lock=threading.Lock())
        finally:
            if prev_device_env is None:
                os.environ.pop(device_env_var, None)
            else:
                os.environ[device_env_var] = prev_device_env

        assert result is sentinel_client
        assert mock_launch.call_args.kwargs["stage_id"] == 3
        assert mock_launch.call_args.kwargs["stage_config"] is plan.stage_cfg

    def test_initialize_diffusion_replica_remote_uses_from_addresses(self, mocker: MockerFixture):
        import vllm_omni.engine.async_omni_engine as engine_mod

        engine = object.__new__(AsyncOmniEngine)
        engine.single_stage_mode = True
        engine.diffusion_batch_size = 4
        engine._omni_master_server = mocker.Mock(spec=OmniMasterServer)
        engine._omni_master_server.get_stage_config.return_value = {"stage_id": 11, "stage_type": "diffusion"}
        engine._omni_master_server.get_zmq_addresses.return_value = SimpleNamespace(
            inputs=["tcp://in"],
            outputs=["tcp://out"],
        )

        remote_metadata = _make_diffusion_plan(1, configured_stage_id=11, launch_mode="remote").replicas[0].metadata
        plan = _make_diffusion_plan(1, configured_stage_id=11, launch_mode="remote").replicas[0]
        sentinel_client = SimpleNamespace()

        mocker.patch.object(engine_mod, "extract_stage_metadata", return_value=remote_metadata)
        mock_from_addresses = mocker.patch.object(engine_mod.StageDiffusionClient, "from_addresses", return_value=sentinel_client)

        result = engine._initialize_diffusion_replica(plan, stage_init_timeout=60, stage_launch_lock=threading.Lock())

        assert result is sentinel_client
        engine._omni_master_server.get_stage_config.assert_called_once_with(11, timeout_s=60)
        mock_from_addresses.assert_called_once()

    def test_initialize_diffusion_replica_single_stage_local_registers_with_master(self, mocker: MockerFixture):
        import vllm_omni.engine.async_omni_engine as engine_mod
        from vllm_omni.platforms import current_omni_platform

        engine = object.__new__(AsyncOmniEngine)
        engine.model = "fake-model"
        engine.single_stage_mode = True
        engine.diffusion_batch_size = 4
        engine.stage_configs = []
        engine._omni_master_server = mocker.Mock(spec=OmniMasterServer)
        engine._omni_master_server.address = "127.0.0.1"
        engine._omni_master_server.port = 25000

        plan = _make_diffusion_plan(0, configured_stage_id=5, launch_mode="local").replicas[0]
        sentinel_client = SimpleNamespace()
        proc = mocker.Mock()

        device_env_var = current_omni_platform.device_control_env_var
        prev_device_env = os.environ.get(device_env_var)
        os.environ[device_env_var] = "0"

        mocker.patch.object(engine_mod, "setup_stage_devices")
        mocker.patch.object(engine_mod, "inject_kv_stage_info")
        mocker.patch.object(engine_mod, "build_diffusion_config", return_value="diffusion-config")
        mock_register = mocker.patch.object(
            engine_mod,
            "register_stage_with_omni_master",
            return_value=("tcp://hs", "tcp://req", "tcp://resp"),
        )
        mock_spawn = mocker.patch.object(
            engine_mod,
            "spawn_diffusion_proc",
            return_value=(proc, None, None, None),
        )
        mock_handshake = mocker.patch.object(engine_mod, "complete_diffusion_handshake")
        mock_from_addresses = mocker.patch.object(
            engine_mod.StageDiffusionClient,
            "from_addresses",
            return_value=sentinel_client,
        )

        try:
            result = engine._initialize_diffusion_replica(plan, stage_init_timeout=60, stage_launch_lock=threading.Lock())
        finally:
            if prev_device_env is None:
                os.environ.pop(device_env_var, None)
            else:
                os.environ[device_env_var] = prev_device_env

        assert result is sentinel_client
        mock_register.assert_called_once_with(
            omni_master_address="127.0.0.1",
            omni_master_port=25000,
            omni_stage_id=5,
            omni_stage_config=plan.stage_cfg,
            return_addresses=True,
        )
        mock_spawn.assert_called_once_with(
            "fake-model",
            "diffusion-config",
            handshake_address="tcp://hs",
            request_address="tcp://req",
            response_address="tcp://resp",
        )
        mock_handshake.assert_called_once_with(proc, "tcp://hs")
        mock_from_addresses.assert_called_once_with(
            plan.metadata,
            request_address="tcp://req",
            response_address="tcp://resp",
            proc=proc,
            batch_size=4,
        )


# ---------------------------------------------------------------------------
# Stage engine startup helpers
# ---------------------------------------------------------------------------


class TestConnectRemoteEngineCoresCoordinator:
    @staticmethod
    def _build_vllm_config(
        mocker: MockerFixture, *, dp_rank: int = 0, offline_mode: bool = False, needs_dp_coordinator: bool = True
    ) -> Any:
        parallel_config = mocker.Mock()
        parallel_config.data_parallel_size_local = 1
        parallel_config.data_parallel_size = 2
        parallel_config.data_parallel_rank = dp_rank
        parallel_config.data_parallel_rank_local = 0 if offline_mode else None

        vllm_config = mocker.Mock()
        vllm_config.parallel_config = parallel_config
        vllm_config.needs_dp_coordinator = needs_dp_coordinator
        vllm_config.model_config = mocker.Mock(is_moe=False)
        return vllm_config

    def test_uses_registered_coordinator_addresses(self, mocker: MockerFixture):
        vllm_config = self._build_vllm_config(mocker, dp_rank=0, offline_mode=False, needs_dp_coordinator=True)

        omni_master_server = mocker.Mock(spec=OmniMasterServer)
        omni_master_server.get_zmq_addresses.return_value = EngineZmqAddresses(
            inputs=["tcp://client-in"],
            outputs=["tcp://client-out"],
        )
        omni_master_server.get_allocation.return_value = mocker.Mock(handshake_bind_address="tcp://127.0.0.1:26001")
        omni_master_server.get_stage_coordinator_addresses.return_value = StageCoordinatorAddresses(
            coordinator_input="tcp://coord-in",
            coordinator_output="tcp://coord-out",
            frontend_stats_publish_address="tcp://stats",
        )

        @contextmanager
        def fake_socket_ctx(*args, **kwargs):
            yield mocker.Mock()

        mocker.patch("vllm_omni.engine.stage_engine_startup.zmq_socket_ctx", return_value=fake_socket_ctx())
        mock_wait = mocker.patch("vllm_omni.engine.stage_engine_startup._wait_for_omni_engine_startup")
        with connect_remote_engine_cores(
            vllm_config=vllm_config,
            omni_master_server=omni_master_server,
            stage_id=7,
        ) as (_, yielded_coordinator, yielded_addresses):
            assert yielded_coordinator is None
            assert yielded_addresses.coordinator_input == "tcp://coord-in"
            assert yielded_addresses.coordinator_output == "tcp://coord-out"
            assert yielded_addresses.frontend_stats_publish_address == "tcp://stats"

        omni_master_server.get_stage_coordinator_addresses.assert_called_once_with(7)
        mock_wait.assert_called_once()


class TestLaunchOmniCoreEngines:
    def test_registers_stage_once_and_reuses_handshake_for_all_local_engines(self, mocker: MockerFixture):
        parallel_config = mocker.Mock(
            data_parallel_size_local=2,
            data_parallel_size=4,
            data_parallel_rank=3,
        )
        vllm_config = mocker.Mock(parallel_config=parallel_config)

        omni_master_server = mocker.Mock(spec=OmniMasterServer)
        omni_master_server.address = "127.0.0.1"
        omni_master_server.port = 26000
        omni_master_server.get_allocation.return_value = mocker.Mock(handshake_bind_address="tcp://127.0.0.1:26001")

        stage_config = {"stage_id": 7, "stage_type": "llm"}
        local_engine_manager = mocker.Mock()

        @contextmanager
        def fake_socket_ctx(*args, **kwargs):
            yield mocker.Mock()

        mock_register = mocker.patch(
            "vllm_omni.engine.stage_engine_startup.register_stage_with_omni_master",
            return_value="tcp://127.0.0.1:26001",
        )
        mocker.patch("vllm_omni.engine.stage_engine_startup.zmq_socket_ctx", return_value=fake_socket_ctx())
        mock_manager_cls = mocker.patch(
            "vllm_omni.engine.stage_engine_startup.CoreEngineProcManager",
            return_value=local_engine_manager,
        )
        mocker.patch("vllm_omni.engine.stage_engine_startup.wait_for_engine_startup")
        with launch_omni_core_engines(
            vllm_config=vllm_config,
            executor_class=mocker.Mock(),
            log_stats=False,
            omni_master_server=omni_master_server,
            stage_id=7,
            stage_config=stage_config,
        ) as (yielded_manager, yielded_coordinator, yielded_addresses):
            assert yielded_manager is local_engine_manager
            assert yielded_coordinator is None
            assert yielded_addresses is not None

        mock_register.assert_called_once_with(
            omni_master_address="127.0.0.1",
            omni_master_port=26000,
            omni_stage_id=7,
            omni_stage_config=stage_config,
            coordinator=None,
        )
        manager_kwargs = mock_manager_cls.call_args.kwargs
        assert manager_kwargs["local_engine_count"] == 2
        assert manager_kwargs["start_index"] == 3
        assert manager_kwargs["local_start_index"] == 0
        assert manager_kwargs["handshake_address"] == "tcp://127.0.0.1:26001"
