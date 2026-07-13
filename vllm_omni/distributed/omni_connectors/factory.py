# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from collections.abc import Callable
from typing import Any

from .utils.logging import get_connector_logger

try:
    from .connectors.base import OmniConnectorBase
    from .utils.config import ConnectorSpec
except ImportError:
    # Fallback for direct execution
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from omni_connectors.connectors.base import OmniConnectorBase
    from omni_connectors.utils.config import ConnectorSpec

logger = get_connector_logger(__name__)


class OmniConnectorFactory:
    """Factory for creating OmniConnectors."""

    _registry: dict[str, Callable[[dict[str, Any]], OmniConnectorBase]] = {}

    @classmethod
    def register_connector(cls, name: str, constructor: Callable[[dict[str, Any]], OmniConnectorBase]) -> None:
        """Register a connector constructor."""
        if name in cls._registry:
            raise ValueError(f"Connector '{name}' is already registered.")
        cls._registry[name] = constructor
        logger.debug(f"Registered connector: {name}")

    @classmethod
    def create_connector(cls, spec: ConnectorSpec) -> OmniConnectorBase:
        """Create a connector from specification."""
        if spec.name not in cls._registry:
            raise ValueError(f"Unknown connector: {spec.name}. Available: {list(cls._registry.keys())}")

        constructor = cls._registry[spec.name]
        try:
            connector = constructor(spec.extra)
            logger.info(f"Created connector: {spec.name}")
            return connector
        except Exception as e:
            logger.error(f"Failed to create connector {spec.name}: {e}")
            raise ValueError(f"Failed to create connector {spec.name}: {e}")

    @classmethod
    def list_registered_connectors(cls) -> list[str]:
        """List all registered connector names."""
        return list(cls._registry.keys())

    @classmethod
    def create_stage_connector(
        cls,
        stage_connector_config: Any,
        *,
        is_transfer_rank: bool = True,
    ) -> OmniConnectorBase | None:
        """Create a stage-level connector.
        Supports all historical `stage_connector_config` formats:

          - `None`                                         -> no connector
          - object with `.name`/`.extra`                   -> single connector
          - `{"name": ..., "extra": ...}`                  -> single connector (shared)
          - `{"input": {...}, "output": {...}}`            -> EdgeRoutedConnector, can specify either side

        Automatically injects `is_transfer_rank` into extra unless already configured.
        """
        config = stage_connector_config
        if config is None:
            return None
        if not isinstance(config, dict):
            config = {
                "name": getattr(config, "name", None),
                "extra": getattr(config, "extra", None),
            }

        def _make(spec_dict: dict[str, Any]) -> OmniConnectorBase:
            name = spec_dict.get("name") or "SharedMemoryConnector"
            if not isinstance(name, str) or not name.strip():
                raise RuntimeError("Invalid stage connector config: missing connector name")
            extra = spec_dict.get("extra") or {}
            if not isinstance(extra, dict):
                raise RuntimeError(f"Invalid extra config for connector {name}: expected dict")
            extra = {"is_transfer_rank": is_transfer_rank, **extra}
            return cls.create_connector(ConnectorSpec(name=name.strip(), extra=extra))

        if "input" in config or "output" in config:
            backends: dict[str, OmniConnectorBase | None] = {}
            for direction in ("input", "output"):
                spec = config.get(direction)
                if spec is not None and not isinstance(spec, dict):
                    raise RuntimeError(
                        f"Invalid {direction!r} connector spec: expected dict, got {type(spec).__name__}"
                    )
                backends[direction] = _make(spec) if spec else None
            from .connectors.edge_routed_connector import EdgeRoutedConnector

            connector = EdgeRoutedConnector(backends["input"], backends["output"])
            logger.info(f"Created stage connector: {connector!r}")
            return connector

        return _make(config)


# Register built-in connectors with lazy imports
def _create_mooncake_store_connector(config: dict[str, Any]) -> OmniConnectorBase:
    try:
        from .connectors.mooncake_store_connector import MooncakeStoreConnector
    except ImportError:
        # Fallback import
        import sys

        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from omni_connectors.connectors.mooncake_store_connector import MooncakeStoreConnector
    return MooncakeStoreConnector(config)


def _create_shm_connector(config: dict[str, Any]) -> OmniConnectorBase:
    try:
        from .connectors.shm_connector import SharedMemoryConnector
    except ImportError:
        # Fallback import
        import sys

        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from omni_connectors.connectors.shm_connector import SharedMemoryConnector
    return SharedMemoryConnector(config)


def _create_yuanrong_connector(config: dict[str, Any]) -> OmniConnectorBase:
    try:
        from .connectors.yuanrong_connector import YuanrongConnector
    except ImportError:
        import sys

        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from omni_connectors.connectors.yuanrong_connector import YuanrongConnector
    return YuanrongConnector(config)


def _create_yuanrong_transfer_engine_connector(config: dict[str, Any]) -> OmniConnectorBase:
    try:
        from vllm_omni.platforms.npu.omni_connectors import YuanrongTransferEngineConnector
    except ImportError as exc:
        raise ImportError(
            "YuanrongTransferEngineConnector is only available in the NPU platform "
            "environment. Install the Ascend/Yuanrong runtime dependencies before "
            "using this connector."
        ) from exc
    return YuanrongTransferEngineConnector(config)


def _create_mooncake_transfer_engine_connector(config: dict[str, Any]) -> OmniConnectorBase:
    try:
        from .connectors.mooncake_transfer_engine_connector import MooncakeTransferEngineConnector
    except ImportError:
        import sys

        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from omni_connectors.connectors.mooncake_transfer_engine_connector import MooncakeTransferEngineConnector
    return MooncakeTransferEngineConnector(config)


def _create_cuda_ipc_connector(config: dict[str, Any]) -> OmniConnectorBase:
    from .connectors.cuda_ipc_connector import CudaIPCConnector

    return CudaIPCConnector(config)


def _create_mori_transfer_engine_connector(config: dict[str, Any]) -> OmniConnectorBase:
    try:
        from .connectors.mori_transfer_engine_connector import MoriTransferEngineConnector
    except ImportError:
        import sys

        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from omni_connectors.connectors.mori_transfer_engine_connector import MoriTransferEngineConnector
    return MoriTransferEngineConnector(config)


# Register connectors
OmniConnectorFactory.register_connector("MooncakeStoreConnector", _create_mooncake_store_connector)
OmniConnectorFactory.register_connector("MooncakeTransferEngineConnector", _create_mooncake_transfer_engine_connector)
OmniConnectorFactory.register_connector("SharedMemoryConnector", _create_shm_connector)
OmniConnectorFactory.register_connector("YuanrongConnector", _create_yuanrong_connector)
OmniConnectorFactory.register_connector("CudaIPCConnector", _create_cuda_ipc_connector)
OmniConnectorFactory.register_connector("YuanrongTransferEngineConnector", _create_yuanrong_transfer_engine_connector)
OmniConnectorFactory.register_connector("MoriTransferEngineConnector", _create_mori_transfer_engine_connector)
# Backward-compatible aliases – will be removed in the future
OmniConnectorFactory.register_connector("MooncakeConnector", _create_mooncake_store_connector)
