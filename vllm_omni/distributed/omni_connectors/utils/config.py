# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from dataclasses import dataclass, field
from typing import Any

from .logging import get_connector_logger

logger = get_connector_logger(__name__)

TRANSFER_ENGINE_CONNECTOR_NAMES = frozenset(
    {
        "MooncakeTransferEngineConnector",
        "MoriTransferEngineConnector",
        "YuanrongTransferEngineConnector",
    }
)


def get_stage_connector_role(model_config: Any) -> str | None:
    """Return the configured stage connector direction, if explicit.

    For the dual ``{"input": ..., "output": ...}`` config shape the input
    edge takes precedence, matching the legacy single-spec resolution where
    ``from_stage_*`` entries were enumerated first: a stage that receives
    chunks keeps reporting ``"receiver"`` even when it also owns an output
    edge.
    """
    connector_config = getattr(model_config, "stage_connector_config", None)
    if isinstance(connector_config, dict) and ("input" in connector_config or "output" in connector_config):
        for direction, default_role in (("input", "receiver"), ("output", "sender")):
            sub = connector_config.get(direction)
            if isinstance(sub, dict) and sub:
                extra = sub.get("extra")
                role = extra.get("role") if isinstance(extra, dict) else None
                return role if isinstance(role, str) else default_role
        return None
    if isinstance(connector_config, dict):
        extra = connector_config.get("extra")
    else:
        extra = getattr(connector_config, "extra", None)
    if isinstance(extra, dict):
        role = extra.get("role")
        return role if isinstance(role, str) else None
    return None


def stage_receives_chunks(model_config: Any) -> bool:
    """Whether connector chunks, rather than the orchestrator, feed a stage."""
    return get_stage_connector_role(model_config) != "sender"


def stage_sends_async_output(model_config: Any) -> bool:
    """Whether async output should be partitioned for connector transport."""
    role = get_stage_connector_role(model_config)
    if role is not None:
        return role == "sender"
    # Preserve legacy partitioning while keeping stage-0 orchestrator bridges
    # on the normal RequestOutput path.
    return getattr(model_config, "stage_id", None) != 0


@dataclass
class ConnectorSpec:
    """Specification for a connector instance."""

    name: str  # e.g., "MooncakeStoreConnector", "SharedMemoryConnector", "YuanrongConnector"
    extra: dict[str, Any] = field(default_factory=dict)  # backend-specific config


def build_stage_connector_config(spec: dict[str, Any], stage_id: int) -> dict[str, Any]:
    """Normalise a stage_connector_spec into stage_connector_config.

    Accepts two formats:
      - Legacy:  ``{"name": "...", "extra": {...}}``
      - Dual:    ``{"input": {"name": ..}, "output": {"name": ..}}``

    Returns a dict consumable by ``OmniConnectorFactory.create_stage_connector``.
    """

    def _with_stage_id(s: dict) -> dict:
        extra = dict(s.get("extra") or {})
        extra["stage_id"] = stage_id
        return {"name": s.get("name", "SharedMemoryConnector"), "extra": extra}

    if "input" in spec or "output" in spec:
        cfg: dict[str, Any] = {}
        for direction in ("input", "output"):
            sub = spec.get(direction)
            if isinstance(sub, dict) and sub:
                cfg[direction] = _with_stage_id(sub)
        if not cfg:
            return {"name": "SharedMemoryConnector", "extra": {"stage_id": stage_id}}
        # Backward-compat: legacy readers do connector_cfg.get("extra"); expose
        # a merged top-level name/extra so they keep working on the dual shape.
        # ``role`` is direction-specific and stripped from the merged view —
        # direction-aware readers use ``get_stage_connector_role`` instead.
        merged = stage_connector_extra(cfg)
        merged.pop("role", None)
        cfg["extra"] = merged
        cfg["name"] = (cfg.get("output") or cfg.get("input")).get("name", "SharedMemoryConnector")
        return cfg

    return _with_stage_id(spec)


def stage_connector_extra(connector_cfg: Any) -> dict[str, Any]:
    """Extract the connector ``extra`` from a stage_connector_config of either
    shape: legacy ``{"name","extra"}`` or dual ``{"input":{...},"output":{...}}``
    (extras merged, output last).  Tolerates a non-dict (object with ``.extra``)
    and missing keys; returns ``{}`` if absent."""
    if connector_cfg is None:
        return {}
    if not isinstance(connector_cfg, dict):
        extra = getattr(connector_cfg, "extra", None)
        return dict(extra) if isinstance(extra, dict) else {}
    if "input" in connector_cfg or "output" in connector_cfg:
        merged: dict[str, Any] = {}
        for direction in ("input", "output"):
            sub = connector_cfg.get(direction)
            if isinstance(sub, dict) and isinstance(sub.get("extra"), dict):
                merged.update(sub["extra"])
        return merged
    extra = connector_cfg.get("extra")
    return dict(extra) if isinstance(extra, dict) else {}


@dataclass
class OmniTransferConfig:
    """
    Top-level configuration for OmniConnector system.
    Members:
        connectors: A dictionary of connectors, keyed by (from_stage, to_stage).
        default_connector: The default connector to use if no connector is specified for an edge.
    """

    # Direct mapping: (from_stage, to_stage) -> connector
    connectors: dict[tuple[str, str], ConnectorSpec] = field(default_factory=dict)
    default_connector: ConnectorSpec | None = None

    def get_connector_for_edge(self, from_stage: str, to_stage: str) -> ConnectorSpec | None:
        """Get connector spec for a specific edge."""
        edge_key = (from_stage, to_stage)
        return self.connectors.get(edge_key, self.default_connector)

    def has_connector_for_edge(self, from_stage: str, to_stage: str) -> bool:
        """Check if there's a connector configured for the edge."""
        return self.get_connector_for_edge(from_stage, to_stage) is not None
