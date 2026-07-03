# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Direction-routing composite connector.

One connector per stage, routing each call by the direction already present
in the connector API: ``put(from_stage, to_stage, ...)`` is this stage
sending (output edge), ``get(from_stage, to_stage, ...)`` is this stage
receiving (input edge).  This keeps the framework (mixin, transfer adapter,
stage input processors) on the single-connector contract while allowing
heterogeneous per-direction backends, e.g. CudaIPC in / SHM out.
"""

from typing import Any

from ..utils.logging import get_connector_logger
from .base import OmniConnectorBase

logger = get_connector_logger(__name__)


class EdgeRoutedConnector(OmniConnectorBase):
    """Route put()/get() to per-direction backend connectors.

    Built by ``OmniConnectorFactory.create_stage_connector`` for the dual
    ``{"input": {...}, "output": {...}}`` config shape.  Either side may be
    ``None`` (edge not configured): ``put`` then reports failure and ``get``
    returns None, mirroring the no-connector no-op behaviour callers already
    handle via ``can_send`` / ``can_recv``.
    """

    def __init__(
        self,
        input_connector: OmniConnectorBase | None,
        output_connector: OmniConnectorBase | None,
    ):
        self._input = input_connector
        self._output = output_connector
        # Shared attrs consumed by adapter/processor helpers.  ``config`` is
        # the merged extra view (output wins on collision — send-edge helpers
        # such as codec sizing read it); ``stage_id`` is identical on both
        # backends (injected from the same stage config).
        in_cfg = getattr(input_connector, "config", None) or {}
        out_cfg = getattr(output_connector, "config", None) or {}
        self.config: dict[str, Any] = {**in_cfg, **out_cfg}
        self.stage_id = getattr(output_connector or input_connector, "stage_id", -1)

    def __repr__(self) -> str:
        _name = lambda c: type(c).__name__ if c is not None else None  # noqa: E731
        return f"EdgeRoutedConnector(input={_name(self._input)}, output={_name(self._output)})"

    # --- Capabilities ---

    @property
    def can_send(self) -> bool:
        return self._output is not None

    @property
    def can_recv(self) -> bool:
        return self._input is not None

    @property
    def supports_gpu_tensor(self) -> bool:  # type: ignore[override]
        # Unqualified capability reads come from send-side payload builders
        # (keep_on_gpu decisions), so answer for the output edge.
        return bool(self._output is not None and self._output.supports_gpu_tensor)

    def supports_gpu_tensor_for(self, from_stage: str, to_stage: str) -> bool:
        backend = self._route(from_stage, to_stage)
        return bool(backend is not None and backend.supports_gpu_tensor)

    def _route(self, from_stage: str, to_stage: str) -> OmniConnectorBase | None:
        """Output edge when this stage is the source, else input edge."""
        if str(from_stage) == str(self.stage_id):
            return self._output
        if str(to_stage) == str(self.stage_id):
            return self._input
        logger.warning(
            "EdgeRoutedConnector(stage=%s): edge (%s -> %s) matches neither direction",
            self.stage_id,
            from_stage,
            to_stage,
        )
        return None

    # --- Data plane ---

    def put(
        self, from_stage: str, to_stage: str, put_key: str, data: Any, **kwargs: Any
    ) -> tuple[bool, int, dict[str, Any] | None]:
        if self._output is None:
            logger.debug("EdgeRoutedConnector.put skipped: no output edge (stage=%s)", self.stage_id)
            return False, 0, None
        return self._output.put(from_stage, to_stage, put_key, data, **kwargs)

    def get(
        self, from_stage: str, to_stage: str, get_key: str, metadata: dict[str, Any] | None = None
    ) -> tuple[Any, int] | None:
        if self._input is None:
            return None
        return self._input.get(from_stage, to_stage, get_key, metadata)

    # --- Lifecycle ---

    def _backends(self) -> list[OmniConnectorBase]:
        """Distinct live backends (legacy single spec may share one instance)."""
        return list(dict.fromkeys(c for c in (self._input, self._output) if c is not None))

    def cleanup(self, request_id: str) -> None:
        for backend in self._backends():
            backend.cleanup(request_id)

    def health(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "input": self._input.health() if self._input is not None else None,
            "output": self._output.health() if self._output is not None else None,
        }

    def close(self) -> None:
        for backend in self._backends():
            try:
                backend.close()
            except Exception:
                logger.warning("EdgeRoutedConnector: backend close failed", exc_info=True)
        self._input = None
        self._output = None
