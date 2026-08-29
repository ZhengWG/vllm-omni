# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Same-node GPU-direct connector built on PyTorch's native CUDA IPC.

``TorchIpcConnector`` extends :class:`SharedMemoryConnector` with a
device-to-device data plane for large payload tensors:

* **Control plane** (unchanged): the msgpack payload travels through the
  existing key-addressed ``/dev/shm`` segments.
* **Data plane**: CUDA tensors in the payload are replaced by small markers
  carrying a :func:`torch.multiprocessing.reductions.reduce_tensor` handle
  (a few hundred bytes).  The receiver rebuilds a zero-copy view of the
  sender's memory and issues one device-to-device copy onto its own device.

PyTorch owns all of the hard parts: CUDA IPC handle export/open (cached per
allocator segment), producer-side event synchronization (recorded at share
time, waited at rebuild time), and cross-process storage refcounting (the
sender's storage stays alive until the receiver's rebuilt view is freed).

Placement contract
------------------
This connector never decides *which* tensors ride the GPU plane — that is
the payload builders' job via :mod:`.gpu_placement`, keyed on the
``gpu_tensor_keys`` configured for the edge (a consumer-locality contract:
list the keys the downstream stage consumes on GPU).  By the time ``put()``
runs, eligible tensors are CUDA and everything else is CPU; ``put()``
simply exports every CUDA tensor it sees.

Stream architecture
-------------------
Both ends run the data plane on dedicated streams, mirroring the async CPU
materialization pipeline, so per-packet transfers never synchronize either
stage's compute stream:

* **Sender**: exports happen under a pack stream that waits one produce
  event recorded on the sender's default stream (which ordered the payload
  writes under the repository's PTDS-off model).  torch's share-time IPC
  event therefore captures only the payload, not the queued step.
* **Receiver**: rebuild (which waits the share-time event) and the D2D copy
  run under a recv stream; the consuming default stream pays exactly one
  device-side ordering edge per ``get()`` — the data dependency itself.
  Rebuilt views are held until the copy event fires; destination blocks are
  pinned to their consumer stream via ``record_stream``.

With no host synchronization and no tensor-byte serialization on either
side, the plane is latency-neutral for small packets and strictly cheaper
than the host round-trip for anything the downstream stage consumes on GPU.

Deployment requirements (opt-in profile, enforced at runtime):

* Sender and receiver stages on the same host.
* The receiver must be able to open the sender's allocation: either both
  stages share one physical GPU, or peer access (NVLink / PCIe P2P) exists
  between the two devices.  When the import fails, ``get()`` raises with an
  actionable message instead of degrading silently — this transport is an
  explicit per-edge choice, not a best-effort default.
"""

from collections import deque
from typing import Any

import torch

from ..utils.logging import get_connector_logger
from .shm_connector import SharedMemoryConnector

logger = get_connector_logger(__name__)

# Marker key for a payload slot whose tensor travels via CUDA IPC.
_TORCH_IPC_MARKER = "__omni_torch_ipc__"


def _is_marker(value: Any) -> bool:
    return isinstance(value, dict) and value.get(_TORCH_IPC_MARKER) is True


def _payload_has_cuda(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return value.is_cuda
    if isinstance(value, dict):
        return any(_payload_has_cuda(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_payload_has_cuda(v) for v in value)
    if hasattr(value, "__struct_fields__"):
        return any(_payload_has_cuda(getattr(value, f)) for f in value.__struct_fields__)
    return False


def _payload_has_marker(value: Any) -> bool:
    if _is_marker(value):
        return True
    if isinstance(value, dict):
        return any(_payload_has_marker(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_payload_has_marker(v) for v in value)
    if hasattr(value, "__struct_fields__"):
        return any(_payload_has_marker(getattr(value, f)) for f in value.__struct_fields__)
    return False


# ---------------------------------------------------------------------- #
#  Restricted codec for torch's CUDA-IPC reduce spec.
#
#  ``reduce_tensor`` returns ``(rebuild_cuda_tensor, args)`` where ``args``
#  contains only primitives, bytes handles, ``torch.dtype``, sizes/strides,
#  and a couple of torch classes.  Encoding these with a closed-world codec
#  (instead of pickle) keeps deserialization of /dev/shm control payloads
#  free of arbitrary-object construction, and a generic walk keeps the
#  marker format independent of the exact tuple layout across torch
#  versions.  Anything outside the whitelist aborts the export, and put()
#  falls back to the host-copy path.
# ---------------------------------------------------------------------- #


def _reduce_spec_classes() -> dict[str, type]:
    classes: dict[str, type] = {
        "Tensor": torch.Tensor,
        "Parameter": torch.nn.Parameter,
        "UntypedStorage": torch.UntypedStorage,
    }
    typed_storage = getattr(torch, "TypedStorage", None)
    if typed_storage is not None:
        classes["TypedStorage"] = typed_storage
    return classes


_NAME_TO_CLASS = _reduce_spec_classes()
_CLASS_TO_NAME = {cls: name for name, cls in _NAME_TO_CLASS.items()}


def _encode_reduce_atom(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    if isinstance(value, torch.dtype):
        return {"__k": "dtype", "v": str(value).removeprefix("torch.")}
    if isinstance(value, (tuple, list)):  # includes torch.Size
        return {"__k": "seq", "v": [_encode_reduce_atom(v) for v in value]}
    if isinstance(value, type):
        name = _CLASS_TO_NAME.get(value)
        if name is None:
            raise TypeError(f"Unsupported class in CUDA IPC reduce spec: {value!r}")
        return {"__k": "cls", "v": name}
    raise TypeError(f"Unsupported value in CUDA IPC reduce spec: {type(value).__name__}")


def _decode_reduce_atom(value: Any) -> Any:
    if isinstance(value, dict):
        kind = value.get("__k")
        if kind == "dtype":
            dtype = getattr(torch, value["v"], None)
            if not isinstance(dtype, torch.dtype):
                raise TypeError(f"Unknown dtype in CUDA IPC reduce spec: {value['v']!r}")
            return dtype
        if kind == "seq":
            return tuple(_decode_reduce_atom(v) for v in value["v"])
        if kind == "cls":
            cls = _NAME_TO_CLASS.get(value["v"])
            if cls is None:
                raise TypeError(f"Unknown class in CUDA IPC reduce spec: {value['v']!r}")
            return cls
        raise TypeError(f"Unknown atom kind in CUDA IPC reduce spec: {kind!r}")
    return value


def _compact_for_share(tensor: torch.Tensor) -> torch.Tensor:
    """Return a tensor that owns exactly its own storage.

    ``reduce_tensor`` shares the *whole* underlying storage.  Payload tensors
    are frequently views into large, step-reused runner buffers, so sharing
    them directly would both over-share memory and race with buffer reuse.

    Runs under the sender's dedicated pack stream: the source may be freed by
    the caller right after ``put()`` returns, so its block must be pinned to
    the pack stream (``record_stream``) until the compaction copy completes.
    """
    tensor = tensor.detach()
    nbytes = tensor.numel() * tensor.element_size()
    if tensor.is_contiguous() and tensor.storage_offset() == 0 and tensor.untyped_storage().nbytes() == nbytes:
        return tensor
    try:
        tensor.record_stream(torch.cuda.current_stream())
    except Exception:
        # Non-allocator-owned source (e.g. graph pool): the defensive copy
        # below still snapshots it before the caller can mutate it.
        pass
    out = torch.empty_like(tensor, memory_format=torch.contiguous_format)
    out.copy_(tensor)
    return out


class TorchIpcConnector(SharedMemoryConnector):
    """SharedMemoryConnector with a torch-native CUDA IPC data plane."""

    supports_gpu_tensor: bool = True
    # Receiver-side pending views are drained per request; opt into the
    # adapter/mixin request-scoped cleanup hook.
    request_scoped_cleanup: bool = True

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self._gpu_keys = frozenset(str(k) for k in (config.get("gpu_tensor_keys") or []))
        # Per-tensor size floor for GPU placement (see gpu_placement module).
        if config.get("gpu_tensor_min_bytes") is not None:
            self.gpu_tensor_min_bytes = int(config["gpu_tensor_min_bytes"])
        self._local_device_cfg = config.get("local_device", "auto")
        self._local_device: torch.device | None = None
        if not torch.cuda.is_available():
            # CPU-only host: behave exactly like the parent SHM connector.
            self.supports_gpu_tensor = False
            logger.warning("TorchIpcConnector: CUDA unavailable; running as plain SharedMemoryConnector.")
        elif not self._gpu_keys:
            # No keys configured: transport still accepts stray CUDA tensors,
            # but payload builders will not keep anything on GPU.
            logger.warning(
                "TorchIpcConnector: no gpu_tensor_keys configured for stage %s; "
                "payloads will follow the CPU pipeline end-to-end.",
                self.stage_id,
            )
        # Dedicated transfer streams (lazy): keep the data plane off the
        # per-process default stream so packet transfers never serialize
        # against either stage's compute (mirrors the async CPU
        # materialization pipeline's dedicated-copy-stream design).
        self._pack_stream: torch.cuda.Stream | None = None
        self._recv_stream: torch.cuda.Stream | None = None
        # Rebuilt sender views must outlive the async D2D copies that read
        # them: (copy-done event, views) drained lazily once the event fires.
        self._pending_views: deque[tuple[torch.cuda.Event, list[torch.Tensor]]] = deque()
        self._metrics.update(
            {
                "ipc_tensors_shared": 0,
                "ipc_bytes_shared": 0,
                "ipc_tensors_imported": 0,
            }
        )

    # ------------------------------------------------------------------ #
    #  Capabilities
    # ------------------------------------------------------------------ #

    @property
    def gpu_tensor_keys(self) -> frozenset[str] | None:
        """Stable per-edge GPU placement key set (None when disabled)."""
        if not self.supports_gpu_tensor or not self._gpu_keys:
            return None
        return self._gpu_keys

    # ------------------------------------------------------------------ #
    #  Data plane: export (sender)
    # ------------------------------------------------------------------ #

    def _ensure_pack_stream(self) -> torch.cuda.Stream:
        if self._pack_stream is None:
            self._pack_stream = torch.cuda.Stream()
        return self._pack_stream

    def _export_payload(self, data: Any) -> Any:
        """Export CUDA tensors under the dedicated pack stream.

        The share-time IPC event that torch records (and that the receiver's
        rebuild waits on) lands on the *current* stream.  Exporting on the
        default stream would make that event capture the whole queued step,
        coupling the receiver's compute stream to the sender's step tail on
        every packet.  Instead, the pack stream waits one produce event
        (recorded on the sender's default stream, which ordered the payload
        writes under the repo's PTDS-off model) so the receiver only ever
        waits for the payload tensors themselves.
        """
        produce_event = torch.cuda.Event()
        produce_event.record()
        stream = self._ensure_pack_stream()
        stream.wait_event(produce_event)
        with torch.cuda.stream(stream):
            return self._export_gpu_tensors(data)

    def _export_one(self, tensor: torch.Tensor) -> dict[str, Any]:
        from torch.multiprocessing.reductions import rebuild_cuda_tensor, reduce_tensor

        compact = _compact_for_share(tensor)
        rebuild_fn, args = reduce_tensor(compact)
        if rebuild_fn is not rebuild_cuda_tensor:
            raise TypeError(f"Unexpected CUDA IPC rebuild function: {rebuild_fn!r}")
        self._metrics["ipc_tensors_shared"] += 1
        self._metrics["ipc_bytes_shared"] += compact.numel() * compact.element_size()
        return {_TORCH_IPC_MARKER: True, "spec": [_encode_reduce_atom(a) for a in args]}

    def _export_gpu_tensors(self, value: Any) -> Any:
        """Replace CUDA tensors with IPC markers (non-mutating walk)."""
        if isinstance(value, torch.Tensor):
            if value.is_cuda:
                return self._export_one(value)
            return value
        if isinstance(value, dict):
            return {k: self._export_gpu_tensors(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._export_gpu_tensors(v) for v in value]
        if isinstance(value, tuple):
            return tuple(self._export_gpu_tensors(v) for v in value)
        if hasattr(value, "__struct_fields__"):
            # msgspec Struct payloads (e.g. OmniPayloadStruct): rebuild with
            # transformed fields instead of mutating the caller's object.
            fields = {f: self._export_gpu_tensors(getattr(value, f)) for f in value.__struct_fields__}
            return type(value)(**fields)
        return value

    def put(
        self,
        from_stage: str,
        to_stage: str,
        put_key: str,
        data: Any,
    ) -> tuple[bool, int, dict[str, Any] | None]:
        if self.supports_gpu_tensor and _payload_has_cuda(data):
            try:
                data = self._export_payload(data)
            except Exception:
                # A failed export leaves ``data`` untouched; the parent path
                # then serializes CUDA tensors through the host as before.
                logger.error(
                    "TorchIpcConnector: GPU export failed for %s; sending via host copy.",
                    put_key,
                    exc_info=True,
                )
        return super().put(from_stage, to_stage, put_key, data)

    # ------------------------------------------------------------------ #
    #  Data plane: import (receiver)
    # ------------------------------------------------------------------ #

    def _resolve_local_device(self) -> torch.device:
        if self._local_device is None:
            if self._local_device_cfg in (None, "auto"):
                # Bare "cuda" resolves to the process's current device; stage
                # processes are single-device (per-stage visibility).
                self._local_device = torch.device("cuda")
            else:
                self._local_device = torch.device(self._local_device_cfg)
        return self._local_device

    def _ensure_recv_stream(self) -> torch.cuda.Stream:
        if self._recv_stream is None:
            self._recv_stream = torch.cuda.Stream()
        return self._recv_stream

    def _import_payload(self, obj: Any) -> Any:
        """Rebuild markers under the dedicated receive stream.

        torch's rebuild waits the sender's share-time event on the *current*
        stream; doing that (and the D2D copy) on the default stream would
        stall the consuming model's compute stream once per packet.  On the
        recv stream both are fully overlapped, and the consumer pays exactly
        one device-side ordering edge per get() — the data dependency itself.
        """
        if not torch.cuda.is_available():
            # Misconfigured edge (CUDA sender, CPU receiver): fall through so
            # _import_one raises its actionable error.
            return self._import_gpu_tensors(obj, [], [])
        stream = self._ensure_recv_stream()
        dsts: list[torch.Tensor] = []
        views: list[torch.Tensor] = []
        with torch.cuda.stream(stream):
            out = self._import_gpu_tensors(obj, dsts, views)
        if views:
            done = torch.cuda.Event()
            done.record(stream)
            ambient = torch.cuda.current_stream()
            ambient.wait_event(done)
            for dst in dsts:
                # dst blocks were allocated on the recv stream but live on in
                # ambient-stream consumers; defer their reuse accordingly.
                dst.record_stream(ambient)
            self._pending_views.append((done, views))
        return out

    def _import_one(self, marker: dict[str, Any], dsts: list[torch.Tensor], views: list[torch.Tensor]) -> torch.Tensor:
        from torch.multiprocessing.reductions import rebuild_cuda_tensor

        try:
            args = [_decode_reduce_atom(a) for a in marker["spec"]]
            view = rebuild_cuda_tensor(*args)
        except Exception as exc:
            raise RuntimeError(
                "TorchIpcConnector: failed to open a CUDA IPC handle from the "
                "sender stage. This edge requires same-host stages whose "
                "devices either coincide or have peer access (NVLink / PCIe "
                "P2P). Use SharedMemoryConnector on this edge otherwise."
            ) from exc
        device = self._resolve_local_device()
        # The current (recv) stream is already fenced on the sender's
        # share-time event by the rebuild above.
        local = torch.empty_like(view, device=device)
        local.copy_(view)
        dsts.append(local)
        views.append(view)
        self._metrics["ipc_tensors_imported"] += 1
        return local

    def _import_gpu_tensors(self, value: Any, dsts: list[torch.Tensor], views: list[torch.Tensor]) -> Any:
        if _is_marker(value):
            return self._import_one(value, dsts, views)
        if isinstance(value, dict):
            return {k: self._import_gpu_tensors(v, dsts, views) for k, v in value.items()}
        if isinstance(value, list):
            return [self._import_gpu_tensors(v, dsts, views) for v in value]
        if isinstance(value, tuple):
            return tuple(self._import_gpu_tensors(v, dsts, views) for v in value)
        if hasattr(value, "__struct_fields__"):
            fields = {f: self._import_gpu_tensors(getattr(value, f), dsts, views) for f in value.__struct_fields__}
            return type(value)(**fields)
        return value

    def _drain_pending_views(self, blocking: bool = False) -> None:
        while self._pending_views:
            event, _views = self._pending_views[0]
            if blocking:
                event.synchronize()
            elif not event.query():
                break
            self._pending_views.popleft()

    def get(
        self,
        from_stage: str,
        to_stage: str,
        get_key: str,
        metadata=None,
    ) -> tuple[Any, int] | None:
        result = super().get(from_stage, to_stage, get_key, metadata)
        if result is None:
            return None
        self._drain_pending_views()
        obj, size = result
        if not _payload_has_marker(obj):
            return obj, size
        return self._import_payload(obj), size

    # ------------------------------------------------------------------ #
    #  Lifecycle
    # ------------------------------------------------------------------ #

    def cleanup(self, request_id: str) -> None:
        super().cleanup(request_id)
        self._drain_pending_views()
        if self.supports_gpu_tensor:
            try:
                # Promptly reclaim sender-side storages whose receivers have
                # already released their views.
                torch.cuda.ipc_collect()
            except Exception:
                pass

    def close(self) -> None:
        try:
            self._drain_pending_views(blocking=True)
        except Exception:
            pass
        self._pending_views.clear()
        super().close()

    def health(self) -> dict[str, Any]:
        report = super().health()
        report["pending_views"] = len(self._pending_views)
        return report
