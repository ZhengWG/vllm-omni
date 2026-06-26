# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CUDA IPC connector (shared-storage): GPU-to-GPU handoff of large CUDA tensors.

Simplified rewrite. Large CUDA tensors go through a resident shared-CUDA
snapshot pool (``SharedCudaSnapshotPool``) built on torch's ``_share_cuda_`` /
``_new_shared_cuda`` storage sharing; small/CPU values, metadata, and the
control plane (tensor descriptors) ride an embedded ``SharedMemoryConnector``,
which also serves as the always-safe fallback.

This replaces the previous byte-pool design's hand-rolled machinery — raw
ctypes IPC handles, the SPSC control ring, credit pool, release board, and TTL
sweep — with:

  * free-safety      : torch cross-process IPC refcount + producer-held slot refs
  * overwrite-safety : ACK-driven slot reuse (receiver writes ack -> sender frees);
                       TTL only reclaims crashed/leaked leases
  * stream-safety    : producer snapshot ready before share; consumer record_stream

Cross-GPU works under CVD isolation via device-index redirect + lazy NVLink peer
(see shared_cuda_tensor_transport.open_shared_tensor).
"""

from __future__ import annotations

import os
import threading
from dataclasses import asdict
from typing import Any

import torch

from ..utils.logging import get_connector_logger
from .base import OmniConnectorBase
from .shared_cuda_tensor_transport import (
    SHARE_SUPPORTED,
    PoolFull,
    SharedCudaSnapshotPool,
    TensorDescriptor,
    open_shared_tensor,
)
from .shm_connector import SharedMemoryConnector

logger = get_connector_logger(__name__)

_REF = "__cuda_ipc_ref__"  # placeholder for a pooled large CUDA tensor
_MARK = "__cuda_ipc_scts__"  # marks a control payload carrying shared-cuda descriptors

_DEFAULT_SLOT_SIZE_MB = 64
_DEFAULT_POOL_CREDITS = 64


def _split(obj: Any, tensors: list, threshold: int) -> Any:
    """Replace large CUDA tensors with refs; collect them. Leave CPU / small
    CUDA tensors / metadata for the embedded-SHM control payload."""
    if isinstance(obj, torch.Tensor):
        if obj.is_cuda and int(obj.nbytes) >= threshold:
            tensors.append(obj)
            return {_REF: len(tensors) - 1}
        return obj
    if isinstance(obj, dict):
        return {k: _split(v, tensors, threshold) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_split(v, tensors, threshold) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_split(v, tensors, threshold) for v in obj)
    if hasattr(obj, "__struct_fields__"):  # msgspec struct -> dict (SHM wire contract)
        return {
            f: _split(getattr(obj, f), tensors, threshold) for f in obj.__struct_fields__ if getattr(obj, f) is not None
        }
    return obj


def _restore(obj: Any, rebuilt: list) -> Any:
    if isinstance(obj, dict):
        if len(obj) == 1 and _REF in obj:
            return rebuilt[obj[_REF]]
        return {k: _restore(v, rebuilt) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_restore(v, rebuilt) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_restore(v, rebuilt) for v in obj)
    return obj


class CudaIPCConnector(OmniConnectorBase):
    supports_gpu_tensor: bool = True

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.stage_id = int(config.get("stage_id", -1))
        self.role = str(config.get("role", "sender")).lower()
        self._is_transfer_rank = bool(config.get("is_transfer_rank", True))
        self.local_device = self._resolve_local_device(config.get("local_device", "auto"))
        # Routing threshold (kept name ``inline_threshold_bytes`` for config compat):
        # CUDA tensors >= this go to the shared pool; everything else rides SHM.
        self.large_threshold = int(config.get("inline_threshold_bytes", 16384))

        # Pool sizing from the existing config keys (ring/credit knobs are accepted
        # and ignored — the new path has no ring or credit board).
        max_num_seqs = int(config.get("max_num_seqs", 0))
        credits = int(
            config.get(
                "pool_credits",
                max(_DEFAULT_POOL_CREDITS, max_num_seqs * 2) if max_num_seqs > 0 else _DEFAULT_POOL_CREDITS,
            )
        )
        slot_mb = int(config.get("slot_size_mb", _DEFAULT_SLOT_SIZE_MB))
        crash_ttl = float(config.get("tensor_lifetime_sec", 30.0))

        self._closed = False
        self._lock = threading.Lock()
        self._seq = 0
        self._inflight: dict[str, list[tuple[int, int]]] = {}
        self._ack_dir = config.get("cudaipc_ack_dir", "/dev/shm/cudaipc_ack")
        os.makedirs(self._ack_dir, exist_ok=True)

        # Embedded SHM: control plane + small/CPU payloads + fallback.
        shm_cfg = dict(config)
        shm_cfg.setdefault("device", str(self.local_device))
        self._shm = SharedMemoryConnector(shm_cfg)

        # Resident shared-CUDA pool — only the sender data-transfer rank owns one.
        self._pool: SharedCudaSnapshotPool | None = None
        if SHARE_SUPPORTED and self.role == "sender" and self._is_transfer_rank and self.local_device.type == "cuda":
            try:
                with torch.cuda.device(self.local_device):
                    self._pool = SharedCudaSnapshotPool(
                        self.local_device,
                        slot_nbytes=slot_mb << 20,
                        num_slots=credits,
                        crash_ttl_sec=crash_ttl,
                    )
            except Exception as e:  # noqa: BLE001 - any pool failure -> SHM-only
                logger.warning("CudaIPCConnector pool init failed (%s); SHM-only fallback", e)
                self._pool = None
        logger.info(
            "CudaIPCConnector(shared-storage) initialized: role=%s, local_device=%s, share_supported=%s, pool=%s",
            self.role,
            self.local_device,
            SHARE_SUPPORTED,
            (self._pool.stats() if self._pool is not None else None),
        )

    @staticmethod
    def _resolve_local_device(local_device_cfg) -> torch.device:
        if local_device_cfg == "auto":
            return torch.device("cuda", torch.accelerator.current_device_index())
        if isinstance(local_device_cfg, int):
            return torch.device("cuda", local_device_cfg)
        if isinstance(local_device_cfg, str):
            if local_device_cfg.startswith("cuda"):
                return torch.device(local_device_cfg)
            return torch.device("cuda", int(local_device_cfg))
        return torch.device("cuda", torch.accelerator.current_device_index())

    def register_producer_stream(self, producer_stream: torch.cuda.Stream | None) -> None:
        """No-op (kept for interface compat). The pool snapshots into its own slot
        on a dedicated stream fenced after the current stream, so pool-D2D ordering
        is handled internally."""
        return

    @staticmethod
    def _safe(s: str) -> str:
        return "".join(c if c.isalnum() else "_" for c in s)

    def _reclaim(self) -> None:
        """Drain receiver acks -> free our slots. Only touch acks we own."""
        if self._pool is None or os is None:
            return
        try:
            names = os.listdir(self._ack_dir)
        except Exception:  # noqa: BLE001 - dir missing / shutdown
            return
        for fn in names:
            with self._lock:
                leases = self._inflight.pop(fn, None)
            if leases is None:
                continue  # another connector's ack -> leave it
            for sid, gen in leases:
                self._pool.ack(sid, gen)
            try:
                os.unlink(os.path.join(self._ack_dir, fn))
            except OSError:
                pass
        self._pool.reclaim_crashed()  # crash/leak only

    # --- producer ---
    def put(self, from_stage: str, to_stage: str, put_key: str, data: Any):
        if self._closed or not self._is_transfer_rank:
            return False, 0, None
        if self._pool is None:
            return self._shm.put(from_stage, to_stage, put_key, data)
        self._reclaim()
        tensors: list[torch.Tensor] = []
        control = _split(data, tensors, self.large_threshold)
        if not tensors:
            return self._shm.put(from_stage, to_stage, put_key, data)
        with self._lock:
            self._seq += 1
            tid = self._safe(f"{from_stage}_{to_stage}_{put_key}_{os.getpid()}_{self._seq}")
        descs: list[TensorDescriptor] = []
        try:
            for t in tensors:
                descs.append(self._pool.put(t))
        except PoolFull:
            for d in descs:  # free partial leases, fall back to SHM
                self._pool.ack(d.slot_id, d.generation)
            return self._shm.put(from_stage, to_stage, put_key, data)
        payload = {_MARK: True, "tid": tid, "control": control, "descriptors": [asdict(d) for d in descs]}
        ok, size, meta = self._shm.put(from_stage, to_stage, put_key, payload)
        if not ok:
            for d in descs:
                self._pool.ack(d.slot_id, d.generation)
            return False, 0, None
        with self._lock:
            self._inflight[tid] = [(d.slot_id, d.generation) for d in descs]
        return ok, size, meta

    # --- consumer ---
    def get(self, from_stage: str, to_stage: str, get_key: str, metadata: dict | None = None):
        if self._closed:
            return None
        res = self._shm.get(from_stage, to_stage, get_key, metadata)
        if res is None:
            return None
        obj, size = res
        if not (isinstance(obj, dict) and obj.get(_MARK)):
            return obj, size  # plain SHM payload (fallback / no large tensor)
        dst = self.local_device.index if self.local_device.index is not None else 0
        rebuilt = [open_shared_tensor(TensorDescriptor(**dd), dst, copy_out=True) for dd in obj["descriptors"]]
        restored = _restore(obj["control"], rebuilt)
        try:  # ack -> sender frees the slots (overwrite-safety)
            open(os.path.join(self._ack_dir, obj["tid"]), "w").close()
        except OSError:
            pass
        return restored, size

    # --- lifecycle ---
    def cleanup(self, request_id: str) -> None:
        self._shm.cleanup(request_id)

    def health(self) -> dict[str, Any]:
        return {
            "status": "healthy" if not self._closed else "closed",
            "connector": "CudaIPCConnector(shared-storage)",
            "role": self.role,
            "local_device": str(self.local_device),
            "share_supported": SHARE_SUPPORTED,
            "pool": (self._pool.stats() if self._pool is not None else None),
            "shm": self._shm.health(),
        }

    def close(self) -> None:
        if getattr(self, "_closed", True):
            return
        self._closed = True
        if self._pool is not None:
            try:
                self._reclaim()
            except Exception:  # noqa: BLE001 - best-effort at shutdown
                pass
        self._shm.close()
        logger.info("CudaIPCConnector closed.")
