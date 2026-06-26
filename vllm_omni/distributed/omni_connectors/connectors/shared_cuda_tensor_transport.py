"""Phase-2 primitive (hardened): resident shared-CUDA snapshot pool.

Five production constraints (per design review):
  1. Normal slot reuse is ACK-driven. TTL only reclaims crashed/leaked leases.
  2. slot generation is carried in the descriptor and validated on ack (stale/late guard).
  3. Producer permanently holds slot tensor refs (clear lifecycle, not relying on
     torch cross-process refcount alone).
  4. Consumer record_stream's the rebuilt tensor (consumer-side allocator/stream safety).
  5. Simple fallback: shared-cuda API unavailable / pool full / ack timeout -> caller uses SHM.

Three safety axes kept separate:
  free-safety      : torch cross-process IPC refcount + producer holds slot refs
  overwrite-safety : ACK-gated reuse (TTL = crash recovery only)
  stream-safety    : producer snapshot ready before share; consumer record_stream
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)

SHARE_SUPPORTED = hasattr(torch.UntypedStorage, "_new_shared_cuda") and hasattr(
    torch.empty(0).untyped_storage(), "_share_cuda_"
)


@dataclass
class TensorDescriptor:
    slot_id: int
    generation: int  # constraint 2: stale-descriptor / late-get guard
    shape: tuple
    dtype: str
    stride: tuple
    storage_offset: int  # elements of dtype, into the slot storage
    nbytes: int
    pool_handle: tuple  # slot storage _share_cuda_(), shared ONCE


class PoolFull(Exception):  # noqa: N818 - public API, imported by cuda_ipc_connector
    """No free slot -> caller should SHM-fallback (constraint 5)."""


class SharedCudaSnapshotPool:
    def __init__(self, device, slot_nbytes: int, num_slots: int, crash_ttl_sec: float = 30.0):
        if not SHARE_SUPPORTED:
            raise RuntimeError("torch CUDA shared-storage API unavailable")  # -> caller SHM fallback
        self.device = torch.device(device)
        self.slot_nbytes = int(slot_nbytes)
        self.crash_ttl_sec = crash_ttl_sec
        self._snap_stream = torch.cuda.Stream(device=self.device)
        self._lock = threading.Lock()
        self._slots = []
        with torch.cuda.device(self.device):
            for i in range(int(num_slots)):
                # constraint 3: pool permanently owns the slot tensor -> storage + handle stay valid
                buf = torch.empty(self.slot_nbytes, dtype=torch.uint8, device=self.device)
                handle = buf.untyped_storage()._share_cuda_()  # shared ONCE
                self._slots.append({"id": i, "buf": buf, "handle": handle, "gen": 0, "state": "free", "ts": 0.0})

    def _acquire_free(self):
        # constraint 1: ONLY free slots. No TTL recycle of leased slots here.
        with self._lock:
            for s in self._slots:
                if s["state"] == "free":
                    s["state"] = "reserved"
                    return s
        return None

    def put(self, tensor: torch.Tensor) -> TensorDescriptor:
        t = tensor.detach()
        if not t.is_contiguous():
            t = t.contiguous()
        nbytes = int(t.nbytes)
        if nbytes > self.slot_nbytes:
            raise PoolFull(f"payload {nbytes} > slot {self.slot_nbytes}")
        slot = self._acquire_free()
        if slot is None:
            raise PoolFull("no free slot")  # constraint 5: caller SHM-fallback
        # Fence the snapshot after the producer's writes on the current stream
        # (PTDS off -> shared default stream), else it can read a half-written clone.
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self._snap_stream):
            self._snap_stream.wait_event(ready)
            slot["buf"][:nbytes].copy_(t.view(torch.uint8).reshape(-1))  # snapshot into resident slot
        self._snap_stream.synchronize()  # prototype: sync. prod: export ready ipc-event for consumer wait
        with self._lock:
            slot["gen"] += 1
            slot["state"] = "leased"
            slot["ts"] = time.monotonic()
            gen = slot["gen"]
        return TensorDescriptor(
            slot_id=slot["id"],
            generation=gen,
            shape=tuple(t.shape),
            dtype=str(t.dtype).removeprefix("torch."),
            stride=tuple(t.stride()),
            storage_offset=0,
            nbytes=nbytes,
            pool_handle=slot["handle"],
        )

    def ack(self, slot_id: int, generation: int) -> bool:
        # constraint 1+2: free a slot only when the acked generation matches the live one
        with self._lock:
            s = self._slots[slot_id]
            if s["state"] == "leased" and s["gen"] == generation:
                s["state"] = "free"
                return True
        return False  # stale/late ack -> ignored

    def reclaim_crashed(self) -> int:
        # TTL is ONLY for crash/leak recovery, never normal reuse.
        now, n = time.monotonic(), 0
        with self._lock:
            for s in self._slots:
                if s["state"] == "leased" and now - s["ts"] > self.crash_ttl_sec:
                    logger.warning("reclaim crashed lease slot=%d gen=%d age=%.1fs", s["id"], s["gen"], now - s["ts"])
                    s["state"] = "free"
                    n += 1
        return n

    def stats(self):
        with self._lock:
            return {st: sum(1 for s in self._slots if s["state"] == st) for st in ("free", "reserved", "leased")}


def open_shared_tensor(
    desc: TensorDescriptor, dst_device_index: int, *, stream: torch.cuda.Stream | None = None, copy_out: bool = True
):
    """Consumer rebuild. device-redirect handle[0] -> consumer device (lazy NVLink peer)."""
    dt = getattr(torch, desc.dtype)
    h = desc.pool_handle
    redirected = (dst_device_index,) + tuple(h)[1:]
    with torch.cuda.device(dst_device_index):
        # Open fresh on every read. The pool overwrites each slot repeatedly; a cached
        # consumer mapping skips the per-open IPC re-sync _new_shared_cuda performs, so it
        # reads a STALE generation across NVLink (proven: cached!=sent, fresh==sent).
        st = torch.UntypedStorage._new_shared_cuda(*redirected)
        view = torch.empty(0, device=f"cuda:{dst_device_index}", dtype=dt).set_(
            st, desc.storage_offset, desc.shape, desc.stride
        )
        cur = stream or torch.cuda.current_stream(dst_device_index)
        view.record_stream(cur)  # constraint 4: consumer stream-safety
        if copy_out:
            with torch.cuda.stream(cur):
                out = view.clone()  # one D2D over NVLink into consumer-owned buffer
            out.record_stream(cur)
            cur.synchronize()
            return out
        return view  # caller must keep ref + ack only after its stream work completes
