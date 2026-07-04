# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Per-edge SPSC keyed mailbox ring — the CudaIPC connector's control plane.

One pre-allocated ring per directed edge (opened once) replaces the per-transfer /dev/shm
round-trip. Sender publishes a fixed-stride entry; receiver looks it up by key — fenceless,
correct on x86 TSO (ARM/POWER would need a store fence).

Lock-free SPSC correctness: body-first / seq-LAST publish (seq is the release marker); a
seq==0 in-progress sentinel written first on (re)claim; a seqlock re-read guarding the body;
a per-slot consumed byte for bounded backpressure; open addressing by key hash.

Pure-Python (struct + POSIX shm), no CUDA — testable on CPU.

Fixed header region (``RING_HEADER_BYTES``, written once): see ``RingHeader``.
Per-slot layout (little-endian):
seq u64@0 | consumed u8@8 | pclass u8@9 | ts u32@10 | keyhash 16B@14 | blen u32@30 | body@34.
"""

import hashlib
import os
import struct
from dataclasses import dataclass
from multiprocessing import shared_memory as shm_pkg

_OFF_SEQ, _OFF_CONSUMED, _OFF_PCLASS, _OFF_TS, _OFF_KEY, _OFF_LEN, _OFF_BODY = 0, 8, 9, 10, 14, 30, 34
_KEY_BYTES = 16

# Edge-constant header blob (pool/event IPC handles + release-board shm name).
RING_HEADER_BYTES = 256  # reserved shm bytes before slots
_RING_MAGIC = b"CIPC"  # rejects uninitialized (all-zero) shm
_RING_VERSION = 1  # bump on incompatible wire layout changes

# Per-entry payload-class tags (ring slot pclass byte).
RING_PCLASS_INLINE = 0  # ring body: serialized payload (small, < inline_threshold)
RING_PCLASS_POOL = 1  # ring body: pool descriptor (big GPU tensor, D2D path)


def make_composite_key(key: str, from_stage: str, to_stage: str) -> str:
    """Per-edge composite key. Change here once if the wire format ever changes."""
    return f"{key}@{from_stage}_{to_stage}"


def key_hash16(composite_key: str) -> bytes:
    return hashlib.sha1(composite_key.encode("utf-8")).digest()[:16]


def ring_shm_name(from_stage, to_stage, replica_id) -> str:
    """Deterministic POSIX shm name for a directed edge.

    Hashes stage ids + replica_id. Pod/process IPC isolation is assumed for
    co-located deployments; replica_id distinguishes parallel pipelines.
    """
    raw = f"{from_stage}:{to_stage}:{replica_id}"
    return f"cudaipc_{hashlib.sha1(raw.encode()).hexdigest()[:20]}"


@dataclass
class RingHeader:
    """Typed view of the ring header. Wire layout (little-endian):
    magic(4) | version(1) | pool_handle(64) | event_handle(64) | board_name_len(1) | board_name.
    """

    pool_handle: bytes
    event_handle: bytes
    board_name: str

    _PREFIX = 4 + 1 + 64 + 64 + 1  # magic + version + two handles + name_len

    def pack(self) -> bytes:
        bn = self.board_name.encode("utf-8")
        if self._PREFIX + len(bn) > RING_HEADER_BYTES:
            raise ValueError(f"board_name {len(bn)}B overflows ring header ({RING_HEADER_BYTES}B)")
        return _RING_MAGIC + bytes([_RING_VERSION]) + self.pool_handle + self.event_handle + bytes([len(bn)]) + bn

    @classmethod
    def try_unpack(cls, blob: bytes) -> "RingHeader | None":
        """Return a RingHeader, or None if not yet written / version-mismatched."""
        # Wire offsets (must match ``pack()`` / class docstring):
        # [0:4) magic | [4:5) version | [5:69) pool_handle | [69:133) event_handle
        # | [133:134) board_name_len | [134:134+len) board_name
        if len(blob) < cls._PREFIX or blob[0:4] != _RING_MAGIC or blob[4] != _RING_VERSION:
            return None
        pool_handle = bytes(blob[5:69])  # cudaIpcMemHandle_t, 64 B
        event_handle = bytes(blob[69:133])  # cudaIpcEventHandle_t, 64 B
        bn_len = blob[133]  # u8 length prefix for board_name UTF-8
        board_name = bytes(blob[134 : 134 + bn_len]).decode("utf-8")
        return cls(pool_handle, event_handle, board_name)


def untrack_shm(name: str) -> None:
    """Drop the resource_tracker registration for a non-owner attach, so this process
    does not unlink the owner's segment at its own exit (only the owner unlinks)."""
    try:
        from multiprocessing.resource_tracker import unregister

        unregister(f"/{name}", "shared_memory")
    except Exception:
        pass


class RingFullError(Exception):
    """Raised by publish() when no free slot exists in the probe window (backpressure)."""


def _pid_alive(pid: int) -> bool:
    """Best-effort owner liveness (same-host IPC): signal-0 probe."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class CudaIpcControlRing:
    """One directed-edge SPSC keyed mailbox. Sender side calls create()+publish();
    receiver side calls open()+poll(). A fixed header region carries edge-constant
    bytes (e.g. the pool/event/board IPC handles) published once at create()."""

    __slots__ = ("_shm", "_buf", "_n", "_slot", "_body_max", "_hdr", "_base", "_pubctr", "_owner")

    def __init__(self, shm, n_slots, body_max, header_bytes, owner):
        self._shm = shm
        self._buf = shm.buf
        self._n = n_slots
        self._body_max = body_max
        self._slot = _OFF_BODY + body_max
        self._hdr = header_bytes
        self._base = 8 + header_bytes  # u32 n_slots + u32 body_max, then header, then slots
        self._pubctr = 0
        self._owner = owner  # sender created it (responsible for unlink)

    @property
    def body_max(self) -> int:
        """Max body bytes per slot (callers route oversize payloads to a fallback)."""
        return self._body_max

    # ---- construction -------------------------------------------------
    @classmethod
    def reclaim_stale(cls, name: str, header_bytes: int = RING_HEADER_BYTES) -> bool:
        """Unlink a leftover ring whose owner process is dead; True if reclaimed.

        Called by the sender at connector init (before any lazy create) so a
        receiver polling during the init->first-put window cannot open and
        cache a stale segment from a previous run. A ring with a LIVE owner is
        left untouched.
        """
        try:
            old = shm_pkg.SharedMemory(name=name)
        except FileNotFoundError:
            return False
        try:
            old_slots, old_body = struct.unpack_from("<II", old.buf, 0)
            owner_pid = 0
            if old_slots > 0 and old_body > 0 and header_bytes >= 4:
                # Fixed offset from the caller's layout — do NOT derive from
                # old.size (tmpfs/macOS round segment sizes up to page size).
                owner_pid = struct.unpack_from("<I", old.buf, 8 + header_bytes - 4)[0]
            if owner_pid and _pid_alive(owner_pid):
                return False
            old.unlink()
            return True
        except Exception:
            return False
        finally:
            old.close()

    @classmethod
    def create(cls, name, n_slots, body_max, header_bytes=0):
        size = 8 + header_bytes + n_slots * (_OFF_BODY + body_max)
        try:
            old = shm_pkg.SharedMemory(name=name)
            try:
                # A pre-existing INITIALIZED ring whose owner is still alive means a
                # second sender-owner is being created on the same edge (dual-putter
                # bug): unlink-recreate would orphan the live owner's segment and
                # strand any receiver that cached the old mapping — a silent
                # pipeline hang. Refuse loudly instead. A dead owner (previous run
                # crashed without cleanup) is reclaimed as before.
                old_slots, old_body = struct.unpack_from("<II", old.buf, 0)
                if old_slots > 0 and old_body > 0:
                    # Fixed offset from this caller's layout — old.size is
                    # page-rounded on some tmpfs/macOS and must not be used.
                    owner_pid = struct.unpack_from("<I", old.buf, 8 + header_bytes - 4)[0] if header_bytes >= 4 else 0
                    if owner_pid and _pid_alive(owner_pid):
                        raise RuntimeError(
                            f"CudaIpcControlRing.create: ring {name!r} already exists with a "
                            f"LIVE owner (pid={owner_pid}). Two sender-owners on one edge are "
                            f"not supported (SPSC): the second create would unlink the first "
                            f"owner's segment and strand receivers that already opened it. "
                            f"Fix the duplicate putter instead of overriding this error."
                        )
                old.unlink()
            finally:
                old.close()
        except FileNotFoundError:
            pass
        shm = shm_pkg.SharedMemory(create=True, size=size, name=name)
        # POSIX shm_open + ftruncate zero-fills the new segment, so every slot already
        # reads seq==0 (empty). No explicit memset — avoids a transient bytes(size) heap
        # spike (a 2048 x 128KB ring = 256MB) on the hot init path.
        struct.pack_into("<II", shm.buf, 0, n_slots, body_max)
        if header_bytes >= 4:
            # Owner pid at the tail of the reserved header region — lets a later
            # create() distinguish "live duplicate owner" from "stale leftover".
            struct.pack_into("<I", shm.buf, 8 + header_bytes - 4, os.getpid() & 0xFFFFFFFF)
        return cls(shm, n_slots, body_max, header_bytes, owner=True)

    @classmethod
    def open(cls, name, header_bytes: int = RING_HEADER_BYTES):
        shm = shm_pkg.SharedMemory(name=name)
        untrack_shm(name)  # non-owner: never unlink the sender's ring at exit
        n_slots, body_max = struct.unpack_from("<II", shm.buf, 0)
        if n_slots <= 0 or body_max <= 0:
            # Opened before the sender's header write (zero-filled) — not ready; caller retries.
            shm.close()
            raise FileNotFoundError(f"ring {name!r} not initialized yet (n_slots={n_slots}, body_max={body_max})")
        # header size is a layout constant shared by both sides of the edge
        # (same code version). Deriving it from shm.size is WRONG on filesystems
        # that round segment sizes up to page granularity (macOS, some tmpfs).
        required = 8 + header_bytes + n_slots * (_OFF_BODY + body_max)
        if shm.size < required:
            shm.close()
            raise FileNotFoundError(
                f"ring {name!r} smaller than its declared layout ({shm.size} < {required}); "
                f"stale or foreign segment — caller retries"
            )
        return cls(shm, n_slots, body_max, header_bytes, owner=False)

    # ---- header (edge-constant, written once by the sender) -----------
    def write_header(self, data: bytes) -> None:
        limit = self._hdr - 4 if self._hdr >= 4 else self._hdr  # last 4B: owner pid
        if len(data) > limit:
            raise ValueError(f"header {len(data)}B exceeds reserved {limit}B (pid tail excluded)")
        self._buf[8 : 8 + len(data)] = data

    def read_header(self, n: int) -> bytes:
        return bytes(self._buf[8 : 8 + n])

    # ---- sender -------------------------------------------------------
    def publish(self, key_hash: bytes, pclass: int, body: bytes, ts: int = 0, ttl_sec: int = 0) -> None:
        """Claim a free slot (open addressing) and publish. RingFullError if full.

        ttl_sec>0 reclaims stale unconsumed slots (aborted / never-polled).
        SPSC — one sender thread only; inline publish is sub-ms and not an E2E bottleneck."""
        if len(body) > self._body_max:
            raise ValueError(f"body {len(body)}B exceeds slot body_max {self._body_max}B")
        buf = self._buf
        home = struct.unpack_from("<Q", key_hash, 0)[0] % self._n
        for p in range(self._n):
            idx = (home + p) % self._n
            off = self._base + idx * self._slot
            seq = struct.unpack_from("<Q", buf, off + _OFF_SEQ)[0]
            free = seq == 0 or buf[off + _OFF_CONSUMED] == 1
            if not free and ttl_sec and ts:  # TTL-reclaim a stale unconsumed entry
                slot_ts = struct.unpack_from("<I", buf, off + _OFF_TS)[0]
                if slot_ts and (ts - slot_ts) > ttl_sec:
                    free = True
            if free:
                struct.pack_into("<Q", buf, off + _OFF_SEQ, 0)  # in-progress sentinel FIRST
                buf[off + _OFF_CONSUMED] = 0
                buf[off + _OFF_PCLASS] = pclass
                struct.pack_into("<I", buf, off + _OFF_TS, ts & 0xFFFFFFFF)
                buf[off + _OFF_KEY : off + _OFF_KEY + _KEY_BYTES] = key_hash[:_KEY_BYTES]
                struct.pack_into("<I", buf, off + _OFF_LEN, len(body))
                buf[off + _OFF_BODY : off + _OFF_BODY + len(body)] = body
                self._pubctr += 1
                struct.pack_into("<Q", buf, off + _OFF_SEQ, self._pubctr)  # publish LAST
                return
        raise RingFullError()

    # ---- receiver -----------------------------------------------------
    def poll(self, key_hash: bytes):
        """Return (pclass, body) for key_hash and mark it consumed, or None if absent.
        Open-addressed probe: stop at an empty (seq==0) slot — under the consumed-gate a
        present key is always found before the first empty slot in its probe path."""
        buf = self._buf
        home = struct.unpack_from("<Q", key_hash, 0)[0] % self._n
        target = key_hash[:_KEY_BYTES]
        for p in range(self._n):
            idx = (home + p) % self._n
            off = self._base + idx * self._slot
            seq_a = struct.unpack_from("<Q", buf, off + _OFF_SEQ)[0]
            if seq_a == 0:
                return None  # empty slot in the probe path => key not present
            if buf[off + _OFF_CONSUMED] == 0 and buf[off + _OFF_KEY : off + _OFF_KEY + _KEY_BYTES] == target:
                pclass = buf[off + _OFF_PCLASS]
                ln = struct.unpack_from("<I", buf, off + _OFF_LEN)[0]
                body = bytes(buf[off + _OFF_BODY : off + _OFF_BODY + ln])
                seq_b = struct.unpack_from("<Q", buf, off + _OFF_SEQ)[0]
                if seq_b != seq_a:
                    return None  # torn (slot reused mid-read); caller retries the poll
                buf[off + _OFF_CONSUMED] = 1  # reclaim (producer may now reuse)
                return pclass, body
        return None

    # ---- lifecycle ----------------------------------------------------
    def close(self) -> None:
        try:
            self._buf = None
            self._shm.close()
            if self._owner:
                self._shm.unlink()
        except Exception:
            pass
