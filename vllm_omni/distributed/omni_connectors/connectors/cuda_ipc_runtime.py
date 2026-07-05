# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Low-level CUDA IPC runtime bindings for the CudaIPC connector.

Thin ctypes wrapper over the ``libcudart`` IPC symbols (mem-handle get/open/close,
event create/get/open/record, stream-wait, memcpy). Kept separate from the
connector so the transport logic doesn't interleave with raw ``ctypes`` plumbing.
"""

import ctypes


class _CudaIpcMemHandle(ctypes.Structure):
    """ctypes wrapper for ``cudaIpcMemHandle_t`` (64-byte opaque struct)."""

    _fields_ = [("reserved", ctypes.c_char * 64)]


class _CudaIpcEventHandle(ctypes.Structure):
    """ctypes wrapper for ``cudaIpcEventHandle_t`` (64-byte opaque struct)."""

    _fields_ = [("reserved", ctypes.c_char * 64)]


# CUDA runtime API constants (fixed by CUDA spec, not configurable).
_CUDA_MEMCPY_D2D = 3  # cudaMemcpyDeviceToDevice
_CUDA_EVENT_INTERPROCESS = 0x04
_CUDA_EVENT_DISABLE_TIMING = 0x02
_CUDA_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 0x01  # cudaIpcMemLazyEnablePeerAccess (only valid open flag)


def load_cudart():
    """Load libcudart with IPC symbols and signatures."""
    import ctypes.util
    import glob

    lib = None
    name = ctypes.util.find_library("cudart")
    if name:
        try:
            lib = ctypes.CDLL(name)
            if not hasattr(lib, "cudaIpcGetMemHandle"):
                lib = None
        except OSError:
            lib = None

    if lib is None:
        candidates = sorted(
            glob.glob("/usr/local/cuda*/lib*/libcudart.so*") + glob.glob("/opt/conda/lib/libcudart.so*"),
            reverse=True,
        )
        for path in candidates:
            try:
                lib = ctypes.CDLL(path)
                if hasattr(lib, "cudaIpcGetMemHandle"):
                    break
                lib = None
            except OSError:
                continue

    if lib is None:
        raise RuntimeError(
            "Cannot find libcudart.so with cudaIpcGetMemHandle. "
            "Ensure CUDA toolkit is installed and libcudart.so is on LD_LIBRARY_PATH."
        )

    lib.cudaIpcGetMemHandle.argtypes = [
        ctypes.POINTER(_CudaIpcMemHandle),
        ctypes.c_void_p,
    ]
    lib.cudaIpcGetMemHandle.restype = ctypes.c_int

    lib.cudaIpcOpenMemHandle.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        _CudaIpcMemHandle,
        ctypes.c_uint,
    ]
    lib.cudaIpcOpenMemHandle.restype = ctypes.c_int

    lib.cudaIpcCloseMemHandle.argtypes = [ctypes.c_void_p]
    lib.cudaIpcCloseMemHandle.restype = ctypes.c_int

    lib.cudaMemcpyAsync.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    lib.cudaMemcpyAsync.restype = ctypes.c_int

    lib.cudaEventCreateWithFlags.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
    lib.cudaEventCreateWithFlags.restype = ctypes.c_int

    lib.cudaIpcGetEventHandle.argtypes = [ctypes.POINTER(_CudaIpcEventHandle), ctypes.c_void_p]
    lib.cudaIpcGetEventHandle.restype = ctypes.c_int

    lib.cudaIpcOpenEventHandle.argtypes = [ctypes.POINTER(ctypes.c_void_p), _CudaIpcEventHandle]
    lib.cudaIpcOpenEventHandle.restype = ctypes.c_int

    lib.cudaEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.cudaEventRecord.restype = ctypes.c_int

    lib.cudaStreamWaitEvent.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]
    lib.cudaStreamWaitEvent.restype = ctypes.c_int

    lib.cudaEventDestroy.argtypes = [ctypes.c_void_p]
    lib.cudaEventDestroy.restype = ctypes.c_int

    return lib


# Thin call wrappers — keep the ctypes.c_void_p/c_size_t boxing and the ret!=0 raise out of
# the connector's hot path.
def memcpy_async_d2d(lib, dst: int, src: int, nbytes: int, stream: int) -> None:
    ret = lib.cudaMemcpyAsync(
        ctypes.c_void_p(dst),
        ctypes.c_void_p(src),
        ctypes.c_size_t(nbytes),
        ctypes.c_int(_CUDA_MEMCPY_D2D),
        ctypes.c_void_p(stream),
    )
    if ret != 0:
        raise RuntimeError(f"cudaMemcpyAsync (D2D) failed: {ret}")


def stream_wait_event(lib, stream: int, event) -> None:
    ret = lib.cudaStreamWaitEvent(ctypes.c_void_p(stream), event, ctypes.c_uint(0))
    if ret != 0:
        raise RuntimeError(f"cudaStreamWaitEvent failed: {ret}")


# --- DLPack view construction (L4 zero-copy receive) ---------------------
# Minimal ctypes DLPack producer: wraps a raw device/host pointer as a torch
# tensor WITHOUT copying. Used by the receiver to view a pool slot in place.

_DL_KDLCPU = 1
_DL_KDLCUDA = 2

_DLPACK_DTYPES = {
    # dtype string (torch, sans "torch.") -> (type_code, bits, lanes)
    "float16": (2, 16, 1),
    "bfloat16": (4, 16, 1),
    "float32": (2, 32, 1),
    "float64": (2, 64, 1),
    "int8": (0, 8, 1),
    "int16": (0, 16, 1),
    "int32": (0, 32, 1),
    "int64": (0, 64, 1),
    "uint8": (1, 8, 1),
    "bool": (6, 8, 1),
}


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]


class _DLDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class _DLManagedTensor(ctypes.Structure):
    pass


_DLManagedTensorDeleter = ctypes.CFUNCTYPE(None, ctypes.POINTER(_DLManagedTensor))
_DLManagedTensor._fields_ = [
    ("dl_tensor", _DLTensor),
    ("manager_ctx", ctypes.c_void_p),
    ("deleter", _DLManagedTensorDeleter),
]

_pycapsule_new = ctypes.pythonapi.PyCapsule_New
_pycapsule_new.restype = ctypes.py_object
_pycapsule_new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]


def make_dlpack_view(ptr: int, shape, dtype_str: str, device_type: str, device_id: int, keepalive: list):
    """Wrap a raw pointer as a torch tensor via DLPack (zero copy).

    Memory lifetime is owned by the caller (pool mapping); append-only
    ``keepalive`` must hold the ctypes structs through from_dlpack()."""
    import torch as _torch

    if dtype_str not in _DLPACK_DTYPES:
        raise ValueError(f"unsupported dtype for zero-copy view: {dtype_str}")
    code, bits, lanes = _DLPACK_DTYPES[dtype_str]
    ndim = len(shape)
    shape_arr = (ctypes.c_int64 * ndim)(*shape)

    managed = _DLManagedTensor()
    managed.dl_tensor.data = ctypes.c_void_p(ptr)
    managed.dl_tensor.device = _DLDevice(_DL_KDLCUDA if device_type == "cuda" else _DL_KDLCPU, device_id)
    managed.dl_tensor.ndim = ndim
    managed.dl_tensor.dtype = _DLDataType(code, bits, lanes)
    managed.dl_tensor.shape = shape_arr
    managed.dl_tensor.strides = None  # compact row-major
    managed.dl_tensor.byte_offset = 0
    managed.manager_ctx = None
    # NULL deleter (DLPack-legal): a Python trampoline deleter segfaults when
    # invoked during interpreter teardown; torch copies shape at construction.
    managed.deleter = ctypes.cast(None, _DLManagedTensorDeleter)
    keepalive.append((managed, shape_arr))
    capsule = _pycapsule_new(ctypes.byref(managed), b"dltensor", None)
    return _torch.from_dlpack(capsule)
