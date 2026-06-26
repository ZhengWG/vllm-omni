#!/usr/bin/env python3
"""Probe PyTorch CUDA shared-storage behavior for connector prototyping.

This script is intentionally standalone: run it inside the serving container to
discover the exact private CUDA storage IPC APIs exposed by that PyTorch build.

Examples:
    python3 examples/online_serving/qwen3_omni/probe_shared_cuda_storage.py
    python3 examples/online_serving/qwen3_omni/probe_shared_cuda_storage.py --roundtrip
"""

from __future__ import annotations

import argparse
import os
import pprint
import sys
from typing import Any


def _summarize(value: Any, depth: int = 0) -> Any:
    if depth > 3:
        return f"<{type(value).__name__}>"
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, bytes):
        return {"type": "bytes", "len": len(value), "head_hex": value[:16].hex()}
    if isinstance(value, bytearray):
        return {"type": "bytearray", "len": len(value), "head_hex": bytes(value[:16]).hex()}
    if isinstance(value, memoryview):
        head = bytes(value[: min(16, len(value))])
        return {"type": "memoryview", "len": len(value), "head_hex": head.hex()}
    if isinstance(value, tuple):
        return tuple(_summarize(v, depth + 1) for v in value)
    if isinstance(value, list):
        return [_summarize(v, depth + 1) for v in value[:8]] + (
            [f"... ({len(value)} total)"] if len(value) > 8 else []
        )
    if isinstance(value, dict):
        return {k: _summarize(v, depth + 1) for k, v in list(value.items())[:16]}
    return repr(value)


def _print_storage_methods(storage: Any, label: str) -> None:
    print(f"\n[{label}] type={type(storage)!r}")
    for name in [
        "_share_cuda_",
        "_new_shared_cuda",
        "_new_shared",
        "_share_filename_cpu_",
        "_new_shared_filename_cpu",
    ]:
        attr = getattr(storage, name, None)
        print(f"  {name}: present={attr is not None} attr={attr!r}")


def _roundtrip_worker(queue, result_queue) -> None:
    import torch

    tensor = queue.get()
    result_queue.put(
        {
            "shape": tuple(tensor.shape),
            "stride": tuple(tensor.stride()),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "sum": float(tensor.float().sum().item()),
            "data_ptr": int(tensor.data_ptr()),
            "is_contiguous": bool(tensor.is_contiguous()),
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roundtrip", action="store_true", help="send CUDA tensor through torch.multiprocessing.Queue")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    try:
        import torch
    except Exception as exc:
        print(f"ERROR: failed to import torch: {exc}", file=sys.stderr)
        return 2

    print("torch.__version__:", torch.__version__)
    print("torch.cuda.is_available:", torch.cuda.is_available())
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("PYTORCH_CUDA_ALLOC_CONF:", os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))

    if not torch.cuda.is_available():
        print("CUDA unavailable; storage IPC probe skipped.")
        return 0

    device = torch.device(args.device)
    torch.cuda.set_device(device)

    base = torch.arange(16, device=device, dtype=torch.float32).reshape(4, 4)
    view = base[:, :2]
    tensors = {"contiguous": base.contiguous(), "strided_view": view}

    for label, tensor in tensors.items():
        print(f"\n=== tensor: {label} ===")
        print("shape:", tuple(tensor.shape))
        print("stride:", tuple(tensor.stride()))
        print("dtype:", tensor.dtype)
        print("device:", tensor.device)
        print("data_ptr:", int(tensor.data_ptr()))
        print("storage_offset:", int(tensor.storage_offset()))
        print("is_contiguous:", bool(tensor.is_contiguous()))
        storage = tensor.untyped_storage()
        _print_storage_methods(storage, f"{label}.untyped_storage")

        share_cuda = getattr(storage, "_share_cuda_", None)
        if share_cuda is not None:
            try:
                shared = share_cuda()
                print("  _share_cuda_() returned:")
                pprint.pp(_summarize(shared), width=120)
            except Exception as exc:
                print(f"  _share_cuda_() ERROR: {type(exc).__name__}: {exc}")

        try:
            from torch.multiprocessing import reductions

            reduced = reductions.reduce_tensor(tensor)
            print("  torch.multiprocessing.reductions.reduce_tensor returned:")
            pprint.pp(_summarize(reduced), width=120)
        except Exception as exc:
            print(f"  reduce_tensor ERROR: {type(exc).__name__}: {exc}")

    if args.roundtrip:
        import torch.multiprocessing as mp

        print("\n=== multiprocessing roundtrip ===")
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        rq = ctx.Queue()
        proc = ctx.Process(target=_roundtrip_worker, args=(q, rq))
        proc.start()
        send_tensor = torch.arange(1024, device=device, dtype=torch.float32)
        q.put(send_tensor)
        result = rq.get(timeout=30)
        proc.join(timeout=30)
        print("worker result:")
        pprint.pp(result, width=120)
        print("producer data_ptr:", int(send_tensor.data_ptr()))
        print("exitcode:", proc.exitcode)
        if proc.exitcode != 0:
            return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
