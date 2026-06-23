"""Tests for the ``payload_already_cloned`` short-circuit in
``vllm_omni.utils.mm_outputs.build_mm_cpu``.

These tests pin the contract used by the async-output GPU snapshot path
(see ``GPUARModelRunner._snapshot_omni_output_tensors_for_async_output``):
when the caller already produced an independent snapshot on the omni
payload copy stream, ``build_mm_cpu(keep_on_gpu=True)`` must NOT issue a
second ``.clone()`` — otherwise the redundant D2D ends up serialized on
the model's compute stream in the background output builder thread, which
is exactly the regression we are fixing.
"""

import pytest
import torch

from vllm_omni.utils.mm_outputs import _detach_tensor, build_mm_cpu

pytestmark = [pytest.mark.cpu]


def _shares_storage(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()


def test_build_mm_cpu_keep_on_gpu_already_cloned_skips_clone():
    src = torch.tensor([1.0, 2.0, 3.0])

    out = build_mm_cpu({"hidden": src}, keep_on_gpu=True, payload_already_cloned=True)

    assert _shares_storage(out["hidden"], src), (
        "When the snapshot has already cloned the tensor, build_mm_cpu must not "
        "re-clone — doing so would re-issue a D2D on the model compute stream."
    )


def test_build_mm_cpu_keep_on_gpu_default_still_clones():
    src = torch.tensor([1.0, 2.0, 3.0])

    out = build_mm_cpu({"hidden": src}, keep_on_gpu=True)

    assert not _shares_storage(out["hidden"], src), (
        "Non-snapshot callers (e.g. the synchronous, non-async output path) must "
        "still get a defensive clone."
    )
    assert torch.equal(out["hidden"], src)


def test_build_mm_cpu_cpu_branch_unaffected_by_already_cloned_flag():
    src = torch.tensor([1.0, 2.0, 3.0])

    out_default = build_mm_cpu({"hidden": src}, keep_on_gpu=False)
    out_flagged = build_mm_cpu({"hidden": src}, keep_on_gpu=False, payload_already_cloned=True)

    for out in (out_default, out_flagged):
        assert out["hidden"].device.type == "cpu"
        assert out["hidden"].is_contiguous()
        assert torch.equal(out["hidden"], src)


def test_detach_tensor_already_cloned_recursive_dict():
    inner_a = torch.tensor([4.0])
    inner_b = torch.tensor([5.0, 6.0])
    nested = {"a": inner_a, "b": [inner_b]}

    out = _detach_tensor(nested, keep_on_gpu=True, payload_already_cloned=True)

    assert _shares_storage(out["a"], inner_a)
    assert _shares_storage(out["b"][0], inner_b)
