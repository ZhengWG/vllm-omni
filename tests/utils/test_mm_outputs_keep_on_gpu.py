"""Tests for the ``payload_already_cloned`` short-circuit in
``vllm_omni.utils.mm_outputs``.

These tests pin the contract used by the async-output GPU snapshot path
(see ``GPUARModelRunner._snapshot_omni_output_tensors_for_async_output``):
when the caller already produced an independent snapshot on the omni
payload copy stream, ``build_mm_cpu(keep_on_gpu=True)`` and
``to_payload_element(payload_already_cloned=True)`` must NOT issue a
second ``.clone()``. Each redundant clone here would land on the
background-thread's ``current_stream()`` (the legacy default = main
thread's compute stream when PTDS is off) and serialise with the next
forward — erasing the ``OmniAsyncGPUModelRunnerOutput`` overlap that
``#4476`` was specifically designed to deliver, and reintroducing the
+67–324 ms / c≥4 tax that motivated this PR's keep_on_gpu work.
"""

import pytest
import torch

from vllm_omni.utils.mm_outputs import (
    _STRIPPED_GPU_TENSOR_MARKER,
    _detach_tensor,
    build_mm_cpu,
    strip_gpu_tensors_for_engine_output,
    to_payload_element,
)

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


# ────────────────────────────────────────────────────────────────────
# ``to_payload_element`` short-circuit (per-request payload derivation)
# ────────────────────────────────────────────────────────────────────


def test_to_payload_element_raw_tensor_already_cloned_skips_clone():
    """The raw-tensor branch is the dominant per-request path for
    list-derived multimodal payloads after ``mm_cpu`` indexing.
    Without this short-circuit, every request would issue a
    redundant GPU D2D in the BG output builder thread on the
    compute stream."""
    src = torch.tensor([1.0, 2.0, 3.0])

    out = to_payload_element(src, idx=0, start=0, end=0, payload_already_cloned=True)

    assert _shares_storage(out, src)


def test_to_payload_element_raw_tensor_default_clones():
    """Non-snapshot callers (synchronous non-async-output path,
    prefix_cache passthrough, NPU runner) must still get the
    defensive clone."""
    src = torch.tensor([1.0, 2.0, 3.0])

    out = to_payload_element(src, idx=0, start=0, end=0)

    assert not _shares_storage(out, src)
    assert torch.equal(out, src)


def test_to_payload_element_list_branch_already_cloned_skips_clone():
    """The list-indexing branch is what the qwen3-omni talker / mm-list
    payloads actually hit per request. Each request indexes a different
    list element, and without the short-circuit each per-request
    derivation re-clones a GPU tensor on the BG-thread's compute
    stream."""
    elements = [torch.tensor([10.0, 20.0]), torch.tensor([30.0, 40.0])]

    out = to_payload_element(elements, idx=1, start=0, end=0, payload_already_cloned=True)

    assert _shares_storage(out, elements[1])


def test_to_payload_element_list_branch_default_still_clones():
    elements = [torch.tensor([10.0, 20.0]), torch.tensor([30.0, 40.0])]

    out = to_payload_element(elements, idx=0, start=0, end=0)

    assert not _shares_storage(out, elements[0])
    assert torch.equal(out, elements[0])


def test_to_payload_element_seq_len_slice_unchanged_by_flag():
    """The ``seq_len`` slice path is shape-driven, not aliasing-driven —
    behaviour must be identical regardless of the flag."""
    seq_tensor = torch.arange(10, dtype=torch.float32)

    out_default = to_payload_element(seq_tensor, idx=0, start=2, end=5, seq_len=10)
    out_flagged = to_payload_element(
        seq_tensor, idx=0, start=2, end=5, seq_len=10, payload_already_cloned=True
    )

    assert torch.equal(out_default, torch.tensor([2.0, 3.0, 4.0]))
    assert torch.equal(out_flagged, out_default)


def test_to_payload_element_dict_recursion_propagates_flag():
    """Nested dict payloads must propagate the flag so the inner
    tensor-leaf branch also short-circuits its clones."""
    inner = torch.tensor([7.0, 8.0])
    nested = {"inner": inner, "deeper": {"leaf": inner}}

    out = to_payload_element(nested, idx=0, start=0, end=0, payload_already_cloned=True)

    assert _shares_storage(out["inner"], inner)
    assert _shares_storage(out["deeper"]["leaf"], inner)


def test_to_payload_element_pass_lists_through_already_cloned_skips_per_element_clone():
    """The ``pass_lists_through`` path is exercised by the prefix-cache
    callsite. When the snapshot has already cloned, returning the list
    as-is is equivalent and saves N redundant GPU clones for an
    N-element list — relevant when the talker stages emit per-request
    audio code lists."""
    elements = [torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0])]

    out = to_payload_element(
        elements, idx=0, start=0, end=0, pass_lists_through=True, payload_already_cloned=True
    )

    assert isinstance(out, list)
    assert len(out) == len(elements)
    for a, b in zip(out, elements):
        assert _shares_storage(a, b)


# ────────────────────────────────────────────────────────────────────
# ``strip_gpu_tensors_for_engine_output`` — Route A structural fix
# ────────────────────────────────────────────────────────────────────


class _CudaLikeTensor(torch.Tensor):
    """Lightweight stand-in: a real CPU tensor that *reports* itself as a
    CUDA tensor via ``device.type``. The strip helper only inspects
    ``device.type``, ``shape``, and ``dtype`` so this is enough to
    exercise its branches without a real GPU."""

    @staticmethod
    def __new__(cls, data: torch.Tensor) -> "_CudaLikeTensor":
        return torch.Tensor._make_subclass(cls, data)

    @property
    def device(self) -> torch.device:  # type: ignore[override]
        return torch.device("cuda", 0)


def _gpu_like(*shape: int, dtype: torch.dtype = torch.float32) -> _CudaLikeTensor:
    return _CudaLikeTensor(torch.zeros(*shape, dtype=dtype))


def test_strip_replaces_cuda_tensor_with_descriptor_dict():
    src = _gpu_like(128, 256, dtype=torch.bfloat16)

    out = strip_gpu_tensors_for_engine_output(src)

    assert isinstance(out, dict)
    assert out[_STRIPPED_GPU_TENSOR_MARKER] is True
    assert out["shape"] == [128, 256]
    assert out["dtype"] == "bfloat16"


def test_strip_passes_cpu_tensor_through_unchanged():
    """Stripping must be a no-op for CPU tensors so terminal stages
    (whose ``mm_output`` is already on CPU) are unaffected when the
    same helper is applied unconditionally."""
    src = torch.tensor([1.0, 2.0, 3.0])

    out = strip_gpu_tensors_for_engine_output(src)

    assert out is src


def test_strip_walks_dict_and_list_recursively():
    """The qwen3-omni wire mm payload is a flat dict mostly, but the
    helper must still walk nested dict/list/tuple containers because
    other models (talker code lists, sparse audio per-request lists)
    use them."""
    cpu_meta = {"req_id": "r0", "finished": False}
    payload = {
        "embed": _gpu_like(4, 8),
        "tts_bos": _gpu_like(1, 8),
        "meta": cpu_meta,
        "code_list": [_gpu_like(2), _gpu_like(2), torch.tensor([1, 2, 3])],
        "nested_tuple": (_gpu_like(3), "label"),
    }

    out = strip_gpu_tensors_for_engine_output(payload)

    assert isinstance(out["embed"], dict) and out["embed"][_STRIPPED_GPU_TENSOR_MARKER] is True
    assert isinstance(out["tts_bos"], dict) and out["tts_bos"]["shape"] == [1, 8]
    # CPU meta dict passes through unchanged content (new dict object due to recursion).
    assert out["meta"] == cpu_meta
    assert isinstance(out["code_list"], list) and len(out["code_list"]) == 3
    assert out["code_list"][0][_STRIPPED_GPU_TENSOR_MARKER] is True
    assert out["code_list"][1][_STRIPPED_GPU_TENSOR_MARKER] is True
    # CPU tensor inside the list passes through.
    assert isinstance(out["code_list"][2], torch.Tensor)
    assert out["code_list"][2].device.type == "cpu"
    assert isinstance(out["nested_tuple"], tuple)
    assert out["nested_tuple"][0][_STRIPPED_GPU_TENSOR_MARKER] is True
    assert out["nested_tuple"][1] == "label"


def test_strip_does_not_mutate_input():
    """Critical: ``save_async`` is called with the *original*
    ``mm_output`` reference and the strip happens in-line for
    ``OmniEngineCoreOutput``. If the helper mutated the input dict,
    the connector save thread would see a corrupted payload."""
    src_tensor = _gpu_like(2, 2)
    payload = {"embed": src_tensor, "meta": {"finished": False}}

    out = strip_gpu_tensors_for_engine_output(payload)

    assert payload["embed"] is src_tensor, "input dict's tensor ref must survive"
    assert isinstance(payload["embed"], torch.Tensor)
    assert payload["embed"].device.type == "cuda"
    assert payload["meta"] == {"finished": False}
    # Output is a new dict.
    assert out is not payload
    assert out["embed"] is not src_tensor


def test_strip_handles_non_tensor_scalars_and_none():
    """Defensive: any scalar / bool / None / bytes that already passes
    msgpack should fall through unchanged so the helper is safe to
    apply unconditionally."""
    payload = {
        "is_seg_finished": True,
        "ttl_sec": 30,
        "rate": 1.5,
        "tag": "text",
        "raw": b"bytes",
        "missing": None,
    }

    out = strip_gpu_tensors_for_engine_output(payload)

    assert out == payload
