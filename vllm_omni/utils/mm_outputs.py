"""Utilities for handling multimodal outputs / building multimodal output
payloads, most of which are shared by the prefix cache / no prefix cache path.
"""

from collections.abc import Mapping
from typing import Any

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

# Flat payload keys partitioned at worker output into inter-stage connector
# payloads vs client-facing multimodal outputs.  Only final output roots are
# listed here; everything else remains available for stage-to-stage transport.
_CLIENT_MM_ROOT_KEYS: frozenset[str] = frozenset(
    {
        "model_outputs",
        "sr",
        "audio",
        "image",
        "images",
        "video",
        "videos",
        "trajectory_latents",
        "latents",
    }
)


def partition_flat_payload(
    payload: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    """Split a flattened per-request payload into inter-stage vs client mm dicts."""
    if not payload:
        return {}, {}
    inter_stage: dict[str, object] = {}
    client_mm: dict[str, object] = {}
    for key, value in payload.items():
        root = key.split(".", 1)[0]
        if root in _CLIENT_MM_ROOT_KEYS:
            client_mm[key] = value
        else:
            inter_stage[key] = value
    return inter_stage, client_mm


def partition_payload_list(
    payloads: list[dict[str, object]],
) -> tuple[list[dict[str, object] | None] | None, list[dict[str, object] | None] | None]:
    inter_stage_list: list[dict[str, object] | None] = []
    client_mm_list: list[dict[str, object] | None] = []
    for payload in payloads:
        inter_stage, client_mm = partition_flat_payload(payload)
        inter_stage_list.append(inter_stage or None)
        client_mm_list.append(client_mm or None)
    return (
        None if all(item is None for item in inter_stage_list) else inter_stage_list,
        None if all(item is None for item in client_mm_list) else client_mm_list,
    )


# Marker on a dict value indicating "this slot held a GPU tensor that was
# stripped before crossing the engine-core -> API msgspec boundary; the real
# payload travelled via the connector pool". Downstream consumers of
# ``OmniEngineCoreOutput.multimodal_output`` for non-terminal stages should
# treat such values as opaque metadata and never call tensor methods on them.
_STRIPPED_GPU_TENSOR_MARKER = "_omni_stripped_gpu_tensor"


def strip_gpu_tensors_for_engine_output(value: Any) -> Any:
    """Replace CUDA tensors with msgspec-safe descriptors recursively.

    Why: the engine-core ``process_output_sockets`` thread serialises
    ``OmniEngineCoreOutput`` via msgpack, which calls ``tensor.detach().cpu()``
    on every tensor it encounters. For a keep_on_gpu mm payload (~57MB/chunk
    for Qwen3-Omni prefill) that call blocks on the GPU compute stream's
    drain (~145ms mean, multi-second p99 at c>=4 in profiling) — a redundant
    D2H tax, because the actual tensor data already travels downstream via
    the connector pool path. Callers gate this on the send edge being a
    GPU-tensor connector (see omni_ar_scheduler).

    Walks dict/list/tuple containers; CPU tensors and non-tensor values pass
    through unchanged. Returns new containers — the input is never mutated,
    so the connector's save_async callsite still sees the original GPU
    tensor refs. Each stripped tensor becomes a small dict carrying shape
    and dtype so metric/trace readers fail soft instead of crashing.
    """
    if isinstance(value, torch.Tensor):
        if value.device.type == "cuda":
            return {
                _STRIPPED_GPU_TENSOR_MARKER: True,
                "shape": list(value.shape),
                "dtype": str(value.dtype).removeprefix("torch."),
            }
        return value
    if isinstance(value, Mapping):
        return {k: strip_gpu_tensors_for_engine_output(v) for k, v in value.items()}
    if isinstance(value, list):
        return [strip_gpu_tensors_for_engine_output(v) for v in value]
    if isinstance(value, tuple):
        return tuple(strip_gpu_tensors_for_engine_output(v) for v in value)
    return value


def build_mm_cpu(
    multimodal_outputs: dict,
    keep_on_gpu: bool = False,
    payload_already_cloned: bool = False,
    gpu_min_bytes: int = 0,
    gpu_keys: "frozenset | None" = None,
) -> dict[str, object]:
    """Pre-copies multimodal tensor to CPU once (not per-request) to avoid
    redundant D2H transfers when gpu_resident_buffer_keys keeps them on GPU.

    In the case of prefix caching, the multimodal outputs provided will
    only contain the passthrough data.

    Args:
        multimodal_outputs: Multimodal dict mapping strings to objects.
        keep_on_gpu: When True, detach tensors but keep them on GPU for
            D2D transfer (e.g. via CudaIPCConnector).
        payload_already_cloned: When True with ``keep_on_gpu``, the input
            tensors are already independent snapshot copies; skip the
            defensive ``.clone()`` so the background output builder does not
            re-issue a redundant D2D on the model's compute stream.
    """
    if not multimodal_outputs:
        return {}

    # Pre-copy multimodal tensors to CPU once (not per-request) to avoid
    # redundant D2H transfers when gpu_resident_buffer_keys keeps them on GPU.
    mm_cpu: dict[str, object] = {}
    # Currently there are some cases where this is true at the
    # moment, which should be fixed.
    if not isinstance(multimodal_outputs, Mapping):
        logger.warning("Multimodal outputs are not a dict and will not be passed")

    for k, v in multimodal_outputs.items():
        if gpu_keys is not None:
            # STABLE per-key placement: a key either always stays on GPU or
            # always drops to CPU. Per-instance size decisions flap the device
            # across chunks and break downstream torch.cat over the stream.
            key_on_gpu = keep_on_gpu and isinstance(k, str) and k.split(".", 1)[0] in gpu_keys
            converted = _detach_tensor(v, key_on_gpu, payload_already_cloned, 0)
        else:
            converted = _detach_tensor(v, keep_on_gpu, payload_already_cloned, gpu_min_bytes)
        if converted is not None:
            mm_cpu[k] = converted
    return mm_cpu


def _detach_tensor(value, keep_on_gpu: bool = False, payload_already_cloned: bool = False, gpu_min_bytes: int = 0):
    """Recursively detach tensors; move to CPU unless keep_on_gpu is set."""
    if isinstance(value, torch.Tensor):
        # Tiered placement: only CUDA tensors >= gpu_min_bytes stay on GPU
        # (0 keeps everything); smaller/control tensors drop to CPU.
        if keep_on_gpu and value.is_cuda and value.nbytes >= gpu_min_bytes:
            # Snapshot is already an independent clone — just detach. Avoids a
            # redundant D2D on the model's compute stream in the async builder.
            if payload_already_cloned:
                return value.detach()
            return value.detach().clone()
        return value.detach().to("cpu").contiguous()
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            converted = _detach_tensor(v, keep_on_gpu, payload_already_cloned, gpu_min_bytes)
            if converted is not None:
                out[k] = converted
        return out or None
    if isinstance(value, list):
        if not value:
            return value
        return [_detach_tensor(v, keep_on_gpu, payload_already_cloned, gpu_min_bytes) for v in value]
    return value


def to_payload_element(
    element: object, idx: int, start: int, end: int, pass_lists_through: bool = False, seq_len: int | None = None
):
    """Build an mm payload element corresponding to one request index
    from an element containing 0 or more CPU tensors.

    Args:
        element: The object to be added to the payload.
        idx: The index of the request.
        start: The start index corresponding to the request idx.
        end: The end index corresponding to the request idx.
        pass_lists_through: bool Whether or not lists should be treated as
            passthrough data; this should be False in normal cases, but True
            if we need to avoid splitting nonempty lists prior to calling
            postprocess, which is the case for prefix cache.
        seq_len: Optional sequence length (i.e., dim 0 of hidden states).
            When set, a tensor whose first dimension equals seq_len is
            sliced per request. The prefix cache passthrough also passes
            the total scheduled token count here so 1D (seq_len,) metadata
            that is intentionally not cached is still split per request.
    """
    # Cached per-token tensors are merged elsewhere; here a first dim
    # equal to seq_len means a per-request slice is required.
    if seq_len is not None and isinstance(element, torch.Tensor) and element.shape[0] == seq_len:
        return element[start:end].contiguous()
    # Every other case is shared between prefix cache (passthrough data)
    # and running a model without prefix caching.
    elif isinstance(element, dict):
        return {
            sk: to_payload_element(sv, idx, start, end, pass_lists_through=pass_lists_through, seq_len=seq_len)
            for sk, sv in element.items()
        }
    elif isinstance(element, list):
        # For lists, clone tensors to avoid cross-request aliasing
        if pass_lists_through:
            return [elem.clone() if isinstance(elem, torch.Tensor) else elem for elem in element]
        element = element[idx] if idx < len(element) else element[0]
        if isinstance(element, torch.Tensor):
            element = element.clone()
        return element
    elif isinstance(element, torch.Tensor):
        # List-derived tensor payloads are request-invariant; clone to
        # avoid accidental cross-request aliasing on downstream mutation.
        return element.clone()
    return element
