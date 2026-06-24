"""Utilities for handling multimodal outputs / building multimodal output
payloads, most of which are shared by the prefix cache / no prefix cache path.
"""

from collections.abc import Mapping

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)


def build_mm_cpu(
    multimodal_outputs: dict,
    keep_on_gpu: bool = False,
    payload_already_cloned: bool = False,
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
            tensors are already independent snapshot copies (taken on the
            omni payload copy stream); skip the redundant ``.clone()`` here
            so the background output builder does not re-issue D2D copies on
            the model's compute stream. Ignored for the CPU branch.
    """
    mm_cpu: dict[str, object] = {}
    # Currently there are some cases where this is true at the
    # moment, which should be fixed.
    if not isinstance(multimodal_outputs, Mapping):
        logger.warning("Multimodal outputs are not a dict and will not be passed")

    if multimodal_outputs:
        for k, v in multimodal_outputs.items():
            converted = _detach_tensor(v, keep_on_gpu, payload_already_cloned)
            if converted is not None:
                mm_cpu[k] = converted
    return mm_cpu


def _detach_tensor(value, keep_on_gpu: bool = False, payload_already_cloned: bool = False):
    """Recursively detach tensors; move to CPU unless keep_on_gpu is set."""
    if isinstance(value, torch.Tensor):
        if keep_on_gpu:
            # Snapshot is already an independent clone — just detach to drop
            # autograd state. Avoids a redundant D2D on the model's compute
            # stream when called from the async output background builder.
            if payload_already_cloned:
                return value.detach()
            return value.detach().clone()
        return value.detach().to("cpu").contiguous()
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            converted = _detach_tensor(v, keep_on_gpu, payload_already_cloned)
            if converted is not None:
                out[k] = converted
        return out or None
    if isinstance(value, list):
        if not value:
            return value
        return [_detach_tensor(v, keep_on_gpu, payload_already_cloned) for v in value]
    return value


def to_payload_element(
    element: object,
    idx: int,
    start: int,
    end: int,
    pass_lists_through: bool = False,
    seq_len: int | None = None,
    payload_already_cloned: bool = False,
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
        payload_already_cloned: When True, the input ``element`` is part of
            an already-snapshotted payload (independent of CUDA-graph reuse
            buffers). Skip the defensive ``.clone()`` calls in this
            function — they would otherwise re-issue a GPU D2D in the
            async-output background thread on the *compute* stream
            (background thread's ``current_stream()`` is the legacy
            default = main thread's compute stream when PTDS is off),
            serialising with the next forward and erasing the
            ``OmniAsyncGPUModelRunnerOutput`` overlap. Cross-request
            aliasing protection is preserved by the snapshot stage's
            initial clone.
    """
    # Cached per-token tensors are merged elsewhere; here a first dim
    # equal to seq_len means a per-request slice is required.
    if seq_len is not None and isinstance(element, torch.Tensor) and element.shape[0] == seq_len:
        sliced = element[start:end]
        # ``contiguous()`` is a no-op when the slice is already contiguous
        # (the common case for a contiguous-dim-0 snapshot tensor); when it
        # isn't, this is a small D2D on the caller's stream — same cost
        # whether the snapshot has been cloned or not.
        return sliced.contiguous()
    # Every other case is shared between prefix cache (passthrough data)
    # and running a model without prefix caching.
    elif isinstance(element, dict):
        return {
            sk: to_payload_element(
                sv,
                idx,
                start,
                end,
                pass_lists_through=pass_lists_through,
                seq_len=seq_len,
                payload_already_cloned=payload_already_cloned,
            )
            for sk, sv in element.items()
        }
    elif isinstance(element, list):
        # For lists, clone tensors to avoid cross-request aliasing — the
        # snapshot stage's ``_clone_cuda_tensor_payload`` walks dict/list/
        # tuple containers, so when ``payload_already_cloned`` is True the
        # list elements are already independent clones and the
        # cross-request aliasing protection is satisfied without another
        # clone per element.
        if pass_lists_through:
            if payload_already_cloned:
                return list(element)
            return [elem.clone() if isinstance(elem, torch.Tensor) else elem for elem in element]
        element = element[idx] if idx < len(element) else element[0]
        if isinstance(element, torch.Tensor) and not payload_already_cloned:
            element = element.clone()
        return element
    elif isinstance(element, torch.Tensor):
        # List-derived tensor payloads are request-invariant; clone to
        # avoid accidental cross-request aliasing on downstream mutation.
        if payload_already_cloned:
            return element
        return element.clone()
    return element
