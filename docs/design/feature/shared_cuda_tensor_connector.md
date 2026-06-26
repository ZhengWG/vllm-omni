# Shared CUDA Tensor Connector Design

## Status

Draft design / prototype proposal.

This document summarizes lessons learned from the CUDA IPC connector
experiments and proposes a simpler connector design for large CUDA tensor
handoff. The proposed connector is not intended to replace
`SharedMemoryConnector` as a general transport. It targets large GPU tensor
handoffs where CPU bounce is visible and hard to overlap.

## Motivation

The current byte-pool `CudaIPCConnector` proved that a CUDA IPC fast path can
work, but also exposed several structural costs:

- CUDA graph outputs are often graph-static buffers, so the producer must take
  a snapshot before the next graph replay can overwrite the same storage.
- The byte-pool connector has to manage pool credits, release boards, TTL
  cleanup, per-slot events, and fallback paths.
- Pool packing copies large tensors into a connector-owned byte pool and builds
  Python/msgpack descriptors.
- Small per-token payloads do not benefit from GPU-direct transfer and are
  often faster and simpler through `SharedMemoryConnector`.

The goal of this proposal is to keep the useful part of CUDA IPC -- large
tensor GPU-direct handoff -- while removing most byte-pool lifecycle machinery.

## Non-goals

- Do not make CUDA IPC the default transport for all payloads.
- Do not optimize small per-token payloads with CUDA IPC.
- Do not share graph-static model output buffers directly.
- Do not promise that Qwen3-Omni high-concurrency E2E latency will beat SHM.
- Do not put model-specific payload routing logic in the transfer adapter.

## High-level design

Introduce an experimental `SharedCudaTensorConnector`:

```text
OmniChunkTransferAdapter
  |
  | put/get(data)
  v
SharedCudaTensorConnector
  |-- PayloadSplitter             # generic tensor-vs-CPU payload split
  |-- SharedMemoryConnector       # small payload / metadata / CPU values
  |-- SharedCudaTensorTransport   # large CUDA tensor snapshot + share
  |-- ControlPlane                # descriptor publication
  |-- AckManager                  # bounded inflight + cleanup
```

The adapter remains transport-agnostic. It calls `connector.put(data)` and
`connector.get(key)` exactly as today.

## Routing policy

Routing is connector-internal and generic:

| Payload value | Route |
| --- | --- |
| Non-tensor metadata / lists / strings | `SharedMemoryConnector` |
| CPU tensor | `SharedMemoryConnector` |
| Small CUDA tensor | `SharedMemoryConnector` |
| Large CUDA tensor | shared CUDA snapshot transport |

Example:

```python
OmniPayloadStruct(
    embed.prefill = cuda_tensor_0,          # large
    hidden_states.output = cuda_tensor_1,   # large
    ids.all = [...],                        # metadata
    ids.prompt = [...],                     # metadata
    meta = ...,
)
```

becomes:

```python
cpu_payload = {
    "embed": {"prefill": TensorRef("t0")},
    "hidden_states": {"output": TensorRef("t1")},
    "ids": {"all": [...], "prompt": [...]},
    "meta": ...,
}

tensor_entries = [
    TensorEntry(ref="t0", path=("embed", "prefill"), tensor=cuda_tensor_0),
    TensorEntry(ref="t1", path=("hidden_states", "output"), tensor=cuda_tensor_1),
]
```

No Qwen3-specific field names are required by the adapter. The connector only
uses generic size/device/type rules.

## Producer flow

### 1. Wait for payload readiness

The correctness boundary learned from previous experiments is retained:

```text
model output ready
  -> save loop waits producer event
  -> custom_process builds payload
  -> payload_ready_event
  -> connector snapshots/shares large tensors
```

The connector must never share or copy from payload tensors before
`payload_ready_event`.

### 2. Snapshot graph-static tensors

The producer must not share original CUDA graph output buffers directly. Those
buffers may be overwritten by the next replay.

For each large CUDA tensor:

```python
with torch.cuda.stream(snapshot_stream):
    snapshot_stream.wait_event(payload_ready_event)
    snapshot = tensor.detach().clone()
    ready_event.record(snapshot_stream)
```

This snapshot copy is unavoidable for graph-static outputs.

### 3. Share CUDA storage

After snapshot creation, the connector shares the snapshot storage using
PyTorch CUDA shared storage APIs, for example:

```python
storage = snapshot.untyped_storage()
handle = storage._share_cuda_()
```

The exact API surface is private and version-dependent. The implementation must
probe runtime support and fail clearly or fall back when unsupported.

The tensor descriptor should include:

```python
{
    "ref": "t0",
    "device": snapshot.device.index,
    "dtype": str(snapshot.dtype),
    "shape": list(snapshot.shape),
    "stride": list(snapshot.stride()),
    "storage_offset": snapshot.storage_offset(),
    "storage_nbytes": snapshot.untyped_storage().nbytes(),
    "cuda_handle": handle,
    "ready_event": optional_cuda_ipc_event_handle,
}
```

### 4. Store CPU payload

The CPU payload, including metadata and `TensorRef`s, is stored through
`SharedMemoryConnector`.

### 5. Publish control descriptor

The control descriptor references the CPU payload and all shared CUDA tensors:

```python
{
    "transfer_id": "...",
    "cpu_payload": shm_meta,
    "tensors": [tensor_descriptor_0, tensor_descriptor_1],
}
```

The descriptor can initially be stored in SHM. A ring notification can be added
later if polling overhead matters.

### 6. Track inflight references

The producer keeps strong references to snapshots until consumer ack or TTL:

```python
inflight[transfer_id] = {
    "snapshots": [snapshot0, snapshot1],
    "created_at": time.monotonic(),
    "bytes": total_bytes,
}
```

This prevents storage release while the consumer is still rebuilding/using the
tensor.

## Receiver flow

### 1. Load control descriptor

The receiver obtains the control descriptor by key.

### 2. Load CPU payload

The CPU payload is read through `SharedMemoryConnector`.

### 3. Rebuild shared CUDA tensors

For each tensor descriptor, rebuild storage using the matching PyTorch private
API, for example:

```python
storage = torch.UntypedStorage._new_shared_cuda(...)
tensor = torch.empty(0, device=device, dtype=dtype).set_(
    storage,
    storage_offset,
    shape,
    stride,
)
```

The exact call signature must be discovered in the target runtime.

### 4. Wait for snapshot readiness

If the descriptor carries a CUDA event handle, the receiver waits on it before
exposing the tensor to the model.

### 5. Rehydrate payload

Replace `TensorRef`s in the CPU payload with rebuilt CUDA tensors.

### 6. Ack

After successful reconstruction/hand-off, the receiver acks the transfer id.
The producer can then release snapshot references.

## Lifecycle model

The design replaces byte-pool credit/board/TTL with:

- bounded inflight transfer count
- bounded inflight bytes
- producer-held snapshot references
- consumer ack
- TTL cleanup for crash recovery

Configuration:

```yaml
max_inflight_transfers: 64
max_inflight_bytes: ...
transfer_ttl_sec: 30
```

If bounds are exceeded, the connector can block, backpressure, or fall back to
SHM depending on configuration.

## Stream correctness

PyTorch storage sharing can help with memory lifetime, but it does not by
itself guarantee stream ordering or content stability.

Required ordering:

```text
payload_ready_event
  -> snapshot clone on producer stream/snapshot stream
  -> snapshot_ready_event
  -> receiver wait snapshot_ready_event
  -> downstream consumption
  -> ack/release
```

`record_stream` may still be required for rebuilt tensors on the consumer side
so the caching allocator does not reuse storage before consumer kernels finish.

## Why not share graph-static output directly?

CUDA graph FULL mode can reuse the same output buffer address on every replay.
Sharing the original output storage would expose the consumer to torn reads when
the producer replays the next graph.

Therefore:

```text
graph-static output -> snapshot clone -> shared storage
```

is mandatory.

## Expected benefits

Compared with the current byte-pool `CudaIPCConnector`:

- remove connector-owned byte pool packing
- remove pool credits and release board
- remove pool slot offset bookkeeping
- remove receiver D2D copy out of pool if consumer can use rebuilt tensor
- reduce descriptor size
- reduce custom lifecycle code
- reduce classes of use-after-free / release-board / TTL bugs

## Expected non-benefits

This design will not automatically:

- eliminate the graph-static snapshot clone
- make small per-token payloads faster than SHM
- guarantee Qwen3-Omni high-concurrency E2E improvement
- remove all stream-ordering concerns

## Performance expectations

### Qwen3-Omni async chunk

Expected:

- correctness and maintainability improve
- large prefill handoff may improve modestly
- E2E may not improve significantly if transfer is overlapped or not critical
- TPOT remains dominated by decode path and small payloads

### Pure large-tensor workloads

Expected:

- better than SHM when CPU bounce is exposed
- cleaner than byte-pool IPC for large tensor handoff
- better fit for diffusion/video latent transfer workloads

### High concurrency

Expected:

- less Python/control overhead than byte-pool connector
- still bounded by snapshot copy and GPU copy bandwidth
- benefit depends on whether transfer is on the E2E critical path

## Runtime validation requirements

Before implementing in production, validate in the target container:

1. PyTorch exposes usable CUDA shared storage APIs:

   ```python
   storage._share_cuda_()
   torch.UntypedStorage._new_shared_cuda(...)
   ```

2. `expandable_segments` behavior is compatible or has a safe fallback.
3. Snapshot sharing works across producer/consumer processes.
4. Consumer can read after producer releases only after ack.
5. TTL cleanup prevents unbounded VRAM retention after consumer crash.

## Soak tests

Minimum soak tests:

1. Single large tensor handoff, compare values.
2. Multiple concurrent handoffs with bounded inflight.
3. Producer repeatedly reuses graph-static output buffer while consumers read
   snapshots.
4. Consumer delayed reads while producer advances many steps.
5. Consumer crash before ack.
6. Producer crash after publishing descriptor.
7. Mixed CPU payload + large CUDA tensor payload.

## Implementation phases

### Phase 0: Runtime probe

Add a script to inspect `_share_cuda_` / `_new_shared_cuda` signatures in the
actual serving container.

### Phase 1: Micro prototype

Two-process standalone prototype:

```text
producer tensor -> snapshot -> share -> receiver rebuild -> compare
```

No Omni connector integration.

### Phase 2: Connector skeleton

Add `SharedCudaTensorConnector` registered in `OmniConnectorFactory`.

Support:

- CPU payload through `SharedMemoryConnector`
- one or more large contiguous CUDA tensors
- bounded inflight
- TTL cleanup

### Phase 3: Qwen prefill experiment

Route only large Qwen prefill tensors through shared CUDA storage. Keep decode
payloads on SHM.

### Phase 4: Benchmark matrix

Compare:

- `SharedMemoryConnector`
- current byte-pool `CudaIPCConnector`
- new `SharedCudaTensorConnector`

Across:

- concurrency 1/2/4/8
- tensor size 64/128/256MB
- Qwen prefill-only
- Qwen full async chunk
- diffusion/video large latent workload

## Decision criteria

Keep the new connector only if it provides at least one of:

- clear performance win on pure large tensor or diffusion/video workload
- substantial code simplification with correctness parity
- lower operational risk than byte-pool CUDA IPC

Do not make it default for Qwen3-Omni unless benchmarks show consistent
high-concurrency wins.
