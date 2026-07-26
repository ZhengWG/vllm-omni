# Speech Pipeline Audit: Qwen3-Omni, Ming-flash-omni-2.0, Ming-omni-tts

## Why this document exists

Three model families in vLLM-Omni share the same "understand, then speak" shape but have arrived at
very different levels of optimization. Qwen3-Omni has absorbed roughly forty bug-fix and performance
commits and is the reference for how a speech pipeline should be built here. Ming-omni-tts is a
younger pipeline that has picked up the same ideas selectively. Ming-flash-omni-2.0 works
end-to-end but has had almost no latency work at all.

This is a stock-take: what each family already has, where they diverge, and which of the remaining
gaps are worth closing. It is written to be read alongside
[Speech Generation on vLLM-Omni](qwen3_omni_tts_performance_optimization.md), which explains *how*
the individual mechanisms work; this document is about *which pipelines have them*.

For cross-checking, it also compares against
[sgl-project/sglang-omni](https://github.com/sgl-project/sglang-omni), which serves Qwen3-Omni and
Ming-Omni on a different runtime and has made some different architectural bets. That project is a
useful mirror precisely because it is not a fork: where both stacks independently converged on the
same mechanism, that mechanism is probably load-bearing; where only one has it, it is worth asking
why.

All file and commit references below point at this repository unless stated otherwise.

## Pipeline shapes

| | Qwen3-Omni | Ming-flash-omni-2.0 | Ming-omni-tts |
| --- | --- | --- | --- |
| Stages | Thinker → Talker → Code2Wav | Thinker → Talker | LLM+CFM → AudioVAE |
| Stage-to-stage payload | hidden states, then codec codes | detokenized text string | latent patches |
| Audio synthesis | RVQ codec tokens + vocoder | CFM flow head + AudioVAE, inside one `forward()` | CFM flow head, then AudioVAE stage |
| Default `async_chunk` | `true` | `false` | `true` |
| Default `max_num_seqs` | 64 / 64 / 64 | 1 / 1 | 1 / 1 |
| Topology | `vllm_omni/model_executor/models/qwen3_omni/pipeline.py` | `.../ming_flash_omni/pipeline.py` | `.../ming_tts/pipeline.py` |
| Deploy | `vllm_omni/deploy/qwen3_omni_moe.yaml` | `vllm_omni/deploy/ming_flash_omni.yaml` | `vllm_omni/deploy/ming_tts.yaml`, `ming_tts_moe.yaml` |

The `max_num_seqs` row is the single biggest structural difference. Qwen3-Omni is a concurrent
serving pipeline; both Ming pipelines are currently single-request pipelines that happen to be
deployed behind a server.

## What has already landed

### Qwen3-Omni

Ordered roughly by how much each one moves the needle.

| Mechanism | Where | Landed in |
| --- | --- | --- |
| Async chunk across all three stages | `stage_input_processors/qwen3_omni.py`, `deploy/qwen3_omni_moe.yaml` | #727, #951, #1656, #2581 |
| Streaming audio output and incremental Code2Wav decode | `qwen3_omni_code2wav.py::chunked_decode_streaming` | #367, #1246 |
| Small first codec chunk to cut TTFP | `initial_codec_chunk_frames` in the connector `extra` block | #4054 |
| Code2Wav CUDA graph with bucketed capture and eager fallback | `qwen3_tts/cuda_graph_decoder_wrapper.py` | #2376, #3732 |
| Talker-MTP CUDA graph capture | `worker/gpu_ar_model_runner.py::_capture_talker_mtp_graphs` | #669, #1005 |
| Code predictor re-prefill (no KV cache) plus `torch.compile` | `models/common/qwen3_code_predictor.py` | #1758, #2012 |
| Talker-MTP batch inference | `worker/gpu_model_runner.py` | #722 |
| Code2Wav batch inference under async chunk | `_batched_chunked_decode` | #1246 |
| GPU-resident decode buffers (no D2H in the talker loop) | `gpu_resident_buffer_keys` in `qwen3_omni.py` | #1758, #2012 |
| Fused QKV / gate-up projections, `SharedFusedMoE` | thinker and talker | #734, #560 |
| Dead `audio_tower` / `visual` removed from the talker stage | `qwen3_omni.py` | #3296 |
| Streaming tail-drop fix (boundary clicks, ~1.2% time compression) | `chunked_decode_streaming` | #4706 |
| Prefix-cache correctness under multi-stage | `worker/` | #3726, #4106 |
| Thinker preemption shape-mismatch fix | `gpu_ar_model_runner.py` | #3147 |
| Single-word / short-answer accuracy fixes | thinker sampling and handoff | #2239, #3385, #1288 |
| PD disaggregation, multi-replica, NPU/ROCm/XPU coverage | deploy + platform overrides | #2220, #484, #3946 |

Qwen3-Omni also has the widest test surface of the three: offline and online E2E, accuracy
benchmarks, DFX reliability and stability suites, and Buildkite entries at every CI level.

### Ming-omni-tts

| Mechanism | Where | Landed in |
| --- | --- | --- |
| Async chunk on latent patches with per-request VAE state cache | `stage_input_processors/ming_tts.py`, `ming_tts_audio_vae.py` | #2906 |
| CFM CUDA graph over the flow head, aggregator and stop head | `ming_tts/fm/cfm_cudagraph.py` | #4341 |
| RoPE/QKV fusion, opt-in `torch.compile` on the DiT, PIECEWISE cudagraph on the MoE backbone | `ming_tts_llm.py`, `deploy/ming_tts_moe.yaml` | #4942 |
| Small first latent chunk to cut TTFP | `initial_latent_chunk_size` | #5011 |
| Speaker-embedding cache for uploaded reference audio | `entrypoints/openai/tts_adapters/ming_tts.py` | #5240 |
| Seed-TTS WER accuracy fix | `models/common/ming/` | #4859 |
| Stage-1 stream-state TTL and LRU eviction | `ming_tts_audio_vae.py` | #2906 |

This family has tracked Qwen3-Omni's playbook closely for a much younger pipeline. The notable
omissions are batching (`max_num_seqs: 1` everywhere) and CUDA graphs on the dense Stage-0 backbone.

### Ming-flash-omni-2.0

| Mechanism | Where | Landed in |
| --- | --- | --- |
| CFM graph executor pool for the talker | `ming_flash_omni/talker_module.py` | #2890 |
| Global CUDA graph pool sharing | `talker_module.py` | #3361 |
| Request-level batching for the ImageGen diffusion stage | `diffusion/` | #4079, #4837, #4866 |
| Transformers 5.x / vLLM 0.22 compatibility | `transformers_utils/configs/ming_flash_omni.py` | #4080 |
| Shared DiT / timestep-embedding layers with Ming-omni-tts | `models/common/ming/` | #3285 |

Everything in that list is either bring-up, compatibility, or ImageGen. The **omni-speech path has
had no latency work at all**, and the default deploy reflects that: `async_chunk: false`,
`enforce_eager: true` on the talker, `max_num_seqs: 1` on both stages.

## Cross-check against sglang-omni

### Where both stacks agree

Both projects independently landed the same set of mechanisms for Qwen3-Omni: hidden-state (not
text) handoff from thinker to talker, per-decode-step streaming of codec frames into the vocoder,
exact-shape CUDA graphs for Code2Wav with eager fallback, suppression of the top-1024 codec vocab
except EOS, and left-context trimming so overlapping audio is not emitted twice. That convergence is
a reasonable signal that the Qwen3-Omni design here is sound.

On Code2Wav graphs specifically, vLLM-Omni's `CUDAGraphDecoderWrapper` is currently the more
complete of the two: it captures a batch × sequence-length matrix rather than batch-1 only, adds an
optional `torch.compile` + graph tier, and its `_trim_replay_output` handles the causal-conv trim
constant that makes Qwen3-Omni's true output shorter than `frames × total_upsample`. sglang-omni's
equivalent (`components/code2wav_cuda_graph.py`, their #1101) captures `batch_size=1` only, but adds
something we do not have: an explicit GPU memory budget with all-or-nothing publish and rollback, so
a capture that would overrun the stage's memory fraction disables the whole runner instead of
leaving a partially-populated graph table.

### Where they diverge, and it matters

**Ming thinker → talker streaming.** This is the largest single gap. sglang-omni runs Ming-Omni
speech in a `streaming_speech` topology with a dedicated segmenter stage between the thinker and the
talker (`models/ming_omni/components/streaming_text.py`, `streaming_segmenter.py`). The thinker
emits UTF-8 text deltas; the segmenter cuts them on sentence punctuation subject to a minimum of 8
tokens and a maximum window of 40, and — the part that actually buys the latency — emits the *first*
segment after only 4 tokens or 450 ms, whichever comes first, even without punctuation. The talker
starts synthesising the opening clause while the thinker is still generating.

vLLM-Omni has no equivalent. `thinker2talker_token_only` in
`stage_input_processors/ming_flash_omni.py` reads `output.text` off a finished thinker output, and
`MingFlashOmniTalkerForConditionalGeneration.forward` then runs the entire AR loop and VAE decode
inside a single call before returning a complete waveform. Time-to-first-audio is therefore the full
thinker latency plus the full talker latency. The talker already segments internally via
`segment_and_normalize`, so the per-segment machinery exists — what is missing is emitting each
segment's audio as it is finished rather than concatenating and returning at the end.

**Ming pipeline-level batching.** Every Ming deploy config pins `max_num_seqs: 1`, and
`ming_tts_moe.yaml` says so explicitly ("Multi-request batching is not yet validated"). Ming-omni-tts
Stage-0 additionally rejects mixed text/audio modes in one batch
(`ming_tts_llm.py::forward`), and the CFM CUDA graph is batch-1 only
(`fm/cfm_cudagraph.py`). sglang-omni is not much better here — its Ming talker also processes one
segment at a time — so this is a genuine gap in both stacks rather than a place where we are behind.
It still caps Ming throughput at roughly one request per GPU.

**Text normalization before TTS.** sglang-omni runs a four-stage pipeline inside the Ming talker:
semantic-length cutting, number normalization for non-Chinese text, a pynini-based `TalkerTN`
normalizer with an identity fallback, and a leading-comma strip. vLLM-Omni's `segment_and_normalize`
is thinner. This shows up as pronunciation quality on numerals, dates and units rather than as
latency, and it is the kind of difference that a WER benchmark surfaces but an E2E test does not.

**Silence and boundary handling in streaming Ming audio.** sglang-omni's `silence_holder` does
frame-based energy thresholding, buffers partial frames across chunks, trims trailing silence, and
deliberately retains a 0.3 s tail on the final chunk. vLLM-Omni trims trailing silence only in the
non-streaming path (`ming_flash_omni_talker.py::_decode_to_output`). If and when Ming streaming
lands, chunk-boundary artefacts will need the same treatment Qwen3-Omni's Code2Wav already got in
#4706.

**Thinker admission control.** sglang-omni's Qwen3-Omni talker supports partial start: it begins
building the talker request once 5 usable thinker chunks have arrived rather than waiting for
`stream_done` (`models/qwen3_omni/talker_scheduler.py`), with `im_end` excluded from the count. Our
async-chunk path streams per token, which achieves a similar effect by a different route, so this is
a design difference rather than a gap.

### Where we are ahead

Multi-stage deployment configuration (`vllm_omni/deploy/*.yaml` with platform overrides for NPU,
ROCm and XPU), PD disaggregation, the unified quantization framework across stages, and the
Code2Wav graph capture matrix described above. sglang-omni has no equivalent of the deploy-config
layer and does not use `torch.compile` anywhere in either model family.

## Defects found while writing this

Distinct from the roadmap below: these are concrete bugs in the current tree, each small enough to
fix and unit-test on its own. They are recorded here because the pattern is more interesting than any
individual fix — five of the seven are places where two layers disagreed about a contract, and none
of them fail loudly.

**Ming-omni-tts**

- `audio_prep.py::coerce_prompt_waveform` flattened `(channels, samples)` reference audio with
  `reshape(1, -1)`, splicing the channels end to end rather than downmixing. Duration doubles, so
  `count_prompt_waveform_patches` also over-budgets the `<audioPatch>` placeholders. The `bytes`
  path already downmixed with `waveform[:1]`; the raw-tensor, tuple and dict paths did not.
- `prompt_assembly.py::resolve_effective_runtime_controls` discarded a prompt's `Duration: Ns` hint
  whenever the caller supplied *either* decode bound, so `max_new_tokens` plus a duration hint left
  the min at 0 and let the stop head finish at the 5-step floor.
- `tts_adapters/ming_tts.py` inherited `max_new_tokens_min = 1`, but the value becomes
  `ming_max_decode_steps` and Stage-0 rejects anything below `stop_head_min_steps + 2`. Values 1–4
  were admitted and then failed inside the engine.
- `stage_input_processors/ming_tts.py::llm2audio_vae_async_chunk` initialised its bookkeeping by
  assigning over `transfer_manager.request_payload[req_id]`, which the connector owns.
- Six `torch.isfinite(x).all()` guards ran per decode step on the Stage-0 hot path, each one a
  device sync (see the *Open items* discussion of host syncs below).

**Ming-flash-omni-2.0**

- The talker pinned `torch.bfloat16` for segment prompt embeddings and for AudioVAE prompt encoding,
  while `self.dtype` was threaded everywhere else. Invisible on the released bf16 checkpoint.
- `_resolve_voice` silently fell through on an unregistered `voice_name`, after which
  `use_zero_spk_emb` substituted a zero speaker vector — plausible audio in an arbitrary voice, with
  no diagnostic. Reachable without a bad request, because presets are registered best-effort during
  `load_weights` and the thinker→talker bridge defaults `voice_name` to `DB30`.
- `CFMGraphExecutor.execute` rebuilt the constant `get_epss_timesteps` schedule and re-copied it into
  a captured placeholder on every step, and allocated its two noise tensors before blitting them into
  the static buffers.

## Open items

Ranked by expected impact per unit of invasiveness. None of these are blocking correctness today.

### 1. Streaming thinker → talker for Ming-flash-omni omni-speech

Currently the largest latency gap in the repo. The work is not a micro-optimization; it needs a
chunked handoff analogous to `thinker2talker_async_chunk`, a text segmenter with a deliberately
short first segment, and a talker path that yields audio per segment instead of concatenating.

Touches: `stage_input_processors/ming_flash_omni.py` (new async-chunk producer),
`ming_flash_omni/pipeline.py` (declare the async-chunk functions),
`ming_flash_omni_talker.py` (per-segment emission),
`deploy/ming_flash_omni.yaml` (`async_chunk: true` plus connector config).

Risks: the talker retokenizes the string itself, so segment boundaries must not split a token
sequence the talker's prompt builder depends on; and per-segment CFM state (`his_lat`) has to carry
across segments or prosody will reset at each boundary.

### 2. Multi-request batching for the Ming families

Raising `max_num_seqs` above 1 requires, at minimum: removing the mixed text/audio batch restriction
in `ming_tts_llm.py::forward`, capturing CFM graphs for more than batch 1 in `fm/cfm_cudagraph.py`,
and validating that Stage-1's per-request VAE state cache (`_past_key_values` / `_stream_state`
keyed by request id) behaves under interleaved chunks from different requests. The state cache
already looks batch-safe; the flow head does not.

### 3. CUDA graphs on the Ming-omni-tts dense Stage-0 backbone

`ming_tts.yaml` sets `enforce_eager: true` on Stage-0 while `ming_tts_moe.yaml` sets
`cudagraph_mode: PIECEWISE`. The dense 0.5B backbone therefore launches every decode step eagerly,
even though the flow head on top of it is already graph-captured. Whether the two graph layers
compose safely is the open question; if they do, this is a config change plus a benchmark.

### 4. Memory-budgeted Code2Wav graph capture

Port the idea from sglang-omni #1101: give `CUDAGraphDecoderWrapper.warmup` a memory budget derived
from the stage's `gpu_memory_utilization`, and on overrun fail the whole runner back to eager rather
than keeping a partial graph table. Today a capture matrix that is too large for the stage's
allocation surfaces as an OOM at capture time or as silent per-shape fallbacks.

### 5. Batched prefill projection for the Qwen3-Omni thinker → talker handoff

`qwen3_omni.py::_thinker_to_talker_prefill` carries an explicit `# Take batch 0 since batched
inference is not supported here.` The pipeline batches fine at request level, so this only costs a
loop over requests during prefill rather than correctness — but it is a per-request Python and
`.item()` cost on the critical path to first audio.

### 6. Text normalization parity for the Ming talkers

Number, date and unit normalization before synthesis, matched against a WER benchmark rather than an
E2E smoke test. Lowest risk of the six and independently landable.

## Reading the code

Entry points, if you want to follow any of the above:

- Pipeline topologies: `vllm_omni/model_executor/models/*/pipeline.py`
- Stage handoff logic: `vllm_omni/model_executor/stage_input_processors/`
- Deploy defaults, including per-platform overrides: `vllm_omni/deploy/*.yaml`
- Multi-stage orchestration: `vllm_omni/engine/orchestrator.py`
- Talker-MTP and Code2Wav graph capture: `vllm_omni/worker/gpu_ar_model_runner.py`,
  `vllm_omni/model_executor/models/qwen3_tts/cuda_graph_decoder_wrapper.py`
