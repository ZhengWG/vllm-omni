# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.
# Copyright 2024 ANT Group and the HuggingFace Inc. team. All rights reserved.
# Adapted from Ming repository modeling_bailingmm2.py
# https://github.com/inclusionAI/Ming
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ming-flash-omni-2.0 unified model (thinker + imagegen + talker)."""

from collections.abc import Iterable

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import (
    SupportsMRoPE,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.utils import (
    init_vllm_registered_model,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.custom_process_mixin import CustomProcessMixin
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.models.utils import add_prefix_to_loaded_weights
from vllm_omni.transformers_utils.configs.ming_flash_omni import BailingMM2Config, MingFlashOmniConfig

from .ming_flash_omni_thinker import (
    MingFlashOmniThinkerDummyInputsBuilder,
    MingFlashOmniThinkerMultiModalProcessor,
    MingFlashOmniThinkerProcessingInfo,
)

logger = init_logger(__name__)


@MULTIMODAL_REGISTRY.register_processor(
    MingFlashOmniThinkerMultiModalProcessor,
    info=MingFlashOmniThinkerProcessingInfo,
    dummy_inputs=MingFlashOmniThinkerDummyInputsBuilder,
)
class MingFlashOmniForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
    SupportsMRoPE,
    CustomProcessMixin,
):
    """Unified Ming-flash-omni-2.0 model combining thinker, imagegen, and talker."""

    supports_multimodal = True
    requires_raw_input_tokens: bool = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False

        config = vllm_config.model_config.hf_config

        self.vllm_config = vllm_config
        self.config = config

        if isinstance(config, MingFlashOmniConfig):
            thinker_config = config.thinker_config
        else:
            thinker_config = config

        self.thinker_config: BailingMM2Config = thinker_config
        self.model_stage = vllm_config.model_config.model_stage

        if self.model_stage == "thinker":
            thinker_vllm_config = vllm_config.with_hf_config(
                thinker_config, architectures=["MingFlashOmniThinkerForConditionalGeneration"]
            )
            self.thinker = init_vllm_registered_model(
                vllm_config=thinker_vllm_config,
                prefix=maybe_prefix(prefix, "thinker"),
                architectures=["MingFlashOmniThinkerForConditionalGeneration"],
            )
            self.model = self.thinker
            self.imagegen = None
            self.talker = None

        elif self.model_stage == "imagegen":
            from .ming_flash_omni_imagegen import MingFlashOmniImageGenModel

            image_gen_cfg = getattr(config, "image_gen_config", None)
            dit_type = getattr(image_gen_cfg, "dit_type", "?") if image_gen_cfg else "?"
            logger.info(
                "[MingFlashOmni] building imagegen stage (dit_type=%s, outer_config=%s)",
                dit_type,
                type(config).__name__,
            )
            self.thinker = None
            self.imagegen = MingFlashOmniImageGenModel(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "imagegen"),
            )
            self.model = self.imagegen
            self.talker = None

        elif self.model_stage == "unified":
            # Single-process Phase 2 mode: thinker (TP=2) + imagegen colocated.
            # Imagegen lives on rank 0 only — TP rank > 0 has it set to None to
            # avoid loading the ~15GB DiT/VAE/connector twice.
            from .ming_flash_omni_imagegen import MingFlashOmniImageGenModel

            logger.info("[MingFlashOmni] building UNIFIED stage (thinker + imagegen)")

            thinker_vllm_config = vllm_config.with_hf_config(
                thinker_config, architectures=["MingFlashOmniThinkerForConditionalGeneration"]
            )
            self.thinker = init_vllm_registered_model(
                vllm_config=thinker_vllm_config,
                prefix=maybe_prefix(prefix, "thinker"),
                architectures=["MingFlashOmniThinkerForConditionalGeneration"],
            )
            self.model = self.thinker
            self.talker = None

            tp_rank = self._resolve_tp_rank()
            if tp_rank == 0:
                logger.info("[MingFlashOmni] tp_rank=0 — constructing imagegen on this worker")
                self.imagegen = MingFlashOmniImageGenModel(
                    vllm_config=vllm_config,
                    prefix=maybe_prefix(prefix, "imagegen"),
                )
            else:
                logger.info(
                    "[MingFlashOmni] tp_rank=%d — skipping imagegen construction (only rank 0 owns it in unified mode)",
                    tp_rank,
                )
                self.imagegen = None

        elif self.model_stage == "talker":
            # TODO: Implement talker (TTS) stage
            raise NotImplementedError(
                "Talker (TTS) stage is not yet implemented. Please use model_stage='thinker' for now."
            )

        else:
            raise ValueError(
                f"Invalid model_stage: {self.model_stage}. Must be one of: 'thinker', 'imagegen', 'talker', 'unified'."
            )

        # Set up intermediate tensors
        self.make_empty_intermediate_tensors = (
            self.thinker.make_empty_intermediate_tensors if self.model_stage in ("thinker", "unified") else lambda: None
        )

    @staticmethod
    def _resolve_tp_rank() -> int:
        """Return the current tensor-parallel rank, or 0 if dist not ready."""
        try:
            from vllm.distributed import get_tensor_model_parallel_rank

            return int(get_tensor_model_parallel_rank())
        except Exception:
            return 0

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> OmniOutput:
        # Dual-stage imagegen path: the stage_input_processor (``thinker2imagegen``)
        # packages the thinker-exported hidden states into
        # ``additional_information[HIDDEN_STATES_PAYLOAD_KEY]`` which vllm-omni
        # surfaces as kwargs["runtime_additional_information"] — a list of
        # dicts, one per request. Bypass the standard AR forward entirely.
        if self.model_stage == "imagegen":
            from vllm_omni.model_executor.stage_input_processors.ming_flash_omni import (
                HIDDEN_STATES_PAYLOAD_KEY,
            )

            # Build a well-shaped dummy hidden_states tensor so vllm's
            # warmup path (compute_logits + _dummy_sampler_run) can
            # index it without triggering CUDA out-of-bounds asserts.
            # Expected shape: [num_tokens, thinker_hidden_size] — we don't
            # know thinker_hidden_size here, but 4096 is Ming's value.
            num_tokens = int(input_ids.numel()) if input_ids is not None else 1
            dummy_hidden_size = 4096
            dummy_hidden_states = torch.zeros(
                (num_tokens, dummy_hidden_size),
                device=input_ids.device if input_ids is not None else "cuda",
                dtype=torch.float32,
            )

            info_dicts = kwargs.get("runtime_additional_information") or []
            if not info_dicts:
                logger.warning(
                    "[MingFlashOmni.imagegen] no runtime_additional_information "
                    "in forward kwargs — returning dummy output (likely "
                    "profile_run / warmup). Wire the stage input processor in "
                    "the YAML for real requests."
                )
                return OmniOutput(
                    text_hidden_states=dummy_hidden_states,
                    multimodal_outputs={"images": []},
                )

            images: list = []
            for req_idx, info in enumerate(info_dicts):
                hidden = info.get(HIDDEN_STATES_PAYLOAD_KEY)
                if hidden is None:
                    logger.warning(
                        "[MingFlashOmni.imagegen] request %d missing %r; skipping",
                        req_idx,
                        HIDDEN_STATES_PAYLOAD_KEY,
                    )
                    continue
                if hidden.dim() == 2:
                    hidden = hidden.unsqueeze(0)  # [N, H] -> [1, N, H]
                # The cross-stage payload went through CPU serialization
                # (numpy.tobytes) so ``hidden`` is a CPU tensor now. Move it
                # to the imagegen model's device and dtype before forward.
                target_device = next(self.imagegen.parameters()).device
                target_dtype = next(self.imagegen.parameters()).dtype
                hidden = hidden.to(device=target_device, dtype=target_dtype)
                f = hidden.detach().float()
                logger.info(
                    "[MingFlashOmni.imagegen] req %d: hidden_states=%s on %s (%s) mean=%+.4f std=%.4f |x|/tok=%.3f",
                    req_idx,
                    tuple(hidden.shape),
                    target_device,
                    target_dtype,
                    f.mean().item(),
                    f.std().item(),
                    f.norm(dim=-1).mean().item(),
                )
                try:
                    gen_out = self.imagegen.forward(hidden)
                    gen_mm = gen_out.multimodal_outputs or {}
                    # MingFlashOmniImageGenModel.forward places a single
                    # [C, H, W] tensor under ``multimodal_outputs["image"]``;
                    # vllm-omni's serving_chat renders it into base64 PNG.
                    img = gen_mm.get("image")
                    if img is not None:
                        images.append(img)
                except Exception:
                    logger.exception(
                        "[MingFlashOmni.imagegen] req %d imagegen forward failed",
                        req_idx,
                    )

            logger.info(
                "[MingFlashOmni.imagegen] produced %d image tensor(s) total",
                len(images),
            )
            # vllm-omni's AR model runner (gpu_ar_model_runner.py:754) copies
            # multimodal_outputs values into the per-request payload. It
            # filters tensors whose ``shape[0] != num_tokens`` (for a whole
            # image ``[C, H, W]`` that check fails), but it **does** surface
            # entries whose value is a ``list`` by taking ``v[idx]`` per
            # request. So we store the image under ``"image"`` as a list of
            # length ``num_reqs`` where each element is a ``[C, H, W]`` tensor.
            # Serving_chat.py:2009 then reads a single tensor out of the
            # resulting ``completion_output.multimodal_output["image"]`` and
            # encodes it to a base64 PNG.
            mm_out: dict[str, list[torch.Tensor]] = {}
            if images:
                mm_out["image"] = images  # list[[C,H,W]] per request
            return OmniOutput(
                text_hidden_states=dummy_hidden_states,
                multimodal_outputs=mm_out,
            )

        output = self.model.forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        # Unified-mode dispatch: if the thinker exported image-gen hidden
        # states and we own the imagegen sub-model on this worker, run it
        # immediately and attach the resulting PIL image(s) into the output.
        if self.model_stage == "unified" and self.imagegen is not None and isinstance(output, OmniOutput):
            mm = output.multimodal_outputs or {}
            gen_hidden = mm.get("ming_imagegen_hidden_states")
            if gen_hidden is not None:
                logger.info(
                    "[MingFlashOmni.unified] dispatching to imagegen with hidden_states=%s",
                    tuple(gen_hidden.shape),
                )
                if gen_hidden.dim() == 2:
                    gen_hidden = gen_hidden.unsqueeze(0)  # [N, H] -> [1, N, H]
                # Read sampling knobs from extra_args if provided.
                ig_kwargs = {}
                extra = kwargs.get("sampling_params") or {}
                if isinstance(extra, dict):
                    ig_extra = extra.get("extra_args", {}).get("image_gen", {})
                    for k in ("steps", "cfg", "seed", "height", "width"):
                        if k in ig_extra:
                            ig_kwargs[
                                {
                                    "steps": "num_inference_steps",
                                    "cfg": "guidance_scale",
                                }.get(k, k)
                            ] = ig_extra[k]
                try:
                    gen_out = self.imagegen.forward(gen_hidden, **ig_kwargs)
                    gen_mm = gen_out.multimodal_outputs or {}
                    images = gen_mm.get("images") or []
                    new_mm = dict(mm)
                    new_mm["images"] = images
                    output = output._replace(multimodal_outputs=new_mm)
                    logger.info(
                        "[MingFlashOmni.unified] imagegen produced %d image(s)",
                        len(images),
                    )
                except Exception:
                    logger.exception("[MingFlashOmni.unified] imagegen forward failed; returning thinker-only output")

        return output

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata=None,
    ) -> torch.Tensor | None:
        # Imagegen stage runs on an AR worker (stage_type: llm) so vllm's
        # warmup / dummy_sampler_run path still calls compute_logits + sample.
        # We have no vocab-space logits to produce — return a dummy tensor
        # with a realistic vocab dimension so vllm's top-k / top-p sampler
        # ops can gather without triggering CUDA index asserts. The result
        # is never consumed because imagegen outputs PIL images via
        # multimodal_outputs, not token ids.
        if self.model_stage == "imagegen":
            num_reqs = hidden_states.shape[0] if hidden_states is not None else 1
            dummy_vocab = 128  # big enough for any default top_k / top_p
            return torch.zeros(
                (num_reqs, dummy_vocab),
                device=hidden_states.device if hidden_states is not None else "cuda",
                dtype=torch.float32,
            )
        if hasattr(self.model, "compute_logits"):
            return self.model.compute_logits(hidden_states, sampling_metadata)
        return None

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata,
    ):
        # Same reasoning as compute_logits — imagegen has nothing to sample.
        # Return zeros; vllm's output processor will ignore them because
        # max_tokens=1 in the imagegen stage config.
        if self.model_stage == "imagegen":
            try:
                from vllm.model_executor.layers.sampler import SamplerOutput
            except ImportError:
                SamplerOutput = None  # type: ignore
            num_reqs = logits.shape[0] if logits is not None else 1
            dummy_ids = torch.zeros((num_reqs, 1), dtype=torch.long, device=logits.device)
            if SamplerOutput is not None:
                return SamplerOutput(outputs=[], sampled_token_ids=dummy_ids)
            return dummy_ids
        if hasattr(self.model, "sample"):
            return self.model.sample(logits, sampling_metadata)
        raise NotImplementedError("sample method not available on current stage")

    def get_mrope_input_positions(self, *args, **kwargs):
        if hasattr(self.model, "get_mrope_input_positions"):
            return self.model.get_mrope_input_positions(*args, **kwargs)
        raise NotImplementedError("get_mrope_input_positions not available on current stage")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded_weights = set()
        thinker_weights = []
        imagegen_weights = []
        talker_weights = []

        for name, value in weights:
            if name.startswith("thinker."):
                thinker_weights.append((name, value))
            elif name.startswith("imagegen."):
                imagegen_weights.append((name, value))
            elif name.startswith("talker."):
                talker_weights.append((name, value))
            else:
                # Weights without prefix go to thinker by default
                thinker_weights.append((name, value))

        if self.model_stage in ("thinker", "unified") and thinker_weights:
            # Remove "thinker." prefix before loading
            thinker_weights_stripped = [
                (name.replace("thinker.", "", 1) if name.startswith("thinker.") else name, value)
                for name, value in thinker_weights
            ]
            thinker_loaded = self.thinker.load_weights(thinker_weights_stripped)
            thinker_loaded = add_prefix_to_loaded_weights(thinker_loaded, "thinker")
            loaded_weights.update(thinker_loaded)

            # ``query_tokens_dict.*`` parameters are pre-loaded inside
            # ``MingFlashOmniThinker.__init__`` from ``<model>/mlp/model.safetensors``
            # (not the main shard index), so vllm's default loader does not
            # see them via the standard ``load_weights`` path. Report them
            # as loaded here to satisfy the post-load completeness check.
            query_tokens_module = getattr(self.thinker, "query_tokens_dict", None)
            if query_tokens_module is not None:
                for scale_name, _param in query_tokens_module.items():
                    loaded_weights.add(f"thinker.query_tokens_dict.{scale_name}")

        if self.model_stage == "unified" and self.imagegen is not None:
            # imagegen's sub-components (ZImagePipeline, condition_encoder)
            # load their own weights from checkpoint subfolders at __init__
            # time — none of them flow through this top-level load_weights.
            # We still need to tell vllm they are "loaded" so the completeness
            # check does not fail.
            for param_name, _ in self.imagegen.named_parameters():
                loaded_weights.add(f"imagegen.{param_name}")

        if self.model_stage == "imagegen":
            # The imagegen stage loads transformer/vae/connector/mlp weights
            # lazily from their own subfolders inside MingFlashOmniImageGenModel
            # (see ming_flash_omni_imagegen.py). Top-level thinker tensors are
            # not consumed on this stage; we pass whatever we got through so
            # load_weights() can log and drop them.
            all_incoming = thinker_weights + imagegen_weights
            if all_incoming:
                imagegen_loaded = self.imagegen.load_weights(all_incoming)
                loaded_weights.update(imagegen_loaded)

            # Same reasoning as the unified branch above: imagegen sub-modules
            # (ZImagePipeline transformer/vae, condition_encoder) are populated
            # inside their own __init__ from_pretrained calls and never flow
            # through top-level load_weights. Report them as loaded here so
            # vllm's default loader's completeness check passes.
            if self.imagegen is not None:
                for param_name, _ in self.imagegen.named_parameters():
                    loaded_weights.add(f"imagegen.{param_name}")

        # TODO: Load talker weights when implemented

        return loaded_weights

    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(
            language_model="thinker.language_model",
            connector=["thinker.linear_proj.", "thinker.linear_proj_audio."],
            tower_model=["thinker.vision.", "thinker.audio."],
        )

    @property
    def sampler(self):
        if hasattr(self.model, "sampler"):
            return self.model.sampler
        return None

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        *,
        is_multimodal=None,
    ) -> torch.Tensor:
        # Imagegen stage: MingFlashOmniImageGenModel has no word embeddings.
        # vllm's AR runner still calls embed_input_ids during _preprocess, so
        # return a shape-correct dummy. The actual hidden states arrive via
        # ``runtime_additional_information`` and are consumed in forward.
        if self.model_stage == "imagegen":
            num_tokens = int(input_ids.numel()) if input_ids is not None else 1
            return torch.zeros(
                (num_tokens, 4096),
                device=input_ids.device if input_ids is not None else "cuda",
                dtype=torch.float32,
            )
        return self.model.embed_input_ids(
            input_ids,
            multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    def embed_multimodal(self, **kwargs):
        if self.model_stage == "imagegen":
            return []
        return self.model.embed_multimodal(**kwargs)
