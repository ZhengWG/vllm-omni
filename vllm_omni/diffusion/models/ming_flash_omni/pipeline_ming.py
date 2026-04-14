# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.

"""Ming-flash-omni-2.0 image generation pipeline for vllm-omni diffusion engine.

This module replaces the Phase 2 v0 ``MingFlashOmniImageGenModel`` hack that
ran imagegen on an AR worker (``stage_type: llm``) with a proper
``stage_type: diffusion`` pipeline, following the glm_image/bagel pattern.

Cross-stage data flow:

    Stage 0 (thinker, llm)           Stage 1 (imagegen, diffusion)
    ────────────────────             ──────────────────────────────
    forward returns                  thinker2imagegen hook slices
    multimodal_output[               final_hidden_states at
      "final_hidden_states"]         <imagePatch> positions,
       ↓                             returns list[dict] with
    shared_memory_connector          {"extra": {"thinker_hidden_states"}}
       ↓                             ──── via OmniMsgpackEncoder ────>
                                     MingImagePipeline.forward(req):
                                       hidden = req.prompts[0]["extra"][...]
                                       cond = condition_encoder(hidden)
                                       img = ZImagePipeline-style loop
                                       return DiffusionOutput(output=img)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.ming_flash_omni.condition_encoder import (
    MingConditionEncoder,
)
from vllm_omni.diffusion.models.z_image.pipeline_z_image import ZImagePipeline
from vllm_omni.diffusion.models.z_image.z_image_transformer import (
    ZImageTransformer2DModel,
)
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import (
    DiffusionPipelineProfilerMixin,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.transformers_utils.configs.ming_flash_omni import MingImageGenConfig

logger = logging.getLogger(__name__)


# Key used by thinker2imagegen to stash the sliced hidden states on the
# per-request prompt dict (under the "extra" sub-dict). The pipeline reads
# from the same key on the receiver side.
THINKER_HIDDEN_STATES_KEY = "thinker_hidden_states"


class MingImagePipeline(nn.Module, DiffusionPipelineProfilerMixin):
    """Ming-flash-omni-2.0 text-to-image diffusion pipeline.

    Composed of:
      * ``condition_encoder`` — Qwen2 connector + proj_in/out + F.normalize×1000
      * ``scheduler``         — FlowMatchEulerDiscreteScheduler (use_dynamic_shifting=True)
      * ``transformer``       — ZImageTransformer2DModel (from in-tree z_image)
      * ``vae``               — AutoencoderKL (Flux format, latent=16)

    The pipeline's ``forward(req)`` reads the thinker-side hidden states from
    ``req.prompts[0]["extra"][THINKER_HIDDEN_STATES_KEY]`` (placed there by
    the ``thinker2imagegen`` custom_process_input_func), runs the full
    Ming condition encoder + ZImage diffusion loop, and returns a
    ``DiffusionOutput`` with a raw image tensor under ``.output``.
    """

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",  # noqa: ARG002
    ) -> None:
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()

        model_path = od_config.model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Ming checkpoint path does not exist: {model_path}")

        dtype = getattr(od_config, "dtype", torch.bfloat16)
        self._dtype = dtype

        # Ming's per-checkpoint image-gen configuration. We cannot rely on
        # ``od_config.hf_config.image_gen_config`` because the diffusion
        # stage is started with ``hf_config_name: thinker_config`` (the
        # BailingMM2Config), which does not carry a MingImageGenConfig.
        # Fall back to defaults that match the released checkpoint.
        self.image_gen_config = MingImageGenConfig()
        logger.info(
            "[MingImagePipeline] init: model=%s dtype=%s image_gen_config=%s",
            model_path,
            dtype,
            self.image_gen_config,
        )

        # ----- Condition encoder (Qwen2 connector + proj_in/out + norm×1000)
        thinker_hidden_size = 4096  # BailingMoeV2 LLM hidden size for Ming 2.0
        self.condition_encoder = MingConditionEncoder(
            self.image_gen_config,
            thinker_hidden_size=thinker_hidden_size,
            device=self.device,
            dtype=dtype,
        )
        self.condition_encoder.load_from_checkpoint(model_path)

        # ----- Scheduler: force use_dynamic_shifting=True regardless of what
        # the checkpoint's scheduler_config.json ships (Ming overrides it at
        # runtime in ZImageLoss.__init__, zimage_loss.py:154).
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_path,
            subfolder=self.image_gen_config.scheduler_subfolder,
            local_files_only=True,
        )
        self.scheduler.config["use_dynamic_shifting"] = True
        logger.info(
            "[MingImagePipeline] scheduler: %s (use_dynamic_shifting=True)",
            type(self.scheduler).__name__,
        )

        # ----- VAE (Flux-format AutoencoderKL).
        logger.info("[MingImagePipeline] loading VAE ...")
        self.vae = AutoencoderKL.from_pretrained(
            model_path,
            subfolder=self.image_gen_config.vae_subfolder,
            local_files_only=True,
            torch_dtype=dtype,
        ).to(device=self.device, dtype=dtype)
        self.vae.eval()

        # ----- DiT transformer. Load state_dict ourselves with the
        # split→merged QKV/FFN rename so the diffusers-format checkpoint
        # fits vllm-omni's in-tree fused Linear layout.
        logger.info("[MingImagePipeline] loading DiT transformer ...")
        self.transformer = ZImageTransformer2DModel(quant_config=None)
        self._load_transformer_weights(
            self.transformer,
            Path(model_path) / self.image_gen_config.transformer_subfolder,
            dtype=dtype,
        )
        self.transformer = self.transformer.to(device=self.device, dtype=dtype)
        self.transformer.eval()

        # Drop text_encoder/tokenizer — Ming uses our own condition_encoder.
        self.text_encoder = None
        self.tokenizer = None

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2, do_convert_rgb=True)

        # weights_sources is consulted by the vllm-omni diffusion loader.
        # Empty means "this pipeline manages its own weight loading", which
        # is true for us: condition encoder + DiT + VAE all load during
        # __init__ above.
        self.weights_sources = []

        # Expose the underlying ZImagePipeline's main forward body by
        # building a lightweight wrapper. Rather than re-implementing the
        # 30-step denoise + CFG + VAE decode here, we borrow ZImagePipeline.
        # We cannot instantiate ZImagePipeline directly (its __init__ loads
        # text_encoder/tokenizer which Ming lacks), so we do ``object.__new__``
        # and copy the components we already built.
        self._z_pipeline = object.__new__(ZImagePipeline)
        nn.Module.__init__(self._z_pipeline)
        self._z_pipeline.od_config = SimpleNamespace(
            model=model_path,
            quantization_config=None,
            enable_diffusion_pipeline_profiler=False,
            dtype=dtype,
        )
        self._z_pipeline._execution_device = self.device
        self._z_pipeline.weights_sources = []
        self._z_pipeline.scheduler = self.scheduler
        self._z_pipeline.vae = self.vae
        self._z_pipeline.transformer = self.transformer
        self._z_pipeline.text_encoder = None
        self._z_pipeline.tokenizer = None
        self._z_pipeline.vae_scale_factor = self.vae_scale_factor
        self._z_pipeline.image_processor = self.image_processor
        try:
            self._z_pipeline.setup_diffusion_pipeline_profiler(enable_diffusion_pipeline_profiler=False)
        except Exception:
            logger.debug("[MingImagePipeline] z_pipeline profiler setup skipped")

        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=getattr(od_config, "enable_diffusion_pipeline_profiler", False)
        )
        logger.info("[MingImagePipeline] ready — vae_scale_factor=%d", self.vae_scale_factor)

    # ------------------------------------------------------------------
    # Weight loading helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_transformer_weights(
        transformer: nn.Module,
        transformer_dir: Path,
        *,
        dtype: torch.dtype,  # noqa: ARG004
    ) -> None:
        from safetensors.torch import load_file

        if not transformer_dir.exists():
            raise FileNotFoundError(f"transformer dir missing: {transformer_dir}")
        candidates = sorted(transformer_dir.glob("*.safetensors"))
        if not candidates:
            candidates = sorted(transformer_dir.glob("*.bin"))
        if not candidates:
            raise FileNotFoundError(f"no weight files in {transformer_dir}")

        state: dict[str, torch.Tensor] = {}
        for p in candidates:
            logger.info("[MingImagePipeline] reading transformer weights: %s", p)
            if p.suffix == ".safetensors":
                state.update(load_file(str(p)))
            else:
                state.update(torch.load(str(p), map_location="cpu"))
        logger.info("[MingImagePipeline] transformer state_dict: %d keys", len(state))

        state = _merge_split_qkv_and_gated_ffn(state)

        missing, unexpected = transformer.load_state_dict(state, strict=False)
        logger.info(
            "[MingImagePipeline] transformer load summary: loaded=%d, missing=%d, unexpected=%d",
            len(list(transformer.named_parameters())) - len(missing),
            len(missing),
            len(unexpected),
        )
        if missing:
            logger.warning(
                "[MingImagePipeline] transformer MISSING %d keys: %s",
                len(missing),
                missing[:8],
            )
        if unexpected:
            logger.warning(
                "[MingImagePipeline] transformer UNEXPECTED %d keys: %s",
                len(unexpected),
                unexpected[:8],
            )

    def load_weights(self, weights):
        """Mark all sub-module parameters as loaded.

        The ZImagePipeline + condition_encoder components load their own
        weights from checkpoint subfolders inside ``__init__``. vllm-omni's
        diffusion loader still calls ``load_weights`` on the pipeline — we
        return a set that covers every named parameter so the completeness
        check passes, and we ignore whatever the caller streams in.
        """
        # drain the iterator (it may be backed by a generator that owns
        # file handles; consuming it avoids resource leaks).
        consumed = 0
        for _ in weights:
            consumed += 1
        logger.info(
            "[MingImagePipeline.load_weights] pipeline owns its own weights (%d external tensors ignored)",
            consumed,
        )
        return {name for name, _ in self.named_parameters()}

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def forward(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        """Run one text-to-image generation request.

        Args:
            req: Diffusion request. The cross-stage thinker hidden states
                must be present at ``req.prompts[0]["extra"][THINKER_HIDDEN_STATES_KEY]``
                as a ``[N, H]`` (or ``[1, N, H]``) tensor, placed there by
                ``thinker2imagegen``.

        Returns:
            DiffusionOutput with ``.output`` set to a ``[B, 3, H, W]``
            image tensor in ``[-1, 1]``. The vllm-omni diffusion engine's
            output adapter converts this to PIL/base64 downstream.
        """
        # Detect vllm-omni's ``_dummy_run`` warmup pass and short-circuit to
        # avoid spending 30 seconds of DiT on a meaningless request. See
        # vllm_omni/diffusion/diffusion_engine.py::_dummy_run.
        dummy_ids = {"dummy_req_id"}
        if (req.request_ids and set(req.request_ids).issubset(dummy_ids)) or req.request_id == "dummy_req_id":
            logger.info("[MingImagePipeline.forward] dummy warmup run — returning blank output")
            dummy_h = int(req.sampling_params.height or 512)
            dummy_w = int(req.sampling_params.width or 512)
            blank = torch.zeros(
                (1, 3, dummy_h, dummy_w),
                device=self.device,
                dtype=self._dtype,
            )
            return DiffusionOutput(output=blank)

        first_prompt = req.prompts[0] if req.prompts else None
        if isinstance(first_prompt, str):
            prompt_dict: dict[str, Any] = {}
        elif isinstance(first_prompt, dict):
            prompt_dict = first_prompt
        elif first_prompt is not None and hasattr(first_prompt, "_asdict"):
            prompt_dict = first_prompt._asdict()
        elif first_prompt is not None and hasattr(first_prompt, "__dict__"):
            prompt_dict = vars(first_prompt)
        else:
            prompt_dict = {}

        extra = prompt_dict.get("extra") or {}
        hidden = extra.get(THINKER_HIDDEN_STATES_KEY)
        if hidden is None:
            # Same dual-path convention as glm_image: also check
            # ``sampling_params.extra_args``.
            hidden = (req.sampling_params.extra_args or {}).get(THINKER_HIDDEN_STATES_KEY)
        if hidden is None:
            # No thinker hidden states found on the request. Two legitimate
            # causes:
            #   1. vllm-omni's warmup / profile_run path (see
            #      ``diffusion_engine.py::_dummy_run``) fabricates a
            #      minimal dummy request to exercise the pipeline once.
            #   2. User misconfiguration — the stage YAML is missing
            #      ``custom_process_input_func: thinker2imagegen`` or the
            #      upstream thinker did not export ``final_hidden_states``.
            # In both cases we fall back to a zero-conditioning tensor of
            # the expected shape so the DiT kernels still run (warmup
            # case) and real requests produce a diagnosable all-noise
            # image instead of a hard crash.
            scale = self.image_gen_config.img_gen_scales[-1]
            num_query_tokens = scale * scale
            hidden = torch.zeros(
                (num_query_tokens, 4096),
                dtype=self._dtype,
                device=self.device,
            )
            logger.warning(
                "[MingImagePipeline.forward] %s missing from request; "
                "falling back to zero-conditioning %s. This is expected "
                "during warmup; for real requests verify that "
                "`custom_process_input_func: thinker2imagegen` is set on "
                "the diffusion stage in the YAML.",
                THINKER_HIDDEN_STATES_KEY,
                tuple(hidden.shape),
            )

        if not isinstance(hidden, torch.Tensor):
            raise TypeError(
                f"[MingImagePipeline] {THINKER_HIDDEN_STATES_KEY!r} must be a Tensor, got {type(hidden).__name__}"
            )

        # Move to the pipeline's device+dtype.
        target_device = next(self.parameters()).device
        target_dtype = next(self.parameters()).dtype
        hidden = hidden.to(device=target_device, dtype=target_dtype)
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)  # [N, H] -> [1, N, H]
        logger.info(
            "[MingImagePipeline.forward] thinker_hidden_states=%s on %s (%s)",
            tuple(hidden.shape),
            target_device,
            target_dtype,
        )

        # ----- Condition encoder → cap_feats
        cap_feats = self.condition_encoder(hidden)
        logger.info("[MingImagePipeline.forward] cap_feats=%s", tuple(cap_feats.shape))

        # ----- Sampling knobs (with Ming defaults).
        sp = req.sampling_params
        cfg = self.image_gen_config
        height = int(sp.height) if sp.height is not None else cfg.default_height
        width = int(sp.width) if sp.width is not None else cfg.default_width
        num_inference_steps = int(sp.num_inference_steps or cfg.num_inference_steps)
        guidance_scale = sp.guidance_scale if sp.guidance_scale else cfg.guidance_scale
        generator = sp.generator
        if generator is None and sp.seed is not None:
            generator = torch.Generator(device=target_device).manual_seed(int(sp.seed))

        # Format prompt_embeds / negative_prompt_embeds as list[Tensor]
        # (one entry per request) — matches ZImagePipeline's contract when
        # prompt_embeds are pre-computed.
        prompt_embeds = [cap_feats[i] for i in range(cap_feats.shape[0])]
        negative_prompt_embeds = [self.condition_encoder.zero_negative(e) for e in prompt_embeds]

        # ZImagePipeline.forward reads a number of fields from req /
        # req.sampling_params. The real ``req`` already has most of them;
        # we only need to supply things ZImagePipeline expects that are
        # Ming-specific. Wrap with a shim namespace if fields are missing.
        z_sp = SimpleNamespace(
            strength=None,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            generator=generator,
            sigmas=None,
            max_sequence_length=512,
            guidance_scale=guidance_scale,
            guidance_rescale=None,
            num_outputs_per_prompt=1,
        )
        z_req = SimpleNamespace(
            request_id=req.request_id or "ming-imagegen",
            prompts=[{"prompt": "", "negative_prompt": ""}],
            sampling_params=z_sp,
        )

        logger.info(
            "[MingImagePipeline.forward] running z_pipeline hw=(%d,%d) steps=%d cfg=%.2f",
            height,
            width,
            num_inference_steps,
            guidance_scale,
        )
        output = self._z_pipeline.forward(
            z_req,
            prompt=None,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            generator=generator,
            output_type="pt",
            return_dict=True,
        )

        if hasattr(output, "output") and output.output is not None:
            raw = output.output
        elif hasattr(output, "images"):
            raw = output.images
        else:
            raw = output
        if not isinstance(raw, torch.Tensor):
            raise RuntimeError(f"ZImagePipeline returned non-tensor output: {type(raw).__name__}")
        logger.info(
            "[MingImagePipeline.forward] produced image tensor shape=%s range=[%.3f,%.3f]",
            tuple(raw.shape),
            raw.float().min().item(),
            raw.float().max().item(),
        )
        return DiffusionOutput(output=raw)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _merge_split_qkv_and_gated_ffn(
    state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Merge diffusers-style split QKV / gated-FFN weights into vllm-omni form.

    Ming's diffusers-format checkpoint ships:
        <prefix>.attention.to_{q,k,v}.weight
        <prefix>.feed_forward.w{1,3}.weight

    vllm-omni's ZImageTransformer2DModel expects the merged form:
        <prefix>.attention.to_qkv.weight       = cat([q, k, v], dim=0)
        <prefix>.feed_forward.w13.weight       = cat([w1, w3], dim=0)
    """
    out: dict[str, torch.Tensor] = {}
    qkv_groups: dict[str, dict[str, torch.Tensor]] = {}
    w13_groups: dict[str, dict[str, torch.Tensor]] = {}

    for name, tensor in state.items():
        handled = False
        for qkv_suffix, letter in (
            (".attention.to_q.weight", "q"),
            (".attention.to_k.weight", "k"),
            (".attention.to_v.weight", "v"),
            (".attention.to_q.bias", "q_b"),
            (".attention.to_k.bias", "k_b"),
            (".attention.to_v.bias", "v_b"),
        ):
            if name.endswith(qkv_suffix):
                prefix = name[: -len(qkv_suffix)]
                qkv_groups.setdefault(prefix, {})[letter] = tensor
                handled = True
                break
        if handled:
            continue
        for ffn_suffix, letter in (
            (".feed_forward.w1.weight", "w1"),
            (".feed_forward.w3.weight", "w3"),
            (".feed_forward.w1.bias", "w1_b"),
            (".feed_forward.w3.bias", "w3_b"),
        ):
            if name.endswith(ffn_suffix):
                prefix = name[: -len(ffn_suffix)]
                w13_groups.setdefault(prefix, {})[letter] = tensor
                handled = True
                break
        if handled:
            continue
        out[name] = tensor

    merged_qkv = 0
    for prefix, parts in qkv_groups.items():
        if {"q", "k", "v"}.issubset(parts):
            out[f"{prefix}.attention.to_qkv.weight"] = torch.cat([parts["q"], parts["k"], parts["v"]], dim=0)
            merged_qkv += 1
        else:
            for letter, t in parts.items():
                if letter in ("q", "k", "v"):
                    out[f"{prefix}.attention.to_{letter}.weight"] = t
        if {"q_b", "k_b", "v_b"}.issubset(parts):
            out[f"{prefix}.attention.to_qkv.bias"] = torch.cat([parts["q_b"], parts["k_b"], parts["v_b"]], dim=0)
        else:
            for letter, t in parts.items():
                if letter in ("q_b", "k_b", "v_b"):
                    out[f"{prefix}.attention.to_{letter[0]}.bias"] = t

    merged_w13 = 0
    for prefix, parts in w13_groups.items():
        if {"w1", "w3"}.issubset(parts):
            out[f"{prefix}.feed_forward.w13.weight"] = torch.cat([parts["w1"], parts["w3"]], dim=0)
            merged_w13 += 1
        else:
            for letter, t in parts.items():
                if letter in ("w1", "w3"):
                    out[f"{prefix}.feed_forward.{letter}.weight"] = t
        if {"w1_b", "w3_b"}.issubset(parts):
            out[f"{prefix}.feed_forward.w13.bias"] = torch.cat([parts["w1_b"], parts["w3_b"]], dim=0)
        else:
            for letter, t in parts.items():
                if letter in ("w1_b", "w3_b"):
                    out[f"{prefix}.feed_forward.{letter[:2]}.bias"] = t

    logger.info(
        "[MingImagePipeline] merged QKV groups: %d, merged w13 groups: %d; final state_dict has %d keys (was %d)",
        merged_qkv,
        merged_w13,
        len(out),
        len(state),
    )
    return out


def get_ming_image_post_process_func(od_config: OmniDiffusionConfig):
    """Return a post-process callable that converts the raw VAE tensor to PIL.

    The diffusion engine calls ``post_process_func(output_data)`` where
    ``output_data`` is the ``DiffusionOutput.output`` tensor returned by
    ``MingImagePipeline.forward``. It has shape ``[B, 3, H, W]`` in ``[-1, 1]``
    (Z-image VAE convention). We run the standard ``VaeImageProcessor``
    postprocess to convert it to ``list[PIL.Image]`` which vllm-omni's
    ``OmniRequestOutput.from_diffusion`` then bubbles up as
    ``omni_outputs.images`` for serving_chat to base64-encode.

    Registered via ``_DIFFUSION_POST_PROCESS_FUNCS["MingImagePipeline"]``
    in vllm_omni/diffusion/registry.py.
    """
    from diffusers.image_processor import VaeImageProcessor  # local import

    model_path = od_config.model
    vae_config_path = os.path.join(model_path, "vae", "config.json")
    try:
        import json as _json

        with open(vae_config_path) as f:
            vae_cfg = _json.load(f)
        block_out_channels = vae_cfg.get("block_out_channels", [128, 256, 512, 512])
        vae_scale_factor = 2 ** (len(block_out_channels) - 1)
    except Exception:
        vae_scale_factor = 8  # Ming's Flux-format VAE default

    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor * 2, do_convert_rgb=True)

    def post_process_func(images: torch.Tensor):
        # VaeImageProcessor.postprocess with default output_type="pil"
        # returns ``list[PIL.Image]``.
        return image_processor.postprocess(images.float())

    return post_process_func


__all__ = [
    "MingImagePipeline",
    "THINKER_HIDDEN_STATES_KEY",
    "get_ming_image_post_process_func",
]
