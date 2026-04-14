# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.

"""Ming-flash-omni-2.0 imagegen stage model.

This module is loaded when ``model_stage == "imagegen"`` in the stage config.
It owns the condition encoder (Qwen2 connector + norm/proj) and the ZImage
diffusion pipeline (transformer + VAE + scheduler).

Inputs:  ``thinker_hidden_states`` [B, N, 4096] + optional attention mask.
Outputs: PIL image list (wrapped in OmniOutput).

NOTE (Phase 1):
  - Weights for the DiT transformer + VAE are loaded by ZImagePipeline via
    diffusers' ``from_pretrained`` path (already done there).
  - Weights for the Qwen2 connector + mlp/ proj-norm are loaded by
    ``MingConditionEncoder.load_from_checkpoint`` using HF transformers.
  - Cross-stage hidden-state transfer is done by the stage input processor
    (see stage_input_processors/ming_flash_omni.py) and arrives on this
    module's forward() as a plain tensor.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import torch
import torch.nn as nn
from vllm.config import VllmConfig

from vllm_omni.diffusion.models.ming_flash_omni.condition_encoder import (
    MingConditionEncoder,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.transformers_utils.configs.ming_flash_omni import (
    MingFlashOmniConfig,
    MingImageGenConfig,
)

logger = logging.getLogger(__name__)


class MingFlashOmniImageGenModel(nn.Module):
    """Imagegen stage of Ming-flash-omni-2.0.

    Composed of:
      * ``condition_encoder`` — Qwen2 connector + norm/proj
      * ``pipeline``          — ``ZImagePipeline`` (DiT + VAE + scheduler)
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):  # noqa: ARG002
        super().__init__()
        self.vllm_config = vllm_config
        self.model_path = vllm_config.model_config.model

        hf_config = vllm_config.model_config.hf_config
        # Three possible shapes for hf_config depending on how this module is
        # instantiated:
        #   (1) MingFlashOmniConfig — outer wrapper; read .image_gen_config
        #       and .thinker_config.llm_config.hidden_size directly.
        #   (2) MingImageGenConfig — caller already passed the image-gen
        #       config directly (unusual but supported).
        #   (3) BailingMM2Config / other — happens in unified mode where the
        #       stage engine uses ``hf_config_name: thinker_config`` and hands
        #       us the thinker config instead of the outer wrapper. Build a
        #       default MingImageGenConfig and fish hidden_size out of the
        #       thinker's llm_config.
        if isinstance(hf_config, MingFlashOmniConfig):
            self.image_gen_config: MingImageGenConfig = hf_config.image_gen_config
            thinker_llm = getattr(hf_config.thinker_config, "llm_config", None)
        elif isinstance(hf_config, MingImageGenConfig):
            self.image_gen_config = hf_config
            thinker_llm = None
        else:
            logger.info(
                "[MingImageGen] hf_config is %s (not MingFlashOmniConfig) — "
                "using MingImageGenConfig() defaults; checkpoint subfolders "
                "(mlp/, transformer/, vae/, scheduler/, connector/) still "
                "read from vllm_config.model_config.model",
                type(hf_config).__name__,
            )
            self.image_gen_config = MingImageGenConfig()
            # hf_config here is the thinker config (BailingMM2Config).
            thinker_llm = getattr(hf_config, "llm_config", None)

        if thinker_llm is not None and getattr(thinker_llm, "hidden_size", None):
            thinker_hidden_size = int(thinker_llm.hidden_size)
        else:
            logger.warning(
                "[MingImageGen] thinker llm_config.hidden_size missing; "
                "falling back to 4096 (BailingMoeV2 default for Ming-flash-omni-2.0)"
            )
            thinker_hidden_size = 4096
        logger.info(
            "[MingImageGen] init: thinker_hidden_size=%d, image_gen_config=%s",
            thinker_hidden_size,
            self.image_gen_config,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = vllm_config.model_config.dtype

        self.condition_encoder = MingConditionEncoder(
            self.image_gen_config,
            thinker_hidden_size=thinker_hidden_size,
            device=device,
            dtype=dtype,
        )

        # Lazily construct the ZImage pipeline so we can fail fast with a
        # clear error if the DiT type is not supported in Phase 1.
        self.pipeline = self._build_diffusion_pipeline(device=device, dtype=dtype)

        # Connector weights live in the unified Ming checkpoint, not inside
        # ZImagePipeline's subfolders. Load them here.
        try:
            self.condition_encoder.load_from_checkpoint(self.model_path)
        except Exception:
            logger.exception(
                "[MingImageGen] condition_encoder.load_from_checkpoint failed — "
                "continuing with uninitialized connector; first real run will "
                "likely produce noise. Fix on real hardware."
            )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _build_diffusion_pipeline(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ):
        dit_type = self.image_gen_config.dit_type.lower()
        logger.info("[MingImageGen] building diffusion pipeline for dit_type=%s", dit_type)

        if dit_type == "zimage":
            return self._build_zimage_pipeline_manually(device=device, dtype=dtype)

        raise NotImplementedError(
            f"dit_type={dit_type!r} is not supported in Phase 1 (only 'zimage' wired so far; sd3/sana come later)."
        )

    def _build_zimage_pipeline_manually(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """Construct a ZImagePipeline without going through its ``__init__``.

        ZImagePipeline's ``__init__`` assumes a z-image-style checkpoint
        layout with ``text_encoder/`` and ``tokenizer/`` subfolders. Ming's
        unified checkpoint does not have those (text conditioning comes from
        our ``MingConditionEncoder`` instead). We therefore bypass the base
        ``__init__`` with ``object.__new__`` and assign only the components
        Ming actually uses:

            * ``scheduler``      from Ming/scheduler/   (FlowMatchEulerDiscrete)
            * ``transformer``    from Ming/transformer/ (ZImageTransformer2DModel)
            * ``vae``            from Ming/vae/         (AutoencoderKL)
            * ``image_processor`` built locally (diffusers VaeImageProcessor)
            * ``text_encoder`` / ``tokenizer`` left as ``None``

        The ``forward`` path on ZImagePipeline happily accepts pre-computed
        ``prompt_embeds`` and never touches ``text_encoder`` / ``tokenizer``
        in that case.
        """
        import os

        from diffusers.image_processor import VaeImageProcessor
        from diffusers.models import AutoencoderKL
        from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

        from vllm_omni.diffusion.models.z_image.pipeline_z_image import ZImagePipeline
        from vllm_omni.diffusion.models.z_image.z_image_transformer import (
            ZImageTransformer2DModel,
        )

        logger.info(
            "[MingImageGen] building ZImagePipeline manually from %s",
            self.model_path,
        )

        local_only = os.path.exists(self.model_path)

        pipeline = object.__new__(ZImagePipeline)
        nn.Module.__init__(pipeline)

        # od_config is consumed by DiffusionPipelineProfilerMixin during setup.
        from types import SimpleNamespace

        pipeline.od_config = SimpleNamespace(
            model=self.model_path,
            quantization_config=None,
            enable_diffusion_pipeline_profiler=False,
            dtype=dtype,
        )
        pipeline._execution_device = device
        pipeline.weights_sources = []

        logger.info("[MingImageGen] loading scheduler from %s/scheduler", self.model_path)
        pipeline.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.model_path,
            subfolder=self.image_gen_config.scheduler_subfolder,
            local_files_only=local_only,
        )
        # Ming overrides ``use_dynamic_shifting=True`` at runtime regardless
        # of what the checkpoint's scheduler_config.json ships (see
        # Ming/diffusion/zimage_loss.py line 154). Without this the
        # FlowMatchEuler sampler uses a uniform timestep schedule and the
        # denoised result is garbage.
        pipeline.scheduler.config["use_dynamic_shifting"] = True
        logger.info(
            "[MingImageGen] scheduler: %s, use_dynamic_shifting=True (forced)",
            type(pipeline.scheduler).__name__,
        )

        logger.info("[MingImageGen] loading VAE from %s/vae", self.model_path)
        pipeline.vae = AutoencoderKL.from_pretrained(
            self.model_path,
            subfolder=self.image_gen_config.vae_subfolder,
            local_files_only=local_only,
            torch_dtype=dtype,
        ).to(device=device, dtype=dtype)

        logger.info(
            "[MingImageGen] loading transformer from %s/transformer",
            self.model_path,
        )
        # ZImageTransformer2DModel is vllm-omni's in-tree implementation; it
        # uses diffusers-style from_pretrained. The class signature expects
        # a quant_config kwarg that we do not use in Phase 1.
        transformer = ZImageTransformer2DModel(quant_config=None)
        # Load weights from the checkpoint. We follow the from_pretrained
        # pattern by reading the safetensors in the transformer subfolder.
        self._load_transformer_weights(
            transformer,
            Path(self.model_path) / self.image_gen_config.transformer_subfolder,
            dtype=dtype,
        )
        pipeline.transformer = transformer.to(device=device, dtype=dtype)

        # No text encoder / tokenizer — Ming uses MingConditionEncoder.
        pipeline.text_encoder = None
        pipeline.tokenizer = None

        pipeline.vae_scale_factor = 2 ** (len(pipeline.vae.config.block_out_channels) - 1)
        pipeline.image_processor = VaeImageProcessor(
            vae_scale_factor=pipeline.vae_scale_factor * 2, do_convert_rgb=True
        )

        # Profiler mixin setup (no-op with enable=False).
        try:
            pipeline.setup_diffusion_pipeline_profiler(enable_diffusion_pipeline_profiler=False)
        except Exception:
            logger.debug("[MingImageGen] profiler setup skipped")

        logger.info(
            "[MingImageGen] ZImagePipeline built — vae_scale_factor=%d",
            pipeline.vae_scale_factor,
        )
        return pipeline

    @staticmethod
    def _load_transformer_weights(
        transformer: nn.Module,
        transformer_dir,
        *,
        dtype: torch.dtype,
    ) -> None:
        """Best-effort loader for the Ming transformer/ safetensors shards.

        Follows the ``diffusion_pytorch_model[.safetensors|.bin]`` naming
        convention used by diffusers. Logs which keys missed.
        """
        from pathlib import Path as _Path

        transformer_dir = _Path(transformer_dir)
        if not transformer_dir.exists():
            logger.error("[MingImageGen] transformer dir missing: %s", transformer_dir)
            return
        candidates = sorted(transformer_dir.glob("*.safetensors"))
        if not candidates:
            candidates = sorted(transformer_dir.glob("*.bin"))
        if not candidates:
            logger.error("[MingImageGen] no transformer weights in %s", transformer_dir)
            return

        state: dict[str, torch.Tensor] = {}
        for p in candidates:
            logger.info("[MingImageGen] reading transformer weights: %s", p)
            if p.suffix == ".safetensors":
                from safetensors.torch import load_file

                state.update(load_file(str(p)))
            else:
                state.update(torch.load(str(p), map_location="cpu"))
        logger.info("[MingImageGen] transformer state_dict: %d keys", len(state))

        # Ming's diffusers-format checkpoint ships attention QKV and the
        # gated FFN as three/two separate tensors (to_q/to_k/to_v,
        # feed_forward.w1/w3), but vllm-omni's in-tree ZImageTransformer2DModel
        # expects the MERGED form (to_qkv, feed_forward.w13) used by
        # vllm's ColumnParallelLinear pattern. Rewrite the state dict so
        # matching triples/pairs are concatenated along dim 0 before load.
        state = _merge_split_qkv_and_gated_ffn(state)

        # Pre-load stats: count total params in the module + list first few
        # parameter names — so we can compare with checkpoint key names.
        module_param_names = [name for name, _ in transformer.named_parameters()]
        logger.info(
            "[MingImageGen] transformer has %d parameters, first 8: %s",
            len(module_param_names),
            module_param_names[:8],
        )
        logger.info(
            "[MingImageGen] checkpoint has %d tensors, first 8: %s",
            len(state),
            list(state.keys())[:8],
        )

        missing, unexpected = transformer.load_state_dict(state, strict=False)
        loaded_count = len(module_param_names) - len(missing)
        logger.info(
            "[MingImageGen] transformer load summary: loaded=%d, missing=%d, unexpected=%d",
            loaded_count,
            len(missing),
            len(unexpected),
        )
        if missing:
            logger.warning(
                "[MingImageGen] transformer MISSING %d keys (first 20): %s",
                len(missing),
                missing[:20],
            )
        if unexpected:
            logger.warning(
                "[MingImageGen] transformer UNEXPECTED %d keys (first 20): %s",
                len(unexpected),
                unexpected[:20],
            )

        # Sanity probe: pick a Linear weight deep in the transformer and log
        # its std. A trained weight should have std ~0.01-0.05; a random
        # Kaiming/Xavier init with fan_in=3840 gives std ~1/sqrt(3840) ~0.016,
        # so also look at ABS max — trained weights usually have abs max
        # around 0.5-2.0, random init is typically <0.1.
        for probe_name, probe_param in transformer.named_parameters():
            if "weight" in probe_name and probe_param.dim() == 2 and probe_param.shape[0] > 256:
                p = probe_param.detach().float()
                logger.info(
                    "[MingImageGen] transformer probe %s shape=%s "
                    "mean=%+.4f std=%.4f |max|=%.3f — trained expect std~0.01-0.05, "
                    "|max| > 0.5; random init has std~%.4f |max| < 0.1",
                    probe_name,
                    tuple(probe_param.shape),
                    p.mean().item(),
                    p.std().item(),
                    p.abs().max().item(),
                    1.0 / (probe_param.shape[1] ** 0.5),
                )
                break

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def forward(
        self,
        thinker_hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        seed: int | None = None,
        **_unused,
    ) -> OmniOutput:
        """Run condition encoder + diffusion loop + VAE decode.

        Args:
            thinker_hidden_states: ``[B, N, H_thinker]`` tensor from stage 0.
            attention_mask: Optional ``[B, N]`` mask.
            height / width: Output resolution. Defaults from config.
            num_inference_steps / guidance_scale / seed: Sampling knobs.
        """
        cfg = self.image_gen_config
        h = height or cfg.default_height
        w = width or cfg.default_width
        steps = num_inference_steps or cfg.num_inference_steps
        cfg_scale = guidance_scale if guidance_scale is not None else cfg.guidance_scale

        logger.info(
            "[MingImageGen.forward] hidden_states=%s, hw=(%d,%d), steps=%d, cfg=%.2f",
            tuple(thinker_hidden_states.shape),
            h,
            w,
            steps,
            cfg_scale,
        )

        cap_feats = self.condition_encoder(thinker_hidden_states, attention_mask=attention_mask)
        logger.info("[MingImageGen.forward] cap_feats=%s", tuple(cap_feats.shape))

        # ZImagePipeline.encode_prompt() normally returns a list[Tensor] where
        # each entry corresponds to one prompt (masked to its valid tokens).
        # We match that contract: one tensor per batch element.
        prompt_embeds = [cap_feats[i] for i in range(cap_feats.shape[0])]
        negative_prompt_embeds = [self.condition_encoder.zero_negative(e) for e in prompt_embeds]

        device = cap_feats.device
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(int(seed))

        # ZImagePipeline.forward reads ``req.prompts`` and many fields from
        # ``req.sampling_params``. Build a fake request that exposes all of
        # them with safe defaults so the pipeline never tries to encode text.
        from types import SimpleNamespace

        sampling_params = SimpleNamespace(
            strength=None,
            height=h,
            width=w,
            num_inference_steps=steps,
            generator=generator,
            sigmas=None,
            max_sequence_length=512,
            guidance_scale=cfg_scale,
            guidance_rescale=None,
            num_outputs_per_prompt=1,
        )
        # ``prompts`` is iterated to derive prompt/negative_prompt strings.
        # We pass empty strings — the real conditioning comes from prompt_embeds.
        req = SimpleNamespace(
            request_id="ming-imagegen",
            prompts=[{"prompt": "", "negative_prompt": ""}],
            sampling_params=sampling_params,
        )

        output = self.pipeline.forward(
            req,
            prompt=None,
            height=h,
            width=w,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            generator=generator,
            output_type="pil",
            return_dict=True,
        )

        # ZImagePipeline.forward returns a DiffusionOutput dataclass whose
        # ``.output`` field is a raw decoded tensor [B, C, H, W] in [-1, 1].
        if hasattr(output, "output") and output.output is not None:
            raw = output.output
        elif hasattr(output, "images"):
            raw = output.images
        else:
            raw = output

        if not isinstance(raw, torch.Tensor):
            raise RuntimeError(f"ZImagePipeline returned non-tensor output: {type(raw).__name__}")
        # Normalize to a single [3, H, W] uint8 tensor in [0, 255] suitable
        # for the API serializer (vllm_omni/entrypoints/openai/serving_chat.py
        # lines 2008-2034 auto-detect tensors and build PIL + base64).
        img_tensor = self.pipeline.image_processor.postprocess(raw.float(), output_type="pt")
        # postprocess(output_type="pt") returns a [B, C, H, W] tensor in [0, 1].
        if img_tensor.dim() == 4:
            img_tensor = img_tensor[0]  # [C, H, W]
        logger.info(
            "[MingImageGen.forward] produced image tensor shape=%s range=[%.3f,%.3f]",
            tuple(img_tensor.shape),
            img_tensor.float().min().item(),
            img_tensor.float().max().item(),
        )
        return OmniOutput(
            text_hidden_states=torch.zeros((1, 1), device=cap_feats.device, dtype=cap_feats.dtype),
            multimodal_outputs={"image": img_tensor.detach().cpu()},
        )

    # ------------------------------------------------------------------
    # Weight loading (vllm contract)
    # ------------------------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Route weights to the right sub-component.

        In Phase 1 the ZImagePipeline loads its transformer/VAE/scheduler
        from disk inside ``__init__`` (via ``from_pretrained``), and the
        condition encoder loads its connector the same way. The main Ming
        checkpoint at the top level only contains the *thinker* weights,
        which the imagegen stage does not consume.

        We therefore silently drop top-level thinker weights here and only
        log what we saw, so callers know loading is effectively a no-op.
        """
        seen: set[str] = set()
        counters: dict[str, int] = {}
        for name, _tensor in weights:
            bucket = name.split(".", 1)[0] if "." in name else name
            counters[bucket] = counters.get(bucket, 0) + 1
            seen.add(name)
        logger.info(
            "[MingImageGen.load_weights] top-level weight buckets observed: %s",
            counters,
        )
        logger.info(
            "[MingImageGen.load_weights] imagegen stage loads its own "
            "transformer/vae/connector via from_pretrained; ignoring %d "
            "top-level tensors.",
            len(seen),
        )
        return seen

    # vllm's runner expects make_empty_intermediate_tensors on every model.
    def make_empty_intermediate_tensors(self, *args, **kwargs):  # noqa: ARG002
        return None


def _merge_split_qkv_and_gated_ffn(
    state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Merge diffusers-style split QKV / gated-FFN weights into vllm-omni form.

    Ming's diffusers-format checkpoint ships:
        <prefix>.attention.to_q.weight
        <prefix>.attention.to_k.weight
        <prefix>.attention.to_v.weight
        <prefix>.feed_forward.w1.weight
        <prefix>.feed_forward.w3.weight

    But vllm-omni's ``ZImageTransformer2DModel`` expects the merged form
    used by its ColumnParallelLinear QKV fusion + gated-FFN fusion:
        <prefix>.attention.to_qkv.weight       = cat([q, k, v], dim=0)
        <prefix>.feed_forward.w13.weight       = cat([w1, w3], dim=0)

    Bias tensors follow the same pattern if present. Keys that are neither
    part of a merge triple/pair pass through unchanged.

    Returns a *new* state dict (does not mutate the input).
    """
    out: dict[str, torch.Tensor] = {}

    # Group by (layer_prefix, suffix) so we can detect triples/pairs.
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
        # Pass through unchanged.
        out[name] = tensor

    # Emit merged QKV tensors.
    merged_qkv = 0
    for prefix, parts in qkv_groups.items():
        if {"q", "k", "v"}.issubset(parts):
            out[f"{prefix}.attention.to_qkv.weight"] = torch.cat([parts["q"], parts["k"], parts["v"]], dim=0)
            merged_qkv += 1
        else:
            # Incomplete group: pass through originals under their real names.
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
        "[MingImageGen] merged QKV groups: %d, merged w13 groups: %d; final state_dict has %d keys (was %d)",
        merged_qkv,
        merged_w13,
        len(out),
        len(state),
    )
    return out


__all__ = ["MingFlashOmniImageGenModel"]
