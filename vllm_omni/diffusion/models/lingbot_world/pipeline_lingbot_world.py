# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LingBot-World causal DMD pipeline: interactive world-model I2V generation.

Offline path (stage 1): first frame + prompt + keyboard actions -> video,
generated autoregressively in 3-latent-frame chunks with a request-local KV
cache. Each chunk runs the 4-step DMD schedule and one cache-refill forward
(``update_cache=True`` at t=0) so the clean chunk becomes history.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from typing import Any, ClassVar

import numpy as np
import PIL.Image
import torch
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from transformers import AutoTokenizer, UMT5EncoderModel
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import DistributedAutoencoderKLWan
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.forward_context import set_forward_context_denoise_step_idx
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.model_loader.hub_prefetch import from_pretrained_with_prefetch, prefetch_subfolders
from vllm_omni.diffusion.models.interface import SupportImageInput, SupportsComponentDiscovery
from vllm_omni.diffusion.models.lingbot_world.camera import camera_chunk_condition, parse_action_string
from vllm_omni.diffusion.models.lingbot_world.lingbot_world_transformer import CausalLingBotWorldTransformer3DModel
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.models.utils import _load_json
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import load_transformer_config, retrieve_latents
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

# LingBot-World fast (v1) DMD distillation timesteps and flow shift; the v2
# checkpoint uses [1000, 750, 500, 250] with flow_shift=5.0 (see extra_args).
DMD_TIMESTEPS = (1000, 821, 642, 321)
FLOW_SHIFT = 10.0
TEXT_LEN = 512


def get_lingbot_world_post_process_func(od_config: OmniDiffusionConfig) -> Callable[..., Any]:
    del od_config
    from diffusers.video_processor import VideoProcessor

    video_processor = VideoProcessor(vae_scale_factor=8)

    def post_process_func(video: torch.Tensor, output_type: str = "np", sampling_params: Any = None) -> Any:
        if sampling_params is not None:
            output_type = getattr(sampling_params, "output_type", None) or output_type
        if output_type == "latent":
            return video
        return {"video": video_processor.postprocess_video(video, output_type=output_type), "custom_output": {}}

    return post_process_func


def _dmd_sigma(timestep: float, flow_shift: float) -> float:
    """Shifted flow-matching noise level for a raw DMD timestep.

    Reference semantics (SelfForcingFlowMatchScheduler + warped DMD steps):
    sigma(t) = shift * x / (1 + (shift - 1) * x) with x = t / 1000; the DiT is
    conditioned on the *warped* timestep sigma * 1000.
    """
    x = timestep / 1000.0
    return flow_shift * x / (1.0 + (flow_shift - 1.0) * x)


class LingBotWorldCausalDMDPipeline(nn.Module, SupportImageInput, SupportsComponentDiscovery, ProgressBarMixin):
    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder"]
    _vae_modules: ClassVar[list[str]] = ["vae"]
    # Warmup cannot synthesize camera actions; skip the generic dummy run.
    dummy_run_num_frames: ClassVar[int] = 0

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__()
        del prefix
        self.od_config = od_config
        self.device = get_local_device()
        dtype = getattr(od_config, "dtype", torch.bfloat16)
        model = od_config.model
        local_only = os.path.exists(model)

        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=model,
                subfolder="transformer",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
            )
        ]
        subfolders = ["tokenizer", "text_encoder", "vae"]
        prefetch_subfolders(model, subfolders, local_files_only=local_only)
        self.tokenizer = from_pretrained_with_prefetch(
            AutoTokenizer.from_pretrained,
            model,
            subfolder="tokenizer",
            prefetch_list=subfolders,
            local_files_only=local_only,
        )
        self.text_encoder = from_pretrained_with_prefetch(
            UMT5EncoderModel.from_pretrained,
            model,
            subfolder="text_encoder",
            prefetch_list=subfolders,
            local_files_only=local_only,
            torch_dtype=dtype,
        ).to(self.device)
        self.vae = from_pretrained_with_prefetch(
            DistributedAutoencoderKLWan.from_pretrained,
            model,
            subfolder="vae",
            prefetch_list=subfolders,
            local_files_only=local_only,
            torch_dtype=dtype,
        ).to(self.device)
        self.transformer = CausalLingBotWorldTransformer3DModel.from_config(
            load_transformer_config(model, "transformer", local_only), prefix="transformer"
        )

        try:
            scheduler_config = _load_json(model, "scheduler/scheduler_config.json", local_only)
        except (OSError, ValueError):
            scheduler_config = {}
        self.flow_shift = float(scheduler_config.get("flow_shift", scheduler_config.get("shift", FLOW_SHIFT)))
        self.vae_temporal = int(getattr(self.vae.config, "scale_factor_temporal", 4))
        self.vae_spatial = int(getattr(self.vae.config, "scale_factor_spatial", 8))

    # ------------------------------------------------------------------ inputs
    def _parse_request(self, req: DiffusionRequestBatch) -> dict[str, Any]:
        assert req.num_reqs == 1, "LingBot-World serves one request at a time"
        sampling = req.sampling_params
        prompt_value = req.prompts[0]
        if isinstance(prompt_value, str):
            prompt, image = prompt_value, None
        else:
            prompt = prompt_value.get("prompt") or ""
            image = (prompt_value.get("multi_modal_data") or {}).get("image")
        if isinstance(image, list):
            image = image[0] if image else None
        if isinstance(image, str | os.PathLike):
            image = PIL.Image.open(image)
        if not isinstance(image, PIL.Image.Image):
            raise ValueError("LingBot-World requires a first-frame image in multi_modal_data.image")

        extra = sampling.extra_args or {}
        actions = extra.get("camera_actions")
        if isinstance(actions, str):
            actions = parse_action_string(actions)
        if not actions:
            raise ValueError("extra_args.camera_actions is required, e.g. 'w-12,iw-6,l-6,none-6'")

        height = int(sampling.height or 480)
        width = int(sampling.width or 832)
        divisor = self.vae_spatial * 2  # spatial patch size
        if height % divisor or width % divisor:
            raise ValueError(f"height/width must be divisible by {divisor}, got {height}x{width}")

        num_frames = int(sampling.num_frames or 81)
        block = self.transformer.config.num_frames_per_block
        if (num_frames - 1) % self.vae_temporal:
            raise ValueError(f"num_frames must satisfy (n-1) % {self.vae_temporal} == 0, got {num_frames}")
        latent_frames = (num_frames - 1) // self.vae_temporal + 1
        if latent_frames % block:
            raise ValueError(
                f"num_frames={num_frames} maps to {latent_frames} latent frames, must be a multiple of {block}"
            )
        # One action per latent frame; pad by holding the last action.
        actions = [list(a) for a in actions[:latent_frames]]
        actions += [list(actions[-1]) for _ in range(latent_frames - len(actions))]

        return dict(
            prompt=" ".join(prompt.split()),
            image=image.convert("RGB"),
            actions=actions,
            height=height,
            width=width,
            num_frames=num_frames,
            latent_frames=latent_frames,
            flow_shift=float(extra.get("flow_shift", self.flow_shift)),
            dmd_timesteps=tuple(extra.get("dmd_timesteps", DMD_TIMESTEPS)),
            output_type=getattr(sampling, "output_type", None) or "np",
            generator=sampling.generator if not isinstance(sampling.generator, list) else sampling.generator[0],
        )

    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        inputs = self.tokenizer(
            [prompt],
            padding="max_length",
            max_length=TEXT_LEN,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = inputs.input_ids.to(self.device)
        mask = inputs.attention_mask.to(self.device)
        embeds = self.text_encoder(input_ids, mask).last_hidden_state.to(dtype=self.transformer.dtype)
        return embeds * mask.unsqueeze(-1).to(embeds.dtype)

    def _latent_stats(self, ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        view = (1, -1, 1, 1, 1)
        mean = torch.as_tensor(self.vae.config.latents_mean, device=ref.device, dtype=ref.dtype).view(view)
        std = torch.as_tensor(self.vae.config.latents_std, device=ref.device, dtype=ref.dtype).view(view)
        return mean, std

    def _encode_condition(self, inputs: dict[str, Any]) -> torch.Tensor:
        """I2V condition [1, 20, latent_frames, h, w]: 4ch temporal mask + 16ch latent."""
        image = inputs["image"].resize((inputs["width"], inputs["height"]), PIL.Image.Resampling.LANCZOS)
        pixels = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1) / 127.5 - 1.0
        video = pixels.new_zeros(1, 3, inputs["num_frames"], inputs["height"], inputs["width"])
        video[:, :, 0] = pixels
        latent = retrieve_latents(
            self.vae.encode(video.to(device=self.device, dtype=self.vae.dtype)), sample_mode="argmax"
        )
        mean, std = self._latent_stats(latent)
        latent = (latent - mean) / std
        mask = latent.new_zeros(1, self.vae_temporal, *latent.shape[2:])
        mask[:, :, 0] = 1.0  # only the first latent frame carries real content
        return torch.cat([mask, latent], dim=1).to(self.transformer.dtype)

    # ---------------------------------------------------------------- denoising
    def _generate_chunk(
        self,
        condition: torch.Tensor,  # [1, 20, block, h, w]
        camera: torch.Tensor,  # [1, 384, block, h, w]
        prompt_embeds: torch.Tensor,
        cache: Any,
        start_frame: int,
        inputs: dict[str, Any],
        progress_bar: Any,
    ) -> torch.Tensor:
        shape = (1, self.transformer.config.out_channels, *condition.shape[2:])
        latents = randn_tensor(shape, generator=inputs["generator"], device=self.device, dtype=condition.dtype)
        sigmas = [_dmd_sigma(t, inputs["flow_shift"]) for t in inputs["dmd_timesteps"]]

        for step, sigma in enumerate(sigmas):
            set_forward_context_denoise_step_idx(step)
            flow_pred = self.transformer(
                torch.cat([latents, condition], dim=1),
                torch.tensor(sigma * 1000.0, device=self.device),
                prompt_embeds,
                camera,
                cache=cache,
                start_frame=start_frame,
                update_cache=False,
            )
            x0 = latents - sigma * flow_pred
            if step + 1 < len(sigmas):  # self-forcing: renoise x0 to the next level
                noise = randn_tensor(shape, generator=inputs["generator"], device=self.device, dtype=latents.dtype)
                latents = (1.0 - sigmas[step + 1]) * x0 + sigmas[step + 1] * noise
            else:
                latents = x0
            progress_bar.update()

        # Context refill: commit this chunk's clean K/V as history for the next chunk.
        self.transformer(
            torch.cat([latents, condition], dim=1),
            torch.tensor(0.0, device=self.device),
            prompt_embeds,
            camera,
            cache=cache,
            start_frame=start_frame,
            update_cache=True,
        )
        return latents

    @torch.no_grad()
    def forward(self, req: DiffusionRequestBatch) -> DiffusionOutput:
        inputs = self._parse_request(req)
        prompt_embeds = self._encode_prompt(inputs["prompt"])
        condition = self._encode_condition(inputs)

        block = self.transformer.config.num_frames_per_block
        cache = self.transformer.allocate_cache(
            batch_size=1,
            latent_height=condition.shape[-2],
            latent_width=condition.shape[-1],
            num_latent_frames=inputs["latent_frames"],
            device=self.device,
            dtype=condition.dtype,
        )

        chunks = []
        num_blocks = inputs["latent_frames"] // block
        with self.progress_bar(total=num_blocks * len(inputs["dmd_timesteps"])) as progress_bar:
            for index in range(num_blocks):
                start = index * block
                camera = camera_chunk_condition(
                    inputs["actions"][: start + block],
                    chunk_size=block,
                    height=inputs["height"],
                    width=inputs["width"],
                    spatial_scale=self.vae_spatial,
                    device=self.device,
                    dtype=condition.dtype,
                )
                chunks.append(
                    self._generate_chunk(
                        condition[:, :, start : start + block],
                        camera,
                        prompt_embeds,
                        cache,
                        start,
                        inputs,
                        progress_bar,
                    )
                )
        latents = torch.cat(chunks, dim=2)

        if inputs["output_type"] == "latent":
            return DiffusionOutput(output=latents)
        mean, std = self._latent_stats(latents)
        video = self.vae.decode((latents * std + mean).to(self.vae.dtype), return_dict=False)[0]
        return DiffusionOutput(output=video)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return set(AutoWeightsLoader(self).load_weights(weights))
