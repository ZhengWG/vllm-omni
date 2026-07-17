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
from vllm.logger import init_logger
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import DistributedAutoencoderKLWan
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.forward_context import set_forward_context_denoise_step_idx
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.model_loader.hub_prefetch import from_pretrained_with_prefetch, prefetch_subfolders
from vllm_omni.diffusion.models.interface import SupportImageInput, SupportsComponentDiscovery
from vllm_omni.diffusion.models.lingbot_world.camera import (
    camera_condition,
    camera_condition_from_poses,
    parse_action_string,
)
from vllm_omni.diffusion.models.lingbot_world.lingbot_world_transformer import CausalLingBotWorldTransformer3DModel
from vllm_omni.diffusion.models.lingbot_world.raw_loader import (
    V2_DMD_TIMESTEPS,
    V2_FLOW_SHIFT,
    build_raw_transformer,
    is_raw_lingbot_checkpoint,
    load_raw_text_encoder,
    load_raw_tokenizer,
    load_raw_vae,
)
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.models.utils import _load_json
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import load_transformer_config, retrieve_latents
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

logger = init_logger(__name__)

# v1-fast DMD distillation defaults (reference warp indices [0, 179, 358, 679]
# and wan_i2v_A14B.py sample_shift). The raw v2 layout carries its own values
# in raw_loader; diffusers checkpoint config or extra_args may override these.
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

        self.is_raw_checkpoint = is_raw_lingbot_checkpoint(model)
        if self.is_raw_checkpoint:
            self._init_raw_checkpoint(model, dtype)
            return
        self._init_diffusers_checkpoint(model, dtype, local_only)

    def _init_raw_checkpoint(self, model: str, dtype: torch.dtype) -> None:
        """Load the original (non-diffusers) LingBot-World v2 layout in place."""
        # The DiT weights live in the raw layout's ``transformers/`` safetensors
        # and load through the engine's weight pass (TP-aware); the remaining
        # components are converted/loaded here.
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=model,
                subfolder="transformers",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
            )
        ]
        self.tokenizer = load_raw_tokenizer(model)
        self.text_encoder = load_raw_text_encoder(model, device=self.device, dtype=dtype)
        self.vae = load_raw_vae(model, device=self.device, dtype=dtype)
        self.transformer = build_raw_transformer(model)
        self.flow_shift = V2_FLOW_SHIFT
        self.dmd_timesteps = V2_DMD_TIMESTEPS
        self.vae_temporal = int(getattr(self.vae.config, "scale_factor_temporal", 4))
        self.vae_spatial = int(getattr(self.vae.config, "scale_factor_spatial", 8))
        logger.info(
            "LingBot-World raw v2 checkpoint loaded: sink=%s window=%s DMD=%s flow_shift=%s",
            self.transformer.config.sink_size,
            self.transformer.config.sliding_window_num_frames,
            self.dmd_timesteps,
            self.flow_shift,
        )

    def _init_diffusers_checkpoint(self, model: str, dtype: torch.dtype, local_only: bool) -> None:
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
        # DMD schedule comes from explicit checkpoint config entries, else the
        # v1-fast defaults; per-request extra_args can override either. The
        # generic "shift" key is deliberately ignored — it is usually the
        # FlowUniPC export default (3.0), not the training value.
        self.flow_shift = float(scheduler_config.get("flow_shift", FLOW_SHIFT))
        self.dmd_timesteps = tuple(int(t) for t in scheduler_config.get("dmd_denoising_steps", DMD_TIMESTEPS))
        logger.info("LingBot-World DMD schedule: %s (flow_shift=%s)", self.dmd_timesteps, self.flow_shift)
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
        camera = None
        if isinstance(prompt_value, dict):
            camera = (prompt_value.get("multi_modal_data") or {}).get("camera")
        actions = extra.get("camera_actions")
        if camera is not None and actions is not None:
            raise ValueError("provide either multi_modal_data.camera (poses) or extra_args.camera_actions, not both")
        if camera is not None:
            camera = self._load_camera_trajectory(camera)
        else:
            if isinstance(actions, str):
                actions = parse_action_string(actions)
            if not actions:
                raise ValueError(
                    "camera control is required: multi_modal_data.camera={'poses': ..., 'intrinsics': ...} "
                    "or extra_args.camera_actions, e.g. 'w-36,iw-24,l-21'"
                )

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
        if camera is not None:
            if camera["poses"].shape[0] < num_frames:
                raise ValueError(f"camera poses cover {camera['poses'].shape[0]} frames but num_frames={num_frames}")
            camera = {
                "poses": camera["poses"][:num_frames],
                "intrinsics": None if camera["intrinsics"] is None else camera["intrinsics"][:num_frames],
            }
        else:
            # One action per *pixel* frame (official semantics); pad by holding
            # the last action, truncate extras.
            actions = [list(a) for a in actions[:num_frames]]
            actions += [list(actions[-1]) for _ in range(num_frames - len(actions))]

        return dict(
            prompt=" ".join(prompt.split()),
            image=image.convert("RGB"),
            actions=actions,
            camera=camera,
            height=height,
            width=width,
            num_frames=num_frames,
            latent_frames=latent_frames,
            flow_shift=float(extra.get("flow_shift", self.flow_shift)),
            dmd_timesteps=tuple(extra.get("dmd_timesteps", self.dmd_timesteps)),
            output_type=getattr(sampling, "output_type", None) or "np",
            generator=sampling.generator if not isinstance(sampling.generator, list) else sampling.generator[0],
        )

    @staticmethod
    def _load_camera_trajectory(camera: Any) -> dict[str, torch.Tensor | None]:
        """Normalize ``multi_modal_data.camera`` into pose/intrinsics tensors.

        Accepts a directory containing ``poses.npy``/``intrinsics.npy`` (the
        official action-path layout) or a mapping with ``poses`` and optional
        ``intrinsics`` given as tensors, arrays, or ``.npy`` paths.
        """
        if isinstance(camera, str | os.PathLike):
            camera = {
                "poses": os.path.join(camera, "poses.npy"),
                "intrinsics": os.path.join(camera, "intrinsics.npy"),
            }
        if not isinstance(camera, dict) or "poses" not in camera:
            raise ValueError("multi_modal_data.camera must be an action dir or a dict with 'poses' [N,4,4]")

        def to_tensor(value: Any, optional: bool = False) -> torch.Tensor | None:
            if value is None or (optional and isinstance(value, str | os.PathLike) and not os.path.isfile(value)):
                return None
            if isinstance(value, str | os.PathLike):
                value = np.load(value, allow_pickle=False)
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(np.asarray(value, dtype=np.float64))
            if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
                raise ValueError("camera poses/intrinsics must be finite tensors, arrays, or .npy paths")
            return value

        poses = to_tensor(camera["poses"])
        intrinsics = to_tensor(camera.get("intrinsics"), optional=True)
        if poses.ndim != 3 or poses.shape[1:] != (4, 4):
            raise ValueError(f"camera poses must have shape [N, 4, 4], got {tuple(poses.shape)}")
        if intrinsics is not None and (
            intrinsics.ndim != 2 or intrinsics.shape[1] != 4 or len(intrinsics) != len(poses)
        ):
            raise ValueError(f"camera intrinsics must have shape [N, 4] matching poses, got {tuple(intrinsics.shape)}")
        return {"poses": poses, "intrinsics": intrinsics}

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
        # Official preprocessing (wan/image2video.py)
        image = inputs["image"].convert("RGB")
        pixels = torch.from_numpy(np.array(image, dtype=np.uint8)).permute(2, 0, 1).to(torch.float32).div_(255)
        pixels = pixels.sub_(0.5).div_(0.5)
        pixels = torch.nn.functional.interpolate(
            pixels[None], size=(inputs["height"], inputs["width"]), mode="bicubic"
        )[0]
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
        initial_latents: torch.Tensor,  # [1, 16, block, h, w] noise slice
        condition: torch.Tensor,  # [1, 20, block, h, w]
        camera: torch.Tensor,  # [1, 384, block, h, w]
        prompt_embeds: torch.Tensor,
        cache: Any,
        start_frame: int,
        inputs: dict[str, Any],
        progress_bar: Any,
    ) -> torch.Tensor:
        # Official numerics: the latent trajectory (noise, x0, renoise) is
        # carried in float32 end to end; only the DiT input is cast down.
        latents = initial_latents
        sigmas = [_dmd_sigma(t, inputs["flow_shift"]) for t in inputs["dmd_timesteps"]]

        for step, sigma in enumerate(sigmas):
            set_forward_context_denoise_step_idx(step)
            flow_pred = self.transformer(
                torch.cat([latents.to(condition.dtype), condition], dim=1),
                torch.tensor(sigma * 1000.0, device=self.device),
                prompt_embeds,
                camera,
                cache=cache,
                start_frame=start_frame,
                update_cache=False,
            )
            # Official semantics: x0 in float64, renoise drawn channel-first
            # without a batch dim so the RNG stream matches the reference.
            x0 = (latents.double() - sigma * flow_pred.double()).float()
            if step + 1 < len(sigmas):  # self-forcing: renoise x0 to the next level
                noise = randn_tensor(
                    x0.shape[1:], generator=inputs["generator"], device=self.device, dtype=torch.float32
                )[None]
                latents = (1.0 - sigmas[step + 1]) * x0 + sigmas[step + 1] * noise
            else:
                latents = x0
            progress_bar.update()

        # Context refill: commit this chunk's clean K/V as history for the next chunk.
        self.transformer(
            torch.cat([latents.to(condition.dtype), condition], dim=1),
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

        # Official offline semantics: build the full-trajectory camera condition
        # once (interpolated to the latent grid, max-norm over the whole
        # sequence), then slice per chunk.
        common = dict(
            num_latent_frames=inputs["latent_frames"],
            height=inputs["height"],
            width=inputs["width"],
            spatial_scale=self.vae_spatial,
            device=self.device,
            dtype=condition.dtype,
        )
        if inputs["camera"] is not None:
            camera = camera_condition_from_poses(inputs["camera"]["poses"], inputs["camera"]["intrinsics"], **common)
        else:
            camera = camera_condition(inputs["actions"], **common)

        # Initial noise is drawn once for the whole video (fp32, channel-first,
        # no batch dim) and sliced per chunk — the reference implementation's
        # RNG order, so the same seed consumes the same stream.
        noise_all = randn_tensor(
            (self.transformer.config.out_channels, inputs["latent_frames"], condition.shape[-2], condition.shape[-1]),
            generator=inputs["generator"],
            device=self.device,
            dtype=torch.float32,
        )

        chunks = []
        num_blocks = inputs["latent_frames"] // block
        with self.progress_bar(total=num_blocks * len(inputs["dmd_timesteps"])) as progress_bar:
            for index in range(num_blocks):
                start = index * block
                chunks.append(
                    self._generate_chunk(
                        noise_all[None, :, start : start + block],
                        condition[:, :, start : start + block],
                        camera[:, :, start : start + block],
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
