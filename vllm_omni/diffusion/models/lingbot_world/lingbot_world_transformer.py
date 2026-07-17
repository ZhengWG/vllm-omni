# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Causal LingBot-World video transformer (Wan2.2-derived, camera-controllable).

Module names follow the diffusers checkpoint parameter names directly
(``blocks.N.self_attn.q`` etc.), so ``load_weights`` is a plain name match.
Block-causal attention: tokens attend bidirectionally inside the current
3-latent-frame chunk and causally to cached history (attention sink + sliding
window), held in a caller-owned request-local KV cache.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
from vllm.model_executor.utils import set_weight_attrs

from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.layers.rope import RotaryEmbeddingWan


@dataclass
class LingBotLayerKVCache:
    """Self-attention K/V history of one layer: [sink | rolling window | current]."""

    key: torch.Tensor  # [B, capacity, local_heads, head_dim]
    value: torch.Tensor
    end: int = 0  # valid tokens in storage
    sink_end: int = 0  # tokens pinned as attention sink
    absolute_end: int = 0  # absolute token offset the cache has seen
    last_start: int | None = None  # None = cache never written


@dataclass
class LingBotTransformerCache:
    self_attention: list[LingBotLayerKVCache]
    cross_attention: list[tuple[torch.Tensor, torch.Tensor] | None] = field(default_factory=list)


class _RMSNormAcrossHeads(nn.Module):
    """RMSNorm over the full projection dim, correct under TP sharding."""

    def __init__(self, local_size: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(local_size))
        set_weight_attrs(self.weight, {"weight_loader": self._load_shard})

    def _load_shard(self, param: torch.Tensor, loaded: torch.Tensor) -> None:
        tp_size = get_tensor_model_parallel_world_size()
        shard = loaded.chunk(tp_size)[get_tensor_model_parallel_rank()] if tp_size > 1 else loaded
        param.data.copy_(shard)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tp_size = get_tensor_model_parallel_world_size()
        x_f = x.float()
        sum_sq = x_f.pow(2).sum(dim=-1, keepdim=True)
        count = x.shape[-1]
        if tp_size > 1:
            sum_sq = tensor_model_parallel_all_reduce(sum_sq)
            count *= tp_size
        return (x_f * torch.rsqrt(sum_sq / count + self.eps) * self.weight.float()).to(x.dtype)


class LingBotSelfAttention(nn.Module):
    """Block-causal self-attention over cached history plus the current chunk."""

    def __init__(self, dim: int, num_heads: int, eps: float, prefix: str) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        self.head_dim = dim // num_heads
        self.num_local_heads = num_heads // tp_size
        local_dim = self.num_local_heads * self.head_dim
        linear = dict(bias=True, gather_output=False, return_bias=False)
        self.q = ColumnParallelLinear(dim, dim, **linear, prefix=f"{prefix}.q")
        self.k = ColumnParallelLinear(dim, dim, **linear, prefix=f"{prefix}.k")
        self.v = ColumnParallelLinear(dim, dim, **linear, prefix=f"{prefix}.v")
        self.o = RowParallelLinear(dim, dim, bias=True, input_is_parallel=True, return_bias=False, prefix=f"{prefix}.o")
        self.norm_q = _RMSNormAcrossHeads(local_dim, eps)
        self.norm_k = _RMSNormAcrossHeads(local_dim, eps)
        self.rotary_embedding = RotaryEmbeddingWan(is_neox_style=False, half_head_dim=True)
        self.attn = Attention(
            num_heads=self.num_local_heads,
            head_size=self.head_dim,
            causal=False,
            softmax_scale=self.head_dim**-0.5,
            num_kv_heads=self.num_local_heads,
            role="self",
            qkv_layout="BSND",
            prefix=prefix,
            skip_sequence_parallel=True,
        )

    def _update_cache(
        self,
        cache: LingBotLayerKVCache,
        key: torch.Tensor,
        value: torch.Tensor,
        current_start: int,
        sink_tokens: int,
        update_cache: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Merge the current chunk with cached history; return the visible K/V.

        Eviction keeps sink tokens pinned and drops the oldest non-sink
        history first. ``update_cache=False`` builds the visible view without
        persisting (denoising passes); the final context-refill pass commits.
        """
        chunk = key.shape[1]
        capacity = cache.key.shape[1]
        incoming_sink = min(chunk, max(sink_tokens - current_start, 0))
        if cache.last_start is None:
            assert current_start == 0 and chunk <= capacity
            next_key, next_value, next_sink = key, value, incoming_sink
        else:
            assert current_start == cache.absolute_end, (
                f"chunks must be contiguous: start={current_start}, cached_end={cache.absolute_end}"
            )
            next_sink = cache.sink_end + incoming_sink
            local_capacity = capacity - next_sink - (chunk - incoming_sink)
            assert local_capacity >= 0, "cache too small for sink tokens plus one full chunk"
            keep = min(cache.end - cache.sink_end, local_capacity)
            parts_k = [cache.key[:, : cache.sink_end], key[:, :incoming_sink]]
            parts_v = [cache.value[:, : cache.sink_end], value[:, :incoming_sink]]
            if keep > 0:
                parts_k.append(cache.key[:, cache.end - keep : cache.end])
                parts_v.append(cache.value[:, cache.end - keep : cache.end])
            parts_k.append(key[:, incoming_sink:])
            parts_v.append(value[:, incoming_sink:])
            next_key, next_value = torch.cat(parts_k, dim=1), torch.cat(parts_v, dim=1)

        if update_cache:
            end = next_key.shape[1]
            with torch.no_grad():
                cache.key[:, :end].copy_(next_key)
                cache.value[:, :end].copy_(next_value)
            cache.end = end
            cache.sink_end = next_sink
            cache.absolute_end = current_start + chunk
            cache.last_start = current_start
        return next_key, next_value

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache: LingBotLayerKVCache,
        current_start: int,
        sink_tokens: int,
        update_cache: bool,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        query = self.norm_q(self.q(hidden_states)).unflatten(2, (self.num_local_heads, self.head_dim))
        key = self.norm_k(self.k(hidden_states)).unflatten(2, (self.num_local_heads, self.head_dim))
        value = self.v(hidden_states).unflatten(2, (self.num_local_heads, self.head_dim))

        cos, sin = rotary_emb
        query = self.rotary_embedding(query, cos, sin)
        key = self.rotary_embedding(key, cos, sin)

        key, value = self._update_cache(cache, key, value, current_start, sink_tokens, update_cache)
        return self.o(self.attn(query, key, value).flatten(2, 3))


class LingBotCrossAttention(nn.Module):
    """Text cross-attention; encoder K/V is projected once and cached per request."""

    def __init__(self, dim: int, num_heads: int, eps: float, prefix: str) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        self.head_dim = dim // num_heads
        self.num_local_heads = num_heads // tp_size
        local_dim = self.num_local_heads * self.head_dim
        linear = dict(bias=True, gather_output=False, return_bias=False)
        self.q = ColumnParallelLinear(dim, dim, **linear, prefix=f"{prefix}.q")
        self.k = ColumnParallelLinear(dim, dim, **linear, prefix=f"{prefix}.k")
        self.v = ColumnParallelLinear(dim, dim, **linear, prefix=f"{prefix}.v")
        self.o = RowParallelLinear(dim, dim, bias=True, input_is_parallel=True, return_bias=False, prefix=f"{prefix}.o")
        self.norm_q = _RMSNormAcrossHeads(local_dim, eps)
        self.norm_k = _RMSNormAcrossHeads(local_dim, eps)
        self.attn = Attention(
            num_heads=self.num_local_heads,
            head_size=self.head_dim,
            causal=False,
            softmax_scale=self.head_dim**-0.5,
            num_kv_heads=self.num_local_heads,
            role="cross",
            qkv_layout="BSND",
            prefix=prefix,
            skip_sequence_parallel=True,
            disable_kv_quant=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        cached_kv: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        query = self.norm_q(self.q(hidden_states)).unflatten(2, (self.num_local_heads, self.head_dim))
        if cached_kv is None:
            key = self.norm_k(self.k(encoder_hidden_states)).unflatten(2, (self.num_local_heads, self.head_dim))
            value = self.v(encoder_hidden_states).unflatten(2, (self.num_local_heads, self.head_dim))
            cached_kv = (key.detach(), value.detach())
        key, value = cached_kv
        return self.o(self.attn(query, key, value).flatten(2, 3)), cached_kv


class LingBotBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_dim: int, eps: float, prefix: str) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.self_attn = LingBotSelfAttention(dim, num_heads, eps, f"{prefix}.self_attn")
        self.norm3 = nn.LayerNorm(dim, eps=eps)  # post-self-attn norm (cross_attn_norm=True)
        self.cross_attn = LingBotCrossAttention(dim, num_heads, eps, f"{prefix}.cross_attn")
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.ffn = nn.Sequential(
            ColumnParallelLinear(
                dim, ffn_dim, bias=True, gather_output=False, return_bias=False, prefix=f"{prefix}.ffn.0"
            ),
            nn.GELU(approximate="tanh"),
            RowParallelLinear(
                ffn_dim, dim, bias=True, input_is_parallel=True, return_bias=False, prefix=f"{prefix}.ffn.2"
            ),
        )
        self.modulation = nn.Parameter(torch.empty(1, 6, dim))
        # Camera conditioning: per-block scale/shift from the Plücker embedding.
        self.cam_injector_layer1 = nn.Linear(dim, dim)
        self.cam_injector_layer2 = nn.Linear(dim, dim)
        self.cam_scale_layer = nn.Linear(dim, dim)
        self.cam_shift_layer = nn.Linear(dim, dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        timestep_projection: torch.Tensor,  # [B, frames, 6, dim]
        camera_hidden_states: torch.Tensor,  # [B, tokens, dim]
        self_cache: LingBotLayerKVCache,
        cross_cached_kv: tuple[torch.Tensor, torch.Tensor] | None,
        current_start: int,
        sink_tokens: int,
        update_cache: bool,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        num_frames = timestep_projection.shape[1]
        tokens_per_frame = hidden_states.shape[1] // num_frames
        modulation = (self.modulation.unsqueeze(1) + timestep_projection.float()).chunk(6, dim=2)
        shift_msa, scale_msa, gate_msa, shift_ffn, scale_ffn, gate_ffn = modulation

        def per_frame(x: torch.Tensor) -> torch.Tensor:
            return x.unflatten(1, (num_frames, tokens_per_frame))

        normed = (per_frame(self.norm1(hidden_states.float())) * (1 + scale_msa) + shift_msa).flatten(1, 2)
        attn_out = self.self_attn(
            normed.to(hidden_states.dtype), self_cache, current_start, sink_tokens, update_cache, rotary_emb
        )
        hidden_states = (per_frame(hidden_states) + per_frame(attn_out) * gate_msa).flatten(1, 2).to(attn_out.dtype)

        cam = self.cam_injector_layer2(F.silu(self.cam_injector_layer1(camera_hidden_states)))
        cam = cam + camera_hidden_states
        hidden_states = (1 + self.cam_scale_layer(cam)) * hidden_states + self.cam_shift_layer(cam)

        attn_out, cross_cached_kv = self.cross_attn(self.norm3(hidden_states), encoder_hidden_states, cross_cached_kv)
        hidden_states = hidden_states + attn_out

        normed = (per_frame(self.norm2(hidden_states.float())) * (1 + scale_ffn) + shift_ffn).flatten(1, 2)
        ffn_out = self.ffn(normed.to(hidden_states.dtype))
        hidden_states = (per_frame(hidden_states) + per_frame(ffn_out) * gate_ffn).flatten(1, 2)
        return hidden_states.to(ffn_out.dtype), cross_cached_kv


class LingBotHead(nn.Module):
    def __init__(self, dim: int, out_channels: int, patch_size: tuple[int, int, int], eps: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_channels * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.empty(1, 2, dim))

    def forward(self, hidden_states: torch.Tensor, timestep_embedding: torch.Tensor) -> torch.Tensor:
        num_frames = timestep_embedding.shape[1]
        tokens_per_frame = hidden_states.shape[1] // num_frames
        shift, scale = (self.modulation.unsqueeze(1) + timestep_embedding.unsqueeze(2).float()).chunk(2, dim=2)
        normed = self.norm(hidden_states.float()).unflatten(1, (num_frames, tokens_per_frame))
        normed = (normed * (1 + scale) + shift).flatten(1, 2).to(hidden_states.dtype)
        return self.head(normed)


def _sinusoidal_embedding(dim: int, timestep: torch.Tensor) -> torch.Tensor:
    half = dim // 2
    freqs = torch.pow(10000, -torch.arange(half, device=timestep.device, dtype=torch.float64) / half)
    phase = torch.outer(timestep.to(torch.float64), freqs)
    return torch.cat([phase.cos(), phase.sin()], dim=1)


def _rope_axis(max_len: int, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    freqs = 1.0 / torch.pow(10000, torch.arange(0, dim, 2, dtype=torch.float64) / dim)
    phase = torch.outer(torch.arange(max_len, dtype=torch.float64), freqs)
    return phase.cos().float(), phase.sin().float()


class CausalLingBotWorldTransformer3DModel(nn.Module):
    """Checkpoint-compatible causal LingBot-World DiT."""

    _layerwise_offload_blocks_attrs = ["blocks"]

    def __init__(
        self,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 36,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        eps: float = 1e-6,
        rope_max_seq_len: int = 1024,
        sink_size: int = 3,
        num_frames_per_block: int = 3,
        sliding_window_num_frames: int = 45,
        camera_in_channels: int = 6 * 8 * 8,
        prefix: str = "",
        **_unused: Any,
    ) -> None:
        super().__init__()
        dim = num_attention_heads * attention_head_dim
        self.dim = dim
        self.config = SimpleNamespace(
            patch_size=tuple(patch_size),
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            sink_size=sink_size,
            num_frames_per_block=num_frames_per_block,
            sliding_window_num_frames=sliding_window_num_frames,
            rope_max_seq_len=rope_max_seq_len,
        )

        self.patch_embedding = nn.Conv3d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.patch_embedding_wancamctrl = nn.Linear(camera_in_channels * math.prod(patch_size), dim)
        self.c2ws_hidden_states_layer1 = nn.Linear(dim, dim)
        self.c2ws_hidden_states_layer2 = nn.Linear(dim, dim)
        self.text_embedding = nn.Sequential(nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim))
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList(
            LingBotBlock(dim, num_attention_heads, ffn_dim, eps, prefix=f"blocks.{i}") for i in range(num_layers)
        )
        self.head = LingBotHead(dim, out_channels, tuple(patch_size), eps)

        self.freq_dim = freq_dim
        temporal_dim = attention_head_dim - 4 * (attention_head_dim // 6)
        spatial_dim = 2 * (attention_head_dim // 6)
        for axis, axis_dim in (("t", temporal_dim), ("h", spatial_dim), ("w", spatial_dim)):
            cos, sin = _rope_axis(rope_max_seq_len, axis_dim)
            self.register_buffer(f"_rope_{axis}_cos", cos, persistent=False)
            self.register_buffer(f"_rope_{axis}_sin", sin, persistent=False)

    @classmethod
    def from_config(cls, config: dict[str, Any], *, quant_config: Any = None, prefix: str = "") -> Any:
        if quant_config is not None:
            raise NotImplementedError("LingBot-World does not support quantization yet.")
        kwargs = {k: v for k, v in config.items() if not k.startswith("_")}
        return cls(**kwargs, prefix=prefix)

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embedding.weight.dtype

    def allocate_cache(
        self,
        *,
        batch_size: int,
        latent_height: int,
        latent_width: int,
        num_latent_frames: int | None = None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> LingBotTransformerCache:
        """Allocate a request-owned KV cache sized to the causal window."""
        cfg = self.config
        tokens_per_frame = (latent_height // cfg.patch_size[1]) * (latent_width // cfg.patch_size[2])
        window = cfg.sliding_window_num_frames
        if num_latent_frames is not None:  # bounded request: no need for the full window
            window = min(window, max(num_latent_frames, cfg.sink_size + cfg.num_frames_per_block))
        shape = (
            batch_size,
            window * tokens_per_frame,
            self.blocks[0].self_attn.num_local_heads,
            cfg.attention_head_dim,
        )
        return LingBotTransformerCache(
            self_attention=[
                LingBotLayerKVCache(
                    key=torch.zeros(shape, device=device, dtype=dtype),
                    value=torch.zeros(shape, device=device, dtype=dtype),
                )
                for _ in range(cfg.num_layers)
            ],
            cross_attention=[None] * cfg.num_layers,
        )

    def _chunk_rotary_emb(
        self, frames: int, height: int, width: int, start_frame: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        def expand(cos_t, sin_t, cos_h, sin_h, cos_w, sin_w):
            parts = []
            for table, view in (
                ((cos_t, sin_t), lambda t: t[start_frame : start_frame + frames].view(frames, 1, 1, -1)),
                ((cos_h, sin_h), lambda t: t[:height].view(1, height, 1, -1)),
                ((cos_w, sin_w), lambda t: t[:width].view(1, 1, width, -1)),
            ):
                parts.append(tuple(view(x).expand(frames, height, width, -1) for x in table))
            cos = torch.cat([p[0] for p in parts], dim=-1).reshape(frames * height * width, -1)
            sin = torch.cat([p[1] for p in parts], dim=-1).reshape(frames * height * width, -1)
            return cos.to(device=device, dtype=dtype), sin.to(device=device, dtype=dtype)

        return expand(
            self._rope_t_cos, self._rope_t_sin, self._rope_h_cos, self._rope_h_sin, self._rope_w_cos, self._rope_w_sin
        )

    def _fold_camera_patches(self, camera: torch.Tensor) -> torch.Tensor:
        """[B, C, F, H, W] -> [B, tokens, C * prod(patch_size)] matching patchify order."""
        b, c, f, h, w = camera.shape
        pt, ph, pw = self.config.patch_size
        camera = camera.view(b, c, f // pt, pt, h // ph, ph, w // pw, pw)
        return camera.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(b, -1, c * pt * ph * pw)

    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, 36, frames, H, W] latent chunk
        timestep: torch.Tensor,  # scalar or [B]; warped (sigma * 1000)
        encoder_hidden_states: torch.Tensor,  # [B, text_len, text_dim]
        camera_hidden_states: torch.Tensor,  # [B, 6*8*8, frames, H, W]
        cache: LingBotTransformerCache,
        start_frame: int,
        update_cache: bool,
    ) -> torch.Tensor:
        cfg = self.config
        batch_size, _, frames, height, width = hidden_states.shape
        assert frames == cfg.num_frames_per_block, f"expected {cfg.num_frames_per_block}-frame chunks, got {frames}"
        patched_h, patched_w = height // cfg.patch_size[1], width // cfg.patch_size[2]
        tokens_per_frame = patched_h * patched_w
        current_start = start_frame * tokens_per_frame
        sink_tokens = cfg.sink_size * tokens_per_frame

        rotary_emb = self._chunk_rotary_emb(
            frames, patched_h, patched_w, start_frame, hidden_states.device, hidden_states.dtype
        )
        hidden = self.patch_embedding(hidden_states).flatten(2).transpose(1, 2)

        camera = self.patch_embedding_wancamctrl(self._fold_camera_patches(camera_hidden_states))
        camera = camera + self.c2ws_hidden_states_layer2(F.silu(self.c2ws_hidden_states_layer1(camera)))

        if timestep.ndim == 0:
            timestep = timestep.reshape(1)
        timestep = timestep.expand(batch_size)[:, None].expand(batch_size, frames).reshape(-1)
        temb = self.time_embedding(_sinusoidal_embedding(self.freq_dim, timestep).to(self.dtype))
        temb = temb.unflatten(0, (batch_size, frames))  # [B, frames, dim]
        timestep_projection = self.time_projection(temb).unflatten(2, (6, self.dim))

        text = self.text_embedding(encoder_hidden_states)
        for index, block in enumerate(self.blocks):
            hidden, cross_kv = block(
                hidden,
                None if cache.cross_attention[index] is not None else text,
                timestep_projection,
                camera,
                self_cache=cache.self_attention[index],
                cross_cached_kv=cache.cross_attention[index],
                current_start=current_start,
                sink_tokens=sink_tokens,
                update_cache=update_cache,
                rotary_emb=rotary_emb,
            )
            cache.cross_attention[index] = cross_kv

        out = self.head(hidden, temb)
        out = out.reshape(batch_size, frames, patched_h, patched_w, *cfg.patch_size, cfg.out_channels)
        out = out.permute(0, 7, 1, 4, 2, 5, 3, 6)
        return out.reshape(batch_size, cfg.out_channels, frames, height, width)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        for name, weight in weights:
            if name not in params:
                raise KeyError(f"unexpected LingBot-World weight: {name}")
            param = params[name]
            loader = getattr(param, "weight_loader", None)
            if loader is not None:
                loader(param, weight)
            else:
                assert param.shape == weight.shape, f"{name}: {tuple(param.shape)} != {tuple(weight.shape)}"
                param.data.copy_(weight)
            loaded.add(name)
        return loaded
