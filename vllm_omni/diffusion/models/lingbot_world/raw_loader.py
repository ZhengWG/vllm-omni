# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load the official (non-diffusers) LingBot-World v2 checkpoint layout.

The released ``robbyant/lingbot-world-v2`` checkpoint ships components in the
original framework format rather than the diffusers layout:

    config.json                          # {"_class_name": "WanXModel", ...}
    transformers/*.safetensors           # DiT (param names already match ours)
    Wan2.1_VAE.pth                       # raw Wan VAE state dict
    models_t5_umt5-xxl-enc-bf16.pth      # raw Wan-umt5 text encoder
    google/umt5-xxl/                     # tokenizer

Conversions here are clean-room (diffusers' Apache-2.0 Wan VAE converter for the
VAE; a published-architecture key remap for the umt5 text encoder) — no code is
copied from the CC BY-NC-SA upstream.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Any

import torch

# v2 causal-fast inference config, from the official quick-start command
# (``--sink_size 6 --local_attn_size 18``) and the v2 DMD/flow-shift values.
V2_SINK_SIZE = 6
V2_LOCAL_ATTN_SIZE = 18
V2_DMD_TIMESTEPS = (1000, 750, 500, 250)
V2_FLOW_SHIFT = 5.0
_UMT5_NUM_LAYERS = 24


def is_raw_lingbot_checkpoint(model_path: str) -> bool:
    """True for the original (non-diffusers) LingBot-World checkpoint layout."""
    if not os.path.isdir(model_path):
        return False
    config = os.path.join(model_path, "config.json")
    if not os.path.isfile(config):
        return False
    try:
        meta = json.load(open(config))
    except (OSError, ValueError):
        return False
    return meta.get("_class_name") == "WanXModel" and os.path.isdir(os.path.join(model_path, "transformers"))


def _load_state_dict(path: str) -> dict[str, torch.Tensor]:
    sd = torch.load(path, map_location="cpu", weights_only=False)
    return sd.get("state_dict", sd) if isinstance(sd, dict) else sd


def load_raw_vae(model_path: str, *, device: torch.device, dtype: torch.dtype) -> Any:
    from diffusers import AutoencoderKLWan
    from diffusers.loaders.single_file_utils import convert_wan_vae_to_diffusers

    raw = _load_state_dict(os.path.join(model_path, "Wan2.1_VAE.pth"))
    vae = AutoencoderKLWan()
    vae.load_state_dict(convert_wan_vae_to_diffusers(raw), strict=True)
    return vae.to(device=device, dtype=dtype).eval()


def _remap_umt5(raw: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Raw Wan-umt5 state dict -> HF UMT5EncoderModel keys.

    Wan-umt5 uses per-layer relative position bias and a gated-GELU FFN
    (``gate`` = GELU branch, ``fc1`` = linear branch), matching HF UMT5's
    ``wi_0`` (activated) / ``wi_1`` (linear) / ``wo``.
    """
    out = {
        "shared.weight": raw["token_embedding.weight"],
        "encoder.embed_tokens.weight": raw["token_embedding.weight"],
        "encoder.final_layer_norm.weight": raw["norm.weight"],
    }
    pairs = [
        ("attn.q", "layer.0.SelfAttention.q"),
        ("attn.k", "layer.0.SelfAttention.k"),
        ("attn.v", "layer.0.SelfAttention.v"),
        ("attn.o", "layer.0.SelfAttention.o"),
        ("pos_embedding.embedding", "layer.0.SelfAttention.relative_attention_bias"),
        ("norm1", "layer.0.layer_norm"),
        ("ffn.gate.0", "layer.1.DenseReluDense.wi_0"),
        ("ffn.fc1", "layer.1.DenseReluDense.wi_1"),
        ("ffn.fc2", "layer.1.DenseReluDense.wo"),
        ("norm2", "layer.1.layer_norm"),
    ]
    for i in range(_UMT5_NUM_LAYERS):
        for src, dst in pairs:
            out[f"encoder.block.{i}.{dst}.weight"] = raw[f"blocks.{i}.{src}.weight"]
    return out


def load_raw_text_encoder(model_path: str, *, device: torch.device, dtype: torch.dtype) -> Any:
    from transformers import UMT5Config, UMT5EncoderModel

    raw = _load_state_dict(os.path.join(model_path, "models_t5_umt5-xxl-enc-bf16.pth"))
    config = UMT5Config(
        d_model=4096,
        d_ff=10240,
        d_kv=64,
        num_layers=_UMT5_NUM_LAYERS,
        num_heads=64,
        vocab_size=256384,
        relative_attention_num_buckets=32,
        relative_attention_max_distance=128,
        is_gated_act=True,
        dense_act_fn="gelu_new",
        layer_norm_epsilon=1e-6,
        tie_word_embeddings=False,
    )
    model = UMT5EncoderModel(config)
    model.load_state_dict(_remap_umt5(raw), strict=True)
    return model.to(device=device, dtype=dtype).eval()


def load_raw_tokenizer(model_path: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(os.path.join(model_path, "google", "umt5-xxl"))


def build_raw_transformer(
    model_path: str,
    *,
    sink_size: int = V2_SINK_SIZE,
    local_attn_size: int = V2_LOCAL_ATTN_SIZE,
) -> Any:
    """Construct the DiT from the raw ``config.json`` without loading weights.

    In the engine path the weights are loaded by ``DiffusersPipelineLoader``
    from the raw layout's ``transformers/`` safetensors (declared via a
    ``ComponentSource``), which also handles TP sharding.
    """
    from vllm_omni.diffusion.models.lingbot_world.lingbot_world_transformer import (
        CausalLingBotWorldTransformer3DModel,
    )

    raw_config = json.load(open(os.path.join(model_path, "config.json")))
    return CausalLingBotWorldTransformer3DModel(
        num_attention_heads=raw_config["num_heads"],
        attention_head_dim=raw_config["dim"] // raw_config["num_heads"],
        in_channels=raw_config["in_dim"],
        out_channels=raw_config["out_dim"],
        ffn_dim=raw_config["ffn_dim"],
        freq_dim=raw_config["freq_dim"],
        num_layers=raw_config["num_layers"],
        eps=raw_config["eps"],
        sink_size=sink_size,
        local_attn_size=local_attn_size,
        prefix="transformer",
    )


def load_raw_transformer(
    model_path: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
    sink_size: int = V2_SINK_SIZE,
    local_attn_size: int = V2_LOCAL_ATTN_SIZE,
) -> Any:
    """Standalone helper: build the DiT and eagerly load weights (no engine)."""
    from safetensors.torch import load_file

    model = build_raw_transformer(model_path, sink_size=sink_size, local_attn_size=local_attn_size)
    model = model.to(device=device, dtype=dtype)
    ckpt: dict[str, torch.Tensor] = {}
    for shard in sorted(glob.glob(os.path.join(model_path, "transformers", "*.safetensors"))):
        ckpt.update(load_file(shard, device="cpu"))
    model.load_weights((name, tensor.to(dtype)) for name, tensor in ckpt.items())
    return model.eval()
