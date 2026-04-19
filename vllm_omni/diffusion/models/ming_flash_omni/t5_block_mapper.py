# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.

"""T5EncoderBlockByT5Mapper — Ming's per-block T5 stack mapping byte5 features
onto the DiT condition space.

Ported from Ming's ``bizgen/custom_diffusers/models/byt5_block_byt5_mapper.py``.
Uses HF transformers' T5 internals (``T5LayerSelfAttention`` / ``T5LayerFF`` /
``T5LayerNorm``) instead of Ming's vendored ``modeling_t5`` copy — semantically
equivalent, no extra vendored code.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from diffusers.models.modeling_utils import ModelMixin
from transformers.models.t5.modeling_t5 import T5LayerFF, T5LayerNorm, T5LayerSelfAttention


class _T5EncoderBlock(nn.Module):
    def __init__(self, config, has_relative_attention_bias: bool = False) -> None:
        super().__init__()
        self.layer = nn.ModuleList()
        self.layer.append(T5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        self.layer.append(T5LayerFF(config))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_bias: torch.Tensor | None = None,
        query_length: int | None = None,
        layer_head_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
    ):
        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            query_length=query_length,
            layer_head_mask=layer_head_mask,
            past_key_value=None,
            use_cache=False,
            output_attentions=output_attentions,
        )
        hidden_states = self_attention_outputs[0]
        attn_extra = self_attention_outputs[2:]

        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        hidden_states = self.layer[-1](hidden_states)

        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        return (hidden_states,) + attn_extra


class T5EncoderBlockByT5Mapper(ModelMixin):
    """Stacks ``num_layers`` T5 encoder blocks on top of byte5 features and
    projects them to ``sdxl_channels`` (= Ming's ``diffusion_c_input_dim``).
    """

    def __init__(self, byte5_config, num_layers: int, sdxl_channels: int | None = None) -> None:
        super().__init__()
        if num_layers > 0:
            self.blocks = nn.ModuleList(
                [_T5EncoderBlock(byte5_config, has_relative_attention_bias=(i == 0)) for i in range(num_layers)]
            )
        else:
            self.blocks = None
        self.layer_norm = T5LayerNorm(byte5_config.d_model, eps=byte5_config.layer_norm_epsilon)
        if sdxl_channels is not None:
            self.channel_mapper = nn.Linear(byte5_config.d_model, sdxl_channels)
            self.final_layer_norm = T5LayerNorm(sdxl_channels, eps=byte5_config.layer_norm_epsilon)
        else:
            self.channel_mapper = None
            self.final_layer_norm = None

    def get_extended_attention_mask(self, attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        if attention_mask.dim() == 3:
            extended = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended = attention_mask[:, None, None, :]
        else:
            raise ValueError(f"Unexpected attention_mask shape {tuple(attention_mask.shape)}")
        extended = extended.to(dtype=dtype)
        return (1.0 - extended) * torch.finfo(dtype).min

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        extended_mask = self.get_extended_attention_mask(attention_mask, dtype=self.dtype)

        hidden_states = inputs_embeds
        position_bias = None
        query_length = inputs_embeds.shape[1]

        if self.blocks is not None:
            for block in self.blocks:
                layer_outputs = block(
                    hidden_states,
                    attention_mask=extended_mask,
                    position_bias=position_bias,
                    query_length=query_length,
                )
                hidden_states, position_bias = layer_outputs[0], layer_outputs[1]

        hidden_states = self.layer_norm(hidden_states)
        if self.channel_mapper is not None:
            hidden_states = self.channel_mapper(hidden_states)
            hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states


__all__ = ["T5EncoderBlockByT5Mapper"]
