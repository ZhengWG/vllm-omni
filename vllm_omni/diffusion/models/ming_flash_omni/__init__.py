# SPDX-License-Identifier: Apache-2.0
"""Diffusion-side components for Ming-flash-omni-2.0 image generation.

Modules:
- ``condition_encoder``: Qwen2 connector + optional RMSNorm that turns
  thinker-side hidden states into DiT conditioning embeds.
- ``byte5_encoder`` / ``t5_block_mapper``: optional ByT5-based glyph
  encoder used as an auxiliary conditioning path.
- ``ming_zimage_transformer``: ZImage DiT specialisation with the Ming
  ref-image fusion hooks.
- ``pipeline_ming``: top-level ``MingImagePipeline`` (registered under
  the same name in ``vllm_omni.diffusion.registry``).
"""
