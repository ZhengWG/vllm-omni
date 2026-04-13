# SPDX-License-Identifier: Apache-2.0
"""Diffusion-side components for Ming-flash-omni-2.0 text-to-image.

Phase 1 contents:
- ``condition_encoder``: Qwen2 connector + optional RMSNorm that turns
  thinker-side hidden states into DiT conditioning embeds (cap_feat_dim=2560).
- ``ming_imagegen_model``: top-level wrapper wiring condition encoder +
  ``ZImagePipeline`` (transformer + VAE + FlowMatchEulerDiscreteScheduler).

The pipeline itself is NOT subclassed — ``ZImagePipeline.forward`` already
accepts pre-computed ``prompt_embeds``, so we can feed our Ming-conditioned
embeds directly and bypass the original text encoder path.
"""
