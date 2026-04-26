# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.

"""Ming-specific subclass of ZImageTransformer2DModel that supports ``ref_x``.

Ming's img2img path concatenates a VAE-encoded reference latent along the
frame axis before patchification, then drops the reference portion from the
unpatchified output. This is a surgical override — everything else (attention,
RoPE, final layer) stays on the parent implementation.

The reference latent is stashed on the instance via ``set_ref_latent`` before
running the denoise loop (rather than threading it through the call signature),
so the parent pipeline's denoise loop can stay untouched.
"""

from __future__ import annotations

import torch

from vllm_omni.diffusion.models.z_image.z_image_transformer import ZImageTransformer2DModel


class MingZImageTransformer2DModel(ZImageTransformer2DModel):
    """ZImage DiT with Ming's reference-latent support.

    Usage::

        transformer.set_ref_latent(vae_encoded_ref)  # [1, C, H, W]
        # ... run diffusion pipeline ...
        transformer.set_ref_latent(None)

    When a ref latent is set, every ``forward`` call concatenates it (broadcast
    to the incoming batch count) onto the frame axis of each image item, and
    the extra frame is dropped at unpatchify time.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._pending_ref_latent: torch.Tensor | None = None  # [1, C, H, W]

    def set_ref_latent(self, ref_latent: torch.Tensor | None) -> None:
        self._pending_ref_latent = ref_latent

    def forward(
        self,
        x: list[torch.Tensor],
        t,
        cap_feats: list[torch.Tensor],
        patch_size=2,
        f_patch_size=1,
    ):
        ref_latent = self._pending_ref_latent
        if ref_latent is not None:
            per_item = ref_latent[0].unsqueeze(1).to(dtype=x[0].dtype, device=x[0].device)  # [C, 1, H, W]
            x = [torch.cat([img, per_item], dim=1) for img in x]
        return super().forward(x, t, cap_feats, patch_size=patch_size, f_patch_size=f_patch_size)

    def unpatchify(
        self,
        x: list[torch.Tensor],
        size: list[tuple],
        patch_size,
        f_patch_size,
    ) -> list[torch.Tensor]:
        out = super().unpatchify(x, size, patch_size, f_patch_size)
        # No-op when F==1 (pure t2i); drops the reference-frame prediction when F==2.
        return [t[:, :1, :, :] for t in out]


__all__ = ["MingZImageTransformer2DModel"]
