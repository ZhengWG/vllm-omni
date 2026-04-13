# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.
#
# Adapted from Ming repository (inclusionAI/Ming) — the ``get_condition_embeds
# _for_image_gen`` path in modeling_bailingmm2.py.

"""Ming-flash-omni-2.0 condition encoder for image generation.

Pipeline (runs inside the imagegen stage):

    thinker hidden states at query-token positions  [B, N, 4096]
                         │
                         ▼
        Qwen2ForCausalLM connector (is_causal=False)
             — loaded from <checkpoint>/connector/
                         │
        final layer hidden states                    [B, N, 1536]
                         │
           RMSNorm  (text_encoder_norm=True)         [B, N, 1536]
                         │
        proj_out (optional; identity when            [B, N, 2560]
                 use_identity_mlp=True in mlp/config.json)
                         │
                         ▼
            cap_feats consumed by ZImageTransformer2DModel
                 (expects cap_feat_dim=2560)

IMPORTANT (to be validated on real hardware):
  - Current implementation assumes ``use_identity_mlp=True`` AND the connector
    hidden_size (1536) aligns with ``diffusion_c_input_dim`` (2560) via a
    single linear ``proj_out``. The real mapping may instead rely on the
    connector's LM head projecting to 2560. The LoadWeights path logs shape
    mismatches loudly so the first real run tells us what to fix.
  - ``text_encoder_norm`` in mlp/config.json is True for the released
    checkpoint. We therefore always apply an RMSNorm here in Phase 1.
  - ``ByT5`` branch is NOT implemented (Phase 2). If enabled in config we
    raise early rather than silently skip.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

from vllm_omni.transformers_utils.configs.ming_flash_omni import MingImageGenConfig

logger = logging.getLogger(__name__)


class MingConditionEncoder(nn.Module):
    """Wraps a Qwen2 connector + norm/projection, producing DiT condition embeds.

    The connector is a ``Qwen2ForCausalLM`` loaded from the ``connector/``
    subfolder of the Ming checkpoint. We run its base model in a non-causal
    (bidirectional) mode, because the connector is used as an encoder over the
    pre-baked query-token hidden states, not as an autoregressive decoder.

    Args:
        image_gen_config: ``MingImageGenConfig`` from ``MingFlashOmniConfig``.
        thinker_hidden_size: Hidden size of the thinker (BailingMoeV2) model.
            Used to build a ``proj_in`` layer when the connector embedding
            dim differs. For the released checkpoint this is 4096.
        device: Placement for the module.
        dtype: Parameter dtype (typically bfloat16 / float16).
    """

    def __init__(
        self,
        image_gen_config: MingImageGenConfig,
        *,
        thinker_hidden_size: int = 4096,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.config = image_gen_config
        self.thinker_hidden_size = thinker_hidden_size
        self._target_device = torch.device(device) if device is not None else None
        self._target_dtype = dtype

        if image_gen_config.enable_byte5:
            raise NotImplementedError("ByT5 text enhancement is Phase 2; set enable_byte5=False in MingImageGenConfig.")

        # Populated lazily by ``load_from_checkpoint`` to keep this module
        # cheap to construct (useful for dummy-init paths and unit tests).
        self.connector: nn.Module | None = None
        self.connector_hidden_size: int | None = None
        self.proj_in: nn.Module = nn.Identity()
        self.proj_out: nn.Module = nn.Identity()
        self.norm: nn.Module = nn.Identity()

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_from_checkpoint(self, model_path: str | Path) -> None:
        """Load the Qwen2 connector + optional projection/norm weights.

        This uses HF transformers directly (not vllm's weight loader) because
        the connector is small (~1.5B params) and only runs once per request
        as an encoder — vllm's distributed loading machinery is overkill.
        """
        from transformers import AutoConfig, Qwen2ForCausalLM

        model_path = Path(model_path)
        connector_path = model_path / self.config.connector_subfolder
        logger.info("[MingConditionEncoder] loading connector from %s", connector_path)

        connector_cfg = AutoConfig.from_pretrained(connector_path, trust_remote_code=True, local_files_only=True)
        # Disable causal masking: the connector is used as a bidirectional
        # encoder over query-token hidden states in image-gen mode.
        connector_cfg.is_decoder = False
        self.connector_hidden_size = int(connector_cfg.hidden_size)
        logger.info(
            "[MingConditionEncoder] connector hidden_size=%d, layers=%d",
            self.connector_hidden_size,
            getattr(connector_cfg, "num_hidden_layers", -1),
        )

        connector = Qwen2ForCausalLM.from_pretrained(
            connector_path,
            config=connector_cfg,
            torch_dtype=self._target_dtype,
            local_files_only=True,
        )
        # Force bidirectional attention in every self-attn module. Qwen2 layers
        # do not branch on ``is_causal`` directly, so we monkey-patch the flag
        # on each attention block (defensive — some transformers versions read
        # ``self_attn.is_causal`` in forward).
        patched = 0
        for module in connector.modules():
            if hasattr(module, "is_causal"):
                module.is_causal = False
                patched += 1
        logger.info("[MingConditionEncoder] disabled is_causal on %d sub-modules", patched)

        # We only need the base encoder (no LM head).
        base = getattr(connector, "model", connector)
        self.connector = base

        # proj_in: align thinker_hidden_size -> connector_hidden_size.
        if self.thinker_hidden_size != self.connector_hidden_size:
            logger.info(
                "[MingConditionEncoder] adding proj_in: %d -> %d",
                self.thinker_hidden_size,
                self.connector_hidden_size,
            )
            self.proj_in = nn.Linear(self.thinker_hidden_size, self.connector_hidden_size, bias=False)
        else:
            self.proj_in = nn.Identity()

        # Norm: text_encoder_norm from mlp/config.json.
        if self.config.text_encoder_norm:
            eps = getattr(connector_cfg, "rms_norm_eps", 1e-6)
            self.norm = nn.RMSNorm(self.connector_hidden_size, eps=eps)
            logger.info("[MingConditionEncoder] using RMSNorm(eps=%g)", eps)

        # proj_out: align connector_hidden_size -> diffusion_c_input_dim.
        c_out = self.config.diffusion_c_input_dim
        if self.config.use_identity_mlp:
            if c_out != self.connector_hidden_size:
                # mlp/config.json says use_identity_mlp=True but the dims
                # don't line up — fall back to a learnable linear and log.
                logger.warning(
                    "[MingConditionEncoder] use_identity_mlp=True but "
                    "connector_hidden_size (%d) != diffusion_c_input_dim (%d); "
                    "inserting a fallback Linear. Verify mlp/ weights on real run.",
                    self.connector_hidden_size,
                    c_out,
                )
                self.proj_out = nn.Linear(self.connector_hidden_size, c_out, bias=False)
            else:
                self.proj_out = nn.Identity()
        else:
            self.proj_out = nn.Linear(self.connector_hidden_size, c_out, bias=False)

        # Attempt to load proj/norm weights from mlp/ subfolder.
        mlp_path = model_path / self.config.mlp_subfolder
        self._load_optional_mlp_weights(mlp_path)

        if self._target_device is not None:
            self.to(self._target_device)
        if self._target_dtype is not None:
            self.to(dtype=self._target_dtype)

        logger.info(
            "[MingConditionEncoder] ready: in=%d -> conn=%d -> out=%d",
            self.thinker_hidden_size,
            self.connector_hidden_size,
            c_out,
        )

    def _load_optional_mlp_weights(self, mlp_path: Path) -> None:
        """Best-effort loader for the mlp/ subfolder (proj + norm weights).

        The exact tensor names inside mlp/ are unknown until we see the real
        checkpoint. We try a few common conventions and log everything.
        """
        if not mlp_path.exists():
            logger.warning(
                "[MingConditionEncoder] mlp/ subfolder missing at %s — proj/norm "
                "will stay randomly initialized. EXPECT BAD IMAGES until this "
                "is fixed on real hardware.",
                mlp_path,
            )
            return

        try:
            from safetensors.torch import load_file  # type: ignore

            candidates = sorted(mlp_path.glob("*.safetensors"))
            if not candidates:
                candidates = sorted(mlp_path.glob("*.bin"))
            if not candidates:
                logger.warning("[MingConditionEncoder] no weight files under %s", mlp_path)
                return
            state: dict[str, torch.Tensor] = {}
            for p in candidates:
                logger.info("[MingConditionEncoder] reading mlp weights: %s", p)
                if p.suffix == ".safetensors":
                    state.update(load_file(str(p)))
                else:
                    state.update(torch.load(str(p), map_location="cpu"))
            logger.info(
                "[MingConditionEncoder] mlp/ keys: %s",
                list(state.keys())[:16],
            )
            # TODO: once we see the real key names, map them onto
            # self.proj_in / self.proj_out / self.norm. For now, log and bail.
            logger.warning(
                "[MingConditionEncoder] mlp/ key mapping is unimplemented in "
                "Phase 1 — using randomly-initialized proj/norm. Patch me once "
                "the real key names are observed on hardware."
            )
        except Exception:  # noqa: BLE001
            logger.exception("[MingConditionEncoder] failed to load mlp/ weights")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        thinker_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode thinker hidden states into DiT condition embeddings.

        Args:
            thinker_hidden_states: ``[B, N, thinker_hidden_size]`` — sliced at
                the learnable query-token positions by the stage input
                processor before being passed here.
            attention_mask: Optional ``[B, N]`` mask. Defaults to all-ones.

        Returns:
            ``[B, N, diffusion_c_input_dim]`` condition tensor ready for the
            ZImage transformer's ``cap_feats`` input.
        """
        if self.connector is None:
            raise RuntimeError("MingConditionEncoder.load_from_checkpoint() must be called before forward().")
        if thinker_hidden_states.dim() != 3:
            raise ValueError(f"expected [B, N, H], got shape {tuple(thinker_hidden_states.shape)}")

        b, n, _ = thinker_hidden_states.shape
        if attention_mask is None:
            attention_mask = torch.ones(
                (b, n),
                dtype=torch.long,
                device=thinker_hidden_states.device,
            )

        x = self.proj_in(thinker_hidden_states)
        # Qwen2 base model accepts inputs_embeds directly, letting us skip the
        # tokenizer/embed-layer path entirely.
        out = self.connector(
            inputs_embeds=x,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden = out.last_hidden_state  # [B, N, conn_hidden]
        hidden = self.norm(hidden)
        cap_feats = self.proj_out(hidden)  # [B, N, diffusion_c_input_dim]
        logger.debug(
            "[MingConditionEncoder.forward] in=%s -> cap_feats=%s",
            tuple(thinker_hidden_states.shape),
            tuple(cap_feats.shape),
        )
        return cap_feats

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @torch.no_grad()
    def zero_negative(
        self,
        cap_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Return a zero tensor shaped like ``cap_feats`` for CFG negatives.

        Phase 1 uses a pure zero embedding as the unconditional branch, which
        is the simplest valid negative for classifier-free guidance.
        """
        return torch.zeros_like(cap_feats)

    def extra_repr(self) -> str:
        return (
            f"thinker_hidden_size={self.thinker_hidden_size}, "
            f"connector_hidden_size={self.connector_hidden_size}, "
            f"diffusion_c_input_dim={self.config.diffusion_c_input_dim}, "
            f"text_encoder_norm={self.config.text_encoder_norm}, "
            f"use_identity_mlp={self.config.use_identity_mlp}"
        )


__all__ = ["MingConditionEncoder"]
