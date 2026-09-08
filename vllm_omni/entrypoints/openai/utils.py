# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from vllm_omni.lora.request import LoRARequest
from vllm_omni.lora.utils import stable_lora_int_id


def get_stage_type(stage_cfg: Any) -> str:
    """Best-effort stage type resolver across dict/omegaconf/object configs."""
    if isinstance(stage_cfg, dict):
        return stage_cfg.get("stage_type", "llm")
    if hasattr(stage_cfg, "get"):
        try:
            return stage_cfg.get("stage_type", "llm")
        except Exception:
            pass
    return getattr(stage_cfg, "stage_type", "llm")


def _stage_final_output(stage: Any) -> tuple[bool, str | None]:
    """Return ``(final_output, final_output_type)`` from a stage config."""
    if isinstance(stage, dict):
        return bool(stage.get("final_output", False)), stage.get("final_output_type")
    if hasattr(stage, "get"):
        try:
            return bool(stage.get("final_output", False)), stage.get("final_output_type")
        except Exception:
            pass
    return bool(getattr(stage, "final_output", False)), getattr(stage, "final_output_type", None)


def is_video_generation_pipeline(stage_configs: list[Any] | None) -> bool:
    """Return whether a pipeline declares a final video output stage."""
    for stage in stage_configs or ():
        final_output, final_output_type = _stage_final_output(stage)
        if final_output and final_output_type in {"video", "videos"}:
            return True
    return False


def is_image_generation_pipeline(stage_configs: list[Any] | None) -> bool:
    """Return whether a pipeline declares a final image output stage."""
    for stage in stage_configs or ():
        final_output, final_output_type = _stage_final_output(stage)
        if final_output and final_output_type in {"image", "images"}:
            return True
    return False


def pipeline_supports_image_generations(stage_configs: list[Any] | None) -> bool:
    """Return whether ``/v1/images/*`` can accept this pipeline.

    Diffusion-typed stages keep the historical serving path. LLM generation
    stages that declare a final image output (for example MammothModa2 DiT or
    Dynin token2image) are also accepted.
    """
    for stage in stage_configs or ():
        if get_stage_type(stage) == "diffusion":
            return True
    return is_image_generation_pipeline(stage_configs)


def image_generation_stage_index(stage_configs: list[Any] | None) -> int | None:
    """Return the stage used for image default sampling, if any.

    Prefer a diffusion-typed stage, then a declared final image-output stage.
    """
    stages = list(stage_configs or ())
    for i, stage in enumerate(stages):
        if get_stage_type(stage) == "diffusion":
            return i
    for i, stage in enumerate(stages):
        final_output, final_output_type = _stage_final_output(stage)
        if final_output and final_output_type in {"image", "images"}:
            return i
    return None


def parse_lora_request(lora_body: Any) -> tuple[LoRARequest | None, float | None]:
    """Parse a request-level LoRA object into a LoRARequest and optional scale.

    Raises:
        ValueError: If the object shape is invalid or required fields are missing.
    """
    if lora_body is None:
        return None, None

    if not isinstance(lora_body, dict):
        raise ValueError("Invalid lora field: expected an object.")

    lora_name = lora_body.get("name") or lora_body.get("lora_name") or lora_body.get("adapter")
    lora_path = (
        lora_body.get("local_path")
        or lora_body.get("path")
        or lora_body.get("lora_path")
        or lora_body.get("lora_local_path")
    )
    lora_scale = lora_body.get("scale")
    if lora_scale is None:
        lora_scale = lora_body.get("lora_scale")
    lora_int_id = lora_body.get("int_id")
    if lora_int_id is None:
        lora_int_id = lora_body.get("lora_int_id")
    if lora_int_id is None and lora_path:
        lora_int_id = stable_lora_int_id(str(lora_path))

    if not lora_name or not lora_path:
        raise ValueError("Invalid lora object: both name and path are required.")

    scale = float(lora_scale) if lora_scale is not None else None
    return LoRARequest(str(lora_name), int(lora_int_id), str(lora_path)), scale


def get_supported_speakers_from_hf_config(hf_config: Any) -> set[str]:
    """Extract supported speaker names from a model hf_config."""
    config = (
        hf_config.get("talker_config") if isinstance(hf_config, dict) else getattr(hf_config, "talker_config", None)
    )
    if config is None:
        return set()

    for spk_attr in ("speaker_id", "spk_id"):
        speakers_dict = config.get(spk_attr) if isinstance(config, dict) else getattr(config, spk_attr, None)
        if speakers_dict and isinstance(speakers_dict, dict):
            return {speaker.lower() for speaker in speakers_dict}
    return set()


def resolve_diffusion_od_config(engine_client: Any, diffusion_engine: Any = None) -> Any:
    """Resolve the OmniDiffusionConfig from the engine or diffusion engine."""
    od_config = None
    if hasattr(engine_client, "get_diffusion_od_config"):
        od_config = engine_client.get_diffusion_od_config()
    if od_config is None and diffusion_engine is not None:
        if hasattr(diffusion_engine, "get_diffusion_od_config"):
            od_config = diffusion_engine.get_diffusion_od_config()
        else:
            od_config = getattr(diffusion_engine, "od_config", None)
    return od_config


def is_single_stage_diffusion(engine_client: Any) -> bool:
    """Return True if the engine is a single-stage diffusion pipeline."""
    stage_configs = getattr(engine_client, "stage_configs", None) or []
    if len(stage_configs) != 1:
        return False
    return getattr(stage_configs[0], "stage_type", None) in ("diffusion", "DIFFUSION")


def validate_requested_speaker(speaker: str | None, supported_speakers: set[str]) -> str | None:
    """Normalize and validate an optional speaker value.

    Returns the normalized speaker string when provided, otherwise ``None``.
    Raises ``ValueError`` when the speaker is not in the supported list.
    """
    if not isinstance(speaker, str) or not speaker.strip():
        return None

    normalized = speaker.lower().strip()
    if supported_speakers and normalized not in supported_speakers:
        raise ValueError(f"Invalid speaker '{speaker}'. Supported: {', '.join(sorted(supported_speakers))}")
    return normalized
