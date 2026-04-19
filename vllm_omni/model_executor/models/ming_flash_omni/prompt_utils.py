# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.

"""Ming-flash-omni-2.0 prompt utilities.

String-level helpers that encode Ming-specific prompt conventions
(image-gen query-token block, etc.). Kept in the Ming module so that
nothing outside needs to know the actual token literals.
"""

from __future__ import annotations

# Ming's thinker uses these tokens to mark a learnable image-generation
# query block inside the text prompt. The thinker substitutes its
# ``query_tokens_dict`` embeddings at each ``<imagePatch>`` position during
# forward; see ``MingFlashOmniThinker._maybe_inject_image_gen_query_embeds``.
_IMAGE_OPEN_TOKEN = "<image>"
_IMAGE_CLOSE_TOKEN = "</image>"
IMAGE_PATCH_TOKEN = "<imagePatch>"

# Default query-token count matches ``MingImageGenConfig(img_gen_scales=[16])``
# (16 * 16 = 256), which is what the released inclusionAI/Ming-flash-omni-2.0
# checkpoint ships.
DEFAULT_NUM_QUERY_TOKENS = 256


def maybe_expand_image_gen_prompt(
    prompt: str,
    num_query_tokens: int = DEFAULT_NUM_QUERY_TOKENS,
) -> str:
    """Append the ``<image><imagePatch>*N</image>`` suffix for text-to-image.

    The thinker expects image-generation requests to end with an N-wide
    block of ``<imagePatch>`` tokens (wrapped in ``<image>``/``</image>``)
    — those positions get substituted with learnable
    ``query_tokens_dict`` embeddings during forward.

    This helper is a no-op (returns the input unchanged) when:

      * ``prompt`` is not a non-empty string, or
      * the prompt already contains an ``<imagePatch>`` block (avoids
        double expansion for tests / manual calls that pre-format the
        prompt).

    Args:
        prompt: Raw user prompt text.
        num_query_tokens: Total number of query tokens to emit. Defaults
            to 256 (the released checkpoint's ``img_gen_scales=[16]``).

    Returns:
        The (possibly expanded) prompt text.
    """
    if not isinstance(prompt, str) or not prompt:
        return prompt
    if IMAGE_PATCH_TOKEN in prompt:
        return prompt

    # TODO(multi-scale): single-block emission assumes img_gen_scales=[16].
    suffix = _IMAGE_OPEN_TOKEN + (IMAGE_PATCH_TOKEN * num_query_tokens) + _IMAGE_CLOSE_TOKEN
    return prompt + suffix


__all__ = [
    "IMAGE_PATCH_TOKEN",
    "DEFAULT_NUM_QUERY_TOKENS",
    "maybe_expand_image_gen_prompt",
]
