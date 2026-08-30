# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ming-flash-omni talker prompt building must follow the model dtype."""

from __future__ import annotations

import pytest

from vllm_omni.model_executor.models.ming_flash_omni.talker_module import build_tts_input

torch = pytest.importorskip("torch")

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_HIDDEN = 8
_VOCAB = 64


class _StubTokenizer:
    """Maps each distinct string to one stable token id."""

    def __init__(self) -> None:
        self._ids: dict[str, int] = {}

    def encode(self, text: str) -> list[int]:
        return [self._ids.setdefault(text, len(self._ids) + 1)]


def _build(dtype: torch.dtype, *, spk_emb=None):
    embed_tokens = torch.nn.Embedding(_VOCAB, _HIDDEN).to(dtype)
    return build_tts_input(
        tokenizer=_StubTokenizer(),
        embed_tokens=embed_tokens,
        device=torch.device("cpu"),
        dtype=dtype,
        text="hello world",
        prompt="say it",
        spk_emb=spk_emb,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_embeddings_follow_the_requested_dtype(dtype: torch.dtype):
    inputs_embeds, input_ids = _build(dtype)

    assert inputs_embeds.dtype == dtype
    assert input_ids.dtype == torch.long
    assert inputs_embeds.shape[:2] == (1, input_ids.shape[1])


def test_speaker_embedding_injection_preserves_the_requested_dtype():
    dtype = torch.float32
    spk_emb = [torch.full((_HIDDEN,), 0.5, dtype=dtype)]

    inputs_embeds, _ = _build(dtype, spk_emb=spk_emb)

    assert inputs_embeds.dtype == dtype
    # The speaker vector lands in the slot after the <|vision_start|> marker.
    assert (inputs_embeds == 0.5).any()


def test_speaker_embedding_is_truncated_when_the_prompt_dtype_disagrees():
    """Why the caller must pass the model dtype rather than a hardcoded one.

    The talker projects speaker embeddings in ``self.dtype``; if the prompt
    buffer is built in a narrower dtype the vector is silently rounded on the
    way in, on top of feeding the backbone embeddings it was not compiled for.
    """
    value = 0.3
    spk_emb = [torch.full((_HIDDEN,), value, dtype=torch.float32)]

    narrow, _ = _build(torch.bfloat16, spk_emb=spk_emb)
    matched, _ = _build(torch.float32, spk_emb=spk_emb)

    assert not (narrow.float() == value).any()
    assert (matched == value).any()
