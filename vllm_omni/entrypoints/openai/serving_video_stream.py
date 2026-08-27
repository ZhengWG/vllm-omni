# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Qwen-Omni streaming video WebSocket handler.

Accepts video frames incrementally via WebSocket, buffers them, and
generates text + optional audio responses using the Qwen3-Omni multi-stage
pipeline (thinker -> talker -> code2wav).

Protocol:
    Client -> Server:
        {"type": "session.config", ...}         # Session config (sent once)
        {"type": "video.frame", "data": "..."}  # base64 JPEG/PNG frame
        {"type": "audio.chunk", "data": "..."}  # base64 PCM16 16kHz mono
        {"type": "video.query", "text": "..."}  # Submit query about buffered frames
        {"type": "video.done"}                  # End of session

    Server -> Client:
        {"type": "response.start"}
        {"type": "response.text.delta", "delta": "..."}
        {"type": "response.text.done", "text": "..."}
        {"type": "response.audio.delta", "data": "...", "format": "wav"}
        {"type": "response.audio.done"}
        {"type": "session.done"}
        {"type": "error", "message": "..."}
"""

from __future__ import annotations

from typing import Any

from vllm_omni.entrypoints.openai.video_stream_base import (
    _BAD_FRAME,
    _DEFAULT_CONFIG_TIMEOUT,
    _DEFAULT_IDLE_TIMEOUT,
    StreamingVideoSessionConfig,
    VideoStreamTurnTrigger,
)
from vllm_omni.entrypoints.openai.video_stream_base import (
    OmniStreamingVideoHandler as OmniStreamingVideoHandlerBase,
)

__all__ = [
    "QwenOmniStreamingVideoHandler",
    "StreamingVideoSessionConfig",
    "create_streaming_video_handler",
]


class QwenOmniStreamingVideoHandler(OmniStreamingVideoHandlerBase):
    """Qwen-Omni pipeline: manual ``video.query`` trigger and image_pil prompts."""

    def should_trigger_turn(self, trigger: VideoStreamTurnTrigger) -> bool:
        return False

    @staticmethod
    def _build_frame_image_parts(
        frames: list[str],
        prewarmed_frames: dict[str, tuple[Any, str]] | None,
    ) -> list[dict[str, Any]]:
        """Frame parts with arrival-time uuids so warmup and query prompts
        hash every frame identically (a hash mismatch would void the warmed
        prefix from that frame onward)."""
        prewarmed = prewarmed_frames or {}
        parts: list[dict[str, Any]] = []
        for frame_b64 in frames:
            cached = prewarmed.get(frame_b64)
            if cached is _BAD_FRAME:
                continue
            if cached is not None:
                pil, mm_uuid = cached
                if pil is not None:
                    parts.append(
                        {
                            "type": "image_pil",
                            "image_pil": pil,
                            "uuid": mm_uuid,
                        }
                    )
                    continue
                # PIL prewarm not finished: same uuid over the base64 payload.
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"},
                        "uuid": mm_uuid,
                    }
                )
                continue
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"},
                }
            )
        return parts

    def build_engine_prompt(
        self,
        config: StreamingVideoSessionConfig,
        frame_buffer: list[str],
        audio_buffer: bytearray,
        message_history: list[dict[str, Any]],
        query_text: str,
        prewarmed_frames: dict[str, tuple[Any, str]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if self._incremental_prefill_active(config):
            # Frames are sampled at arrival time only (EVS + max_frames), so
            # the buffer already is the warmed sequence. Query-time
            # subsampling would pick a subset, which is not a prefix.
            frames = list(frame_buffer)
        else:
            n_buf = len(frame_buffer)
            if n_buf <= config.num_frames:
                frames = list(frame_buffer)
            else:
                stride = max(1, n_buf // config.num_frames)
                idx = [i * stride for i in range(config.num_frames - 1)] + [n_buf - 1]
                frames = [frame_buffer[i] for i in idx]

        user_content: list[dict] = self._build_frame_image_parts(frames, prewarmed_frames)

        if len(audio_buffer) > 0:
            wav_b64 = self._pcm_to_wav_b64(bytes(audio_buffer))
            user_content.append(
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": wav_b64,
                        "format": "wav",
                    },
                }
            )

        if query_text:
            user_content.append({"type": "text", "text": query_text})

        user_message: dict[str, Any] = {"role": "user", "content": user_content}

        messages = self._history_prefix_messages(config, message_history)
        messages.append(user_message)

        return messages, user_message

    def _history_prefix_messages(
        self,
        config: StreamingVideoSessionConfig,
        message_history: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Shared prompt prefix (system + compressed text-only history).

        Warmup and query MUST build this identically: any divergence before
        the frame parts voids the warmed KV from that token onward.
        """
        messages: list[dict[str, Any]] = []
        if config.system_prompt:
            messages.append({"role": "system", "content": config.system_prompt})
        recent_history = message_history[-2:] if len(message_history) > 2 else message_history
        for hist_msg in recent_history:
            messages.append(self._text_only_message(hist_msg))
        return messages

    def on_turn_complete(
        self,
        message_history: list[dict[str, Any]],
        user_message: dict[str, Any],
        response_text: str,
    ) -> None:
        message_history.append(user_message)
        message_history.append({"role": "assistant", "content": response_text})

    def build_warmup_messages(
        self,
        config: StreamingVideoSessionConfig,
        frame_buffer: list[str],
        message_history: list[dict[str, Any]],
        prewarmed_frames: dict[str, tuple[Any, str]],
    ) -> list[dict[str, Any]] | None:
        """History prefix + a frames-only user turn: a strict prefix (up to
        the last vision token) of the prompt the next query will submit."""
        frame_parts = self._build_frame_image_parts(frame_buffer, prewarmed_frames)
        if not frame_parts:
            return None

        messages = self._history_prefix_messages(config, message_history)
        messages.append({"role": "user", "content": frame_parts})
        return messages

    _build_messages = build_engine_prompt


def create_streaming_video_handler(
    chat_service: Any,
    idle_timeout: float = _DEFAULT_IDLE_TIMEOUT,
    config_timeout: float = _DEFAULT_CONFIG_TIMEOUT,
    engine_client: Any | None = None,
) -> OmniStreamingVideoHandlerBase:
    """Create the handler for ``/v1/video/chat/stream``.

    Returns :class:`QwenOmniStreamingVideoHandler` today. Additional pipelines
    can be selected here in follow-up PRs.
    """
    return QwenOmniStreamingVideoHandler(
        chat_service=chat_service,
        idle_timeout=idle_timeout,
        config_timeout=config_timeout,
        engine_client=engine_client,
    )
