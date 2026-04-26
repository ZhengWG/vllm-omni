# Ming-flash-omni 2.0 for omni-speech chat and standalone TTS

## Summary

- Vendor: inclusionAI
- Model: `Jonathan1909/Ming-flash-omni-2.0`
- Task: Multimodal chat with text, image, audio, or video input; standalone text-to-speech (TTS);
and image generation
- Mode: Online serving with the OpenAI-compatible API
- Maintainer: Community

## When to use this recipe

Use this recipe when you want a known-good starting point for serving
`Jonathan1909/Ming-flash-omni-2.0` with vLLM-Omni in one of four modes:

- **Thinker only** — multimodal understanding with text output.
- **Thinker + Talker (omni-speech)** — multimodal understanding with text and spoken output.
- **Talker only (TTS)** — standalone text-to-speech via the OpenAI `/v1/audio/speech` endpoint.
- **Thinker + Diffusion (image generation)** — text-to-image and image
  edit via the OpenAI `/v1/chat/completions` endpoint with
  `modalities: ["image"]`.

## References

- Upstream model:
  [`inclusionAI/Ming`](https://github.com/inclusionAI/Ming)
- For offline inference and additional client variants, see
  `examples/offline_inference/ming_flash_omni{,_tts}/` and
  `examples/online_serving/ming_flash_omni{,_tts}/`.


## Hardware Support

This recipe documents reference GPU configurations for the two-stage
omni-speech deployment and the standalone TTS deployment.
Other hardware and configurations are welcome as community validation lands.

## GPU

### 4x H100 80GB — omni-speech/chat (thinker + talker)

The bundled `ming_flash_omni.yaml` runs the thinker with tensor parallel size
4 on GPUs 0–3 and the talker on GPU 3.
Adjust `devices` in the YAML to match your hardware.

#### Environment

- OS: Linux
- Python: 3.10+
- CUDA Driver Version: 590.48.01
- CUDA 12.5
- vLLM version: 0.19.0
- vLLM-Omni version or commit: 0.19.0rc1

#### Command

Thinker only (text output):

```bash
vllm serve Jonathan1909/Ming-flash-omni-2.0 --omni --port 8091
```

Thinker + talker (text and/or audio output):

```bash
vllm serve Jonathan1909/Ming-flash-omni-2.0 \
    --omni \
    --port 8091 \
    --stage-configs-path vllm_omni/model_executor/stage_configs/ming_flash_omni.yaml \
    --log-stats
```

`--log-stats` is optional but recommended while validating the deployment.

#### Verification

Text output from a multimodal (image) input:

```bash
curl http://localhost:8091/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "Jonathan1909/Ming-flash-omni-2.0",
      "messages": [
        {"role": "system", "content": [{"type": "text", "text": "你是一个友好的AI助手。\n\ndetailed thinking off"}]},
        {"role": "user", "content": [
          {"type": "image_url", "image_url": {"url": "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/cherry_blossom.jpg"}},
          {"type": "text", "text": "Describe this image in detail."}
        ]}
      ],
      "modalities": ["text"]
    }'
```

Spoken response from a text query (save the WAV bytes):

```bash
curl http://localhost:8091/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "Jonathan1909/Ming-flash-omni-2.0",
      "messages": [
        {"role": "system", "content": [{"type": "text", "text": "你是一个友好的AI助手。\n\ndetailed thinking off"}]},
        {"role": "user", "content": "请详细介绍鹦鹉的生活习性。"}
      ],
      "modalities": ["audio"]
    }' | jq -r '.choices[0].message.audio.data' | base64 -d > ming_omni_parrot.wav
```

Text + audio output from an audio input (swap `audio_url` for `video_url`
or `image_url` to exercise the other multimodal input paths):

```bash
curl http://localhost:8091/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "Jonathan1909/Ming-flash-omni-2.0",
      "messages": [
        {"role": "system", "content": [{"type": "text", "text": "你是一个友好的AI助手。\n\ndetailed thinking off"}]},
        {"role": "user", "content": [
          {"type": "audio_url", "audio_url": {"url": "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/mary_had_lamb.ogg"}},
          {"type": "text", "text": "Please recognize the language of this speech and transcribe it. Format: oral."}
        ]}
      ],
      "modalities": ["text", "audio"]
    }' | jq -r '.choices[0].message.content'
```

Streaming text output via SSE (set `"stream": true`):

```bash
curl -N http://localhost:8091/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "Jonathan1909/Ming-flash-omni-2.0",
      "messages": [
        {"role": "system", "content": [{"type": "text", "text": "你是一个友好的AI助手。\n\ndetailed thinking off"}]},
        {"role": "user", "content": "请详细介绍鹦鹉的生活习性。"}
      ],
      "modalities": ["text"],
      "stream": true
    }'
```

Each SSE event carries a `data:` line with a chat-completion chunk; text
deltas appear at `choices[0].delta.content`.

#### Notes

- Output modality is selected by the request body: `"modalities": ["text"]`,
  `["audio"]`, or `["text", "audio"]`. The two-stage omni-speech server must be launched
  for any request containing `audio`.
- Reasoning mode: flip the system prompt suffix from `detailed thinking off`
  to `detailed thinking on` in any request above.
- Memory usage: size depends on output modalities and multimodal input; leave
  headroom for video frames and audio caches.

### 1x H100 80GB — standalone TTS (talker only)

The bundled `ming_flash_omni_tts.yaml` runs the talker on a single GPU and exposes the OpenAI `/v1/audio/speech` endpoint.

#### Environment

- OS: Linux
- Python: 3.10+
- CUDA Driver Version: 590.48.01
- CUDA 12.5
- vLLM version: 0.19.0
- vLLM-Omni version or commit: 0.19.0rc1

#### Command

```bash
vllm serve Jonathan1909/Ming-flash-omni-2.0 \
    --omni \
    --stage-configs-path vllm_omni/model_executor/stage_configs/ming_flash_omni_tts.yaml \
    --port 8091 \
    --log-stats
```

`--log-stats` is optional but recommended while validating the deployment.

#### Verification

Basic curl:

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
      "model": "Jonathan1909/Ming-flash-omni-2.0",
      "input": "我会一直在这里陪着你。",
      "response_format": "wav"
    }' --output ming_online.wav
```

Speaker selection (e.g. `lingguang`):

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
      "model": "Jonathan1909/Ming-flash-omni-2.0",
      "input": "春天来了，万物复苏，大地一片生机盎然。田野里的油菜花开得金灿灿的，蜜蜂在花丛中忙碌地采蜜。远处的山坡上，桃花和杏花竞相绽放，粉的白的交织在一起，美不胜收。清晨的微风带着泥土的芬芳，轻轻拂过脸颊，让人感到无比惬意。孩子们在田间小路上追逐嬉戏，老人们坐在门前晒太阳，享受着这份宁静与美好。",
      "speaker": "lingguang",
      "response_format": "wav"
    }' --output ming_online_lingguang.wav
```

#### Notes

- The OpenAI `instructions` field is forwarded to the talker as the caption JSON — pass a raw string for `风格` (style) only, or a JSON-encoded object for multiple entries such as `方言` (dialect) and `情感` (emotion).

### 5x H100 80GB — image generation (thinker + diffusion)

The bundled `ming_flash_omni_dual.yaml` runs the AR thinker with tensor
parallel size 4 on GPUs 0–3 and the diffusion stage (`MingImagePipeline`)
on GPU 4. Adjust `devices` in the YAML for your hardware. Image
generation can be co-located with the thinker (e.g. `devices: "3"` on
the diffusion stage) at the cost of higher peak memory.

#### Environment

- OS: Linux
- Python: 3.10+
- CUDA Driver Version: 590.48.01
- CUDA 12.5
- vLLM version: 0.19.0
- vLLM-Omni version or commit: 0.19.0rc1

#### Command

```bash
vllm serve Jonathan1909/Ming-flash-omni-2.0 \
    --omni \
    --stage-configs-path vllm_omni/model_executor/stage_configs/ming_flash_omni_dual.yaml \
    --trust-remote-code \
    --port 8188 \
    --log-stats
```

`--log-stats` is optional but recommended while validating the deployment.

#### Verification

Text-to-image:

```bash
curl http://127.0.0.1:8188/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Jonathan1909/Ming-flash-omni-2.0",
        "messages": [{"role": "user", "content": "Please draw a cute cat."}],
        "modalities": ["image"]
    }' -o /tmp/ming_response.json

python -c "
import base64, json
r = json.load(open('/tmp/ming_response.json'))
url = r['choices'][0]['message']['content'][0]['image_url']['url']
png = base64.b64decode(url.split(',')[1])
open('/tmp/ming_cat.png', 'wb').write(png)
print('PNG bytes:', len(png))
"
```

Image edit (img2img) — include a reference image and an edit instruction:

```bash
curl http://127.0.0.1:8188/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Jonathan1909/Ming-flash-omni-2.0",
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,'"$(base64 -w0 ./input.jpg)"'"}},
            {"type": "text", "text": "Turn this into a watercolour painting."}
        ]}],
        "modalities": ["image"]
    }' -o /tmp/ming_edit.json
```

Optional generation knobs (passed via `extra_body`):

| Field | Default | Notes |
|-------|---------|-------|
| `height`, `width` | 1024×1024 | Target image resolution. |
| `num_inference_steps` | 30 | DiT denoise step count. |
| `negative_prompt` | None | Triggers the CFG companion via `expand_cfg_prompts`. |
| `cfg_text_scale` | 2.0 | Classifier-free-guidance scale. |
| `seed` | request seed | Deterministic generation. |

#### Notes

- Image generation is requested by the chat client via
  `"modalities": ["image"]`. The dual-stage server in this section
  must be launched for any image-output request — the thinker-only or
  omni-speech servers do not load the diffusion stage.
- When a request carries a `negative_prompt`, the AR thinker spawns one
  CFG companion in parallel (see
  `vllm_omni/model_executor/stage_input_processors/ming_flash_omni.py::expand_cfg_prompts`).
  Both the parent's and the companion's hidden states are forwarded to
  `MingImagePipeline` for true CFG.
- Image edit (img2img) requests prepend an `<IMAGE>` placeholder to the
  text prompt so the thinker's ref-image substitution still fires when
  the multimodal cache is warm.
