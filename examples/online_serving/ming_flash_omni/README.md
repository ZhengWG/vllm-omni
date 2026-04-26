# Ming-flash-omni 2.0

## Installation

Please refer to [README.md](../../../README.md)

## Deployment modes

| Mode | Launch command | Output |
|------|---------------|--------|
| Thinker only (multimodal understanding) | `vllm serve ... --omni` | Text |
| Thinker + Talker (omni-speech) | `vllm serve ... --omni --stage-configs-path ming_flash_omni.yaml` | Text + Audio |
| Thinker + Diffusion (image generation) | `vllm serve ... --omni --stage-configs-path ming_flash_omni_dual.yaml` | Image |

For standalone TTS (talker only), see [`examples/online_serving/ming_flash_omni_tts/`](../ming_flash_omni_tts/).
For image generation, see the [Image generation](#image-generation-thinker--diffusion) section below.

## Run examples (Ming-flash-omni 2.0)

### Launch the Server

**Thinker only (text output):**
```bash
vllm serve Jonathan1909/Ming-flash-omni-2.0 --omni --port 8091
```

**Thinker + Talker (omni-speech, text + audio output):**
```bash
vllm serve Jonathan1909/Ming-flash-omni-2.0 --omni --port 8091 \
    --stage-configs-path vllm_omni/model_executor/stage_configs/ming_flash_omni.yaml
```

Pass `--stage-configs-path /path/to/your_config.yaml` to use a custom stage
config.

### Send Multi-modal Request

Shared Python client (supports `text | use_image | use_audio | use_video |
use_mixed_modalities`; pass `--image-path` / `--audio-path` / `--video-path`
for local files or URLs, `--modalities text` for output, `--help` for the
full flag list):

```bash
python examples/online_serving/openai_chat_completion_client_for_multimodal_generation.py \
    --model Jonathan1909/Ming-flash-omni-2.0 \
    --query-type use_mixed_modalities \
    --port 8091 --host localhost \
    --modalities text
```

Parameterized curl wrapper in this directory:

```bash
bash run_curl_multimodal_generation.sh text
bash run_curl_multimodal_generation.sh use_image
bash run_curl_multimodal_generation.sh use_audio
bash run_curl_multimodal_generation.sh use_video
bash run_curl_multimodal_generation.sh use_mixed_modalities
```

## Modality control

| `modalities` | Server config | Output |
|-------------|--------------|--------|
| `["text"]` or omitted | Thinker only | Text |
| `["audio"]` | Thinker + Talker | Audio (speech) |
| `["text", "audio"]` | Thinker + Talker | Text + Audio |

For ready-to-copy curl examples (text / audio / multimodal input, SSE
streaming, reasoning mode), see the recipe at
[`recipes/inclusionAI/Ming-flash-omni-2.0.md`](../../../recipes/inclusionAI/Ming-flash-omni-2.0.md).

## Image generation (thinker + diffusion)

Launch the dual-stage image-generation server (AR thinker on TP=4 +
diffusion stage; adjust `devices` inside the YAML to match your hardware):

```bash
vllm serve Jonathan1909/Ming-flash-omni-2.0 \
    --omni \
    --stage-configs-path vllm_omni/model_executor/stage_configs/ming_flash_omni_dual.yaml \
    --trust-remote-code \
    --port 8188
```

### Text-to-image

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

### Image edit (img2img)

Include a reference image as the first content item; the chat endpoint
detects the reference image and routes the request through
`MingImagePipeline` in img2img mode (the `<IMAGE>` placeholder is
prepended automatically so the thinker can still locate the
ref-image position when its multimodal cache is warm).

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

### Optional generation knobs (passed via `extra_body`)

| Field | Default | Notes |
|-------|---------|-------|
| `height`, `width` | 1024×1024 | Target image resolution. |
| `num_inference_steps` | 30 | DiT denoise step count. |
| `negative_prompt` | None | Triggers the CFG companion via `expand_cfg_prompts`. |
| `cfg_text_scale` | 2.0 | Classifier-free-guidance scale. |
| `seed` | request seed | Deterministic generation. |

## OpenAI Python SDK — streaming

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8091/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="Jonathan1909/Ming-flash-omni-2.0",
    messages=[
        {"role": "system", "content": [{"type": "text", "text": "你是一个友好的AI助手。\n\ndetailed thinking off"}]},
        {"role": "user", "content": "请详细介绍鹦鹉的生活习性。"},
    ],
    modalities=["text"],
    stream=True,
)
for chunk in response:
    for choice in chunk.choices:
        if hasattr(choice, "delta") and choice.delta.content:
            print(choice.delta.content, end="", flush=True)
print()
```

The `--stream` flag on the Python client script above shows the same pattern
driven by the shared multimodal client.
