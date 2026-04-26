# Ming-flash-omni 2.0

[Ming-flash-omni-2.0](https://github.com/inclusionAI/Ming) is an omni-modal model supporting text, image, video, and audio understanding, with text and speech outputs.

vLLM-Omni supports three deployment modes:

| Mode | Stage config | Output |
|------|-------------|--------|
| Thinker only (multimodal understanding) | `ming_flash_omni_thinker.yaml` (default `--omni`) | Text |
| Thinker + Talker (omni-speech) | `ming_flash_omni.yaml` | Text + Audio |
| Thinker + Diffusion (image generation, text-to-image / image edit) | `ming_flash_omni_dual.yaml` | Image |

For standalone TTS (talker only), see [`examples/offline_inference/ming_flash_omni_tts/`](../ming_flash_omni_tts/).
For image generation, see the [Image generation](#image-generation-thinker--diffusion) section below.

## Setup

Please refer to the [stage configuration documentation](https://docs.vllm.ai/projects/vllm-omni/en/latest/configuration/stage_configs/) to configure memory allocation appropriately for your hardware setup.

The default `--omni` flag runs thinker only.  For omni-speech, pass the two-stage config explicitly:

```bash
--stage-configs-path vllm_omni/model_executor/stage_configs/ming_flash_omni.yaml
```

## Run examples

The end-to-end script defaults to built-in assets; pass `--image-path`,
`--audio-path`, or `--video-path` to override.

```bash
# Text-only
python examples/offline_inference/ming_flash_omni/end2end.py --query-type text

# Image / audio / video / mixed understanding
python examples/offline_inference/ming_flash_omni/end2end.py --query-type use_image
python examples/offline_inference/ming_flash_omni/end2end.py --query-type use_audio
python examples/offline_inference/ming_flash_omni/end2end.py --query-type use_video --num-frames 16
python examples/offline_inference/ming_flash_omni/end2end.py --query-type use_mixed_modalities \
    --image-path /path/to/image.jpg --audio-path /path/to/audio.wav
```

#### Reasoning (Thinking Mode)

Reasoning ("detailed thinking on") is applied by the script when
`--query-type reasoning` is set. The default prompt matches Ming's cookbook
and expects the reference figure from the upstream repo — see
`get_reasoning_query` in `end2end.py`.

```bash
python examples/offline_inference/ming_flash_omni/end2end.py -q reasoning --image-path ./3_0.png
```

### Omni-speech (thinker + talker)

To enable spoken output, use the two-stage config and request `audio` (or `text,audio`) modalities.
The thinker processes your multimodal input, generates text, then the talker synthesises the response as speech.

**Audio-only output** (speech response, no text):
```bash
python examples/offline_inference/ming_flash_omni/end2end.py \
    --query-type text \
    --stage-configs-path vllm_omni/model_executor/stage_configs/ming_flash_omni.yaml \
    --modalities audio \
    --output-dir output_ming_omni_speech
```

**Both text and audio output**:
```bash
python examples/offline_inference/ming_flash_omni/end2end.py \
    --query-type use_audio \
    --stage-configs-path vllm_omni/model_executor/stage_configs/ming_flash_omni.yaml \
    --modalities text,audio \
    --output-dir output_ming_omni_speech
```

Generated `.wav` files are saved to `--output-dir` (default `output_ming`), one per request.

The stage config allocates thinker on GPUs 0–3 and talker on GPU 3 by default. Adjust `devices` in the YAML to match your hardware.

### Modality control

| `--modalities` | Thinker output | Talker | Saved files |
|---------------|----------------|--------|-------------|
| `text` (default) | Text | Not run | `<id>.txt` |
| `audio` | Text (internal) | Runs | `<id>.wav` |
| `text,audio` | Text | Runs | `<id>.txt` + `<id>.wav` |

Pass `--stage-configs-path /path/to/your_config.yaml` to any of the commands
above to override the stage config.

### Image generation (thinker + diffusion)

Ming-flash-omni-2.0 also exposes an image-generation stage backed by the
ZImage DiT and a Qwen2 connector. The diffusion stage is wired up via
`ming_flash_omni_dual.yaml`, which runs the AR thinker on TP=4 across
GPUs 0–3 and the diffusion stage on GPU 4. Adjust `devices` in the YAML
to match your hardware.

Online launch (OpenAI-compatible server):

```bash
vllm serve Jonathan1909/Ming-flash-omni-2.0 \
    --omni \
    --stage-configs-path vllm_omni/model_executor/stage_configs/ming_flash_omni_dual.yaml \
    --trust-remote-code \
    --port 8188
```

Send a text-to-image request via curl:

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

Optional knobs accepted in `extra_body`:

| Field | Default | Notes |
|-------|---------|-------|
| `height`, `width` | 1024×1024 | Target image resolution. |
| `num_inference_steps` | 30 | DiT denoise step count. |
| `negative_prompt` | None | Triggers the CFG companion via `expand_cfg_prompts`. |
| `cfg_text_scale` | 2.0 | Classifier-free-guidance scale. |
| `seed` | request seed | Deterministic generation. |

Image edit (img2img): include a single reference `image_url` in the
chat message; the chat endpoint will prepend `<IMAGE>` to the prompt
so the thinker's ref-image placeholder substitution still fires when
the multimodal cache is warm.

## Online serving

For online serving via the OpenAI-compatible API, see
[examples/online_serving/ming_flash_omni/README.md](../../online_serving/ming_flash_omni/README.md).
