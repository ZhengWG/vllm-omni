# LingBot-World: Interactive World Model

Offline demo for [LingBot-World](https://github.com/robbyant/lingbot-world) — a
Wan2.2-derived causal world model that turns a first-frame image, a text prompt,
and keyboard actions into an explorable video. Generation is autoregressive in
3-latent-frame chunks with a request-local KV cache (attention sink + sliding
window) and a 4-step causal DMD schedule.

> The LingBot-World checkpoints are licensed **CC BY-NC-SA (non-commercial)**;
> this integration code is Apache-2.0.

## Usage

```bash
python lingbot_world.py \
    --model robbyant/lingbot-world-fast-diffusers \
    --image scene.png \
    --prompt "A first-person walk through a sunlit forest" \
    --actions "w-36,lw-24,w-21" \
    --num-frames 81 \
    --output lingbot_world.mp4
```

## Camera control

Two mutually exclusive inputs:

- `--actions "w-36,lw-24,w-21"` — keyboard script (below);
- `--action-dir path/` — official camera trajectory: `poses.npy` `[N,4,4]`
  c2w poses (+ optional `intrinsics.npy` `[N,4]` fx,fy,cx,cy at 832x480
  reference), e.g. the `examples/NN/` dirs from Robbyant/lingbot-world.
  Programmatically: `multi_modal_data["camera"] = {"poses": ..., "intrinsics": ...}`
  (tensors, arrays, or `.npy` paths).

## Keyboard actions

`--actions` is a comma-separated script with one key set per **video frame**
(official semantics; poses are interpolated onto the latent grid internally):

| Keys | Effect |
|------|--------|
| `w` / `s` | move forward / backward |
| `a` / `d` | strafe left / right |
| `i` / `k` | pitch up / down (2°/frame, clamped ±85°) |
| `j` / `l` | yaw left / right (2°/frame) |
| `none` | idle |

`w-36` holds `w` for 36 video frames; `lw-24` combines yaw-right + forward.
If the script is shorter than the video, the last action is held. 81 output
frames = 21 latent frames = 7 chunks.

## Constraints

- `(num_frames - 1) % 4 == 0` and the latent frame count must be a multiple of 3.
- `height`/`width` divisible by 16; the checkpoint is trained at 832x480.
- One request at a time (world-model session semantics); `--tensor-parallel-size`
  2/4 splits the 14B DiT and the KV cache across GPUs.

## Checkpoints

The official raw v2 layout (`robbyant/lingbot-world-v2`, with `config.json` +
`transformers/` + `Wan2.1_VAE.pth`) is auto-detected and loaded with the v2
schedule built in. For diffusers-layout checkpoints, the DMD schedule comes
from the checkpoint's `scheduler_config.json` (`flow_shift` /
`dmd_denoising_steps`) and can be overridden per request:

```python
extra_args={"camera_actions": "...", "dmd_timesteps": [1000, 750, 500, 250], "flow_shift": 5.0}
```

(The generic `shift` key in diffusers scheduler configs is ignored — it is
usually the FlowUniPC export default (3.0), not the official training value.)

## Regression test

`tests/e2e/offline_inference/test_lingbot_world.py` is an opt-in GPU E2E: a
seeded 21-frame (2-chunk) run through the full engine path with sanity and
temporal-continuity checks. See its docstring for all options (poses input,
golden/PSNR comparison via `LINGBOT_REFERENCE`).

```bash
LINGBOT_E2E=1 LINGBOT_MODEL=/path/to/model LINGBOT_IMAGE=first_frame.png \
    pytest tests/e2e/offline_inference/test_lingbot_world.py -s
```

On air-gapped machines: `pytest-asyncio` must be installed (the repo's
`--strict-config` otherwise aborts silently after collection), and plugin
autoload may hang at exit without network — run with
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -p asyncio ...`.
