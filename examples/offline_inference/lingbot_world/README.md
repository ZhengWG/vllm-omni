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
    --actions "w-9,lw-6,w-6" \
    --num-frames 81 \
    --output lingbot_world.mp4
```

## Keyboard actions

`--actions` is a comma-separated script with one key set per **latent frame**
(4 pixel frames each, 3 latent frames per chunk):

| Keys | Effect |
|------|--------|
| `w` / `s` | move forward / backward |
| `a` / `d` | strafe left / right |
| `i` / `k` | pitch up / down (4°/frame, clamped ±85°) |
| `j` / `l` | yaw left / right (6°/frame) |
| `none` | idle |

`w-9` holds `w` for 9 latent frames; `lw-6` combines yaw-right + forward.
If the script is shorter than the video, the last action is held. 81 output
frames = 21 latent frames = 7 chunks.

## Constraints

- `(num_frames - 1) % 4 == 0` and the latent frame count must be a multiple of 3.
- `height`/`width` divisible by 16; the checkpoint is trained at 832x480.
- One request at a time (world-model session semantics); `--tensor-parallel-size`
  2/4 splits the 14B DiT and the KV cache across GPUs.

## v2 checkpoint

`robbyant/lingbot-world-v2-diffusers` shares this pipeline; its DMD schedule is
selected via sampling `extra_args`:

```python
extra_args={"camera_actions": "...", "dmd_timesteps": [1000, 750, 500, 250], "flow_shift": 5.0}
```

(When the checkpoint's `scheduler_config.json` declares `flow_shift`, it is
picked up automatically.)
