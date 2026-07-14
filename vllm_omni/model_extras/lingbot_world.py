# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

# LingBot-World reads its model-specific knobs from sampling_params.extra_args
# in pipeline_lingbot_world.py; declaring them here routes request `extra_body`
# fields into OmniDiffusionSamplingParams.extra_args so the model can be driven
# through the online /v1/videos path and the shared examples.
LINGBOT_WORLD_EXTRA_BODY_PARAMS = frozenset(
    {
        # keyboard action script: "w-9,lw-6,..." or per-frame key lists
        "camera_actions",
        # DMD schedule overrides (v2 checkpoint: [1000, 750, 500, 250] + 5.0)
        "dmd_timesteps",
        "flow_shift",
    }
)
LINGBOT_WORLD_EXTRA_OUTPUT_PARAMS = frozenset()
