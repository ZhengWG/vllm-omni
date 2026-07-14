# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keyboard-action camera geometry for LingBot-World.

Turns per-latent-frame keyboard actions (w/a/s/d move, i/k pitch, j/l yaw)
into camera-to-world poses and folded Plücker-ray embeddings, matching the
reference semantics of Robbyant/lingbot-world.
"""

from __future__ import annotations

import numpy as np
import torch

_MOVE_SPEED = 0.05
_PITCH_STEP_RAD = np.deg2rad(4.0)
_YAW_STEP_RAD = np.deg2rad(6.0)
_PITCH_LIMIT_RAD = np.deg2rad(85.0)
_ACTION_KEYS = frozenset("wasdijkl")


def parse_action_string(action_string: str) -> list[list[str]]:
    """Parse ``"w-10,iw-5,none-3"`` into per-latent-frame key lists.

    Each ``<keys>-<count>`` token repeats the key set for ``count`` latent
    frames; ``none`` (or an empty key set) means no action.
    """
    actions: list[list[str]] = []
    for token in action_string.split(","):
        token = token.strip()
        if not token:
            continue
        keys, _, count = token.rpartition("-")
        if not keys or not count.isdigit():
            raise ValueError(f"invalid action token {token!r}, expected '<keys>-<count>'")
        frame_keys = [] if keys == "none" else sorted(set(keys))
        unknown = set(frame_keys) - _ACTION_KEYS
        if unknown:
            raise ValueError(f"unknown action keys {sorted(unknown)} in token {token!r}")
        actions.extend([list(frame_keys) for _ in range(int(count))])
    return actions


def _rotation(axis: str, angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])  # axis == "y"


def actions_to_c2ws(action_history: list[list[str]]) -> torch.Tensor:
    """Integrate keyboard actions into camera-to-world poses, one per frame."""
    c2w = np.eye(4)
    pitch = 0.0
    poses = []
    for frame_keys in action_history:
        rot = c2w[:3, :3]

        pitch_delta = _PITCH_STEP_RAD * (("i" in frame_keys) - ("k" in frame_keys))
        if abs(pitch + pitch_delta) <= _PITCH_LIMIT_RAD:
            pitch += pitch_delta
        else:
            pitch_delta = 0.0
        yaw_delta = _YAW_STEP_RAD * (("l" in frame_keys) - ("j" in frame_keys))
        rot = _rotation("y", yaw_delta) @ rot @ _rotation("x", pitch_delta)

        # Move on the ground plane along the (flattened) camera axes.
        forward = rot[:, 2] * np.array([1.0, 0.0, 1.0])
        right = rot[:, 0] * np.array([1.0, 0.0, 1.0])
        forward = forward / (np.linalg.norm(forward) + 1e-6)
        right = right / (np.linalg.norm(right) + 1e-6)
        move = _MOVE_SPEED * (
            forward * (("w" in frame_keys) - ("s" in frame_keys)) + right * (("d" in frame_keys) - ("a" in frame_keys))
        )

        c2w = np.eye(4)
        c2w[:3, :3] = rot
        c2w[:3, 3] = poses[-1][:3, 3] + move if poses else move
        poses.append(c2w)
    return torch.from_numpy(np.stack(poses))


def _se3_inverse(mats: torch.Tensor) -> torch.Tensor:
    rot_t = mats[:, :3, :3].transpose(-1, -2)
    inv = torch.eye(4, dtype=mats.dtype)[None].repeat(mats.shape[0], 1, 1)
    inv[:, :3, :3] = rot_t
    inv[:, :3, 3:] = -torch.bmm(rot_t, mats[:, :3, 3:])
    return inv


def compute_relative_poses(c2ws: torch.Tensor) -> torch.Tensor:
    """First-frame-relative framewise pose deltas, translation max-normalized."""
    relative = torch.matmul(_se3_inverse(c2ws[:1]), c2ws)
    relative[0] = torch.eye(4, dtype=c2ws.dtype)
    if relative.shape[0] > 1:
        relative[1:] = torch.bmm(_se3_inverse(relative[:-1].clone()), relative[1:].clone())
    translations = relative[:, :3, 3]
    max_norm = torch.norm(translations, dim=-1).max()
    if max_norm > 0:
        relative[:, :3, 3] = translations / max_norm
    return relative


def _plucker_embeddings(c2ws: torch.Tensor, intrinsics: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Per-pixel (ray origin, ray direction) channels, shape [F, H, W, 6]."""
    num_frames = c2ws.shape[0]
    x = torch.arange(width, dtype=c2ws.dtype) + 0.5
    y = torch.arange(height, dtype=c2ws.dtype) + 0.5
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    grid = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2)[None].repeat(num_frames, 1, 1)

    fx, fy, cx, cy = intrinsics.chunk(4, dim=-1)
    directions = torch.stack(
        [(grid[..., 0] - cx) / fx, (grid[..., 1] - cy) / fy, torch.ones_like(grid[..., 0])],
        dim=-1,
    )
    directions = directions / directions.norm(dim=-1, keepdim=True)
    rays_d = directions @ c2ws[:, :3, :3].transpose(-1, -2)
    rays_o = c2ws[:, :3, 3][:, None, :].expand_as(rays_d)
    return torch.cat([rays_o, rays_d], dim=-1).view(num_frames, height, width, 6)


def camera_chunk_condition(
    action_history: list[list[str]],
    *,
    chunk_size: int,
    height: int,
    width: int,
    spatial_scale: int = 8,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Folded Plücker condition for the trailing chunk of the action history.

    Poses are integrated over the *full* history so framewise deltas and the
    translation normalization stay consistent across chunks (reference
    behavior), then only the last ``chunk_size`` frames are embedded.
    Returns ``[1, 6*scale*scale, chunk_size, height/scale, width/scale]``.
    """
    c2ws = compute_relative_poses(actions_to_c2ws(action_history).double())[-chunk_size:]
    intrinsics = torch.tensor([[500.0, 500.0, width / 2, height / 2]], dtype=c2ws.dtype).repeat(chunk_size, 1)
    plucker = _plucker_embeddings(c2ws, intrinsics, height, width)

    latent_height, latent_width = height // spatial_scale, width // spatial_scale
    plucker = plucker.view(chunk_size, latent_height, spatial_scale, latent_width, spatial_scale, 6)
    plucker = plucker.permute(0, 1, 3, 5, 2, 4).reshape(
        chunk_size, latent_height, latent_width, 6 * spatial_scale * spatial_scale
    )
    return plucker.permute(3, 0, 1, 2)[None].contiguous().to(device=device, dtype=dtype)
