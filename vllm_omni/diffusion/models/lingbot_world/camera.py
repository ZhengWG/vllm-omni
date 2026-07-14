# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keyboard-action camera geometry for LingBot-World.

Follows the official reference (``wan/utils/wasd_ijkl_to_c2ws.py`` +
``wan/utils/cam_utils.py`` in Robbyant/lingbot-world): actions are per *pixel*
frame, integrated at 2°/frame rotation into camera-to-world poses (with a
leading identity pose), SLERP-interpolated down to the latent-frame grid,
converted to max-norm-normalized framewise deltas, and embedded as folded
Plücker rays.
"""

from __future__ import annotations

import numpy as np
import torch

_MOVE_SPEED = 0.05
_ROTATE_STEP_RAD = np.deg2rad(2.0)  # official: same 2° step for pitch (i/k) and yaw (j/l)
_PITCH_LIMIT_RAD = np.deg2rad(85.0)
_ACTION_KEYS = frozenset("wasdijkl")


def parse_action_string(action_string: str) -> list[list[str]]:
    """Parse ``"w-10,iw-5,none-3"`` into per-pixel-frame key lists.

    Each ``<keys>-<count>`` token holds the key set for ``count`` video frames;
    ``none`` (or an empty key set) means no action.
    """
    actions: list[list[str]] = []
    for token in action_string.replace("，", ",").split(","):
        token = "".join(token.split())
        if not token:
            continue
        keys, _, count = token.rpartition("-")
        if not keys or not count.isdigit():
            raise ValueError(f"invalid action token {token!r}, expected '<keys>-<count>'")
        frame_keys = [] if keys.lower() == "none" else sorted(set(keys.lower()))
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


def actions_to_c2ws(action_history: list[list[str]]) -> np.ndarray:
    """Integrate per-frame keyboard actions into camera-to-world poses.

    Returns ``len(action_history) + 1`` poses — the leading identity is kept,
    matching the official ``generate_and_save_trajectory``.
    """
    c2w = np.eye(4)
    pitch = 0.0
    poses = [c2w]
    for frame_keys in action_history:
        rot, trans = c2w[:3, :3], c2w[:3, 3]

        pitch_delta = _ROTATE_STEP_RAD * (("i" in frame_keys) - ("k" in frame_keys))
        if abs(pitch + pitch_delta) <= _PITCH_LIMIT_RAD:
            pitch += pitch_delta
        else:
            pitch_delta = 0.0
        yaw_delta = _ROTATE_STEP_RAD * (("l" in frame_keys) - ("j" in frame_keys))
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
        c2w[:3, 3] = trans + move
        poses.append(c2w)
    return np.stack(poses)


def _slerp_rotations(rot0: np.ndarray, rot1: np.ndarray, frac: float) -> np.ndarray:
    """Geodesic interpolation between two rotation matrices (axis-angle form)."""
    delta = rot0.T @ rot1
    cos_angle = np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    if angle < 1e-8:
        return rot0
    axis = np.array([delta[2, 1] - delta[1, 2], delta[0, 2] - delta[2, 0], delta[1, 0] - delta[0, 1]])
    axis = axis / (2.0 * np.sin(angle))
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    theta = angle * frac
    return rot0 @ (np.eye(3) + np.sin(theta) * k + (1 - np.cos(theta)) * (k @ k))


def interpolate_poses(c2ws: np.ndarray, num_target: int) -> torch.Tensor:
    """Resample poses onto ``num_target`` uniform points (rotation SLERP, translation lerp).

    Matches the official ``interpolate_camera_poses`` (scipy Slerp + interp1d)
    without the scipy dependency.
    """
    num_source = c2ws.shape[0]
    targets = np.linspace(0.0, num_source - 1.0, num_target)
    out = np.tile(np.eye(4), (num_target, 1, 1))
    for i, t in enumerate(targets):
        lo = min(int(np.floor(t)), num_source - 1)
        hi = min(lo + 1, num_source - 1)
        frac = t - lo
        out[i, :3, :3] = _slerp_rotations(c2ws[lo, :3, :3], c2ws[hi, :3, :3], frac) if hi > lo else c2ws[lo, :3, :3]
        out[i, :3, 3] = (1 - frac) * c2ws[lo, :3, 3] + frac * c2ws[hi, :3, 3]
    return torch.from_numpy(out)


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


def camera_condition(
    action_history: list[list[str]],
    *,
    num_latent_frames: int,
    height: int,
    width: int,
    spatial_scale: int = 8,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Folded Plücker condition for the full video, one entry per latent frame.

    Official offline semantics: integrate all pixel-frame actions (identity
    first), SLERP down to the latent grid, framewise deltas with full-sequence
    max-norm, then per-pixel Plücker rays pixel-unshuffled onto the latent
    grid. Returns ``[1, 6*scale*scale, num_latent_frames, H/scale, W/scale]``.
    """
    c2ws = interpolate_poses(actions_to_c2ws(action_history), num_latent_frames).double()
    c2ws = compute_relative_poses(c2ws)
    intrinsics = torch.tensor([[500.0, 500.0, width / 2, height / 2]], dtype=c2ws.dtype).repeat(num_latent_frames, 1)
    plucker = _plucker_embeddings(c2ws, intrinsics, height, width)

    latent_height, latent_width = height // spatial_scale, width // spatial_scale
    plucker = plucker.view(num_latent_frames, latent_height, spatial_scale, latent_width, spatial_scale, 6)
    plucker = plucker.permute(0, 1, 3, 5, 2, 4).reshape(
        num_latent_frames, latent_height, latent_width, 6 * spatial_scale * spatial_scale
    )
    return plucker.permute(3, 0, 1, 2)[None].contiguous().to(device=device, dtype=dtype)
