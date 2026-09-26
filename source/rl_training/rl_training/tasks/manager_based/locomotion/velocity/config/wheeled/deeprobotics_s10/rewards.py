# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

from rl_training.tasks.manager_based.locomotion.velocity.mdp.rewards import joint_pos_penalty


def joint_pos_penalty_turn_side(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
    command_threshold: float,
    ang_cmd_threshold: float = 0.1,
    y_cmd_threshold: float = 0.1,
    xy_norm_max: float = 0.1,
    xz_norm_max: float = 0.1,
    sensor_cfg: SceneEntityCfg | None = None,
    terrain_height_threshold: float = 0.05,
    high_terrain_penalty_scale: float = 0.5,
) -> torch.Tensor:
    """Retain 20% of the S10 posture penalty for turn and sideways commands.

    Reduce when either:
    1) |cmd_z| > ang_cmd_threshold and ||[cmd_x, cmd_y]|| < xy_norm_max
    2) |cmd_y| > y_cmd_threshold and ||[cmd_x, cmd_z]|| < xz_norm_max

    If ``sensor_cfg`` is provided, reduce the penalty on high terrain (e.g. stairs)
    so hip-y/knee joints can deviate more from default posture.
    """
    reward = joint_pos_penalty(
        env=env,
        command_name=command_name,
        asset_cfg=asset_cfg,
        stand_still_scale=stand_still_scale,
        velocity_threshold=velocity_threshold,
        command_threshold=command_threshold,
    )

    cmd = env.command_manager.get_command(command_name)
    gate_turn = (torch.abs(cmd[:, 2]) > ang_cmd_threshold) & (torch.linalg.norm(cmd[:, :2], dim=1) < xy_norm_max)
    gate_side = (torch.abs(cmd[:, 1]) > y_cmd_threshold) & (
        torch.linalg.norm(torch.stack((cmd[:, 0], cmd[:, 2]), dim=1), dim=1) < xz_norm_max
    )
    reduced_gate = gate_turn | gate_side
    reward = reward * (~reduced_gate).float() + reward * (reduced_gate).float() * 0.2

    if sensor_cfg is not None:
        height_sensor: RayCaster = env.scene[sensor_cfg.name]
        ray_hits = height_sensor.data.ray_hits_w[..., 2]
        valid_hits = torch.isfinite(ray_hits) & (torch.abs(ray_hits) <= 1e6)
        valid_count = torch.sum(valid_hits, dim=1)
        safe_hits = torch.where(valid_hits, ray_hits, torch.zeros_like(ray_hits))
        terrain_height = torch.sum(safe_hits, dim=1) / torch.clamp(valid_count, min=1)
        terrain_height = terrain_height - env.scene.env_origins[:, 2]
        high_terrain = (terrain_height > terrain_height_threshold) & (valid_count > 0)
        scale = torch.where(
            high_terrain,
            torch.full_like(reward, high_terrain_penalty_scale),
            torch.ones_like(reward),
        )
        reward = reward * scale

    return reward
