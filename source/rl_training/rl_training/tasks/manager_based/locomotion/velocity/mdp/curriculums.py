# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause
# 
# # Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terrain curriculum with synchronized global gait_level update."""
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")

    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up

    terrain.update_env_origins(env_ids, move_up, move_down)

    mean_level = torch.mean(terrain.terrain_levels.float())
    # Local import avoids module-load circular dependency.
    from .rewards import update_gait_level_from_terrain_mean

    update_gait_level_from_terrain_mean(mean_level)
    return mean_level


def gait_level_curve(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    """Return current global gait_level for logging in curriculum curves."""
    from .rewards import gait_level

    return torch.tensor(gait_level, device=env.device)


def command_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    range_multiplier: Sequence[float] = (0.1, 1.0),
) -> None:
    """command_levels_vel"""
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges
    # Get original velocity ranges (ONLY ON FIRST EPISODE)
    if env.common_step_counter == 0:
        env._original_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device)
        env._original_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device)
        env._initial_vel_x = env._original_vel_x * range_multiplier[0]
        env._final_vel_x = env._original_vel_x * range_multiplier[1]
        env._initial_vel_y = env._original_vel_y * range_multiplier[0]
        env._final_vel_y = env._original_vel_y * range_multiplier[1]

        # Initialize command ranges to initial values
        base_velocity_ranges.lin_vel_x = env._initial_vel_x.tolist()
        base_velocity_ranges.lin_vel_y = env._initial_vel_y.tolist()

    # avoid updating command curriculum at each step since the maximum command is common to all envs
    if env.common_step_counter % env.max_episode_length == 0:
        episode_sums = env.reward_manager._episode_sums[reward_term_name]
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        delta_command = torch.tensor([-0.1, 0.1], device=env.device)

        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(episode_sums[env_ids]) / env.max_episode_length_s > 0.8 * reward_term_cfg.weight:
            new_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device) + delta_command
            new_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device) + delta_command

            # Clamp to ensure we don't exceed final ranges
            new_vel_x = torch.clamp(new_vel_x, min=env._final_vel_x[0], max=env._final_vel_x[1])
            new_vel_y = torch.clamp(new_vel_y, min=env._final_vel_y[0], max=env._final_vel_y[1])

            # Update ranges
            base_velocity_ranges.lin_vel_x = new_vel_x.tolist()
            base_velocity_ranges.lin_vel_y = new_vel_y.tolist()

    return torch.tensor(base_velocity_ranges.lin_vel_x[1], device=env.device)
