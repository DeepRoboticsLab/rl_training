# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Reward functions for hierarchical navigation environment."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
import rl_training.tasks.manager_based.locomotion.hierarchical_nav.mdp.observations as obs_funcs
from rl_training.tasks.manager_based.locomotion.hierarchical_nav.mdp.observations import (
    distance_to_goal,
    robot_position_2d,
    goal_position_2d,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def goal_reaching_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "goal_position",
) -> torch.Tensor:
    """Reward for reaching the goal using exponential kernel.
    
    The reward decreases exponentially with distance to goal.
    
    Args:
        env: The environment instance
        std: Standard deviation for exponential kernel
        asset_cfg: Configuration for the robot asset
        command_name: Name of the command term for goal position
        
    Returns:
        Reward tensor of shape [num_envs]
    """
    distance = obs_funcs.distance_to_goal(env, asset_cfg, command_name)
    reward = torch.exp(-distance / std**2)
    return reward.squeeze(-1)


def goal_reaching_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "goal_position",
) -> torch.Tensor:
    """Penalty for distance to goal using L2 norm.
    
    Args:
        env: The environment instance
        asset_cfg: Configuration for the robot asset
        command_name: Name of the command term for goal position
        
    Returns:
        Penalty tensor of shape [num_envs] (negative values)
    """
    distance = obs_funcs.distance_to_goal(env, asset_cfg, command_name)
    return -distance.squeeze(-1)


def goal_reaching_progress(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "goal_position",
) -> torch.Tensor:
    """Reward for making progress towards the goal.
    
    This compares current distance to goal with previous distance,
    rewarding the agent for moving closer to the goal.
    
    Args:
        env: The environment instance
        asset_cfg: Configuration for the robot asset
        command_name: Name of the command term for goal position
        
    Returns:
        Progress reward tensor of shape [num_envs]
    """
    # Get current distance
    current_distance = obs_funcs.distance_to_goal(env, asset_cfg, command_name)
    
    # Get previous distance (stored in environment buffer)
    if not hasattr(env, "_prev_distance_to_goal"):
        env._prev_distance_to_goal = current_distance.clone()
    
    # Compute progress (positive if getting closer)
    progress = env._prev_distance_to_goal - current_distance
    
    # Update previous distance
    env._prev_distance_to_goal = current_distance.clone()
    
    return progress.squeeze(-1)

