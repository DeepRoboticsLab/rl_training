# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Observation functions for hierarchical navigation environment."""

from __future__ import annotations

import torch
import math
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def robot_position_2d(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Robot position in 2D (x, y) in world frame.
    
    Args:
        env: The environment instance
        asset_cfg: Configuration for the robot asset
        
    Returns:
        Robot position tensor of shape [num_envs, 2] (x, y)
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    pos_w = asset.data.root_pos_w
    return pos_w[:, :2]


def robot_yaw(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Robot yaw angle in world frame.
    
    Args:
        env: The environment instance
        asset_cfg: Configuration for the robot asset
        
    Returns:
        Robot yaw tensor of shape [num_envs, 1] (radians)
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    quat = asset.data.root_quat_w
    # Extract yaw from quaternion (w, x, y, z)
    qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return yaw.unsqueeze(-1)


def goal_position_2d(
    env: ManagerBasedRLEnv,
    command_name: str = "goal_position",
) -> torch.Tensor:
    """Goal position in 2D (x, y) in world frame.
    
    Args:
        env: The environment instance
        command_name: Name of the command term for goal position
        
    Returns:
        Goal position tensor of shape [num_envs, 2] (x, y)
    """
    goal_cmd = env.command_manager.get_command(command_name)
    return goal_cmd[:, :2]


def distance_to_goal(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "goal_position",
) -> torch.Tensor:
    """Distance to goal in 2D plane.
    
    Args:
        env: The environment instance
        asset_cfg: Configuration for the robot asset
        command_name: Name of the command term for goal position
        
    Returns:
        Distance tensor of shape [num_envs, 1]
    """
    robot_pos = robot_position_2d(env, asset_cfg)
    goal_pos = goal_position_2d(env, command_name)
    distance = torch.norm(goal_pos - robot_pos, dim=1, keepdim=True)
    return distance


def direction_to_goal(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "goal_position",
) -> torch.Tensor:
    """Direction to goal in robot frame (cos, sin).
    
    Args:
        env: The environment instance
        asset_cfg: Configuration for the robot asset
        command_name: Name of the command term for goal position
        
    Returns:
        Direction tensor of shape [num_envs, 2] (cos, sin) in robot frame
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    robot_pos = robot_position_2d(env, asset_cfg)
    goal_pos = goal_position_2d(env, command_name)
    # Extract yaw from quaternion (w, x, y, z)
    quat = asset.data.root_quat_w
    qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    robot_yaw_val = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    
    # Vector from robot to goal in world frame
    direction_w = goal_pos - robot_pos
    distance = torch.norm(direction_w, dim=1, keepdim=True)
    
    # Avoid division by zero
    direction_w = direction_w / torch.clamp(distance, min=1e-6)
    
    # Rotate to robot frame
    cos_yaw = torch.cos(robot_yaw_val)
    sin_yaw = torch.sin(robot_yaw_val)
    direction_x = direction_w[:, 0:1] * cos_yaw + direction_w[:, 1:2] * sin_yaw
    direction_y = -direction_w[:, 0:1] * sin_yaw + direction_w[:, 1:2] * cos_yaw
    
    return torch.cat([direction_x, direction_y], dim=1)


def goal_reached(
    env: ManagerBasedRLEnv,
    threshold: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "goal_position",
) -> torch.Tensor:
    """Check if goal is reached (distance < threshold).
    
    Args:
        env: The environment instance
        threshold: Distance threshold for goal reaching
        asset_cfg: Configuration for the robot asset
        command_name: Name of the command term for goal position
        
    Returns:
        Boolean tensor of shape [num_envs] indicating goal reached
    """
    distance = distance_to_goal(env, asset_cfg, command_name)
    return (distance.squeeze(-1) < threshold).bool()

