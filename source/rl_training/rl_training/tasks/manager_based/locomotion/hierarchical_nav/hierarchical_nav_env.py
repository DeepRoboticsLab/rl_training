# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Hierarchical navigation environment.

This environment wraps a low-level locomotion environment and provides
a high-level interface for goal-reaching navigation tasks.
"""

from __future__ import annotations

import torch
import numpy as np
import gymnasium as gym
from gymnasium.core import ActType, ObsType
from typing import TYPE_CHECKING, Any

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

if TYPE_CHECKING:
    from .hierarchical_nav_env_cfg import HierarchicalNavEnvCfg

from rl_training.utils.frozen_policy import FrozenLocomotionPolicy
import rl_training.tasks.manager_based.locomotion.hierarchical_nav.mdp as mdp


class HierarchicalNavEnv:
    """Hierarchical navigation environment wrapper.
    
    This wrapper takes a low-level locomotion environment and provides
    a high-level interface for goal-reaching navigation tasks. High-level
    actions are velocity commands [vx, vy, vyaw] that are converted to
    joint actions by a frozen low-level policy.
    
    Args:
        env: Low-level locomotion environment (wrapped with RslRlVecEnvWrapper)
        frozen_policy_wrapper: FrozenLocomotionPolicy instance
        decimation: Number of low-level steps per high-level step
    """
    
    def __init__(
        self,
        env,
        frozen_policy_wrapper: FrozenLocomotionPolicy,
        decimation: int = 10,
    ):
        """Initialize the hierarchical navigation environment wrapper.
        
        Args:
            env: Low-level locomotion environment (wrapped with RslRlVecEnvWrapper)
            frozen_policy_wrapper: FrozenLocomotionPolicy instance
            decimation: Number of low-level steps per high-level step
        """
        self.env = env
        self.frozen_policy_wrapper = frozen_policy_wrapper
        self.decimation = decimation
        
        # High-level action space: velocity commands [vx, vy, vyaw]
        self.action_space = gym.spaces.Box(
            low=np.array([-2.0, -2.0, -2.0], dtype=np.float32),
            high=np.array([2.0, 2.0, 2.0], dtype=np.float32),
            dtype=np.float32,
        )
        
        # High-level observation space: [robot_pos_2d, robot_yaw, goal_pos_2d, distance, direction]
        # robot_pos_2d: 2, robot_yaw: 1, goal_pos_2d: 2, distance: 1, direction: 2
        obs_dim = 2 + 1 + 2 + 1 + 2  # 8 dimensions
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        
        # Store previous distance for progress reward
        self._prev_distance_to_goal = None
        
        # Goal positions for each environment [num_envs, 2] (x, y)
        self._goal_positions = torch.zeros(self.num_envs, 2, device=self.device)
        
        # Goal resampling parameters
        self._goal_distance_range = (1.0, 5.0)
        self._goal_resample_time_range = (20.0, 20.0)  # Resample every 20 seconds
        self._last_goal_resample_time = torch.zeros(self.num_envs, device=self.device)
    
    @property
    def device(self):
        """Get device from wrapped environment."""
        return self.env.unwrapped.device
    
    @property
    def num_envs(self):
        """Get number of environments."""
        return self.env.unwrapped.num_envs
    
    @property
    def unwrapped(self):
        """Get unwrapped environment."""
        return self.env.unwrapped
    
    def reset(self, **kwargs) -> tuple[ObsType, dict[str, Any]]:
        """Reset the environment and return high-level observations.
        
        Returns:
            High-level observations and info dict
        """
        # Reset low-level environment
        obs, info = self.env.reset(**kwargs)
        
        # Resample goal positions for all environments
        self._resample_goals(torch.arange(self.num_envs, device=self.device))
        
        # Get high-level observations
        high_level_obs = self._get_high_level_observations()
        
        # Reset previous distance
        self._prev_distance_to_goal = None
        
        # Initialize high-level metrics in info (for logging)
        if not isinstance(info, dict):
            info = {}
        info["Episode_Termination/goal_reached"] = 0.0
        info["Episode_Reward/goal_reaching"] = 0.0
        info["Episode_Reward/progress"] = 0.0
        info["hierarchical/distance_to_goal"] = 0.0
        
        return high_level_obs, info
    
    def _resample_goals(self, env_ids: torch.Tensor):
        """Resample goal positions for specified environments.
        
        Args:
            env_ids: Environment indices to resample goals for
        """
        # Sample distance and angle
        distance = torch.empty(len(env_ids), device=self.device).uniform_(
            self._goal_distance_range[0], self._goal_distance_range[1]
        )
        angle = torch.empty(len(env_ids), device=self.device).uniform_(
            -torch.pi, torch.pi
        )
        
        # Get robot position
        robot_pos = self.unwrapped.scene["robot"].data.root_pos_w[env_ids, :2]
        
        # Compute goal position
        goal_x = robot_pos[:, 0] + distance * torch.cos(angle)
        goal_y = robot_pos[:, 1] + distance * torch.sin(angle)
        
        self._goal_positions[env_ids, 0] = goal_x
        self._goal_positions[env_ids, 1] = goal_y
    
    def step(self, action: ActType) -> tuple[ObsType, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Step the environment with high-level action.
        
        Args:
            action: High-level action tensor of shape [num_envs, 3]
                where columns are [vx, vy, vyaw]
        
        Returns:
            High-level observations, rewards, terminated, truncated, info
        """
        # Convert action to tensor if needed
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, device=self.device, dtype=torch.float32)
        
        # Ensure correct shape
        if action.dim() == 1:
            action = action.unsqueeze(0)
        
        # Expand to all environments if needed
        if action.shape[0] == 1 and self.num_envs > 1:
            action = action.expand(self.num_envs, -1)
        
        # Step low-level environment multiple times (decimation)
        info = {}
        last_dones = None
        
        for _ in range(self.decimation):
            # Convert high-level velocity command to low-level joint actions
            low_level_actions = self.frozen_policy_wrapper(action)
            
            # Step low-level environment
            # RslRlVecEnvWrapper.step() returns: (obs, rew, dones, extras)
            obs, rewards, dones, extras = self.env.step(low_level_actions)
            
            # Store info from last step
            info = extras
            last_dones = dones
        
        # Compute high-level observations
        high_level_obs = self._get_high_level_observations()
        
        # Compute high-level rewards
        high_level_reward = self._compute_high_level_rewards()
        
        # Compute high-level terminations
        # RslRlVecEnvWrapper returns dones (terminated | truncated), we need to separate them
        # For now, we'll use dones as both terminated and truncated, and check goal reached separately
        high_level_terminated, high_level_truncated = self._compute_high_level_terminations(last_dones)
        
        # Add high-level metrics to info dict for logging
        low_env = self.unwrapped
        robot_pos_2d = mdp.robot_position_2d(low_env)
        distance_vec = self._goal_positions - robot_pos_2d
        distance = torch.norm(distance_vec, dim=1)  # [num_envs]
        goal_reached = (distance < 0.5)  # threshold
        
        # Compute individual reward components for logging
        std = 0.5
        goal_reward = torch.exp(-distance / std**2)  # [num_envs]
        if self._prev_distance_to_goal is not None:
            progress_reward = (self._prev_distance_to_goal.squeeze(-1) - distance)  # [num_envs]
        else:
            progress_reward = torch.zeros(self.num_envs, device=self.device)
        
        # Add high-level metrics to info dict in RSL-RL logging format
        if not isinstance(info, dict):
            info = {}
        # RSL-RL logs Episode_Termination/ and Episode_Reward/ prefixed metrics
        info["Episode_Termination/goal_reached"] = goal_reached.float().mean().item()
        info["Episode_Reward/goal_reaching"] = goal_reward.mean().item()
        info["Episode_Reward/progress"] = progress_reward.mean().item()
        # Also add as hierarchical metrics for clarity
        info["hierarchical/distance_to_goal"] = distance.mean().item()
        
        return high_level_obs, high_level_reward, high_level_terminated, high_level_truncated, info
    
    def _get_high_level_observations(self) -> torch.Tensor:
        """Compute high-level observations from current state.
        
        Returns:
            High-level observation tensor of shape [num_envs, 8]
        """
        # Get low-level environment (unwrapped ManagerBasedRLEnv)
        low_env = self.unwrapped
        
        # Get robot position and yaw
        robot_pos_2d = mdp.robot_position_2d(low_env)
        robot_yaw = mdp.robot_yaw(low_env)
        
        # Use stored goal positions
        goal_pos_2d = self._goal_positions  # [num_envs, 2]
        
        # Compute distance to goal
        distance_vec = goal_pos_2d - robot_pos_2d
        distance = torch.norm(distance_vec, dim=1, keepdim=True)  # [num_envs, 1]
        
        # Compute direction to goal in robot frame
        direction_w = distance_vec / torch.clamp(distance, min=1e-6)  # [num_envs, 2]
        robot_yaw_val = robot_yaw.squeeze(-1)  # [num_envs]
        cos_yaw = torch.cos(robot_yaw_val)
        sin_yaw = torch.sin(robot_yaw_val)
        direction_x = direction_w[:, 0:1] * cos_yaw.unsqueeze(-1) + direction_w[:, 1:2] * sin_yaw.unsqueeze(-1)
        direction_y = -direction_w[:, 0:1] * sin_yaw.unsqueeze(-1) + direction_w[:, 1:2] * cos_yaw.unsqueeze(-1)
        direction = torch.cat([direction_x, direction_y], dim=1)  # [num_envs, 2]
        
        # Concatenate observations
        obs = torch.cat([
            robot_pos_2d,  # [num_envs, 2]
            robot_yaw,  # [num_envs, 1]
            goal_pos_2d,  # [num_envs, 2]
            distance,  # [num_envs, 1]
            direction,  # [num_envs, 2]
        ], dim=1)  # [num_envs, 8]
        
        return obs
    
    def _compute_high_level_rewards(self) -> torch.Tensor:
        """Compute high-level rewards.
        
        Returns:
            High-level reward tensor of shape [num_envs]
        """
        # Get robot position
        low_env = self.unwrapped
        robot_pos_2d = mdp.robot_position_2d(low_env)
        
        # Compute distance to goal
        distance_vec = self._goal_positions - robot_pos_2d
        current_distance = torch.norm(distance_vec, dim=1, keepdim=True)  # [num_envs, 1]
        
        # Goal reaching reward (exponential kernel)
        std = 0.5
        goal_reward = torch.exp(-current_distance / std**2).squeeze(-1)  # [num_envs]
        
        # Progress reward
        if self._prev_distance_to_goal is None:
            self._prev_distance_to_goal = current_distance.clone()
        
        progress = (self._prev_distance_to_goal - current_distance).squeeze(-1)  # [num_envs]
        self._prev_distance_to_goal = current_distance.clone()
        
        # Total reward
        total_reward = goal_reward + 0.5 * progress
        
        return total_reward
    
    def _compute_high_level_terminations(
        self,
        low_level_dones: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute high-level terminations.
        
        Args:
            low_level_dones: Low-level done flags (terminated | truncated from RslRlVecEnvWrapper)
        
        Returns:
            High-level terminated and truncated flags
        """
        # Get robot position
        low_env = self.unwrapped
        robot_pos_2d = mdp.robot_position_2d(low_env)
        
        # Compute distance to goal
        distance_vec = self._goal_positions - robot_pos_2d
        distance = torch.norm(distance_vec, dim=1)  # [num_envs]
        
        # Check if goal is reached
        threshold = 0.5
        goal_reached = (distance < threshold)
        
        # High-level termination: goal reached or low-level done
        high_level_terminated = goal_reached | low_level_dones.bool()
        
        # High-level truncation: same as low-level (for now, we don't distinguish)
        # In RslRlVecEnvWrapper, time_outs (truncations) are in extras["time_outs"]
        # For simplicity, we'll use low_level_dones for truncated as well
        high_level_truncated = low_level_dones.bool()
        
        return high_level_terminated, high_level_truncated
