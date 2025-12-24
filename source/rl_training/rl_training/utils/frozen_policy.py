# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Utility for freezing and using trained locomotion policies.

This module provides utilities to load and freeze trained policies
following the same pattern as play.py.
"""

from __future__ import annotations

import torch
from typing import Optional

from tensordict import TensorDict
from rsl_rl.runners import OnPolicyRunner


def freeze_policy(runner: OnPolicyRunner) -> torch.nn.Module:
    """Freeze a policy from an OnPolicyRunner.
    
    This function sets the policy to eval mode and disables gradients,
    following the same pattern as play.py.
    
    Args:
        runner: OnPolicyRunner instance with loaded checkpoint
        
    Returns:
        Frozen policy module
    """
    # Extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic
    
    # Set to eval mode and freeze all parameters
    policy_nn.eval()
    for param in policy_nn.parameters():
        param.requires_grad = False
    
    return policy_nn


def is_frozen(policy_nn: torch.nn.Module) -> bool:
    """Check if a policy is frozen (all parameters have requires_grad=False).
    
    Args:
        policy_nn: Policy neural network module
        
    Returns:
        True if all parameters have requires_grad=False, False otherwise
    """
    return all(not param.requires_grad for param in policy_nn.parameters())


def count_parameters(policy_nn: torch.nn.Module) -> int:
    """Count total number of parameters in a policy.
    
    Args:
        policy_nn: Policy neural network module
        
    Returns:
        Total number of parameters
    """
    return sum(p.numel() for p in policy_nn.parameters())


class FrozenLocomotionPolicy:
    """Wrapper for frozen low-level locomotion policy.
    
    This class provides a simple interface to convert velocity commands to joint actions
    using a frozen low-level policy. The policy is expected to be loaded externally
    and the inference function is passed to this class.
    
    Args:
        inference_policy: Callable that takes observation tensor and returns actions
        env: The low-level locomotion environment (ManagerBasedRLEnv)
    """
    
    def __init__(
        self,
        inference_policy: callable,
        env,  # ManagerBasedRLEnv - avoiding circular import
    ):
        self.inference_policy = inference_policy
        self.env = env
    
    def __call__(self, velocity_command: torch.Tensor) -> torch.Tensor:
        """Convert velocity command to joint actions.
        
        This method temporarily sets the velocity command in the environment,
        gets the observation, and returns the policy actions.
        
        Args:
            velocity_command: Velocity command tensor of shape [num_envs, 3]
                where columns are [vx, vy, vyaw]
                
        Returns:
            Joint actions tensor of shape [num_envs, num_joints]
        """
        # Store original command
        cmd_term = self.env.command_manager.get_term("base_velocity")
        original_cmd = cmd_term.command.clone()
        
        # Set new command (only first 3 dims: vx, vy, vyaw)
        cmd_term.command[:, :3] = velocity_command
        
        # Get observations with new command using observation manager
        obs_dict = self.env.observation_manager.compute()
        # Convert to TensorDict format that policy expects (only "policy" group)
        obs = TensorDict({"policy": obs_dict["policy"]}, batch_size=[obs_dict["policy"].shape[0]])
        
        # Get actions from frozen policy
        with torch.no_grad():
            actions = self.inference_policy(obs)
        
        # Restore original command
        cmd_term.command[:, :3] = original_cmd
        
        return actions
