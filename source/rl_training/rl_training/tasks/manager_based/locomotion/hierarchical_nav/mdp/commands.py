# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Command functions for hierarchical navigation environment."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class UniformGoalPositionCommand(CommandTerm):
    """Command generator that generates goal positions uniformly in a circular area."""

    cfg: "UniformGoalPositionCommandCfg"
    """The configuration of the command generator."""

    def __init__(self, cfg: "UniformGoalPositionCommandCfg", env: ManagerBasedEnv):
        """Initialize the command generator."""
        super().__init__(cfg, env)
        # Command is [goal_x, goal_y] in world frame
        self._command = torch.zeros(self.num_envs, 2, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """Return the current command. Shape is (num_envs, 2)."""
        return self._command

    def _update_command(self):
        """Update and store the current commands."""
        # Commands are already stored in self._command, nothing to do here
        pass

    def _update_metrics(self):
        """Update metrics for the command generator."""
        # No metrics to update for goal position commands
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample goal positions for specified environments."""
        # Sample distance and angle
        distance = torch.empty(len(env_ids), device=self.device).uniform_(
            self.cfg.distance_range[0], self.cfg.distance_range[1]
        )
        angle = torch.empty(len(env_ids), device=self.device).uniform_(
            -torch.pi, torch.pi
        )
        
        # Get robot position
        robot_pos = self.env.scene["robot"].data.root_pos_w[env_ids, :2]
        
        # Compute goal position
        goal_x = robot_pos[:, 0] + distance * torch.cos(angle)
        goal_y = robot_pos[:, 1] + distance * torch.sin(angle)
        
        self._command[env_ids, 0] = goal_x
        self._command[env_ids, 1] = goal_y


@configclass
class UniformGoalPositionCommandCfg(CommandTermCfg):
    """Configuration for uniform goal position command generator."""

    class_type: type = UniformGoalPositionCommand
    
    distance_range: tuple[float, float] = (1.0, 5.0)
    """Range of distances from robot to goal (min, max) in meters."""


