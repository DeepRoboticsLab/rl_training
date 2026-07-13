# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause
# 
# # Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class UniformThresholdVelocityCommand(mdp.UniformVelocityCommand):
    """Command generator that generates a velocity command in SE(2) from uniform distribution with threshold.

    Supports fixed-proportion special-case samples that always use the original (full)
    velocity ranges, preventing catastrophic forgetting of basic skills as the curriculum
    narrows the command range.
    """

    cfg: mdp.UniformThresholdVelocityCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: mdp.UniformThresholdVelocityCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        # Additional metrics for TensorBoard.
        self.metrics["base_z"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["knee_pos"] = torch.zeros(self.num_envs, device=self.device)
        self._metric_step_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        knee_joint_ids = self.robot.find_joints(".*[Kk]nee.*")[0]
        self._knee_joint_ids = torch.tensor(knee_joint_ids, dtype=torch.long, device=self.device)

        # ---- Store original full ranges for special-case sampling ----
        # These are used by special-case envs so they always sample from the
        # full range regardless of curriculum changes.
        self._full_ranges_lin_vel_x = tuple(cfg.ranges.lin_vel_x)
        self._full_ranges_lin_vel_y = tuple(cfg.ranges.lin_vel_y)
        self._full_ranges_ang_vel_z = tuple(cfg.ranges.ang_vel_z)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        if env_ids is None:
            env_ids = slice(None)

        extras = {}
        for metric_name, metric_value in self.metrics.items():
            if metric_name in {"base_z", "knee_pos"}:
                step_count = torch.clamp(self._metric_step_counter[env_ids].float(), min=1.0)
                extras[metric_name] = torch.mean(metric_value[env_ids] / step_count).item()
            else:
                extras[metric_name] = torch.mean(metric_value[env_ids]).item()
            metric_value[env_ids] = 0.0

        self._metric_step_counter[env_ids] = 0
        self.command_counter[env_ids] = 0
        self._resample(env_ids)
        return extras

    def _update_metrics(self):
        super()._update_metrics()

        # 1) base_z metric: root_pos_w[:, 2]
        base_z = self.robot.data.root_pos_w[:, 2]

        # 2) knee_pos metric: same formulation as joint_pos_penalty for knee joints
        cmd = torch.linalg.norm(self.vel_command_b, dim=1)
        body_vel = torch.linalg.norm(self.robot.data.root_lin_vel_b[:, :2], dim=1)

        if self._knee_joint_ids.numel() > 0:
            running_reward = torch.linalg.norm(
                self.robot.data.joint_pos[:, self._knee_joint_ids]
                - self.robot.data.default_joint_pos[:, self._knee_joint_ids],
                dim=1,
            )
        else:
            running_reward = torch.zeros(self.num_envs, device=self.device)

        knee_pos = torch.where(
            torch.logical_or(cmd > 0.1, body_vel > 0.5),
            running_reward,
            5.0 * running_reward,
        )

        self.metrics["base_z"] += base_z
        self.metrics["knee_pos"] += knee_pos
        self._metric_step_counter += 1

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        # set small commands to zero
        self.vel_command_b[env_ids, :2] *= (torch.norm(self.vel_command_b[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

        # ---- Fixed-proportion special-case samples ----
        # These override the sampled command for a fixed fraction of envs,
        # always using the original (full) velocity ranges so they are not
        # affected by curriculum narrowing.  This prevents catastrophic
        # forgetting of basic skills.
        n = len(env_ids)
        if n == 0:
            return

        r = torch.empty(n, device=self.device)
        # Assign each env a slot in [0, 1).  Slots are laid out as:
        #   [0,                          rel_zero_vel          ) -> zero velocity
        #   [rel_zero_vel,              +rel_only_lin_y        ) -> only lin_y
        #   [rel_zero_vel+rel_only_lin_y, +rel_only_lin_x      ) -> only lin_x
        #   [...,                        +rel_only_ang_z        ) -> only ang_z
        #   [...,                        1.0                    ) -> normal (no override)
        slot = r.uniform_(0.0, 1.0)
        cum = 0.0

        # --- Zero velocity (standing) ---
        rel_zero_vel = self.cfg.rel_zero_vel_envs
        mask_zero = slot < cum + rel_zero_vel
        cum += rel_zero_vel

        # --- Only linear y ---
        rel_only_lin_y = self.cfg.rel_only_lin_y_envs
        mask_only_y = (slot >= cum) & (slot < cum + rel_only_lin_y)
        cum += rel_only_lin_y

        # --- Only linear x ---
        rel_only_lin_x = self.cfg.rel_only_lin_x_envs
        mask_only_x = (slot >= cum) & (slot < cum + rel_only_lin_x)
        cum += rel_only_lin_x

        # --- Only angular z ---
        rel_only_ang_z = self.cfg.rel_only_ang_z_envs
        mask_only_ang = (slot >= cum) & (slot < cum + rel_only_ang_z)
        cum += rel_only_ang_z

        ids_zero = env_ids[mask_zero]
        ids_only_y = env_ids[mask_only_y]
        ids_only_x = env_ids[mask_only_x]
        ids_only_ang = env_ids[mask_only_ang]

        # Apply zero velocity
        if len(ids_zero) > 0:
            self.vel_command_b[ids_zero, :] = 0.0
            self.is_standing_env[ids_zero] = True
            self.is_heading_env[ids_zero] = False

        # Apply only-lin_y: vx=0, vy from full range, wz=0
        if len(ids_only_y) > 0:
            r_y = torch.empty(len(ids_only_y), device=self.device)
            self.vel_command_b[ids_only_y, 0] = 0.0
            self.vel_command_b[ids_only_y, 1] = r_y.uniform_(*self._full_ranges_lin_vel_y)
            self.vel_command_b[ids_only_y, 2] = 0.0
            self.is_standing_env[ids_only_y] = False
            self.is_heading_env[ids_only_y] = False

        # Apply only-lin_x: vx from full range, vy=0, wz=0
        if len(ids_only_x) > 0:
            r_x = torch.empty(len(ids_only_x), device=self.device)
            self.vel_command_b[ids_only_x, 0] = r_x.uniform_(*self._full_ranges_lin_vel_x)
            self.vel_command_b[ids_only_x, 1] = 0.0
            self.vel_command_b[ids_only_x, 2] = 0.0
            self.is_standing_env[ids_only_x] = False
            self.is_heading_env[ids_only_x] = False

        # Apply only-ang_z: vx=0, vy=0, heading from full range
        if len(ids_only_ang) > 0:
            r_h = torch.empty(len(ids_only_ang), device=self.device)
            self.vel_command_b[ids_only_ang, 0] = 0.0
            self.vel_command_b[ids_only_ang, 1] = 0.0
            self.is_standing_env[ids_only_ang] = False
            self.is_heading_env[ids_only_ang] = True
            if self.cfg.heading_command:
                self.heading_target[ids_only_ang] = r_h.uniform_(*self.cfg.ranges.heading)


@configclass
class UniformThresholdVelocityCommandCfg(mdp.UniformVelocityCommandCfg):
    """Configuration for the uniform threshold velocity command generator."""

    class_type: type = UniformThresholdVelocityCommand

    rel_zero_vel_envs: float = 0.07
    """Fraction of environments that always receive a zero-velocity command.

    These samples prevent forgetting of standing-still behavior and are not
    affected by the command-level curriculum.
    """

    rel_only_lin_y_envs: float = 0.07
    """Fraction of environments that receive only a lateral (y) velocity command.

    vx and wz are forced to zero; vy is sampled from the original full range.
    Not affected by the command-level curriculum.
    """

    rel_only_lin_x_envs: float = 0.07
    """Fraction of environments that receive only a forward/backward (x) velocity command.

    vy and wz are forced to zero; vx is sampled from the original full range.
    Not affected by the command-level curriculum.
    """

    rel_only_ang_z_envs: float = 0.07
    """Fraction of environments that receive only an angular-velocity (heading) command.

    vx and vy are forced to zero; heading is sampled from the full range.
    Not affected by the command-level curriculum.
    """


class DiscreteCommandController(CommandTerm):
    """
    Command generator that assigns discrete commands to environments.

    Commands are stored as a list of predefined integers.
    The controller maps these commands by their indices (e.g., index 0 -> 10, index 1 -> 20).
    """

    cfg: DiscreteCommandControllerCfg
    """Configuration for the command controller."""

    def __init__(self, cfg: DiscreteCommandControllerCfg, env: ManagerBasedEnv):
        """
        Initialize the command controller.

        Args:
            cfg: The configuration of the command controller.
            env: The environment object.
        """
        # Initialize the base class
        super().__init__(cfg, env)

        # Validate that available_commands is non-empty
        if not self.cfg.available_commands:
            raise ValueError("The available_commands list cannot be empty.")

        # Ensure all elements are integers
        if not all(isinstance(cmd, int) for cmd in self.cfg.available_commands):
            raise ValueError("All elements in available_commands must be integers.")

        # Store the available commands
        self.available_commands = self.cfg.available_commands

        # Create buffers to store the command
        # -- command buffer: stores discrete action indices for each environment
        self.command_buffer = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # -- current_commands: stores a snapshot of the current commands (as integers)
        self.current_commands = [self.available_commands[0]] * self.num_envs  # Default to the first command

    def __str__(self) -> str:
        """Return a string representation of the command controller."""
        return (
            "DiscreteCommandController:\n"
            f"\tNumber of environments: {self.num_envs}\n"
            f"\tAvailable commands: {self.available_commands}\n"
        )

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """Return the current command buffer. Shape is (num_envs, 1)."""
        return self.command_buffer

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        """Update metrics for the command controller."""
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample commands for the given environments."""
        sampled_indices = torch.randint(
            len(self.available_commands), (len(env_ids),), dtype=torch.int32, device=self.device
        )
        sampled_commands = torch.tensor(
            [self.available_commands[idx.item()] for idx in sampled_indices], dtype=torch.int32, device=self.device
        )
        self.command_buffer[env_ids] = sampled_commands

    def _update_command(self):
        """Update and store the current commands."""
        self.current_commands = self.command_buffer.tolist()


@configclass
class DiscreteCommandControllerCfg(CommandTermCfg):
    """Configuration for the discrete command controller."""

    class_type: type = DiscreteCommandController

    available_commands: list[int] = []
    """
    List of available discrete commands, where each element is an integer.
    Example: [10, 20, 30, 40, 50]
    """
