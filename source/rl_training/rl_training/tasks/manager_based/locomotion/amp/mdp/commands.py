# AMP velocity command generator for CR1 humanoid
# Migrated from AMPTrainEnv.py command logic (_resample_commands, _halfway_resample_commands, _update_vel_commands)

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class AmpVelocityCommand(CommandTerm):
    """AMP velocity command generator for humanoid locomotion.

    Command has 5 dimensions: [lin_vel_x, lin_vel_y, ang_vel_yaw, heading, raw_ang_vel_yaw].

    Features migrated from AMPTrainEnv.py:
    - Heading-based angular velocity control with tracking_strength
    - Zero command rate (randomly zero commands)
    - Pure angular velocity envs (15% of envs use raw ang_vel directly)
    - In-place turning after threshold time
    - Lateral velocity scaling (vy reduced when vx is large)
    - Small command thresholding
    - cmd_flag binary indicator (non-zero command)

    The ``cmd_flag`` property is used by observation functions to indicate
    whether the current command is non-zero.
    """

    cfg: "AmpVelocityCommandCfg"

    def __init__(self, cfg: "AmpVelocityCommandCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)

        # robot asset
        self.robot: Articulation = env.scene[cfg.asset_name]

        # command buffer: [vx, vy, wz, heading, raw_wz]
        self.vel_command_b = torch.zeros(self.num_envs, 5, device=self.device)

        # pure angular velocity env mask
        self.pure_ang_vel_env_mask = torch.rand(self.num_envs, device=self.device) < cfg.pure_ang_vel_env_ratio

        # cmd_flag: binary indicator of non-zero command
        self.cmd_flag = torch.zeros(self.num_envs, 1, device=self.device)

        # metrics
        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "AmpVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tHeading command: {self.cfg.heading_command}\n"
        msg += f"\tZero command rate: {self.cfg.zero_command_rate}\n"
        msg += f"\tPure ang vel env ratio: {self.cfg.pure_ang_vel_env_ratio}"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired velocity command. Shape is (num_envs, 5).

        [lin_vel_x, lin_vel_y, ang_vel_yaw, heading, raw_ang_vel_yaw]
        """
        return self.vel_command_b

    """
    Implementation specific functions
    """

    def _update_metrics(self):
        # Update cmd_flag
        condition1 = torch.norm(self.vel_command_b[:, :2], dim=1) > self.cfg.zero_cmd_threshold_xy
        condition2 = torch.abs(self.vel_command_b[:, 2]) > self.cfg.zero_cmd_threshold_z
        self.cmd_flag = torch.logical_or(condition1, condition2).unsqueeze(-1).float()

        # Error metrics
        self.metrics["error_vel_xy"] += torch.norm(
            self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2], dim=-1
        )
        self.metrics["error_vel_yaw"] += torch.abs(
            self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2]
        )

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return

        r = torch.empty(len(env_ids), device=self.device)

        # Sample linear velocity
        self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)

        # Zero command rate logic
        zero_cmd_dec = torch.rand(len(env_ids), device=self.device) < self.cfg.zero_command_rate
        nonzero_cmd = torch.norm(self.vel_command_b[env_ids, :2], dim=1) > 0
        # Since we're in resample, is_halfway is always True -> shall_keep_moving = nonzero_cmd
        shall_stop = zero_cmd_dec & ~nonzero_cmd

        stop_ids = env_ids[shall_stop] if shall_stop.any() else torch.empty(0, dtype=torch.long, device=self.device)
        if len(stop_ids) > 0:
            self.vel_command_b[stop_ids, :] = 0.0

        # Heading / angular velocity
        if self.cfg.heading_command:
            self.vel_command_b[env_ids, 3] = r.uniform_(*self.cfg.ranges.heading)
            self.vel_command_b[env_ids, 4] = r.uniform_(*self.cfg.ranges.ang_vel_yaw)
            self.vel_command_b[env_ids, 4] *= torch.abs(self.vel_command_b[env_ids, 4]) > self.cfg.zero_cmd_threshold_z

            # Zero heading for stopped envs
            if len(stop_ids) > 0:
                base_quat = self.robot.data.root_quat_w[stop_ids]
                forward = math_utils.quat_apply(
                    base_quat,
                    torch.tensor([1.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(len(stop_ids), 1),
                )
                heading = torch.atan2(forward[:, 1], forward[:, 0])
                self.vel_command_b[stop_ids, 3] = heading
        else:
            self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_yaw)
            if len(stop_ids) > 0:
                self.vel_command_b[stop_ids, 2] = 0.0

        # Lateral velocity scaling
        self.vel_command_b[env_ids, 1] /= torch.clip(
            3.0 * torch.abs(self.vel_command_b[env_ids, 0]), min=1.0
        )

        # Small command thresholding
        self.vel_command_b[env_ids, :2] *= (
            torch.norm(self.vel_command_b[env_ids, :2], dim=1) > self.cfg.zero_cmd_threshold_xy
        ).unsqueeze(1)

    def _update_command(self):
        """Update angular velocity from heading error (called every step)."""
        if not self.cfg.heading_command:
            return

        # Compute current heading from robot forward direction
        base_quat = self.robot.data.root_quat_w
        forward = math_utils.quat_apply(
            base_quat,
            torch.tensor([1.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1),
        )
        heading = torch.atan2(forward[:, 1], forward[:, 0])

        # Compute angular velocity from heading error (non-pure-ang-vel envs)
        update_mask = ~self.pure_ang_vel_env_mask
        ranges = self.cfg.ranges

        cmd_speed = torch.clip(torch.norm(self.vel_command_b[:, :2], dim=-1), min=1.0)
        ang_vel_yaw_lower = ranges.ang_vel_yaw[0] / cmd_speed
        ang_vel_yaw_upper = ranges.ang_vel_yaw[1] / cmd_speed

        self.vel_command_b[update_mask, 2] = torch.clip(
            self.cfg.tracking_strength * math_utils.wrap_to_pi(
                self.vel_command_b[update_mask, 3] - heading[update_mask]
            ),
            ang_vel_yaw_lower[update_mask],
            ang_vel_yaw_upper[update_mask],
        )

        # Pure angular velocity envs use raw_ang_vel directly
        self.vel_command_b[self.pure_ang_vel_env_mask, 2] = self.vel_command_b[self.pure_ang_vel_env_mask, 4]

        # In-place turning: if speed is near-zero and enough time has passed
        inplace_turn_ids = (
            (torch.norm(self.vel_command_b[:, :2], dim=-1) < self.cfg.zero_cmd_threshold_xy)
            & (self._env.episode_length_buf >= int(self.cfg.inplace_turn_time / self._env.step_dt))
        ).nonzero(as_tuple=False).flatten()
        self.vel_command_b[inplace_turn_ids, 2] = self.vel_command_b[inplace_turn_ids, 4]

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass


@configclass
class AmpVelocityCommandCfg(CommandTermCfg):
    """Configuration for the AMP velocity command generator."""

    class_type: type = AmpVelocityCommand

    asset_name: str = MISSING
    """Name of the robot asset in the environment scene."""

    heading_command: bool = True
    """Whether to use heading-based angular velocity control."""

    tracking_strength: float = 0.5
    """Scale factor for heading error to angular velocity."""

    zero_command_rate: float = 0.2
    """Probability of zeroing commands at each resample."""

    inplace_turn_time: float = 12.0
    """Time (s) before in-place turning activates for zero-speed commands."""

    zero_cmd_threshold_xy: float = 0.15
    """Threshold below which xy commands are zeroed."""

    zero_cmd_threshold_z: float = 0.15
    """Threshold below which z angular velocity commands are zeroed."""

    pure_ang_vel_env_ratio: float = 0.15
    """Fraction of envs that use raw angular velocity instead of heading control."""

    @configclass
    class Ranges:
        """Distribution ranges for the velocity commands."""

        lin_vel_x: tuple[float, float] = MISSING
        """Range for linear-x velocity command (m/s)."""

        lin_vel_y: tuple[float, float] = MISSING
        """Range for linear-y velocity command (m/s)."""

        ang_vel_yaw: tuple[float, float] = MISSING
        """Range for angular-yaw velocity command (rad/s)."""

        heading: tuple[float, float] | None = None
        """Range for heading command (rad). Used only if heading_command is True."""

    ranges: Ranges = MISSING
    """Distribution ranges for the velocity commands."""
