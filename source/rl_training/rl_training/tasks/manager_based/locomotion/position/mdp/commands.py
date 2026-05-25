# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause
# 
# # Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import torch
from dataclasses import MISSING
from typing import TYPE_CHECKING, Sequence

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

import rl_training.tasks.manager_based.locomotion.position.mdp as mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class UniformThresholdVelocityCommand(mdp.UniformVelocityCommand):
    """Command generator that generates a velocity command in SE(2) from uniform distribution with threshold."""

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

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        if env_ids is None:
            env_ids = slice(None)

        extras = {}
        for metric_name, metric_value in self.metrics.items():
            if metric_name == "success_point_num":
                success_mask = self._success_reached[env_ids]
                if torch.any(success_mask):
                    extras[metric_name] = torch.mean(self._reached_point_count[env_ids][success_mask].float()).item()
                else:
                    extras[metric_name] = 0.0
            elif metric_name in {"base_z", "knee_pos"}:
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


@configclass
class UniformThresholdVelocityCommandCfg(mdp.UniformVelocityCommandCfg):
    """Configuration for the uniform threshold velocity command generator."""

    class_type: type = UniformThresholdVelocityCommand


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


class WaypointPositionCommand(CommandTerm):
    """Command generator that produces a sequence of waypoints for each environment.

    Waypoints are sampled using polar coordinates relative to the previous point:
    each new point is placed at distance ``r`` from the last point, with a uniformly
    random angle theta.  The command output is the **relative position** (dx, dy)
    and **heading error** (dheading) from the robot's base frame to the current
    target waypoint.  When the robot gets within ``reach_threshold`` of the current
    waypoint, the target advances to the next one.

    Command shape: (num_envs, 4) → [dx_b, dy_b, dz_b, dheading]
    """

    cfg: WaypointPositionCommandCfg

    def __init__(self, cfg: WaypointPositionCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.robot = env.scene[cfg.asset_name]

        n = self.num_envs
        max_wp = cfg.max_waypoints

        # ---- waypoint storage (per-env) ----
        # positions in local frame (relative to env origin): (num_envs, max_wp, 2)
        self.wp_pos = torch.zeros(n, max_wp, 2, device=self.device)
        # desired velocity at each waypoint: (num_envs, max_wp, 2)
        self.wp_vel = torch.zeros(n, max_wp, 2, device=self.device)
        # motion velocity of each waypoint itself: (num_envs, max_wp, 2)
        self.wp_move_vel = torch.zeros(n, max_wp, 2, device=self.device)
        # acceleration/linear coefficients and direction for waypoint self-motion
        self.wp_move_a = torch.zeros(n, max_wp, device=self.device)
        self.wp_move_b = torch.zeros(n, max_wp, device=self.device)
        self.wp_move_dir = torch.zeros(n, max_wp, 2, device=self.device)
        # elapsed motion time for each waypoint
        self.wp_move_t = torch.zeros(n, max_wp, device=self.device)
        # number of waypoints per env
        self.wp_count = torch.zeros(n, dtype=torch.long, device=self.device)
        # current target index per env
        self.wp_idx = torch.zeros(n, dtype=torch.long, device=self.device)

        # ---- command output buffers ----
        # target position in world frame
        self.pos_command_w = torch.zeros(n, 3, device=self.device)
        # relative position in base frame (dx, dy, dz)
        self.pos_command_b = torch.zeros(n, 3, device=self.device)
        # heading error in base frame
        self.heading_command_b = torch.zeros(n, device=self.device)

        # ---- metrics ----
        self.metrics["error_pos"] = torch.zeros(n, device=self.device)
        self.metrics["error_heading"] = torch.zeros(n, device=self.device)
        self.metrics["success_point_num"] = torch.zeros(n, device=self.device)
        self.metrics["continuous_point_motion_dt_scale"] = torch.zeros(n, device=self.device)

        # Track how many waypoints have been reached in the current episode.
        self._reached_point_count = torch.zeros(n, dtype=torch.long, device=self.device)
        self._success_reached = torch.zeros(n, dtype=torch.bool, device=self.device)

        # ---- curriculum for continuous_point_motion_dt_scale ----
        # Runtime value starts from config default and increases as success improves.
        self._continuous_point_motion_dt_scale: float = cfg.continuous_point_motion_dt_scale
        # EMA of per-reset-group mean_success; smooths noise from partial resets.
        self._ema_mean_success: float = 0.0
        # After each increment, block further increments until EMA drops below
        # threshold (proving the current difficulty actually challenges the policy).
        self._curriculum_can_advance: bool = True

    def __str__(self) -> str:
        return (
            "WaypointPositionCommand:\n"
            f"\tCommand dim: {tuple(self.command.shape[1:])}\n"
            f"\tMax waypoints: {self.cfg.max_waypoints}\n"
            f"\tReach threshold: {self.cfg.reach_threshold}\n"
            f"\tResampling time: {self.cfg.resampling_time_range}"
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Reset metrics, run curriculum check, and resample commands."""
        if env_ids is None:
            env_ids = slice(None)

        extras = {}
        for metric_name, metric_value in self.metrics.items():
            if metric_name == "continuous_point_motion_dt_scale":
                # Log the current curriculum value (global scalar, not per-env mean)
                extras[metric_name] = self._continuous_point_motion_dt_scale
                # Keep metric tensor in sync (don't reset to 0 – it's a curriculum value)
                metric_value.fill_(self._continuous_point_motion_dt_scale)
            else:
                extras[metric_name] = torch.mean(metric_value[env_ids]).item()
                metric_value[env_ids] = 0.0

        # ---- Curriculum for continuous_point_motion_dt_scale ----
        # Use an EMA of per-reset-group mean_success to smooth noise from
        # partial resets.  Only advance dt_scale when the EMA (which reflects
        # the global trend) crosses the threshold from below.  After each
        # advance, block further advances until the EMA drops below threshold
        # (proving the current difficulty actually challenges the policy).
        if self.cfg.continuous_point_motion_dt_scale_curriculum:
            mean_success = extras.get("success_point_num", 0.0)
            # Update EMA
            alpha = self.cfg.continuous_point_motion_dt_scale_curriculum_ema_alpha
            self._ema_mean_success = alpha * mean_success + (1 - alpha) * self._ema_mean_success

            threshold = self.cfg.continuous_point_motion_dt_scale_curriculum_threshold * (self.cfg.num_waypoints - 1)
            increment = self.cfg.continuous_point_motion_dt_scale_curriculum_increment
            if self._ema_mean_success < threshold:
                self._curriculum_can_advance = True
            elif self._curriculum_can_advance:
                self._continuous_point_motion_dt_scale += increment
                self._curriculum_can_advance = False

        self.command_counter[env_ids] = 0
        self._resample(env_ids)
        return extras

    # ------------------------------------------------------------------ #
    # Properties                                                          #
    # ------------------------------------------------------------------ #

    @property
    def command(self) -> torch.Tensor:
        """(num_envs, 4): [dx_b, dy_b, dz_b, dheading]."""
        return torch.cat(
            [self.pos_command_b, self.heading_command_b.unsqueeze(1)], dim=1
        )

    # ------------------------------------------------------------------ #
    # Implementation-specific functions                                    #
    # ------------------------------------------------------------------ #

    def _update_metrics(self):
        self.metrics["error_pos"] += torch.norm(self.pos_command_b[:, :2], dim=1)
        self.metrics["error_heading"] += torch.abs(self.heading_command_b)
        self.metrics["success_point_num"] = self._reached_point_count.float()

        # Keep the metric tensor in sync for logging
        self.metrics["continuous_point_motion_dt_scale"].fill_(
            self._continuous_point_motion_dt_scale
        )

    def _resample_command(self, env_ids: Sequence[int]):
        """Generate a new waypoint sequence for the given envs."""
        n_env = len(env_ids)
        cfg = self.cfg

        # ---- generate waypoints ----
        num_wp = cfg.num_waypoints  # how many points to generate (fixed)
        r_range = cfg.r_range
        theta_min = cfg.theta_min
        theta_max = cfg.theta_max
        device = self.device

        # positions (local, relative to env origin)
        pos = torch.zeros(n_env, num_wp, 2, device=device)
        # velocities
        vel = torch.zeros(n_env, num_wp, 2, device=device)

        prev_heading = torch.empty(n_env, device=device).uniform_(0.0, 2.0 * math.pi)
        for i in range(1, num_wp):
            # Sample radius and heading change. For i >= 2, theta is the angle
            # between (p_i - p_{i-1}) and (p_{i-1} - p_{i-2}).
            r = torch.empty(n_env, device=device).uniform_(*r_range)
            if i == 1:
                heading = prev_heading
            else:
                dtheta = torch.empty(n_env, device=device).uniform_(
                    theta_min * 2.0 * math.pi, theta_max * 2.0 * math.pi
                )
                heading = prev_heading + dtheta
                prev_heading = heading

            pos[:, i, 0] = pos[:, i - 1, 0] + r * torch.cos(heading)
            pos[:, i, 1] = pos[:, i - 1, 1] + r * torch.sin(heading)

            # desired velocity at point i-1: direction → point i, magnitude uniform
            dir_vec = pos[:, i, :] - pos[:, i - 1, :]
            dir_norm = torch.norm(dir_vec, dim=1, keepdim=True).clamp(min=1e-6)
            dir_unit = dir_vec / dir_norm
            v_mag = torch.empty(n_env, 1, device=device).uniform_(0.0, 1.0)
            vel[:, i - 1, :] = v_mag * dir_unit

        # Move the generated points in-place.
        move_dt = cfg.point_move_dt
        for current_idx in range(num_wp):
            old_pos = pos[:, current_idx, :].clone()

            a = torch.empty(n_env, device=device).uniform_(0.0, cfg.point_move_a_max)
            b = torch.empty(n_env, device=device).uniform_(0.0, cfg.point_move_b_max)
            theta = torch.empty(n_env, device=device).uniform_(0, 2 * math.pi)
            distance = a * (move_dt**2) + b * move_dt

            pos[:, current_idx, 0] = old_pos[:, 0] + distance * torch.cos(theta)
            pos[:, current_idx, 1] = old_pos[:, 1] + distance * torch.sin(theta)

            move_dir = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
            self.wp_move_dir[env_ids, current_idx, :] = move_dir
            self.wp_move_a[env_ids, current_idx] = a
            self.wp_move_b[env_ids, current_idx] = b
            # initial instantaneous velocity at t=0 is b * dir
            self.wp_move_vel[env_ids, current_idx, :] = b.unsqueeze(1) * move_dir
            self.wp_move_t[env_ids, current_idx] = 0.0

            if current_idx > 0:
                last_idx = current_idx - 1
                dir_vec = pos[:, current_idx, :] - pos[:, last_idx, :]
                dir_norm = torch.norm(dir_vec, dim=1, keepdim=True).clamp(min=1e-6)
                dir_unit = dir_vec / dir_norm
                v_mag = torch.empty(n_env, 1, device=device).uniform_(cfg.point_move_vE_min, cfg.point_move_vE_max)
                vel[:, last_idx, :] = v_mag * dir_unit

        # last point velocity = 0 (already zero from init)
        if num_wp > 0:
            last_idx = num_wp - 1
            self.wp_move_vel[env_ids, last_idx, :] = 0.0
            self.wp_move_a[env_ids, last_idx] = 0.0
            self.wp_move_b[env_ids, last_idx] = 0.0
            self.wp_move_t[env_ids, last_idx] = 0.0

        # write to buffers
        self.wp_pos[env_ids, :num_wp, :] = pos
        self.wp_vel[env_ids, :num_wp, :] = vel
        self.wp_count[env_ids] = num_wp
        self.wp_idx[env_ids] = 1  # start targeting the 1st real waypoint (index 1)
        self._reached_point_count[env_ids] = 0
        self._success_reached[env_ids] = False
        self.wp_move_t[env_ids, :] = 0.0

        # set current target world position
        self._set_target_from_index(env_ids)

    def _update_command(self):
        """Compute relative position & heading from robot to current target waypoint.

        Also checks whether the robot has reached the current target and, if so,
        advances to the next waypoint.
        """
        if self.cfg.continuous_point_motion:
            self._move_waypoints_continuously()
            self._set_target_from_index(torch.arange(self.num_envs, device=self.device))

        self._update_relative_command_buffers()

        # --- check reach & advance ---
        dist = torch.norm(self.pos_command_b[:, :2], dim=1)
        reached = dist < self.cfg.reach_threshold
        final_target = self.wp_idx >= self.wp_count - 1
        newly_successful = reached & final_target & ~self._success_reached
        if torch.any(newly_successful):
            self._reached_point_count[newly_successful] += 1
            self._success_reached[newly_successful] = True

        can_advance = reached & (self.wp_idx < self.wp_count - 1)
        advance_ids = can_advance.nonzero(as_tuple=False).flatten()
        if len(advance_ids) > 0:
            self._reached_point_count[advance_ids] += 1
            self.wp_idx[advance_ids] += 1
            # new target starts its own motion clock from 0
            self.wp_move_t[advance_ids, self.wp_idx[advance_ids]] = 0.0
            self._set_target_from_index(advance_ids)
            # Refresh command buffers immediately so the next waypoint becomes
            # visible in the same simulation step.
            self._update_relative_command_buffers(advance_ids)

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _set_target_from_index(self, env_ids: Sequence[int]):
        """Set pos_command_w from wp_pos[env_ids, wp_idx]."""
        idx = self.wp_idx[env_ids]  # (len(env_ids),)
        local_xy = self.wp_pos[env_ids, idx]  # (len(env_ids), 2)
        # world = env_origin + local
        self.pos_command_w[env_ids, :2] = (
            self._env.scene.env_origins[env_ids, :2] + local_xy
        )
        # z = fixed target height from config
        self.pos_command_w[env_ids, 2] = self.cfg.target_z

    def _update_relative_command_buffers(self, env_ids: Sequence[int] | None = None):
        """Update relative target position and heading buffers."""
        if env_ids is None:
            env_ids = slice(None)

        from isaaclab.utils.math import quat_apply_inverse, wrap_to_pi, yaw_quat

        target_vec = self.pos_command_w[env_ids] - self.robot.data.root_pos_w[env_ids, :3]
        self.pos_command_b[env_ids] = quat_apply_inverse(
            yaw_quat(self.robot.data.root_quat_w[env_ids]), target_vec
        )

        target_heading = torch.atan2(target_vec[:, 1], target_vec[:, 0])
        self.heading_command_b[env_ids] = wrap_to_pi(
            target_heading - self.robot.data.heading_w[env_ids]
        )

    def _move_waypoints_continuously(self):
        """Move only the current target waypoint for each environment."""
        dt = self._env.step_dt * self._continuous_point_motion_dt_scale
        env_ids = torch.arange(self.num_envs, device=self.device)
        target_idx = self.wp_idx
        final_idx = self.wp_count - 1
        movable = target_idx < final_idx
        if not torch.any(movable):
            return
        env_ids = env_ids[movable]
        target_idx = target_idx[movable]
        self.wp_move_t[env_ids, target_idx] += dt
        t = self.wp_move_t[env_ids, target_idx]
        a = self.wp_move_a[env_ids, target_idx]
        b = self.wp_move_b[env_ids, target_idx]
        speed = (a * t + b).clamp(min=0.0)
        move_dir = self.wp_move_dir[env_ids, target_idx, :]
        self.wp_move_vel[env_ids, target_idx, :] = speed.unsqueeze(1) * move_dir
        self.wp_pos[env_ids, target_idx, :] += self.wp_move_vel[env_ids, target_idx, :] * dt


@configclass
class WaypointPositionCommandCfg(CommandTermCfg):
    """Configuration for :class:`WaypointPositionCommand`."""

    class_type: type = WaypointPositionCommand

    asset_name: str = MISSING
    """Name of the robot asset in the scene."""

    num_waypoints: int = 10
    """Number of waypoints to generate per resample."""

    max_waypoints: int = 50
    """Maximum number of waypoints (buffer size). Must be >= num_waypoints."""

    r_range: tuple[float, float] = (0.5, 3.0)
    """Range of polar radius for waypoint sampling."""

    theta_min: float = 0.0
    """Minimum signed turn angle in cycles between consecutive segments."""

    theta_max: float = 1.0
    """Maximum signed turn angle in cycles between consecutive segments."""

    reach_threshold: float = 0.3
    """Distance threshold (m) to consider a waypoint reached."""

    target_z: float = 0.45
    """Fixed world-frame z target used for waypoint command."""

    point_move_dt: float = 1.0
    """Time step used by the waypoint movement rule."""

    point_move_a_max: float = 1.0
    """Maximum quadratic coefficient for waypoint displacement."""

    point_move_b_max: float = 0.5
    """Maximum linear coefficient for waypoint displacement."""

    point_move_vE_max: float = 2.0
    """Maximum speed magnitude when updating the previous waypoint velocity."""

    point_move_vE_min: float = 0.0
    """Minimum speed magnitude when updating the previous waypoint velocity."""
    continuous_point_motion: bool = False
    """Whether to move waypoints continuously at every simulation step."""

    continuous_point_motion_dt_scale: float = 1.0
    """Scale factor applied to env step_dt for continuous waypoint movement."""

    continuous_point_motion_dt_scale_curriculum: bool = False
    """Whether to enable curriculum learning for continuous_point_motion_dt_scale.

    When enabled, dt_scale is incremented by ``increment`` each time the EMA of
    mean success_point_num crosses the threshold from below.  After each increment,
    further advances are blocked until the EMA drops below threshold (proving the
    current difficulty actually challenges the policy) and then recovers.
    """

    continuous_point_motion_dt_scale_curriculum_threshold: float = 0.6
    """Ratio of (num_waypoints - 1) used as the curriculum threshold.

    The actual threshold value is ``threshold * (num_waypoints - 1)``.
    For example, with threshold=0.6 and num_waypoints=12, the trigger value is 6.6.
    """

    continuous_point_motion_dt_scale_curriculum_increment: float = 0.2
    """Amount to add to continuous_point_motion_dt_scale when the curriculum threshold is reached."""

    continuous_point_motion_dt_scale_curriculum_ema_alpha: float = 0.05
    """Smoothing factor for the EMA of mean success_point_num.

    Lower values make the EMA smoother (less sensitive to per-reset noise from
    partial environment resets).  A typical range is 0.01–0.1.
    """

    # Make resampling_time_range optional with a large default so waypoints
    # persist for the whole episode unless explicitly resampled.
    resampling_time_range: tuple[float, float] = (1000.0, 1000.0)
