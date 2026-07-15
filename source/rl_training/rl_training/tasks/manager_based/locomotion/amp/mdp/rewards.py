# AMP reward functions for humanoid locomotion
# Migrated from AMPTrainEnv.py _reward_* methods
#
# All reward state buffers (contact_filt, feet_air_time, foot_contact_trajs,
# last_actions, last_last_actions) are managed by RewardComputeHelperManager,
# accessed via env.reward_compute_helper_manager.
#
# The manager's lifecycle hooks ensure correct execution order:
#   - pre_reward_update() is called BEFORE reward computation (contact state)
#   - post_step_update() is called AFTER everything (action history)

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg

def _rsm(env: ManagerBasedEnv):
    """Shorthand for reward_compute_helper_manager."""
    return env.reward_compute_helper_manager


# ---------------------------------------------------------------------------
# Task rewards (Gaussian: weight * exp(-error / decay))
# ---------------------------------------------------------------------------
def lin_vel_tracking(env: ManagerBasedEnv, command_name: str, decay: float = 0.3) -> torch.Tensor:
    """Linear velocity tracking reward (Gaussian).

    0.65 * base_vel + 0.35 * torso_vel, then exp(-sum(square(cmd - vel)) / decay).
    """
    rsm = _rsm(env)
    robot = env.scene["robot"]
    cmd_term = env.command_manager.get_term(command_name)
    body_vel = 0.65 * robot.data.root_lin_vel_b[:, :2] + 0.35 * rsm.get_torso_lin_vel()[:, :2]
    error = torch.sum(torch.square(cmd_term.command[:, :2] - body_vel), dim=1)
    return torch.exp(-error / decay)


def ang_vel_tracking(env: ManagerBasedEnv, command_name: str, decay: float = 0.3) -> torch.Tensor:
    """Angular velocity tracking reward (Gaussian)."""
    robot = env.scene["robot"]
    cmd_term = env.command_manager.get_term(command_name)
    error = torch.square(cmd_term.command[:, 2] - robot.data.root_ang_vel_b[:, 2])
    return torch.exp(-error / decay)


# ---------------------------------------------------------------------------
# Posture rewards
# ---------------------------------------------------------------------------
def base_height(env: ManagerBasedEnv, target_height: float, decay: float = 0.02) -> torch.Tensor:
    """Base height reward (Gaussian). For flat terrain, measured_base_heights=0."""
    robot = env.scene["robot"]
    base_height = robot.data.root_pos_w[:, 2]  # flat terrain: ground = 0
    error = torch.square(base_height - target_height)
    return torch.exp(-error / decay)


def orientation(env: ManagerBasedEnv, decay: float = 0.01) -> torch.Tensor:
    """Orientation reward (Gaussian). Penalizes non-flat base."""
    robot = env.scene["robot"]
    error = torch.sum(torch.square(robot.data.projected_gravity_b[:, :2]), dim=1)
    return torch.exp(-error / decay)


def ang_vel_xy(env: ManagerBasedEnv, decay: float = 0.2) -> torch.Tensor:
    """Angular velocity XY penalty (Gaussian)."""
    robot = env.scene["robot"]
    error = torch.sum(torch.square(robot.data.root_ang_vel_b[:, :2]), dim=1)
    return torch.exp(-error / decay)


# ---------------------------------------------------------------------------
# Gait rewards
# ---------------------------------------------------------------------------
def feet_air_time(env: ManagerBasedEnv, contact_sensor_cfg: SceneEntityCfg, threshold: float = 0.32) -> torch.Tensor:
    """Reward long steps. Returns positive value (use negative weight).

    Updates feet_air_time inside the reward function to match the original
    AMPTrainEnv._reward_feet_air_time() execution order:
      1. first_contact = (feet_air_time > 0) * contact_filt  (before increment)
      2. feet_air_time += step_dt                             (increment)
      3. reward = sum(clip(threshold - feet_air_time) * first_contact)
      4. feet_air_time *= ~contact_filt                       (reset on contact)
    """
    rsm = _rsm(env)
    first_contact = (rsm.feet_air_time > 0.0) * rsm.contact_filt
    rsm.feet_air_time += env.step_dt
    reward = torch.sum(torch.clip(threshold - rsm.feet_air_time, min=0.0) * first_contact, dim=1)
    rsm.feet_air_time *= ~rsm.contact_filt
    return reward


def feet_contact(env: ManagerBasedEnv, contact_sensor_cfg: SceneEntityCfg, command_name: str) -> torch.Tensor:
    """Reward single-foot contact pattern."""
    rsm = _rsm(env)
    single_contact_steps = torch.sum(rsm.foot_contact_trajs, dim=1) == 1
    single_contact_ratio = single_contact_steps.float().mean(dim=1)
    single_contact = (single_contact_ratio > 0.01).float()

    cmd_term = env.command_manager.get_term(command_name)
    condition1 = torch.norm(cmd_term.command[:, :2], dim=1) <= 0.15  # zero_cmd_threshold_xy
    condition2 = torch.abs(cmd_term.command[:, 2]) <= 0.15  # zero_cmd_threshold_z
    zerocmd_mask = torch.logical_and(condition1, condition2)

    return torch.where(zerocmd_mask, torch.ones_like(single_contact), single_contact)


def feet_distance(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg, target_distance: float, decay: float = 0.03) -> torch.Tensor:
    """Foot distance reward (Gaussian)."""
    rsm = _rsm(env)
    robot = env.scene[asset_cfg.name]
    base_quat = robot.data.root_quat_w
    foot0 = math_utils.quat_apply_inverse(base_quat, robot.data.body_pos_w[:, rsm.r_feet_ids[0]])
    foot1 = math_utils.quat_apply_inverse(base_quat, robot.data.body_pos_w[:, rsm.r_feet_ids[1]])
    foot_distance = foot0 - foot1
    error = torch.clip(target_distance - torch.abs(foot_distance[:, 1]), min=0.0)
    return torch.exp(-error / decay)


def feet_slippage(env: ManagerBasedEnv, contact_sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize feet horizontal velocity when in contact. Returns positive (use negative weight)."""
    rsm = _rsm(env)
    robot = env.scene[asset_cfg.name]
    foot_vel_xy = robot.data.body_lin_vel_w[:, rsm.r_feet_ids, :2]
    slip_speed = torch.norm(foot_vel_xy, dim=-1)
    return torch.sum(torch.square(slip_speed) * rsm.contact_filt.float(), dim=1)


# ---------------------------------------------------------------------------
# Penalty rewards
# ---------------------------------------------------------------------------
def dof_pos_limits(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize DOF positions near soft limits."""
    robot = env.scene[asset_cfg.name]
    indices = asset_cfg.joint_ids
    soft_limits = robot.data.soft_joint_pos_limits[:, indices]
    upper = (robot.data.joint_pos[:, indices] > soft_limits[:, :, 1]).float()
    lower = (robot.data.joint_pos[:, indices] < soft_limits[:, :, 0]).float()
    return torch.sum(upper + lower, dim=1)


def dof_vel_limits(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg, soft_ratio: float = 0.8) -> torch.Tensor:
    """Penalize DOF velocities near soft limits."""
    robot = env.scene[asset_cfg.name]
    indices = asset_cfg.joint_ids
    return torch.sum(
        (torch.abs(robot.data.joint_vel[:, indices]) > robot.data.soft_joint_vel_limits[:, indices] * soft_ratio).float(),
        dim=1,
    )


def dof_torque_limits(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg, soft_ratio: float = 0.95) -> torch.Tensor:
    """Penalize torques near soft limits."""
    robot = env.scene[asset_cfg.name]
    indices = asset_cfg.joint_ids
    limits = robot.data.joint_effort_limits[:, indices]
    return torch.sum(
        (torch.abs(robot.data.applied_torque[:, indices]) > limits * soft_ratio).float(),
        dim=1,
    )


# ---------------------------------------------------------------------------
# Regularization rewards
# ---------------------------------------------------------------------------
def dof_vel_l2(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize high DOF velocity."""
    robot = env.scene[asset_cfg.name]
    return torch.sum(torch.square(robot.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def dof_torque_l2(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg, torque_coeffs: dict | None = None) -> torch.Tensor:
    """Penalize DOF torque with per-joint coefficients."""
    robot = env.scene[asset_cfg.name]
    indices = asset_cfg.joint_ids
    torques = robot.data.applied_torque[:, indices]

    if torque_coeffs is not None:
        # Build coefficient tensor from robot's joint names
        joint_names = robot.data.joint_names
        coeffs_full = torch.ones(len(joint_names), device=env.device)
        for i, name in enumerate(joint_names):
            if name in torque_coeffs:
                coeffs_full[i] = torque_coeffs[name]
        coeffs = coeffs_full[indices]
        return torch.sum(torch.square(torques) * coeffs, dim=1)
    return torch.sum(torch.square(torques), dim=1)


def power(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Cost of transportation (torque * velocity)."""
    robot = env.scene[asset_cfg.name]
    indices = asset_cfg.joint_ids
    return torch.sum(
        torch.abs(robot.data.applied_torque[:, indices] * robot.data.joint_vel[:, indices]),
        dim=1,
    )


def action_l2(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg, action_coeffs: dict | None = None) -> torch.Tensor:
    """Penalize actions with per-joint coefficients."""
    actions = env.action_manager.action

    if action_coeffs is not None:
        # Build coefficient tensor from robot's joint names
        robot = env.scene[asset_cfg.name]
        joint_names = robot.data.joint_names
        coeffs = torch.ones(len(joint_names), device=env.device)
        for i, name in enumerate(joint_names):
            if name in action_coeffs:
                coeffs[i] = action_coeffs[name]
        return torch.sum(torch.square(actions) * coeffs, dim=1)
    return torch.sum(torch.square(actions), dim=1)


def action_rate_l2(env: ManagerBasedEnv) -> torch.Tensor:
    """Penalize changes in actions."""
    rsm = _rsm(env)
    return torch.sum(torch.square(rsm.last_actions - env.action_manager.action), dim=1)


def smoothness_l2(env: ManagerBasedEnv) -> torch.Tensor:
    """Penalize jerk (second-order action changes)."""
    rsm = _rsm(env)
    actions = env.action_manager.action
    return torch.sum(torch.square(actions - 2 * rsm.last_actions + rsm.last_last_actions), dim=1)


# ---------------------------------------------------------------------------
# Constraint rewards
# ---------------------------------------------------------------------------
def hipz_deviation(
    env: ManagerBasedEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=[".*hip_z.*"]),
) -> torch.Tensor:
    """Penalize hip_z joint deviation from default (scaled by angular velocity command).

    Args:
        asset_cfg: SceneEntityCfg specifying the hip_z joints.  Defaults to
            ``joint_names=[".*hip_z.*"]`` which matches most humanoid robots.
            Override in robot-specific config with explicit joint names.
    """
    robot = env.scene[asset_cfg.name]
    cmd_term = env.command_manager.get_term(command_name)

    z_cmd_abs = torch.abs(cmd_term.command[:, 2])
    yaw_abs = torch.abs(robot.data.root_ang_vel_b[:, 2])

    cmd_normalized = torch.clamp(z_cmd_abs / 1.2, 0.0, 1.0)
    yaw_normalized = torch.clamp(yaw_abs / 1.2, 0.0, 1.0)
    coeff = torch.clamp(0.9 * cmd_normalized + 0.1 * yaw_normalized, 0.0, 1.0)

    indices = asset_cfg.joint_ids
    base_penalty = torch.sum(
        torch.square(
            robot.data.joint_pos[:, indices] - robot.data.default_joint_pos[:, indices]
        ),
        dim=1,
    )
    penalty_scale = 1.0 - 0.999 * coeff
    return base_penalty * penalty_scale


def dof_err(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg, jerr_coeffs: dict | None = None) -> torch.Tensor:
    """Penalize joint deviation from default with per-joint coefficients."""
    robot = env.scene[asset_cfg.name]
    indices = asset_cfg.joint_ids
    dof_err = robot.data.joint_pos[:, indices] - robot.data.default_joint_pos[:, indices]

    if jerr_coeffs is not None:
        # Build coefficient tensor from robot's joint names
        joint_names = robot.data.joint_names
        coeffs_full = torch.ones(len(joint_names), device=env.device)
        for i, name in enumerate(joint_names):
            if name in jerr_coeffs:
                coeffs_full[i] = jerr_coeffs[name]
        coeffs = coeffs_full[indices]
        return torch.sum(torch.square(dof_err * coeffs), dim=1)
    return torch.sum(torch.square(dof_err), dim=1)


def stand_still(env: ManagerBasedEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint deviation from default at zero commands."""
    robot = env.scene[asset_cfg.name]
    cmd_term = env.command_manager.get_term(command_name)

    condition1 = torch.norm(cmd_term.command[:, :2], dim=1) <= 0.15
    condition2 = torch.abs(cmd_term.command[:, 2]) <= 0.15
    zerocmd_mask = torch.logical_and(condition1, condition2)

    indices = asset_cfg.joint_ids
    dof_err = robot.data.joint_pos[:, indices] - robot.data.default_joint_pos[:, indices]
    return torch.sum(torch.abs(dof_err), dim=1) * zerocmd_mask.float()


def collision(env: ManagerBasedEnv, contact_sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    """Penalize undesired contacts (hands/wrists)."""
    contact_sensor = env.scene[contact_sensor_cfg.name]
    # Penalize contact bodies: hands and wrist links
    penalize_ids, _ = contact_sensor.find_bodies(".*hand_link|.*wrist_*_link")
    net_contact_forces = contact_sensor.data.net_forces_w
    penalize_contacts = torch.norm(net_contact_forces[:, penalize_ids], dim=-1) > threshold
    return torch.sum(penalize_contacts.float(), dim=1)


# ---------------------------------------------------------------------------
# AMP reward (placeholder)
# ---------------------------------------------------------------------------
def amp_reward(env: ManagerBasedEnv) -> torch.Tensor:
    """AMP discriminator reward. Returns zeros until discriminator is integrated."""
    return torch.zeros(env.num_envs, device=env.device)
