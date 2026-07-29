from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg

def _ahm(env: ManagerBasedEnv):
    """Shorthand for amp_helper_manager."""
    return env.amp_helper_manager


# ---------------------------------------------------------------------------
# Task rewards (Gaussian: weight * exp(-error / decay))
# ---------------------------------------------------------------------------
def lin_vel_tracking(env: ManagerBasedEnv, command_name: str, decay: float = 0.3) -> torch.Tensor:
    """Linear velocity tracking reward (Gaussian)."""
    ahm = _ahm(env)
    robot = env.scene["robot"]
    cmd_term = env.command_manager.get_term(command_name)
    body_vel = 0.65 * robot.data.root_lin_vel_b[:, :2] + 0.35 * ahm.get_torso_lin_vel()[:, :2]
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


def orientation(env: ManagerBasedEnv) -> torch.Tensor:
    """Orientation penalty (linear). Returns positive value; use negative weight."""
    robot = env.scene["robot"]
    return torch.sum(torch.square(robot.data.projected_gravity_b[:, :2]), dim=1)


def orientation_gaussian(env: ManagerBasedEnv, decay: float = 0.02) -> torch.Tensor:
    """Orientation reward (Gaussian). Use positive weight."""
    robot = env.scene["robot"]
    error = torch.sum(torch.square(robot.data.projected_gravity_b[:, :2]), dim=1)
    return torch.exp(-error / decay)


def ang_vel_xy(env: ManagerBasedEnv) -> torch.Tensor:
    """Angular velocity XY penalty (linear). Returns positive value; use negative weight."""
    robot = env.scene["robot"]
    return torch.sum(torch.square(robot.data.root_ang_vel_b[:, :2]), dim=1)


def ang_vel_xy_gaussian(env: ManagerBasedEnv, decay: float = 0.02) -> torch.Tensor:
    """Angular velocity XY reward (Gaussian). Use positive weight."""
    robot = env.scene["robot"]
    error = torch.sum(torch.square(robot.data.root_ang_vel_b[:, :2]), dim=1)
    return torch.exp(-error / decay)


# ---------------------------------------------------------------------------
# Gait rewards
# ---------------------------------------------------------------------------
def feet_air_time(env: ManagerBasedEnv, contact_sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps with dynamic target. Returns positive value (use negative weight)."""
    ahm = _ahm(env)
    cmd_term = env.command_manager.get_term("base_velocity")
    first_contact = (ahm.feet_air_time > 0.0) * ahm.contact
    ahm.feet_air_time += env.step_dt

    # Dynamic target: 0.36s at zero cmd, 0.22s at max cmd
    cmd_speed = torch.norm(cmd_term.command[:, :2], dim=1)
    yaw_speed = torch.abs(cmd_term.command[:, 2])
    v_th = 0.15  # zero_cmd_threshold_xy
    w_th = 0.15  # zero_cmd_threshold_z
    speed_norm = torch.clamp((cmd_speed - v_th) / (1.4 - v_th), 0.0, 1.0)
    yaw_norm = torch.clamp((yaw_speed - w_th) / (1.2 - w_th), 0.0, 1.0)
    cmd_level = torch.max(speed_norm, yaw_norm)
    target = (0.36 - cmd_level * (0.36 - 0.22)).unsqueeze(1)
    reward = torch.sum(torch.clip(target - ahm.feet_air_time, min=0.0) / target * first_contact, dim=1)
    ahm.feet_air_time *= ~ahm.contact
    return reward


def feet_contact(env: ManagerBasedEnv, contact_sensor_cfg: SceneEntityCfg, command_name: str) -> torch.Tensor:
    """Reward single-foot contact pattern."""
    ahm = _ahm(env)
    single_contact_steps = torch.sum(ahm.foot_contact_trajs, dim=1) == 1
    single_contact_ratio = single_contact_steps.float().mean(dim=1)
    single_contact = (single_contact_ratio >= 0.2).float()

    cmd_term = env.command_manager.get_term(command_name)
    condition1 = torch.norm(cmd_term.command[:, :2], dim=1) <= 0.15  # zero_cmd_threshold_xy
    condition2 = torch.abs(cmd_term.command[:, 2]) <= 0.15  # zero_cmd_threshold_z
    zerocmd_mask = torch.logical_and(condition1, condition2)

    return torch.where(zerocmd_mask, torch.ones_like(single_contact), single_contact)


def feet_distance(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg, target_distance: float, decay: float = 0.03) -> torch.Tensor:
    """Foot distance reward (Gaussian)."""
    ahm = _ahm(env)
    robot = env.scene[asset_cfg.name]
    base_quat = robot.data.root_quat_w
    foot0 = math_utils.quat_apply_inverse(base_quat, robot.data.body_pos_w[:, ahm.r_feet_ids[0]])
    foot1 = math_utils.quat_apply_inverse(base_quat, robot.data.body_pos_w[:, ahm.r_feet_ids[1]])
    foot_distance = foot0 - foot1
    error = torch.clip(target_distance - torch.abs(foot_distance[:, 1]), min=0.0)
    return torch.exp(-error / decay)


def feet_slippage(env: ManagerBasedEnv, contact_sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize feet horizontal velocity when in contact."""
    ahm = _ahm(env)
    robot = env.scene[asset_cfg.name]
    foot_vel_xy = robot.data.body_lin_vel_w[:, ahm.r_feet_ids, :2]
    slip_speed = torch.norm(foot_vel_xy, dim=-1)
    return torch.sum(torch.square(slip_speed) * ahm.contact.float(), dim=1)


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

    ahm = _ahm(env)
    coeffs = getattr(ahm, "torque_coeffs_vec", None)
    if coeffs is not None:
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

    ahm = _ahm(env)
    coeffs = getattr(ahm, "action_coeffs_vec", None)
    if coeffs is not None:
        return torch.sum(torch.square(actions) * coeffs, dim=1)
    return torch.sum(torch.square(actions), dim=1)


def action_rate_l2(env: ManagerBasedEnv) -> torch.Tensor:
    """Penalize changes in actions."""
    ahm = _ahm(env)
    return torch.sum(torch.square(ahm.last_actions - env.action_manager.action), dim=1)


def smoothness_l2(env: ManagerBasedEnv) -> torch.Tensor:
    """Penalize jerk (second-order action changes)."""
    ahm = _ahm(env)
    actions = env.action_manager.action
    return torch.sum(torch.square(actions - 2 * ahm.last_actions + ahm.last_last_actions), dim=1)


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

    # Use randomized default from AmpHelperManager if available
    ahm = _ahm(env)
    default_random = getattr(ahm, "default_dof_pos_random", None)
    if default_random is not None:
        dof_err = robot.data.joint_pos[:, indices] - default_random[:, indices]
    else:
        dof_err = robot.data.joint_pos[:, indices] - robot.data.default_joint_pos[:, indices]

    coeffs = getattr(ahm, "jerr_coeffs_vec", None)
    if coeffs is not None:
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
    """Penalize undesired contacts on specified bodies."""
    contact_sensor = env.scene[contact_sensor_cfg.name]
    penalize_ids = contact_sensor_cfg.body_ids
    net_contact_forces = contact_sensor.data.net_forces_w
    penalize_contacts = torch.norm(net_contact_forces[:, penalize_ids], dim=-1) > threshold
    return torch.sum(penalize_contacts.float(), dim=1)


# ---------------------------------------------------------------------------
# Foot impact and orientation rewards
# ---------------------------------------------------------------------------
def feet_impact_vel(env: ManagerBasedEnv, contact_sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize feet downward velocity at landing. Returns positive; use negative weight."""
    ahm = _ahm(env)
    prev_foot_vel_z = ahm.last_foot_velocities[:, :, 2]  # (num_envs, num_feet)
    downward_vel = torch.clamp(prev_foot_vel_z, max=0.0)  # negative or zero
    return torch.sum(ahm.contact.float() * torch.square(downward_vel), dim=1)


def foot_orientation(env: ManagerBasedEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[".*foot.*|.*ankle.*"])) -> torch.Tensor:
    """Penalize feet x-axis misalignment with torso x-axis. Returns positive; use negative weight."""
    ahm = _ahm(env)
    robot = env.scene[asset_cfg.name]
    cmd_term = env.command_manager.get_term(command_name)

    base_quat = robot.data.root_quat_w  # (num_envs, 4)
    forward_vec = torch.zeros(env.num_envs, 3, device=env.device)
    forward_vec[:, 0] = 1.0  # x-axis unit vector
    base_forward = math_utils.quat_apply(base_quat, forward_vec)  # (num_envs, 3)
    base_forward_xy = base_forward[:, :2]  # (num_envs, 2)
    base_forward_xy = base_forward_xy / (torch.norm(base_forward_xy, dim=1, keepdim=True) + 1e-6)

    foot_quats = robot.data.body_quat_w[:, ahm.r_feet_ids]  # (num_envs, num_feet, 4)
    penalty = torch.zeros(env.num_envs, device=env.device)
    for i in range(len(ahm.r_feet_ids)):
        foot_forward = math_utils.quat_apply(foot_quats[:, i], forward_vec)  # (num_envs, 3)
        foot_forward_xy = foot_forward[:, :2]  # (num_envs, 2)
        foot_forward_xy = foot_forward_xy / (torch.norm(foot_forward_xy, dim=1, keepdim=True) + 1e-6)
        cos_sim = torch.sum(base_forward_xy * foot_forward_xy, dim=1)
        penalty += 1.0 - cos_sim

    z_cmd_abs = torch.abs(cmd_term.command[:, 2])
    yaw_scale = torch.clamp(1.0 - z_cmd_abs / 1.2, 0.0, 1.0)
    penalty = penalty * yaw_scale
    return penalty


# ---------------------------------------------------------------------------
# AMP reward (discriminator)
# ---------------------------------------------------------------------------
def amp_reward(env: ManagerBasedEnv) -> torch.Tensor:
    """AMP discriminator reward."""
    amp_disc = getattr(env, "amp_discriminator", None)
    amp_norm = getattr(env, "amp_normalizer", None)
    amp_obs_buf = getattr(env, "amp_obs_history_buf", None)

    if amp_disc is not None and amp_obs_buf is not None:
        return amp_disc.compute_amp_reward(amp_obs_buf, normalizer=amp_norm)
    return torch.zeros(env.num_envs, device=env.device)
