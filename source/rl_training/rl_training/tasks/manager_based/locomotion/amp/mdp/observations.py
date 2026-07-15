# AMP observation functions for humanoid locomotion
#
# Individual observation terms for use with ObservationManager.
# Noise, clip, scale, and history are handled by the ObservationManager
# via ObsTerm configuration — these functions return raw or already-scaled
# values and do NOT manage any buffers or indices themselves.
#
# Joint/body reordering is handled by SceneEntityCfg(joint_names=...,
# preserve_order=True) in the ObsTerm params, which resolves joint_ids
# and body_ids automatically when the manager initializes.
#
# Functions that apply internal scaling (joint_pos_action_scaled,
# torques_normalized) accept scale values as parameters, set via
# ObsTerm params in amp_env_cfg.py.

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg


# ---------------------------------------------------------------------------
# Command observations
# ---------------------------------------------------------------------------

def velocity_command(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Velocity command (first 3 dims: vx, vy, wz). Shape: (num_envs, 3).

    Returns raw command — scale is applied via ``ObsTerm(scale=...)``.
    """
    cmd_term = env.command_manager.get_term(command_name)
    return cmd_term.command[:, :3]


def cmd_flag(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Binary command flag (1 if non-zero command). Shape: (num_envs, 1)."""
    cmd_term = env.command_manager.get_term(command_name)
    return cmd_term.cmd_flag


# ---------------------------------------------------------------------------
# Joint observations (already-scaled — no ObsTerm scale needed)
# ---------------------------------------------------------------------------

def joint_pos_action_scaled(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
    action_scale: float | dict[str, float] = 0.25,
) -> torch.Tensor:
    """Joint position relative to default, divided by action scale.

    Shape: (num_envs, N).

    Returns already-scaled value ``(joint_pos - default) / action_scale``.
    Used for policy obs where noise must match the original amplitude
    (±0.0025) — if ObsTerm ``scale`` were used instead, the noise applied
    before scale would be amplified by 1/action_scale.

    Args:
        action_scale: The action scale used by JointPositionActionCfg.
            Can be a float (uniform) or a dict of regex→float (per-joint).
            Must match the action config's ``scale`` value.
    """
    robot = env.scene[asset_cfg.name]
    dof_pos = robot.data.joint_pos[:, asset_cfg.joint_ids]
    default_pos = robot.data.default_joint_pos[:, asset_cfg.joint_ids]
    rel_pos = dof_pos - default_pos
    if isinstance(action_scale, dict):
        joint_names = [robot.data.joint_names[i] for i in asset_cfg.joint_ids]
        index_list, _, value_list = string_utils.resolve_matching_names_values(
            action_scale, joint_names
        )
        scales = torch.ones(len(joint_names), device=env.device)
        scales[index_list] = torch.tensor(value_list, device=env.device)
        return rel_pos / scales
    return rel_pos / action_scale


def torques_normalized(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
    torque_scale: float = 5.0,
) -> torch.Tensor:
    """Applied torques normalized by effort limits and scaled.

    Shape: (num_envs, N).

    Returns already-scaled value ``torque / limit × torque_scale``.
    Uses ``robot.data.joint_effort_limits`` (set from PhysX max forces).

    Args:
        torque_scale: Scale factor applied after normalization.
    """
    robot = env.scene[asset_cfg.name]
    torques = robot.data.applied_torque[:, asset_cfg.joint_ids]
    limits = robot.data.joint_effort_limits[:, asset_cfg.joint_ids]
    return torques / limits * torque_scale


# ---------------------------------------------------------------------------
# Foot / hand observations (raw — clip & scale via ObsTerm)
# ---------------------------------------------------------------------------

def foot_contact_forces(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Foot contact forces in base frame. Shape: (num_envs, 3×N_feet).

    Returns raw forces — clip and scale are applied via ``ObsTerm`` config
    (e.g. ``clip=(-5000, 5000), scale=0.002``).
    """
    robot = env.scene["robot"]
    contact_sensor = env.scene[sensor_cfg.name]
    base_quat = robot.data.root_quat_w
    forces = torch.cat([
        math_utils.quat_apply_inverse(base_quat, contact_sensor.data.net_forces_w[:, idx, :])
        for idx in sensor_cfg.body_ids
    ], dim=1)
    return forces


def foot_velocities(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Foot velocities in base frame. Shape: (num_envs, 3×N_feet).

    Returns raw velocities — clip and scale are applied via ``ObsTerm``
    config (e.g. ``clip=(-10, 10), scale=0.5``).
    """
    robot = env.scene[asset_cfg.name]
    base_quat = robot.data.root_quat_w
    vels = torch.cat([
        math_utils.quat_apply_inverse(base_quat, robot.data.body_lin_vel_w[:, idx, :])
        for idx in asset_cfg.body_ids
    ], dim=1)
    return vels


def body_pos_in_base_frame(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Body positions relative to base in body frame.

    Shape: (num_envs, 3×N_bodies).

    Returns raw positions — clip and scale are applied via ``ObsTerm``
    config.  Used for both hand and foot positions (specified via
    ``SceneEntityCfg(body_names=...)``).
    """
    robot = env.scene[asset_cfg.name]
    base_quat = robot.data.root_quat_w
    root_pos = robot.data.root_pos_w
    body_pos_w = robot.data.body_pos_w[:, asset_cfg.body_ids, :3]
    pos_to_base = body_pos_w - root_pos.unsqueeze(1)
    pos = torch.stack([
        math_utils.quat_apply_inverse(base_quat, pos_to_base[:, i, :])
        for i in range(pos_to_base.shape[1])
    ], dim=1).flatten(start_dim=1)
    return pos
