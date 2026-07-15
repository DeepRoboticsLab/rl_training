# AMP event functions for the manager-based framework.
#
# Migrated from:
#   - IsaacLabExtension/exts/deeprobotics/deeprobotics/Env/utils/functions.py
#       (randomize_rigid_body_mass)
#   - IsaacLabExtension/exts/deeprobotics/deeprobotics/Env/AMPTrainEnv.py
#       (_push_robots, _check_push_finish, _reset_dofs, _reset_root_states,
#        _reset_idx buffer reset logic)
#
# Note: ``randomize_rigid_body_mass`` and ``randomize_rigid_body_com`` are
# re-exported from IsaacLab's native mdp module.

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs.mdp import randomize_rigid_body_com  # noqa: F401  (re-export)
from isaaclab.envs.mdp import randomize_rigid_body_mass  # noqa: F401  (re-export)
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


# ────────────────────────────────────────────────────────────────────
# Push robots
# ────────────────────────────────────────────────────────────────────


def push_robots(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    force_range: tuple[float, float, float] = (200.0, 100.0, 50.0),
    torque_range: tuple[float, float, float] = (25.0, 50.0, 25.0),
):
    """Apply random external forces and torques to the robot's base link.

    Adapted from ``AMPTrainEnv._push_robots``.  In the manager-based
    framework this is called as an ``interval`` event (every
    ``interval_range_s`` seconds).  The force/torque is applied to the base
    body and persists until the next call or a reset.

    The force and torque are sampled as 1-D random values (per axis) in the
    base frame and rotated to the world frame via ``quat_apply``.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)
    else:
        env_ids = env_ids.to(env.device)

    # base link quaternion (world frame)
    base_quat = asset.data.root_quat_w[env_ids]

    max_force = torch.tensor(force_range, device=env.device, dtype=torch.float)
    max_torque = torch.tensor(torque_range, device=env.device, dtype=torch.float)

    # sample 1-D random forces/torques in base frame, rotate to world frame
    random_forces = torch.randn((len(env_ids), 1), device=env.device) * max_force
    forces_w = math_utils.quat_apply(base_quat, random_forces)

    random_torques = torch.randn((len(env_ids), 1), device=env.device) * max_torque
    torques_w = math_utils.quat_apply(base_quat, random_torques)

    # apply external force/torque on the base body (body index 0)
    asset.set_external_force_and_torque(
        forces=forces_w.unsqueeze(1),  # (N, 1, 3)
        torques=torques_w.unsqueeze(1),  # (N, 1, 3)
        body_ids=asset_cfg.body_ids,
        env_ids=env_ids,
    )


# ────────────────────────────────────────────────────────────────────
# Reset: DOF positions with randomization
# ────────────────────────────────────────────────────────────────────


def reset_dof_pos_randomized(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    pos_range: tuple[float, float] = (0.5, 1.5),
    pos_offset: tuple[float, float] = (-0.25, 0.25),
    vel_range: tuple[float, float] = (0.0, 0.0),
    randomize: bool = True,
):
    """Reset DOF positions and velocities with optional randomization.

    Adapted from ``AMPTrainEnv._reset_dofs``:
    - If ``randomize`` is True, the default joint positions are scaled by a
      random factor sampled from ``pos_range`` and then biased by a random
      offset sampled from ``pos_offset``.
    - The result is clamped to 95 % of the joint position limits.
    - Joint velocities are set to zero (or sampled from ``vel_range``).
    """
    asset: Articulation = env.scene[asset_cfg.name]

    if len(env_ids) == 0:
        return

    # 95 % of joint position limits
    dof_upper = 0.95 * asset.data.soft_joint_pos_limits[env_ids, :, 1]
    dof_lower = 0.95 * asset.data.soft_joint_pos_limits[env_ids, :, 0]

    if randomize:
        # scale default positions
        init_dof_pos = asset.data.default_joint_pos[env_ids] * math_utils.sample_uniform(
            pos_range[0], pos_range[1],
            (len(env_ids), asset.num_joints),
            device=asset.device,
        )
        # add random offset
        init_dof_pos += math_utils.sample_uniform(
            pos_offset[0], pos_offset[1],
            (len(env_ids), asset.num_joints),
            device=asset.device,
        )
        joint_pos = torch.clamp(init_dof_pos, dof_lower, dof_upper)
    else:
        joint_pos = asset.data.default_joint_pos[env_ids].clone()

    joint_vel = math_utils.sample_uniform(
        vel_range[0], vel_range[1],
        (len(env_ids), asset.num_joints),
        device=asset.device,
    )

    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


# ────────────────────────────────────────────────────────────────────
# Reset: root state with randomization
# ────────────────────────────────────────────────────────────────────


def reset_root_state_randomized(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]] | None = None,
    velocity_range: dict[str, tuple[float, float]] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    randomize: bool = True,
):
    """Reset the robot root state (position + orientation + velocity) with randomization.

    Adapted from ``AMPTrainEnv._reset_root_states``:
    - Root position is set to the default + env_origins + random xy offset.
    - Root orientation uses the default (no random rotation on flat terrain).
    - If ``randomize`` is True, base linear/angular velocities are sampled
      from the specific ranges used by the original AMP config.
    - If ``randomize`` is False, velocities are zeroed.

    Args:
        pose_range: Optional dict with keys ``"x"`` and ``"y"`` for xy
            position randomization.  Defaults to ``(-1.5, 1.5)`` on both axes.
        velocity_range: Optional dict with velocity ranges.  When provided,
            the keys ``"x"``, ``"y"``, ``"z"``, ``"roll"``, ``"pitch"``,
            ``"yaw"`` are used.  Defaults to the AMP-specific values.
        randomize: Whether to randomize the initial velocities.
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    if len(env_ids) == 0:
        return

    # default root state
    default_root_state = asset.data.default_root_state[env_ids].clone()
    default_root_state[:, :3] += env.scene.env_origins[env_ids]

    # xy position randomization
    if pose_range is None:
        pose_range = {"x": (-1.5, 1.5), "y": (-1.5, 1.5)}
    for key in ("x", "y"):
        if key in pose_range:
            idx = 0 if key == "x" else 1
            default_root_state[:, idx] += math_utils.sample_uniform(
                pose_range[key][0], pose_range[key][1],
                (len(env_ids),), device=asset.device,
            )

    if randomize:
        # AMP-specific velocity randomization (from _reset_root_states)
        if velocity_range is not None:
            # use provided velocity ranges
            range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
            ranges = torch.tensor(range_list, device=asset.device)
            rand_samples = math_utils.sample_uniform(
                ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device
            )
            default_root_state[:, 7:13] = rand_samples
        else:
            # AMP-specific default velocity ranges
            n = len(env_ids)
            default_root_state[:, 7] = torch.rand(n, device=asset.device) - 0.5       # vx: [-0.5, 0.5]
            default_root_state[:, 8] = 0.6 * torch.rand(n, device=asset.device) - 0.3  # vy: [-0.3, 0.3]
            default_root_state[:, 9] = 0.4 * torch.rand(n, device=asset.device) - 0.2  # vz: [-0.2, 0.2]
            default_root_state[:, 10:13] = math_utils.sample_uniform(
                -0.2, 0.2, (n, 3), device=asset.device
            )
    else:
        default_root_state[:, 7:13] = 0.0

    # set into the physics simulation
    asset.write_root_pose_to_sim(default_root_state[:, :7], env_ids=env_ids)
    asset.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids=env_ids)

