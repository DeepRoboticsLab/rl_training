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
    """Reset DOF positions and velocities with optional randomization."""
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
    """Reset the robot root state (position + orientation + velocity) with randomization."""
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


# ────────────────────────────────────────────────────────────────────
# Reset: AMP reference state initialization
# ────────────────────────────────────────────────────────────────────


def reset_amp_reference(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    sampling_probability: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    amp_dof_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    pose_range: tuple[float, float] = (-1.5, 1.5),
):
    """Reset robot state from AMP reference motion data (subset of envs)."""
    # Access the AMP dataset (set by runner via env.amp_dataset)
    amp_dataset = getattr(env, "amp_dataset", None)
    if amp_dataset is None:
        return

    # Resolve env_ids (EventManager may pass slice(None) when env_ids is None)
    if isinstance(env_ids, slice):
        env_ids = torch.arange(env.scene.num_envs, device=env.device)
    else:
        env_ids = env_ids.to(env.device)

    if len(env_ids) == 0:
        return

    # Sample which envs use AMP reference
    use_amp = torch.rand(len(env_ids), device=env.device) < sampling_probability
    amp_env_ids = env_ids[use_amp]
    if len(amp_env_ids) == 0:
        return

    # Get reference frames from the AMP dataset
    frames = amp_dataset.get_full_frame_batch(len(amp_env_ids))

    asset: Articulation = env.scene[asset_cfg.name]

    # ── DOF reset ─────────────────────────────────────────────────
    # Resolve AMP DOF indices (cached on the env to avoid repeated lookups)
    if not hasattr(env, "_amp_dof_indices_resolved"):
        if amp_dof_cfg.joint_names:
            indices, _ = asset.find_joints(amp_dof_cfg.joint_names, preserve_order=True)
        else:
            indices = list(range(asset.num_joints))
        env._amp_dof_indices_resolved = indices
    amp_indices = env._amp_dof_indices_resolved

    # Use randomized default dof pos if available
    ahm = getattr(env, "amp_helper_manager", None)
    default_random = getattr(ahm, "default_dof_pos_random", None) if ahm is not None else None
    if default_random is not None:
        joint_pos = default_random[amp_env_ids].clone()
    else:
        joint_pos = asset.data.default_joint_pos[amp_env_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)
    joint_pos[:, amp_indices] = amp_dataset.get_joint_pose_batch(frames)
    joint_vel[:, amp_indices] = amp_dataset.get_joint_vel_batch(frames)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=amp_env_ids)

    # ── Root state reset ─────────────────────────────────────────
    root_pos = amp_dataset.get_root_pos_batch(frames)
    root_pos[:, 2] = torch.clip(root_pos[:, 2], 0.9, 1.0)
    root_pos[:, :2] = 0
    root_pos = root_pos + env.scene.env_origins[amp_env_ids]

    # xy position randomization
    n = len(amp_env_ids)
    root_pos[:, 0] += torch.rand(n, device=env.device) * (pose_range[1] - pose_range[0]) + pose_range[0]
    root_pos[:, 1] += torch.rand(n, device=env.device) * (pose_range[1] - pose_range[0]) + pose_range[0]

    root_orn = amp_dataset.get_root_rot_batch(frames)
    # AMP dataset stores quaternions in (x, y, z, w) format,
    # but IsaacLab expects (w, x, y, z) for write_root_pose_to_sim.
    root_orn = math_utils.convert_quat(root_orn, to="wxyz")
    root_lin_vel = math_utils.quat_apply(
        root_orn, amp_dataset.get_linear_vel_batch(frames)
    )
    root_ang_vel = math_utils.quat_apply(
        root_orn, amp_dataset.get_angular_vel_batch(frames)
    )

    default_root_state = asset.data.default_root_state[amp_env_ids].clone()
    default_root_state[:, :3] = root_pos
    default_root_state[:, 3:7] = root_orn
    default_root_state[:, 7:10] = root_lin_vel
    default_root_state[:, 10:13] = root_ang_vel

    asset.write_root_pose_to_sim(default_root_state[:, :7], env_ids=amp_env_ids)
    asset.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids=amp_env_ids)

