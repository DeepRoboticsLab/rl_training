"""Symmetry functions for the CR1 humanoid robot.

This module implements left-right symmetry augmentation for the CR1-B2-STD
humanoid (21 DOF).  The interface follows
``isaaclab_tasks.manager_based.locomotion.velocity.mdp.symmetry.anymal``.

CR1 DOF order (21 joints):
    0-5:   left leg  (hip_y, hip_x, hip_z, knee, ankle_y, ankle_x)
    6-11:  right leg (hip_y, hip_x, hip_z, knee, ankle_y, ankle_x)
    12:    waist_z
    13-16: left arm  (shoulder_y, shoulder_x, shoulder_z, elbow)
    17-20: right arm (shoulder_y, shoulder_x, shoulder_z, elbow)

Left-right symmetry swaps left/right limbs and negates joints whose positive
direction is lateral (hip_x, hip_z, ankle_x, shoulder_x, shoulder_z).  The
waist_z joint is also negated.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["compute_symmetric_states"]


# ────────────────────────────────────────────────────────────────────
# CR1 joint left-right swap indices
# ────────────────────────────────────────────────────────────────────

# Joints whose sign does NOT change under left-right symmetry
_LEFT_POS = [0, 3, 4, 13, 16]      # left  hip_y, knee, ankle_y, shoulder_y, elbow
_RIGHT_POS = [6, 9, 10, 17, 20]    # right hip_y, knee, ankle_y, shoulder_y, elbow

# Joints whose sign IS negated under left-right symmetry
_LEFT_NEG = [1, 2, 5, 14, 15]      # left  hip_x, hip_z, ankle_x, shoulder_x, shoulder_z
_RIGHT_NEG = [7, 8, 11, 18, 19]    # right hip_x, hip_z, ankle_x, shoulder_x, shoulder_z

_WAIST_IDX = 12  # waist_z


# ────────────────────────────────────────────────────────────────────
# Main entry point
# ────────────────────────────────────────────────────────────────────


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    """Augment observations and actions by applying left-right symmetry.

    For the CR1 biped, only left-right symmetry is meaningful (no front-back
    symmetry like quadrupeds).  This function produces **2×** augmented data:
    the original batch followed by the left-right mirrored batch.

    Args:
        env: The environment instance.
        obs: Original observation TensorDict (keys: ``"policy"``, ``"critic"``,
            etc.).  Can be ``None`` if only actions are augmented.
        actions: Original action tensor of shape ``(B, 21)``.  Can be ``None``
            if only observations are augmented.

    Returns:
        ``(obs_aug, actions_aug)`` — each doubled along the batch dimension.
        ``None`` for whichever input was ``None``.
    """
    # ── observations ──────────────────────────────────────────────
    if obs is not None:
        batch_size = obs.batch_size[0]
        obs_aug = obs.repeat(2)

        # policy group (73 dims)
        if "policy" in obs:
            obs_aug["policy"][:batch_size] = obs["policy"][:]
            obs_aug["policy"][batch_size:] = _transform_policy_obs_left_right(
                env.unwrapped, obs["policy"]
            )

        # critic group (121 dims)
        if "critic" in obs:
            obs_aug["critic"][:batch_size] = obs["critic"][:]
            obs_aug["critic"][batch_size:] = _transform_critic_obs_left_right(
                env.unwrapped, obs["critic"]
            )

        # obs_history group (730 dims = 10 × 73)
        # Each frame has the same structure as policy obs, so we reshape
        # to (batch, num_frames, 73) and transform per-frame.
        if "obs_history" in obs:
            oh = obs["obs_history"]
            num_frames = oh.shape[-1] // 73
            obs_aug["obs_history"][:batch_size] = oh[:]
            mirrored = _transform_policy_obs_left_right(
                env.unwrapped, oh.reshape(-1, 73)
            ).reshape(batch_size, num_frames * 73)
            obs_aug["obs_history"][batch_size:] = mirrored

        # obs_future group (69 dims = 3+3+21+21+21)
        # Structure: [ang_vel(3), proj_grav(3), dof_pos(21), dof_vel(21), actions(21)]
        if "obs_future" in obs:
            obs_aug["obs_future"][:batch_size] = obs["obs_future"][:]
            obs_aug["obs_future"][batch_size:] = _transform_future_obs_left_right(
                env.unwrapped, obs["obs_future"]
            )

        # vel_est / vel group (3 dims: body linear velocity)
        # Under left-right symmetry, the y-component (lateral) is negated.
        for vel_key in ("vel_est", "vel"):
            if vel_key in obs:
                obs_aug[vel_key][:batch_size] = obs[vel_key][:]
                mirrored_vel = obs[vel_key].clone()
                mirrored_vel[:, 1] = mirrored_vel[:, 1] * -1.0
                obs_aug[vel_key][batch_size:] = mirrored_vel
    else:
        obs_aug = None

    # ── actions ───────────────────────────────────────────────────
    if actions is not None:
        batch_size = actions.shape[0]
        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device)
        actions_aug[:batch_size] = actions[:]
        actions_aug[batch_size:] = _transform_actions_left_right(actions)
    else:
        actions_aug = None

    return obs_aug, actions_aug


# ────────────────────────────────────────────────────────────────────
# Observation transforms
# ────────────────────────────────────────────────────────────────────


def _transform_policy_obs_left_right(env: ManagerBasedRLEnv, obs: torch.Tensor) -> torch.Tensor:
    """Apply left-right symmetry to the 73-dim policy observation.

    Structure: ``[ang_vel(3), proj_grav(3), cmd(3), cmd_flag(1),
    dof_pos(21), dof_vel(21), actions(21)]``
    """
    obs = obs.clone()
    device = obs.device

    # ang_vel: negate x and z (y stays)
    obs[:, 0] *= -1.0
    obs[:, 2] *= -1.0
    # projected_gravity: negate y
    obs[:, 4] *= -1.0
    # velocity command: negate y and z (yaw)
    obs[:, 7] *= -1.0
    obs[:, 8] *= -1.0

    # DOF pos / vel / actions (21 each)
    obs[:, 10:31] = _switch_cr1_joints_left_right(obs[:, 10:31])
    obs[:, 31:52] = _switch_cr1_joints_left_right(obs[:, 31:52])
    obs[:, 52:73] = _switch_cr1_joints_left_right(obs[:, 52:73])

    return obs


def _transform_critic_obs_left_right(env: ManagerBasedRLEnv, obs: torch.Tensor) -> torch.Tensor:
    """Apply left-right symmetry to the 121-dim critic observation.

    Structure: ``[body_vel(3), policy_obs(73), torques(21),
    foot_force(6), foot_vel(6), end_pos(12)]``
    """
    obs = obs.clone()
    device = obs.device

    # body_vel: negate y
    obs[:, 1] *= -1.0

    # policy_obs subset (indices 3:76) — same structure as policy obs
    # ang_vel: negate x and z
    obs[:, 3] *= -1.0
    obs[:, 5] *= -1.0
    # projected_gravity: negate y
    obs[:, 7] *= -1.0
    # velocity command: negate y and z
    obs[:, 10] *= -1.0
    obs[:, 11] *= -1.0

    # DOF pos / vel / actions / torques (21 each)
    obs[:, 13:34] = _switch_cr1_joints_left_right(obs[:, 13:34])
    obs[:, 34:55] = _switch_cr1_joints_left_right(obs[:, 34:55])
    obs[:, 55:76] = _switch_cr1_joints_left_right(obs[:, 55:76])
    obs[:, 76:97] = _switch_cr1_joints_left_right(obs[:, 76:97])

    # foot_force (97:103) — swap left/right, negate y
    _swap_lr_xyz(obs, 97, 100)

    # foot_vel (103:109) — swap left/right, negate y
    _swap_lr_xyz(obs, 103, 106)

    # end_pos (109:121) — two pairs of left/right 3-vectors
    _swap_lr_xyz(obs, 109, 112)  # hands
    _swap_lr_xyz(obs, 115, 118)  # feet

    return obs


def _transform_future_obs_left_right(env: ManagerBasedRLEnv, obs: torch.Tensor) -> torch.Tensor:
    """Apply left-right symmetry to the 69-dim future observation.

    Structure: ``[ang_vel(3), proj_grav(3), dof_pos(21), dof_vel(21), actions(21)]``
    """
    obs = obs.clone()

    # ang_vel: negate x and z (y stays)
    obs[:, 0] *= -1.0
    obs[:, 2] *= -1.0
    # projected_gravity: negate y
    obs[:, 4] *= -1.0

    # DOF pos / vel / actions (21 each)
    obs[:, 6:27] = _switch_cr1_joints_left_right(obs[:, 6:27])
    obs[:, 27:48] = _switch_cr1_joints_left_right(obs[:, 27:48])
    obs[:, 48:69] = _switch_cr1_joints_left_right(obs[:, 48:69])

    return obs


# ────────────────────────────────────────────────────────────────────
# Action transform
# ────────────────────────────────────────────────────────────────────


def _transform_actions_left_right(actions: torch.Tensor) -> torch.Tensor:
    """Apply left-right symmetry to the 21-dim action vector."""
    actions = actions.clone()
    actions[:] = _switch_cr1_joints_left_right(actions[:])
    return actions


# ────────────────────────────────────────────────────────────────────
# Helper: CR1 joint left-right swap
# ────────────────────────────────────────────────────────────────────


def _switch_cr1_joints_left_right(joint_data: torch.Tensor) -> torch.Tensor:
    """Swap left/right joints and negate lateral-sign joints.

    Args:
        joint_data: tensor with last dim == 21 (CR1 DOF count).

    Returns:
        Swapped and sign-corrected tensor of the same shape.
    """
    out = joint_data.clone()

    # waist_z is negated
    out[..., _WAIST_IDX] = joint_data[..., _WAIST_IDX] * (-1.0)

    # POS: left ← right (same sign), right ← left (same sign)
    out[..., _LEFT_POS] = joint_data[..., _RIGHT_POS]
    out[..., _RIGHT_POS] = joint_data[..., _LEFT_POS]

    # NEG: left ← right × (-1), right ← left × (-1)
    out[..., _LEFT_NEG] = joint_data[..., _RIGHT_NEG] * (-1.0)
    out[..., _RIGHT_NEG] = joint_data[..., _LEFT_NEG] * (-1.0)

    return out


# ────────────────────────────────────────────────────────────────────
# Helper: swap a pair of 3-vectors (left xyz, right xyz) and negate y
# ────────────────────────────────────────────────────────────────────


def _swap_lr_xyz(tensor: torch.Tensor, left_start: int, right_start: int) -> None:
    """In-place swap of two consecutive 3-element blocks with y-negation.

    After the call:
    - ``tensor[..., left_start:left_start+3]``  ← original right block (y negated)
    - ``tensor[..., right_start:right_start+3]`` ← original left block  (y negated)
    """
    tmp_left = tensor[..., left_start:left_start + 3].clone()
    tmp_right = tensor[..., right_start:right_start + 3].clone()

    tensor[..., left_start:left_start + 3] = tmp_right
    tensor[..., left_start + 1] *= -1.0       # negate y of new left

    tensor[..., right_start:right_start + 3] = tmp_left
    tensor[..., right_start + 1] *= -1.0      # negate y of new right
