"""Symmetry functions for DR02.

DR02 dof observation order (21 joints):
    0-5:   left leg  (hip_y, hip_x, hip_z, knee, ankle_y, ankle_x)
    6-11:  right leg (hip_y, hip_x, hip_z, knee, ankle_y, ankle_x)
    12:    waist_z
    13-16: left arm  (shoulder_y, shoulder_x, shoulder_z, elbow)
    17-20: right arm (shoulder_y, shoulder_x, shoulder_z, elbow)

Observation layouts:
    policy (73): [ang_vel(3), proj_grav(3), cmd(3), cmd_flag(1), dof_pos(21), dof_vel(21), actions(21)]
    critic (121): [body_vel(3), policy(73), torques(21), foot_force(6), foot_vel(6), hand_pos(6), foot_pos(6)]
    obs_future (69): [ang_vel(3), proj_grav(3), dof_pos(21), dof_vel(21), actions(21)]
    obs_history (730 = 10 x 73): per-frame policy obs
"""

from __future__ import annotations

import torch

__all__ = [
    "compute_symmetric_states",
    "symmetrize_policy_obs",
    "symmetrize_critic_obs",
    "symmetrize_actions",
    "symmetrize_obs_future",
    "symmetrize_obs_history",
]

# --------------------------------------------------------------------------- #
# DOF-level symmetry
# --------------------------------------------------------------------------- #

# Joints that keep the same sign under left-right swap
_LEFT_POS = [0, 3, 4, 13, 16]   # hip_y, knee, ankle_y, shoulder_y, elbow
_RIGHT_POS = [6, 9, 10, 17, 20]

# Joints that need sign negation under left-right swap
_LEFT_NEG = [1, 2, 5, 14, 15]   # hip_x, hip_z, ankle_x, shoulder_x, shoulder_z
_RIGHT_NEG = [7, 8, 11, 18, 19]

# Self-symmetric joint (negate only)
_WAIST_Z = 12


def symmetrize_dof(dof_data: torch.Tensor) -> torch.Tensor:
    """Apply left-right symmetry to 21-DOF data (any leading batch dims)."""
    out = dof_data.clone()
    out[..., _WAIST_Z] = dof_data[..., _WAIST_Z] * -1.0
    out[..., _LEFT_POS] = dof_data[..., _RIGHT_POS]
    out[..., _RIGHT_POS] = dof_data[..., _LEFT_POS]
    out[..., _LEFT_NEG] = dof_data[..., _RIGHT_NEG] * -1.0
    out[..., _RIGHT_NEG] = dof_data[..., _LEFT_NEG] * -1.0
    return out


# --------------------------------------------------------------------------- #
# Observation-level symmetry
# --------------------------------------------------------------------------- #

def symmetrize_policy_obs(obs: torch.Tensor) -> torch.Tensor:
    """Apply left-right symmetry to policy obs (73 dims).

    Layout: [ang_vel(3), proj_grav(3), cmd(3), cmd_flag(1), dof_pos(21), dof_vel(21), actions(21)]
    """
    out = obs.clone()
    out[:, 0] *= -1.0   # ang_vel x
    out[:, 2] *= -1.0   # ang_vel z
    out[:, 4] *= -1.0   # proj_grav y
    out[:, 7] *= -1.0   # cmd y
    out[:, 8] *= -1.0   # cmd z (ang_vel_yaw)
    out[:, 10:31] = symmetrize_dof(obs[:, 10:31])    # dof_pos
    out[:, 31:52] = symmetrize_dof(obs[:, 31:52])    # dof_vel
    out[:, 52:73] = symmetrize_dof(obs[:, 52:73])    # actions
    return out


def symmetrize_critic_obs(obs: torch.Tensor) -> torch.Tensor:
    """Apply left-right symmetry to critic obs (121 dims).

    Layout: [body_vel(3), policy(73), torques(21), foot_force(6), foot_vel(6),
             hand_pos(6), foot_pos(6)]
    """
    out = obs.clone()
    # body_vel (0:3): negate y
    out[:, 1] *= -1.0
    # ang_vel (3:6): negate x, z
    out[:, 3] *= -1.0
    out[:, 5] *= -1.0
    # proj_grav (6:9): negate y
    out[:, 7] *= -1.0
    # cmd (9:12): negate y, z
    out[:, 10] *= -1.0
    out[:, 11] *= -1.0
    # cmd_flag (12): unchanged
    # dof_pos (13:34), dof_vel (34:55), actions (55:76), torques (76:97)
    out[:, 13:34] = symmetrize_dof(obs[:, 13:34])
    out[:, 34:55] = symmetrize_dof(obs[:, 34:55])
    out[:, 55:76] = symmetrize_dof(obs[:, 55:76])
    out[:, 76:97] = symmetrize_dof(obs[:, 76:97])

    # foot_force (97:103): swap L/R, negate y
    out[:, 97:100] = obs[:, 100:103]
    out[:, 98] *= -1.0
    out[:, 100:103] = obs[:, 97:100]
    out[:, 101] *= -1.0

    # foot_vel (103:109): swap L/R, negate y
    out[:, 103:106] = obs[:, 106:109]
    out[:, 104] *= -1.0
    out[:, 106:109] = obs[:, 103:106]
    out[:, 107] *= -1.0

    # hand_pos (109:115): swap L/R, negate y
    out[:, 109:112] = obs[:, 112:115]
    out[:, 110] *= -1.0
    out[:, 112:115] = obs[:, 109:112]
    out[:, 113] *= -1.0

    # foot_pos (115:121): swap L/R, negate y
    out[:, 115:118] = obs[:, 118:121]
    out[:, 116] *= -1.0
    out[:, 118:121] = obs[:, 115:118]
    out[:, 119] *= -1.0

    return out


def symmetrize_actions(actions: torch.Tensor) -> torch.Tensor:
    """Apply left-right symmetry to actions (21 dims)."""
    return symmetrize_dof(actions)


def symmetrize_obs_future(obs: torch.Tensor) -> torch.Tensor:
    """Apply left-right symmetry to obs_future (69 dims).

    Layout: [ang_vel(3), proj_grav(3), dof_pos(21), dof_vel(21), actions(21)]
    """
    out = obs.clone()
    out[:, 0] *= -1.0   # ang_vel x
    out[:, 2] *= -1.0   # ang_vel z
    out[:, 4] *= -1.0   # proj_grav y
    out[:, 6:27] = symmetrize_dof(obs[:, 6:27])     # dof_pos
    out[:, 27:48] = symmetrize_dof(obs[:, 27:48])    # dof_vel
    out[:, 48:69] = symmetrize_dof(obs[:, 48:69])    # actions
    return out


def symmetrize_obs_history(obs_history: torch.Tensor, num_actor_obs: int = 73) -> torch.Tensor:
    """Apply left-right symmetry to obs_history (num_history * num_actor_obs).

    Each frame of the history is a full policy obs (73 dims) and is symmetrized
    independently.
    """
    batch_size = obs_history.shape[0]
    reshaped = obs_history.reshape(batch_size, -1, num_actor_obs)  # (B, T, 73)
    sym_frames = torch.stack([symmetrize_policy_obs(f) for f in reshaped.unbind(dim=1)], dim=1)
    return sym_frames.reshape(batch_size, -1)


# --------------------------------------------------------------------------- #
# TensorDict-level augmentation
# --------------------------------------------------------------------------- #

def compute_symmetric_states(
    env,
    obs=None,
    actions=None,
):
    """Augment observations and actions by applying left-right symmetry.

    Doubles the batch by concatenating the original and the symmetrized version.

    Args:
        env: The environment instance (unused, kept for API compatibility with g1.py).
        obs: Original observation TensorDict. Defaults to None.
        actions: Original actions tensor. Defaults to None.

    Returns:
        (obs_aug, actions_aug) — each doubled in batch dimension, or None.
    """
    if obs is not None:
        batch_size = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        obs_aug["policy"][batch_size:] = symmetrize_policy_obs(obs["policy"][:])
        if "critic" in obs_aug:
            obs_aug["critic"][batch_size:] = symmetrize_critic_obs(obs["critic"][:])
        if "obs_history" in obs_aug:
            obs_aug["obs_history"][batch_size:] = symmetrize_obs_history(obs["obs_history"][:])
        if "obs_future" in obs_aug:
            obs_aug["obs_future"][batch_size:] = symmetrize_obs_future(obs["obs_future"][:])
    else:
        obs_aug = None

    if actions is not None:
        batch_size = actions.shape[0]
        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device)
        actions_aug[:batch_size] = actions[:]
        actions_aug[batch_size:] = symmetrize_actions(actions[:])
    else:
        actions_aug = None

    return obs_aug, actions_aug
