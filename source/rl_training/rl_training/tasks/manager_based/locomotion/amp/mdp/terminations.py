# AMP custom termination functions for the manager-based framework.
#
# Migrated from AMPTrainEnv._get_dones() termination logic.

from __future__ import annotations

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

import isaaclab.utils.math as math_utils

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def bad_orientation_pitch_roll(
    env: "ManagerBasedRLEnv",
    pitch_limit: float = 1.1,
    roll_limit: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when pitch or roll exceeds independent thresholds.

    Mirrors AMPTrainEnv._get_dones() lines 847-851:
        roll, pitch, yaw = euler_xyz_from_quat(base_quat)
        incline_flag = (abs(pitch) > 1.1) | (abs(roll) > 1.0)

    Unlike IsaacLab's built-in ``bad_orientation`` which uses a combined
    tilt angle (``acos(-projected_gravity_b[:, 2]) > limit_angle``), this
    function uses separate pitch and roll thresholds matching the original
    AMP training environment.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    base_quat = asset.data.root_quat_w
    roll, pitch, _ = math_utils.euler_xyz_from_quat(base_quat)
    return (torch.abs(pitch) > pitch_limit) | (torch.abs(roll) > roll_limit)
