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
    """Terminate when pitch or roll exceeds independent thresholds."""
    asset: RigidObject = env.scene[asset_cfg.name]
    base_quat = asset.data.root_quat_w
    roll, pitch, _ = math_utils.euler_xyz_from_quat(base_quat)
    return (torch.abs(pitch) > pitch_limit) | (torch.abs(roll) > roll_limit)
