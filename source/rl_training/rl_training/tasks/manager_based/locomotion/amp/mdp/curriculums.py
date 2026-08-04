from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def push_curriculum(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int],
    reward_term_name: str = "lin_vel_tracking",
    reward_ratio: float = 0.8,
    velocity_range: dict | None = None,
):
    """One-way switch: permanently enable push_robot once reward threshold is met.
    """
    if velocity_range is None:
        velocity_range = {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}

    episode_sums = env.reward_manager._episode_sums[reward_term_name]
    reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)

    normalized_reward = torch.mean(episode_sums[env_ids]) / env.max_episode_length_s
    threshold = reward_ratio * reward_term_cfg.weight
    try:
        term_cfg = env.event_manager.get_term_cfg("push_robots")
        if not getattr(env, "_push_enabled", False) and normalized_reward > threshold:
            term_cfg.params["velocity_range"] = velocity_range
            env._push_enabled = True
    except ValueError:
        pass  # push_robot term not found

    return float(getattr(env, "_push_enabled", False))
