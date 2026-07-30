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
    force_range: tuple[float, float, float] = (200.0, 100.0, 50.0),
    torque_range: tuple[float, float, float] = (25.0, 50.0, 25.0),
):
    """Enable push_robots when a reward term reaches a ratio of its max weight.

    Compares ``mean(episode_sums) / max_episode_length_s`` against
    ``reward_ratio * reward_term_cfg.weight``. When exceeded, sets push_robots
    event params to real force/torque ranges. Otherwise sets them to zero.
    """
    # Directly access reward_manager buffers (always available after __init__)
    episode_sums = env.reward_manager._episode_sums[reward_term_name]
    reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)

    normalized_reward = torch.mean(episode_sums[env_ids]) / env.max_episode_length_s
    threshold = reward_ratio * reward_term_cfg.weight

    # Modify push_robots event term params
    try:
        term_cfg = env.event_manager.get_term_cfg("push_robots")
        if normalized_reward > threshold:
            term_cfg.params["force_range"] = force_range
            term_cfg.params["torque_range"] = torque_range
        else:
            term_cfg.params["force_range"] = (0.0, 0.0, 0.0)
            term_cfg.params["torque_range"] = (0.0, 0.0, 0.0)
    except ValueError:
        pass  # push_robots term not found

    return float(normalized_reward)
