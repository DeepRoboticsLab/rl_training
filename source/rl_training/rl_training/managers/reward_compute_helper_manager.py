# Reward Compute Helper Manager
#
# Manages reward-related state buffers (contact state, air time, action history)
# that need precise lifecycle control within the environment's step() method.
#
# The manager exposes two update hooks:
#   - pre_reward_update():  Called AFTER physics stepping, BEFORE termination/reward computation.
#                           Updates contact_filt and foot_contact_trajs.
#   - post_step_update():   Called AFTER observations are computed (end of step).
#                           Updates last_actions and last_last_actions.
#
# feet_air_time is NOT updated here — it is updated inside the feet_air_time
# reward function to match the original AMPTrainEnv._reward_feet_air_time()
# execution order (first_contact before increment, reset after reward value).

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import ManagerBase
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class RewardComputeHelperManager(ManagerBase):
    """Manager for reward computation state buffers.

    Maintains the following state used by reward functions:

    - **contact_filt**: Boolean contact state for feet (num_envs, num_feet)
    - **feet_air_time**: Air time accumulator for feet (num_envs, num_feet)
    - **foot_contact_trajs**: Boolean contact trajectory history (num_envs, num_feet, traj_len)
    - **last_actions**: Previous step's actions (num_envs, num_actions)
    - **last_last_actions**: Action from two steps ago (num_envs, num_actions)

    Body indices (resolved lazily on first use):

    - **r_feet_ids**: Robot body indices for feet (ankle_x_link)
    - **c_feet_ids**: Contact sensor body indices for feet
    - **r_torso_ids**: Robot body index for torso ("body" link)

    Lifecycle hooks (called by the custom AmpLocomotionEnv.step()):

    1. ``pre_reward_update()`` — after physics, before terminations/rewards
    2. ``post_step_update()`` — after observations (end of step)
    3. ``reset(env_ids)`` — called from ``_reset_idx()``
    """

    def __init__(self, cfg, env: "ManagerBasedEnv"):
        self._initialized = False
        super().__init__(cfg, env)

    @property
    def active_terms(self) -> list[str]:
        return []

    def _prepare_terms(self):
        pass  # No terms — this manager only holds state buffers

    # ------------------------------------------------------------------
    # Lazy initialization (called on first use when scene is ready)
    # ------------------------------------------------------------------

    def _ensure_initialized(self):
        if self._initialized:
            return
        robot = self._env.scene["robot"]
        contact_sensor = self._env.scene["contact_sensor"]

        self.r_feet_ids, _ = robot.find_bodies(".*ankle_x_link", preserve_order=True)
        self.c_feet_ids, _ = contact_sensor.find_bodies(".*ankle_x_link", preserve_order=True)
        self.r_torso_ids, _ = robot.find_bodies("body")

        self._init_buffers()
        self._initialized = True

    def _init_buffers(self):
        num_envs = self.num_envs
        device = self.device
        num_feet = len(self.r_feet_ids)
        num_actions = self._env.action_manager.total_action_dim
        traj_len = round(0.2 / self._env.step_dt)  # foot_contact_window = 0.2s

        self.contact_filt = torch.zeros(num_envs, num_feet, dtype=torch.bool, device=device)
        self.feet_air_time = torch.zeros(num_envs, num_feet, dtype=torch.float, device=device)
        self.foot_contact_trajs = torch.zeros(num_envs, num_feet, traj_len, dtype=torch.bool, device=device)
        self.last_actions = torch.zeros(num_envs, num_actions, dtype=torch.float, device=device)
        self.last_last_actions = torch.zeros(num_envs, num_actions, dtype=torch.float, device=device)

    # ------------------------------------------------------------------
    # Step lifecycle hooks
    # ------------------------------------------------------------------

    def pre_reward_update(self):
        """Update contact state — called AFTER physics stepping, BEFORE terminations/rewards.

        Mirrors AMPTrainEnv.step() lines 728-735:
            contact = norm(net_forces_w[:, c_feet_ids]) > threshold
            contact_filt = contact
            foot_contact_trajs = cat(contact, foot_contact_trajs[..., :-1])

        Note: feet_air_time is NOT updated here — it is updated inside the
        feet_air_time reward function to match the original execution order.
        """
        self._ensure_initialized()
        contact_sensor = self._env.scene["contact_sensor"]
        contact = torch.norm(
            contact_sensor.data.net_forces_w[:, self.c_feet_ids], dim=-1
        ) > contact_sensor.cfg.force_threshold
        self.contact_filt = contact
        self.foot_contact_trajs = torch.cat(
            (contact.unsqueeze(-1), self.foot_contact_trajs[..., :-1]), dim=-1
        )

    def post_step_update(self):
        """Update action history — called AFTER observations (end of step).

        Mirrors AMPTrainEnv.step() lines 762-764:
            last_last_actions[:] = last_actions[:]
            last_actions[:] = actions[:]
        """
        self._ensure_initialized()
        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self._env.action_manager.action[:]

    # ------------------------------------------------------------------
    # Reset (called from _reset_idx)
    # ------------------------------------------------------------------

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Reset reward state buffers for the given environments.

        Mirrors AMPTrainEnv._reset_idx() lines 1222-1236:
            contact = norm(net_forces_w[env_ids][:, c_feet_ids]) > threshold
            contact_filt[env_ids] = contact
            foot_contact_trajs[env_ids] = 0
            feet_air_time[env_ids] = 0
            last_actions[env_ids] = 0
            last_last_actions[env_ids] = 0
        """
        self._ensure_initialized()
        if env_ids is None:
            env_ids = slice(None)

        contact_sensor = self._env.scene["contact_sensor"]
        contact = torch.norm(
            contact_sensor.data.net_forces_w[env_ids][:, self.c_feet_ids], dim=-1
        ) > contact_sensor.cfg.force_threshold
        self.contact_filt[env_ids] = contact
        self.feet_air_time[env_ids] = 0.0
        self.foot_contact_trajs[env_ids] = 0
        self.last_actions[env_ids] = 0.0
        self.last_last_actions[env_ids] = 0.0
        return {}

    # ------------------------------------------------------------------
    # Helper for reward functions
    # ------------------------------------------------------------------

    def get_torso_lin_vel(self) -> torch.Tensor:
        """Get torso linear velocity in base frame.

        Used by the lin_vel_tracking reward function.
        """
        self._ensure_initialized()
        robot = self._env.scene["robot"]
        if len(self.r_torso_ids) > 0:
            torso_vel_w = robot.data.body_lin_vel_w[:, self.r_torso_ids[0], :]
            base_quat = robot.data.root_quat_w
            return math_utils.quat_apply_inverse(base_quat, torso_vel_w)
        return robot.data.root_lin_vel_b


@configclass
class RewardComputeHelperCfg:
    """Configuration for RewardComputeHelperManager (no parameters needed)."""
    pass
