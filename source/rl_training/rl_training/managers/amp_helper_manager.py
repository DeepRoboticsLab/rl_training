# AMP Helper Manager
#
# Manages AMP-related state buffers (contact state, air time, action history,
# obs history) that need precise lifecycle control within the environment's
# step() method.
#
# The manager exposes three update hooks:
#   - pre_reward_update():       Called AFTER physics stepping, BEFORE termination/reward.
#                                Updates contact_filt and foot_contact_trams.
#   - update_obs_history():      Called AFTER observations are computed.
#                                Maintains obs_history buffer using obs_buf["policy"].
#   - post_step_update():        Called AFTER obs_history update (end of step).
#                                Updates last_actions and last_last_actions.
#
# feet_air_time is NOT updated here — it is updated inside the feet_air_time
# reward function to match the original AMPTrainEnv._reward_feet_air_time()
# execution order (first_contact before increment, reset after reward value).

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.managers import ManagerBase
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# Number of policy observation history frames (matches AMPTrainEnv)
_OBS_HISTORY_LENGTH = 10


class AmpHelperManager(ManagerBase):
    """Manager for AMP-related state buffers.

    Maintains the following state used by reward functions and the training
    runner:

    **Reward state buffers:**
    - **contact_filt**: Boolean contact state for feet (num_envs, num_feet)
    - **feet_air_time**: Air time accumulator for feet (num_envs, num_feet)
    - **foot_contact_trajs**: Boolean contact trajectory history (num_envs, num_feet, traj_len)
    - **last_actions**: Previous step's actions (num_envs, num_actions)
    - **last_last_actions**: Action from two steps ago (num_envs, num_actions)
    - **last_dof_vel**: Previous step's joint velocities (num_envs, num_joints)
    - **last_foot_velocities**: Previous step's foot velocities in world frame (num_envs, num_feet, 3)

    **Observation history buffer:**
    - **_obs_history_buf**: Policy observation history (num_envs, history_length, policy_obs_dim)
      Maintained by ``update_obs_history()`` using the already-computed
      ``obs_buf["policy"]`` tensor (frame-by-frame layout, 10×73 = 730 dims).
      Written into ``obs_buf["obs_history"]`` as a flattened tensor.

    Body indices (resolved lazily on first use):
    - **r_feet_ids**: Robot body indices for feet (ankle_x_link)
    - **c_feet_ids**: Contact sensor body indices for feet
    - **r_torso_ids**: Robot body index for torso ("body" link)

    Lifecycle hooks (called by the custom AmpLocomotionEnv.step()):

    1. ``pre_reward_update()`` — after physics, before terminations/rewards
    2. ``update_obs_history()`` — after observations (maintains obs_history)
    3. ``post_step_update()`` — after obs_history (updates action history)
    4. ``reset(env_ids)`` — called from ``_reset_idx()``
    """

    def __init__(self, cfg, env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        robot = self._env.scene["robot"]
        contact_sensor = self._env.scene["contact_sensor"]

        self.r_feet_ids, _ = robot.find_bodies(".*ankle_x_link", preserve_order=True)
        self.c_feet_ids, _ = contact_sensor.find_bodies(".*ankle_x_link", preserve_order=True)
        self.r_torso_ids, _ = robot.find_bodies("body")

        # Pre-init coefficient vectors to None so that observation functions
        # can safely check getattr(ahm, 'action_scale_vec', None) during
        # ObservationManager._prepare_terms() (which runs before _init_buffers()).
        self.action_scale_vec = None
        self.torque_coeffs_vec = None
        self.action_coeffs_vec = None
        self.jerr_coeffs_vec = None
        self.default_dof_pos_random = None

    @property
    def active_terms(self) -> list[str]:
        return []

    def _prepare_terms(self):
        pass  # No terms — this manager only holds state buffers

    # ------------------------------------------------------------------
    # Lazy initialization (called on first use when scene is ready)
    # ------------------------------------------------------------------

    # def _ensure_initialized(self):
    #     if self._initialized:
    #         return


    #     self._init_buffers()
    #     self._initialized = True

    def _init_buffers(self):
        num_envs = self.num_envs
        device = self.device
        num_feet = len(self.r_feet_ids)
        num_actions = self._env.action_manager.total_action_dim
        traj_len = round(0.2 / self._env.step_dt)  # foot_contact_window = 0.2s
        robot = self._env.scene["robot"]

        self.contact_filt = torch.zeros(num_envs, num_feet, dtype=torch.bool, device=device)
        self.feet_air_time = torch.zeros(num_envs, num_feet, dtype=torch.float, device=device)
        self.foot_contact_trajs = torch.zeros(num_envs, num_feet, traj_len, dtype=torch.bool, device=device)
        self.last_actions = torch.zeros(num_envs, num_actions, dtype=torch.float, device=device)
        self.last_last_actions = torch.zeros(num_envs, num_actions, dtype=torch.float, device=device)
        self.last_dof_vel = torch.zeros(num_envs, robot.num_joints, dtype=torch.float, device=device)
        self.last_foot_velocities = torch.zeros(num_envs, num_feet, 3, dtype=torch.float, device=device)
        self._obs_history_buf = torch.zeros(self.num_envs, _OBS_HISTORY_LENGTH, 73, device=self.device, dtype=torch.float)

        # randomize_default_dof_pos: per-env randomized default joint positions
        # used as action offset, observation normalization, and dof_err reward
        # baseline.  Mirrors AMPTrainEnv._init_buffers() lines 289-293:
        #   default_dof_pos_random = default_joint_pos + uniform(-0.01, 0.01)
        num_joints = robot.num_joints
        self.default_dof_pos_random = robot.data.default_joint_pos.clone() + (
            torch.rand(num_envs, num_joints, device=device, dtype=torch.float) * 2.0 - 1.0
        ) * 0.01  # range [-0.01, 0.01]

        # Override the JointPositionAction term's offset with the randomized
        # version so that ``processed_action = raw_action * scale + offset``
        # uses per-env randomized defaults.  The action term stores its offset
        # as ``_offset`` (shape: num_envs, num_joints_in_term).
        try:
            joint_pos_term = self._env.action_manager.get_term("joint_pos")
            joint_ids = joint_pos_term._joint_ids
            if isinstance(joint_ids, slice):
                joint_pos_term._offset = self.default_dof_pos_random.clone()
            else:
                joint_pos_term._offset = self.default_dof_pos_random[:, joint_ids].clone()
        except (KeyError, AttributeError):
            pass  # Action term not found or structure differs — skip override

        # Precompute per-joint coefficient vectors (avoids per-step for-loops)
        # Vectors are pre-initialized to None in __init__; _init_coeff_vectors
        # fills them in after all managers are initialized.
        self._init_coeff_vectors()

    # ------------------------------------------------------------------
    # Coefficient precomputation
    # ------------------------------------------------------------------

    def _build_coeff_vec_from_dict(
        self,
        joint_names: list[str],
        coeffs_dict: dict[str, float],
    ) -> torch.Tensor:
        """Build a coefficient tensor from an exact-name dict (one-time at init)."""
        vec = torch.ones(len(joint_names), device=self.device)
        for i, name in enumerate(joint_names):
            if name in coeffs_dict:
                vec[i] = coeffs_dict[name]
        return vec

    def _build_scale_vec_from_regex(
        self,
        joint_names: list[str],
        scale_dict: dict[str, float],
    ) -> torch.Tensor:
        """Build a scale tensor from a regex->value dict (one-time at init)."""
        index_list, _, value_list = string_utils.resolve_matching_names_values(
            scale_dict, joint_names
        )
        vec = torch.ones(len(joint_names), device=self.device)
        vec[index_list] = torch.tensor(value_list, device=self.device)
        return vec

    def _init_coeff_vectors(self):
        """Precompute per-joint coefficient tensors from reward/obs manager configs.

        Reads coefficient dicts and SceneEntityCfg from already-initialized
        reward_manager and observation_manager term configs, resolves them to
        tensors indexed by the correct joint order, and caches the results.

        Cached tensors (all indexed to match their term's asset_cfg.joint_ids):
        - ``action_scale_vec``:  per-joint action scale (from obs joint_pos term)
        - ``torque_coeffs_vec``: per-joint torque coefficients (from dof_torque_l2 reward)
        - ``action_coeffs_vec``: per-joint action coefficients (from action_l2 reward)
        - ``jerr_coeffs_vec``:   per-joint joint-error coefficients (from dof_err reward)
        """
        robot = self._env.scene["robot"]
        all_joint_names = robot.data.joint_names

        # --- Reward coefficient vectors ---
        reward_mapping = {
            "dof_torque_l2": ("torque_coeffs", "torque_coeffs_vec"),
            "action_l2":      ("action_coeffs", "action_coeffs_vec"),
            "dof_err":        ("jerr_coeffs",   "jerr_coeffs_vec"),
        }
        for term_name, (param_key, attr_name) in reward_mapping.items():
            try:
                term_cfg = self._env.reward_manager.get_term_cfg(term_name)
                coeffs_dict = term_cfg.params.get(param_key)
                asset_cfg = term_cfg.params.get("asset_cfg")
                if coeffs_dict is not None and asset_cfg is not None:
                    indices = asset_cfg.joint_ids
                    joint_names = [all_joint_names[i] for i in indices]
                    vec = self._build_coeff_vec_from_dict(joint_names, coeffs_dict)
                    setattr(self, attr_name, vec)
            except (KeyError, AttributeError):
                pass  # Term not found or params missing -- leave as None

        # --- Observation action_scale vector ---
        # Find the joint_pos term (uses action_scale with regex patterns)
        obs_mgr = self._env.observation_manager
        for group_name in ("policy", "critic", "obs_future"):
            term_names = getattr(obs_mgr, "_group_obs_term_names", {}).get(group_name, [])
            term_cfgs = getattr(obs_mgr, "_group_obs_term_cfgs", {}).get(group_name, [])
            for term_name, term_cfg in zip(term_names, term_cfgs):
                if term_name != "joint_pos":
                    continue
                asset_cfg = term_cfg.params.get("asset_cfg")
                action_scale = term_cfg.params.get("action_scale")
                if asset_cfg is None or action_scale is None:
                    continue
                indices = asset_cfg.joint_ids
                joint_names = [all_joint_names[i] for i in indices]
                if isinstance(action_scale, dict):
                    self.action_scale_vec = self._build_scale_vec_from_regex(
                        joint_names, action_scale
                    )
                else:
                    self.action_scale_vec = torch.full(
                        (len(joint_names),), float(action_scale), device=self.device
                    )
                break  # Found -- stop searching
            if self.action_scale_vec is not None:
                break

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
        # self._ensure_initialized()
        contact_sensor = self._env.scene["contact_sensor"]
        contact = torch.norm(
            contact_sensor.data.net_forces_w[:, self.c_feet_ids], dim=-1
        ) > contact_sensor.cfg.force_threshold
        self.contact_filt = contact
        self.foot_contact_trajs = torch.cat(
            (contact.unsqueeze(-1), self.foot_contact_trajs[..., :-1]), dim=-1
        )

    def update_obs_history(self):
        """Update the obs_history buffer using obs_buf["policy"].

        Mirrors the original AMPTrainEnv._compute_observations():

        .. code-block:: python

            obs_buf = observation_vector_assemble()  # with scale
            obs_buf += noise                        # add noise
            obs_history_buf = cat(obs_history_buf[:, 1:, :], obs_buf.unsqueeze(1))
            obs_history = flatten(obs_history_buf)  # frame-by-frame

        For reset envs, the buffer was already zeroed in ``reset()``, so the
        sliding window produces ``[0, 0, ..., 0, current_obs]`` — matching
        the original AMPTrainEnv where ``_reset_idx`` sets
        ``obs_history_buf[env_ids] = 0`` and then ``_compute_observations``
        appends one frame.

        The policy observation already has noise, scale, and clip applied
        by the ObservationManager — no recomputation needed.

        Args:
            reset_env_ids: Env indices that were just reset (unused now —
                reset zeroing happens in ``reset()``). Kept for API
                compatibility with the calling code.
        """
        # self._ensure_initialized()
        policy_obs = self._env.obs_buf["policy"]  # (num_envs, 73)

        # Sliding window: shift left, append new frame
        # For reset envs, the buffer was zeroed in reset(), so this produces
        # [0, 0, ..., 0, current_obs] — matching the original AMPTrainEnv.
        self._obs_history_buf = torch.cat(
            (self._obs_history_buf[:, 1:, :], policy_obs.unsqueeze(1)), dim=1
        )

        # Store flattened in obs_buf: [frame0(73), frame1(73), ..., frame9(73)]
        self._env.obs_buf["obs_history"] = self._obs_history_buf.reshape(self.num_envs, -1)

    def post_step_update(self):
        """Update action history — called AFTER obs_history update (end of step).

        Mirrors AMPTrainEnv.step() lines 762-764:
            last_last_actions[:] = last_actions[:]
            last_actions[:] = actions[:]
        """
        # self._ensure_initialized()
        robot = self._env.scene["robot"]
        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self._env.action_manager.action[:]
        self.last_dof_vel[:] = robot.data.joint_vel[:]
        self.last_foot_velocities[:] = robot.data.body_lin_vel_w[:, self.r_feet_ids, :]

    # ------------------------------------------------------------------
    # Reset (called from _reset_idx)
    # ------------------------------------------------------------------

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Reset AMP helper state buffers for the given environments.

        Mirrors AMPTrainEnv._reset_idx() lines 1222-1236:
            contact = norm(net_forces_w[env_ids][:, c_feet_ids]) > threshold
            contact_filt[env_ids] = contact
            foot_contact_trajs[env_ids] = 0
            feet_air_time[env_ids] = 0
            last_actions[env_ids] = 0
            last_last_actions[env_ids] = 0

        Also zeroes obs_history_buf for the reset envs (mirrors
        AMPTrainEnv._reset_idx lines 1243-1244):
            obs_history_buf[env_ids] = 0
            obs_history[env_ids] = 0
        The subsequent ``update_obs_history()`` call then produces
        ``[0, 0, ..., 0, current_obs]`` via the sliding window.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)

        contact_sensor = self._env.scene["contact_sensor"]
        contact = torch.norm(
            contact_sensor.data.net_forces_w[env_ids][:, self.c_feet_ids], dim=-1
        ) > contact_sensor.cfg.force_threshold
        self.contact_filt[env_ids] = contact
        self.feet_air_time[env_ids] = 0.0
        self.foot_contact_trajs[env_ids] = 0
        self.last_actions[env_ids] = 0.0
        self.last_last_actions[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0
        self.last_foot_velocities[env_ids] = 0.0
        self._obs_history_buf[env_ids] = 0.0
        self._env.obs_buf["obs_history"] = self._obs_history_buf.reshape(self.num_envs, -1)
        return {}

    # ------------------------------------------------------------------
    # Helper for reward functions
    # ------------------------------------------------------------------

    def get_torso_lin_vel(self) -> torch.Tensor:
        """Get torso linear velocity in base frame.

        Used by the lin_vel_tracking reward function.
        """
        # self._ensure_initialized()
        robot = self._env.scene["robot"]
        if len(self.r_torso_ids) > 0:
            torso_vel_w = robot.data.body_lin_vel_w[:, self.r_torso_ids[0], :]
            base_quat = robot.data.root_quat_w
            return math_utils.quat_apply_inverse(base_quat, torso_vel_w)
        return robot.data.root_lin_vel_b


@configclass
class AmpHelperCfg:
    """Configuration for AmpHelperManager (no parameters needed)."""
    pass
