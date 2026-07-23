# AMP Locomotion Environment
#
# Custom ManagerBasedRLEnv subclass that integrates AmpHelperManager
# and overrides step() to match the original AMPTrainEnv.step() execution order.
#
# AMPTrainEnv.step() order:
#   1. Process actions
#   2. Physics stepping (decimation loop)
#   3. Update episode counters
#   4. Update contact state (contact_filt, foot_contact_trams)
#   5. Post-physics callback (push robots, heights) -> interval events in manager-based
#   6. Compute terminations
#   7. Compute rewards (feet_air_time updated inside reward function)
#   8. Resample commands
#   9. Reset terminated envs
#  10. Compute observations
#  11. Update obs_history buffer (using obs_buf["policy"])
#  12. Update last_* buffers (last_actions, last_last_actions, last_dof_vel, last_foot_velocities)
#
# AMP reference state initialization is handled by EventTerm(reset_amp_reference)
# in the EventManager, not by custom env methods.

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

from rl_training.managers import AmpHelperCfg, AmpHelperManager

# Constants (must match amp_env_cfg.py — kept here to avoid circular import)
_AMP_NUM_FRAMES = 5
_AMP_HISTORY_STRIDE = 2
# Number of policy observation history frames (must match AmpHelperManager)
_OBS_HISTORY_LENGTH = 10


class AmpLocomotionEnvCfg(ManagerBasedRLEnvCfg):
    """Extended config with AMP helper and AMP-specific fields.

    Subclass this in your environment config (e.g. AmpLocomotionEnvCfg in
    amp_env_cfg.py) and add:
        amp_helper: AmpHelperCfg = AmpHelperCfg()
    """

    amp_helper: AmpHelperCfg = AmpHelperCfg()

    # Reward clipping: if True, clip total reward to min=0 (matches original
    # AMPTrainEnv's only_positive_rewards)
    only_positive_rewards: bool = True


class AmpLocomotionEnv(ManagerBasedRLEnv):
    """Manager-based RL environment for AMP locomotion training.

    Extends :class:`ManagerBasedRLEnv` with:

    - :class:`AmpHelperManager` for AMP state buffer management
      (contact state, action history, obs_history)
    - Custom :meth:`step()` matching the original AMPTrainEnv execution order
    - Custom :meth:`_reset_idx()` that resets the AMP helper and
      spreads out episode lengths on full reset
    - :meth:`_set_amp_discriminator()` for runner integration

    AMP reference state initialization is handled entirely by the
    ``reset_amp_reference`` EventTerm in the EventManager config.
    """

    cfg: AmpLocomotionEnvCfg

    def load_managers(self):
        # Create AMP helper BEFORE super().load_managers()
        # so it's available when reward functions are registered.
        
        self.amp_helper_manager = AmpHelperManager(self.cfg.amp_helper, self)
        print("[INFO] AMP Helper Manager: ", self.amp_helper_manager)

        # AMP dataset reference (set later by runner via _set_amp_discriminator)
        self.amp_dataset = None

        super().load_managers()

        self.amp_helper_manager._init_buffers()

    def step(self, action: torch.Tensor):
        """Execute one time-step matching AMPTrainEnv.step() execution order.

        Key differences from :meth:`ManagerBasedRLEnv.step`:

        1. ``amp_helper_manager.pre_reward_update()`` is called
           AFTER physics stepping and BEFORE termination/reward computation
           (updates contact_filt and foot_contact_trams).
        2. ``only_positive_rewards`` clips total reward to min=0 after
           reward computation (matches original AMPTrainEnv).
        3. ``amp_helper_manager.update_obs_history()`` is called
           AFTER observations are computed (maintains obs_history buffer
           using obs_buf["policy"], frame-by-frame layout).
        4. ``amp_helper_manager.post_step_update()`` is called
           AFTER obs_history update (updates last_actions, last_dof_vel,
           last_foot_velocities).
        """
        # 1. Process actions
        self.action_manager.process_action(action.to(self.device))

        self.recorder_manager.record_pre_step()

        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # 2. Physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)

        # 3. Update episode counters
        self.episode_length_buf += 1
        self.common_step_counter += 1

        # 4. Update AMP helper (contact_filt, foot_contact_trams)
        #    Mirrors AMPTrainEnv.step() lines 728-735
        self.amp_helper_manager.pre_reward_update()

        # 5. Compute terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs

        # 6. Compute rewards (feet_air_time updated inside reward function)
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        # Clip negative rewards (matches AMPTrainEnv only_positive_rewards)
        if self.cfg.only_positive_rewards:
            self.reward_buf = torch.clip(self.reward_buf, min=0.0)

        # Recorder post-step (if needed)
        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        # 7. Reset terminated envs
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.recorder_manager.record_pre_reset(reset_env_ids)
            self._reset_idx(reset_env_ids)
            if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()
            self.recorder_manager.record_post_reset(reset_env_ids)

        # 8. Update commands
        self.command_manager.compute(dt=self.step_dt)

        # 9. Step interval events (push robots)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        # 10. Compute observations
        self.obs_buf = self.observation_manager.compute(update_history=True)

        # 11. Update obs_history buffer using already-computed obs_buf["policy"].
        #     Layout: [frame0(73), frame1(73), ..., frame9(73)] = 730 dims.
        #     Matches original AMPTrainEnv._compute_observations().
        self.amp_helper_manager.update_obs_history()

        # 12. Update last_* buffers (after rewards and observations)
        #     Mirrors AMPTrainEnv.step() lines 762-766
        self.amp_helper_manager.post_step_update()
        # Add time_outs to extras for PPO timeout bootstrapping (matches AMPTrainEnv)
        self.extras["time_outs"] = self.reset_time_outs
        self.extras["episode"] = self.extras["log"]
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset environments, including AMP helper.

        Order:
        1. ``super()._reset_idx()`` — resets scene, applies reset events
           (including ``reset_amp_reference`` if configured), resets all
           standard managers, sets ``episode_length_buf[env_ids] = 0``.
        2. Episode length randomization for full reset (spread out timeouts).
        3. ``amp_helper_manager.reset(env_ids)`` — resets AMP state
           buffers using post-reset contact data.

        Note: obs_history buffer reset is handled in
        ``amp_helper_manager.update_obs_history()`` after observations are
        computed (matching the original AMPTrainEnv which fills the history
        buffer with the post-reset observation).
        """
        super()._reset_idx(env_ids)

        # Spread out resets for full reset (mirrors AMPTrainEnv._reset_idx)
        if len(env_ids) == self.num_envs:
            self.episode_length_buf[:] = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length)
            )
    
        # Reset AMP helper state buffers
        self.amp_helper_manager.reset(env_ids)

    # ------------------------------------------------------------------
    # VecEnv interface: get_observations (called by deeprobotics runner)
    # ------------------------------------------------------------------
    def reset(self, **kwargs):
        obs_dict, extras = super().reset(**kwargs)
        self.amp_helper_manager.reset()
        return self.obs_buf, extras

    def get_observations(self) -> tuple[dict, dict]:
        """Return current observations and extras.

        Called by AMPOnPolicyRunner.learn() at initialization and during training.
        Returns ``(obs_dict, extras)`` where ``obs_dict`` is a mapping from
        observation group name (e.g. ``"policy"``, ``"critic"``,
        ``"obs_history"``, ``"amp_obs_history"``, ``"obs_future"``) to
        the corresponding tensor.
        """
        return self.obs_buf, self.extras

    # ------------------------------------------------------------------
    # AMP discriminator setter (called by runner)
    # ------------------------------------------------------------------

    def _set_amp_discriminator(self, amp_discriminator, amp_normalizer, amp_dataset):
        """Set AMP discriminator, normalizer, and dataset references.

        Called by AMPOnPolicyRunner after initializing these components.
        The AMP dataset is stored on the env so that the ``reset_amp_reference``
        EventTerm can access it via ``env.amp_dataset``.
        """
        self.amp_discriminator = amp_discriminator
        self.amp_normalizer = amp_normalizer
        self.amp_dataset = amp_dataset
        print("[AmpLocomotionEnv-SetAMPDiscriminator]: success")

    # ------------------------------------------------------------------
    # Compatibility properties for deeprobotics rsl_rl AMPOnPolicyRunner
    # ------------------------------------------------------------------

    @property
    def dt(self) -> float:
        """Step dt (alias for step_dt, used by deeprobotics runner)."""
        return self.step_dt

    @property
    def num_obs(self) -> int:
        """Policy observation dimension (used by deeprobotics runner)."""
        return self.observation_manager.group_obs_dim["policy"][0]

    @property
    def num_privileged_obs(self) -> int:
        """Critic observation dimension (used by deeprobotics runner)."""
        dim = self.observation_manager.group_obs_dim.get("critic")
        return dim[0] if dim is not None else self.num_obs

    @property
    def num_actions(self) -> int:
        """Action dimension (used by deeprobotics runner)."""
        return self.action_manager.total_action_dim

    @property
    def num_amp_discriminator_obs(self) -> int:
        """Single-frame AMP observation dimension.

        AMP obs = [proj_grav(3), lin_vel(3), ang_vel(3), joint_pos(N),
                   joint_vel(N), hand_pos(3*Nh), foot_pos(3*Nf)]
        We read this from the amp_obs_history group's per-frame dim.
        """
        amp_dim = self.observation_manager.group_obs_dim.get("amp_obs_history")
        if amp_dim is not None:
            # amp_obs_history is (num_envs, amp_num_frames, amp_obs_dim)
            # group_obs_dim may report the total or per-frame dim depending
            # on how the ObservationManager handles 3D terms.
            # We compute it from the last dimension.
            if isinstance(amp_dim, (list, tuple)) and len(amp_dim) >= 1:
                shape = amp_dim[0] if isinstance(amp_dim[0], (list, tuple)) else amp_dim
                if len(shape) >= 1:
                    return shape[-1] if isinstance(shape, (list, tuple)) else shape
            return amp_dim[-1] if isinstance(amp_dim, (list, tuple)) else amp_dim
        # Fallback: compute from known structure
        return 61

    @property
    def num_obs_history(self) -> int:
        """Policy observation history dimension (flattened).

        obs_history is NOT an ObservationManager group — it is maintained
        by ``amp_helper_manager.update_obs_history()`` using ``obs_buf["policy"]``.
        Total = ``num_obs * history_length`` = 73 * 10 = 730.
        """
        return self.num_obs * _OBS_HISTORY_LENGTH

    @property
    def num_obs_future(self) -> int:
        """Future observation dimension."""
        dim = self.observation_manager.group_obs_dim.get("obs_future")
        if dim is not None:
            return dim[0] if isinstance(dim, (list, tuple)) else dim
        return 69

    @property
    def num_history_frames(self) -> int:
        """Number of policy history frames (used by deeprobotics runner for encoder).

        The encoder input dim = ``num_obs * num_history_frames``.
        """
        return _OBS_HISTORY_LENGTH

    @property
    def current_iteration(self) -> int:
        """Current training iteration (set by runner)."""
        return getattr(self, "_current_iteration", 0)

    @current_iteration.setter
    def current_iteration(self, value: int):
        self._current_iteration = value
