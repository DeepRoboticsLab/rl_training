# AMP Locomotion Environment
#
# Custom ManagerBasedRLEnv subclass that integrates RewardComputeHelperManager
# and overrides step() to match the original AMPTrainEnv.step() execution order.
#
# AMPTrainEnv.step() order:
#   1. Process actions
#   2. Physics stepping (decimation loop)
#   3. Update episode counters
#   4. Update contact state (contact_filt, foot_contact_trajs)
#   5. Post-physics callback (push robots, heights) → interval events in manager-based
#   6. Compute terminations
#   7. Compute rewards (feet_air_time updated inside reward function)
#   8. Resample commands
#   9. Reset terminated envs
#  10. Compute observations
#  11. Update last_* buffers (last_actions, last_last_actions)

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import (
    CommandManager,
    CurriculumManager,
    EventManager,
    RewardManager,
    TerminationManager,
)

from rl_training.managers import RewardComputeHelperCfg, RewardComputeHelperManager


class AmpLocomotionEnvCfg(ManagerBasedRLEnvCfg):
    """Extended config with reward compute helper field.

    Subclass this in your environment config (e.g. AmpLocomotionEnvCfg in
    amp_env_cfg.py) and add:
        reward_compute_helper: RewardComputeHelperCfg = RewardComputeHelperCfg()
    """

    reward_compute_helper: RewardComputeHelperCfg = RewardComputeHelperCfg()


class AmpLocomotionEnv(ManagerBasedRLEnv):
    """Manager-based RL environment for AMP locomotion training.

    Extends :class:`ManagerBasedRLEnv` with:

    - :class:`RewardComputeHelperManager` for reward state buffer management
    - Custom :meth:`step()` matching the original AMPTrainEnv execution order
    - Custom :meth:`_reset_idx()` that resets the reward compute helper
    """

    cfg: AmpLocomotionEnvCfg

    def load_managers(self):
        # Create reward compute helper BEFORE super().load_managers()
        # so it's available when reward functions are registered.
        self.reward_compute_helper_manager = RewardComputeHelperManager(
            self.cfg.reward_compute_helper, self
        )
        print("[INFO] Reward Compute Helper Manager: ", self.reward_compute_helper_manager)

        super().load_managers()

    def step(self, action: torch.Tensor):
        """Execute one time-step matching AMPTrainEnv.step() execution order.

        Key differences from :meth:`ManagerBasedRLEnv.step`:

        1. ``reward_compute_helper_manager.pre_reward_update()`` is called
           AFTER physics stepping and BEFORE termination/reward computation
           (updates contact_filt and foot_contact_trajs).
        2. ``reward_compute_helper_manager.post_step_update()`` is called
           AFTER observations are computed (updates last_actions).
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

        # 4. Update reward compute helper (contact_filt, foot_contact_trajs)
        #    Mirrors AMPTrainEnv.step() lines 728-735
        self.reward_compute_helper_manager.pre_reward_update()

        # 5. Compute terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs

        # 6. Compute rewards (feet_air_time updated inside reward function)
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

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

        # 11. Update last_* buffers (after rewards and observations)
        #     Mirrors AMPTrainEnv.step() lines 762-764
        self.reward_compute_helper_manager.post_step_update()

        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset environments, including reward compute helper.

        Order:
        1. ``super()._reset_idx()`` — resets scene, applies reset events,
           resets all standard managers.
        2. ``reward_compute_helper_manager.reset(env_ids)`` — resets reward
           state buffers using post-reset contact data.
        """
        super()._reset_idx(env_ids)
        self.reward_compute_helper_manager.reset(env_ids)
