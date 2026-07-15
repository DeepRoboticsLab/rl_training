"""Random PD actuator with motor randomization and velocity-dependent torque saturation.

This actuator extends :class:`DelayedPDActuator` to add:
- Per-episode randomization of PD gains (stiffness/damping)
- Motor strength randomization (output torque scaling)
- Position bias randomization (joint zero offset)
- Battery voltage simulation affecting velocity-dependent torque limits
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.utils.math as math_utils
from isaaclab.actuators.actuator_pd import DelayedPDActuator
from isaaclab.actuators.actuator_pd_cfg import DelayedPDActuatorCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions


class RandomPDActuator(DelayedPDActuator):
    """Delayed PD actuator with per-episode randomization.

    Randomizes the following parameters at each reset:
    - **PD gains**: stiffness and damping are scaled by a random factor
      sampled from ``PD_random_range``.
    - **Motor strength**: output torque is scaled by a random factor sampled
      from ``motor_strength``.
    - **Position bias**: a constant offset added to the joint position error,
      sampled from ``pos_bias_range``.
    - **Battery voltage**: affects the velocity-dependent torque saturation
      curve. A random factor ``t_n_vel`` is sampled from ``t_n_vel_range``
      and used to compute the saturation effort. A random battery voltage
      further scales the velocity limit.
    """

    cfg: "RandomPDActuatorCfg"

    def __init__(self, cfg: "RandomPDActuatorCfg", *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        # Randomized PD gains (initialized to nominal, randomized on reset)
        self.random_stiffness = self.stiffness.clone()
        self.random_damping = self.damping.clone()

        # Motor strength scaling factor
        self.motor_strength = torch.ones_like(self.stiffness)

        # Position bias (constant offset on joint position error)
        self.pos_bias = torch.zeros_like(self.stiffness)

        # Cached joint velocity for torque-velocity clipping
        self._joint_vel = torch.zeros_like(self.computed_effort)

        # Battery / velocity-limit randomization
        self.t_n_vel = torch.ones((self._num_envs, 1), device=self._device)
        self._saturation_effort = self.computed_effort.clone()
        self._zeros_effort = torch.zeros_like(self.computed_effort)

        # Minimum velocity limit (80% of nominal) for battery simulation
        self.min_velocity_limit = self.velocity_limit[[0]] * 0.8
        self.battery_v = torch.zeros((self._num_envs, 1), device=self._device)
        self.random_velocity_limit = self.velocity_limit.clone()

    def reset(self, env_ids: Sequence[int]):
        super().reset(env_ids)

        if env_ids is None or env_ids == slice(None):
            num_envs = self._num_envs
        else:
            num_envs = len(env_ids)

        # Randomize PD gains
        pd_scale = math_utils.sample_uniform(
            self.cfg.PD_random_range[0],
            self.cfg.PD_random_range[1],
            (num_envs, self.num_joints),
            device=self._device,
        )
        self.random_stiffness[env_ids] = self.stiffness[env_ids] * pd_scale
        self.random_damping[env_ids] = self.damping[env_ids] * pd_scale

        # Randomize motor strength
        self.motor_strength[env_ids] = math_utils.sample_uniform(
            self.cfg.motor_strength[0],
            self.cfg.motor_strength[1],
            (num_envs, self.num_joints),
            device=self._device,
        )

        # Randomize position bias
        self.pos_bias[env_ids] = math_utils.sample_uniform(
            self.cfg.pos_bias_range[0],
            self.cfg.pos_bias_range[1],
            (num_envs, self.num_joints),
            device=self._device,
        )

        # Randomize battery voltage coefficient (affects torque-velocity curve)
        self.t_n_vel[env_ids] = math_utils.sample_uniform(
            self.cfg.t_n_vel_range[0],
            self.cfg.t_n_vel_range[1],
            (num_envs, 1),
            device=self._device,
        )
        self._saturation_effort[env_ids] = self.effort_limit[env_ids] / (
            1 - self.t_n_vel[env_ids]
        ).clip(0.01, 1)

        # Randomize battery voltage → randomize effective velocity limit
        self.battery_v[env_ids] = torch.rand((num_envs, 1), device=self._device)
        self.random_velocity_limit[env_ids] = self.battery_v[env_ids] * (
            self.velocity_limit[env_ids] - self.min_velocity_limit
        ) + self.min_velocity_limit

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        # Apply delay to control commands
        control_action.joint_positions = self.positions_delay_buffer.compute(
            control_action.joint_positions
        )
        control_action.joint_velocities = self.velocities_delay_buffer.compute(
            control_action.joint_velocities
        )
        control_action.joint_efforts = self.efforts_delay_buffer.compute(
            control_action.joint_efforts
        )

        # Cache joint velocity for torque-velocity clipping
        self._joint_vel[:] = joint_vel

        # Compute PD errors (with position bias)
        error_pos = control_action.joint_positions - (joint_pos + self.pos_bias)
        error_vel = control_action.joint_velocities - joint_vel

        # Compute desired torque (feedforward efforts omitted for efficiency)
        self.computed_effort = self.random_stiffness * error_pos + self.random_damping * error_vel

        # Clip with velocity-dependent saturation and apply motor strength
        self.applied_effort = self._clip_effort(self.computed_effort) * self.motor_strength

        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        """Clip torque based on the velocity-dependent motor torque-speed curve.

        Uses a DC-motor-style saturation model where the maximum torque
        decreases linearly with joint velocity, modulated by the randomized
        saturation effort and velocity limit.
        """
        # Max torque (positive direction)
        max_effort = self._saturation_effort * (1.0 - self._joint_vel / self.random_velocity_limit)
        max_effort = torch.clip(max_effort, min=self._zeros_effort, max=self.effort_limit)

        # Min torque (negative direction)
        min_effort = self._saturation_effort * (-1.0 - self._joint_vel / self.random_velocity_limit)
        min_effort = torch.clip(min_effort, min=-self.effort_limit, max=self._zeros_effort)

        return torch.clip(effort, min=min_effort, max=max_effort)


@configclass
class RandomPDActuatorCfg(DelayedPDActuatorCfg):
    """Configuration for :class:`RandomPDActuator`.

    Inherits all fields from :class:`DelayedPDActuatorCfg` (which includes
    ``effort_limit``, ``effort_limit_sim``, ``velocity_limit``,
    ``velocity_limit_sim``, ``stiffness``, ``damping``, ``friction``,
    ``armature``, ``min_delay``, ``max_delay`` from
    :class:`ActuatorBaseCfg` and :class:`DelayedPDActuatorCfg`).

    Additional fields for per-episode randomization:
    """

    class_type: type = RandomPDActuator

    # Range for PD gain randomization: stiffness and damping are scaled by
    # a uniform sample in [low, high].
    motor_strength: tuple = (0.9, 1.1)
    """Range for motor output strength randomization."""

    PD_random_range: tuple = (0.9, 1.1)
    """Range for PD gain scaling factor."""

    pos_bias_range: tuple = (-0.04, 0.04)
    """Range for joint position bias (zero offset) randomization."""

    t_n_vel_range: tuple = (1 / 3, 2 / 3)
    """Range for battery voltage coefficient (affects torque-velocity curve)."""
