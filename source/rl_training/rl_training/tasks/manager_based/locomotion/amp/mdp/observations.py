# AMP observation functions for humanoid locomotion
#
# Individual observation terms for use with ObservationManager.
# Noise, clip, scale, and history are handled by the ObservationManager
# via ObsTerm configuration — these functions return raw or already-scaled
# values and do NOT manage any buffers or indices themselves.
#
# Joint/body reordering is handled by SceneEntityCfg(joint_names=...,
# preserve_order=True) in the ObsTerm params, which resolves joint_ids
# and body_ids automatically when the manager initializes.
#
# Functions that apply internal scaling (joint_pos_action_scaled,
# torques_normalized) accept scale values as parameters, set via
# ObsTerm params in amp_env_cfg.py.

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.managers import ManagerTermBaseCfg


# ---------------------------------------------------------------------------
# Command observations
# ---------------------------------------------------------------------------

def velocity_command(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Velocity command (first 3 dims: vx, vy, wz). Shape: (num_envs, 3).

    Returns raw command — scale is applied via ``ObsTerm(scale=...)``.
    """
    cmd_term = env.command_manager.get_term(command_name)
    return cmd_term.command[:, :3]


def cmd_flag(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Binary command flag (1 if non-zero command). Shape: (num_envs, 1)."""
    cmd_term = env.command_manager.get_term(command_name)
    return cmd_term.cmd_flag


# ---------------------------------------------------------------------------
# Joint observations (already-scaled — no ObsTerm scale needed)
# ---------------------------------------------------------------------------

def joint_pos_action_scaled(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
    action_scale: float | dict[str, float] = 0.25,
) -> torch.Tensor:
    """Joint position relative to default, divided by action scale.

    Shape: (num_envs, N).

    Returns already-scaled value ``(joint_pos - default) / action_scale``.
    Used for policy obs where noise must match the original amplitude
    (±0.0025) — if ObsTerm ``scale`` were used instead, the noise applied
    before scale would be amplified by 1/action_scale.

    The per-joint scale vector is precomputed once at init by
    ``AmpHelperManager._init_coeff_vectors()`` and cached as
    ``ahm.action_scale_vec``.  If the cached vector is not available
    (e.g. manager not initialized), it falls back to per-call computation.

    Args:
        action_scale: The action scale used by JointPositionActionCfg.
            Can be a float (uniform) or a dict of regex→float (per-joint).
            Must match the action config's ``scale`` value.
    """
    robot = env.scene[asset_cfg.name]
    dof_pos = robot.data.joint_pos[:, asset_cfg.joint_ids]

    # Use randomized default from AmpHelperManager if available
    # (matches AMPTrainEnv randomize_default_dof_pos=True)
    ahm = getattr(env, "amp_helper_manager", None)
    default_random = getattr(ahm, "default_dof_pos_random", None) if ahm is not None else None
    if default_random is not None:
        default_pos = default_random[:, asset_cfg.joint_ids]
    else:
        default_pos = robot.data.default_joint_pos[:, asset_cfg.joint_ids]
    rel_pos = dof_pos - default_pos

    # Use precomputed scale vector from AmpHelperManager if available
    scale_vec = getattr(ahm, "action_scale_vec", None) if ahm is not None else None
    if scale_vec is not None:
        return rel_pos / scale_vec

    # Fallback: compute on the fly (rare — only if manager not initialized)
    if isinstance(action_scale, dict):
        joint_names = [robot.data.joint_names[i] for i in asset_cfg.joint_ids]
        index_list, _, value_list = string_utils.resolve_matching_names_values(
            action_scale, joint_names
        )
        scales = torch.ones(len(joint_names), device=env.device)
        scales[index_list] = torch.tensor(value_list, device=env.device)
        return rel_pos / scales
    return rel_pos / action_scale


def torques_normalized(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
    torque_scale: float = 5.0,
) -> torch.Tensor:
    """Applied torques normalized by effort limits and scaled.

    Shape: (num_envs, N).

    Returns already-scaled value ``torque / limit × torque_scale``.
    Uses ``robot.data.joint_effort_limits`` (set from PhysX max forces).

    Args:
        torque_scale: Scale factor applied after normalization.
    """
    robot = env.scene[asset_cfg.name]
    torques = robot.data.applied_torque[:, asset_cfg.joint_ids]
    limits = robot.data.joint_effort_limits[:, asset_cfg.joint_ids]
    return torques / limits * torque_scale


# ---------------------------------------------------------------------------
# Foot / hand observations (raw — clip & scale via ObsTerm)
# ---------------------------------------------------------------------------

def foot_contact_forces(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Foot contact forces in base frame. Shape: (num_envs, 3×N_feet).

    Returns raw forces — clip and scale are applied via ``ObsTerm`` config
    (e.g. ``clip=(-5000, 5000), scale=0.002``).
    """
    robot = env.scene["robot"]
    contact_sensor = env.scene[sensor_cfg.name]
    base_quat = robot.data.root_quat_w
    forces = torch.cat([
        math_utils.quat_apply_inverse(base_quat, contact_sensor.data.net_forces_w[:, idx, :])
        for idx in sensor_cfg.body_ids
    ], dim=1)
    return forces


def foot_velocities(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Foot velocities in base frame. Shape: (num_envs, 3×N_feet).

    Returns raw velocities — clip and scale are applied via ``ObsTerm``
    config (e.g. ``clip=(-10, 10), scale=0.5``).
    """
    robot = env.scene[asset_cfg.name]
    base_quat = robot.data.root_quat_w
    vels = torch.cat([
        math_utils.quat_apply_inverse(base_quat, robot.data.body_lin_vel_w[:, idx, :])
        for idx in asset_cfg.body_ids
    ], dim=1)
    return vels


def body_pos_in_base_frame(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Body positions relative to base in body frame.

    Shape: (num_envs, 3×N_bodies).

    Returns raw positions — clip and scale are applied via ``ObsTerm``
    config.  Used for both hand and foot positions (specified via
    ``SceneEntityCfg(body_names=...)``).
    """
    robot = env.scene[asset_cfg.name]
    base_quat = robot.data.root_quat_w
    root_pos = robot.data.root_pos_w
    body_pos_w = robot.data.body_pos_w[:, asset_cfg.body_ids, :3]
    pos_to_base = body_pos_w - root_pos.unsqueeze(1)
    pos = torch.stack([
        math_utils.quat_apply_inverse(base_quat, pos_to_base[:, i, :])
        for i in range(pos_to_base.shape[1])
    ], dim=1).flatten(start_dim=1)
    return pos


# ---------------------------------------------------------------------------
# AMP observation history with stride sampling
# ---------------------------------------------------------------------------

class AmpObsHistoryTerm(ManagerTermBase):
    """AMP observation history with stride-sampled sliding window.

    Maintains a buffer of ``amp_num_frames * amp_history_stride`` consecutive
    frames, then subsamples with ``amp_history_stride`` to produce
    ``amp_num_frames`` frames that are temporally separated by
    ``amp_history_stride`` simulation steps — matching the expert motion
    dataset's ``time_between_frames = dt * amp_history_stride``.

    This mirrors the original AMPTrainEnv implementation:

    .. code-block:: python

        amp_obs_history = torch.zeros(num_envs, num_frames * stride, obs_dim)
        # each step: sliding window
        amp_obs_history = cat(amp_obs_history[:, 1:, :], amp_obs.unsqueeze(1))
        # stride subsample
        amp_obs_history_buf = amp_obs_history[:, stride-1::stride, :]

    The single-frame AMP observation consists of:
    ``[projected_gravity(3), base_lin_vel(3), base_ang_vel(3),
       joint_pos(N), joint_vel(N), hand_pos(3×N_h), foot_pos(3×N_f)]``

    Output shape: ``(num_envs, amp_num_frames * amp_obs_dim)`` (flattened).
    """

    def __init__(self, cfg: "ManagerTermBaseCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._buffer: torch.Tensor | None = None
        self._amp_obs_dim: int | None = None
        # cached params from __call__ for use in reset()
        self._amp_joint_cfg: SceneEntityCfg | None = None
        self._hand_cfg: SceneEntityCfg | None = None
        self._feet_cfg: SceneEntityCfg | None = None
        self._amp_history_stride: int = 2
        self._amp_num_frames: int = 5

    def __call__(
        self,
        env: ManagerBasedEnv,
        amp_joint_cfg: SceneEntityCfg,
        hand_cfg: SceneEntityCfg,
        feet_cfg: SceneEntityCfg,
        amp_history_stride: int = 2,
        amp_num_frames: int = 5,
    ) -> torch.Tensor:
        """Compute stride-sampled AMP observation history."""
        robot = env.scene[amp_joint_cfg.name]

        # -- single-frame AMP observation (same terms as former AmpObsCfg) --
        amp_obs = torch.cat(
            [
                robot.data.projected_gravity_b,
                robot.data.root_lin_vel_b,
                robot.data.root_ang_vel_b,
                robot.data.joint_pos[:, amp_joint_cfg.joint_ids],
                robot.data.joint_vel[:, amp_joint_cfg.joint_ids],
                body_pos_in_base_frame(env, hand_cfg),
                body_pos_in_base_frame(env, feet_cfg),
            ],
            dim=-1,
        )
        # clip per-frame (same as ObsTerm clip=(-100, 100))
        amp_obs = torch.clamp(amp_obs, -100.0, 100.0)

        # -- lazy buffer init --
        if self._buffer is None:
            self._amp_obs_dim = amp_obs.shape[-1]
            buffer_size = amp_num_frames * amp_history_stride
            self._buffer = torch.zeros(
                env.num_envs, buffer_size, self._amp_obs_dim,
                device=env.device, dtype=torch.float,
            )

        # cache params for reset()
        self._amp_joint_cfg = amp_joint_cfg
        self._hand_cfg = hand_cfg
        self._feet_cfg = feet_cfg
        self._amp_history_stride = amp_history_stride
        self._amp_num_frames = amp_num_frames

        # -- sliding window: shift left, append new frame --
        self._buffer = torch.cat(
            (self._buffer[:, 1:, :], amp_obs.unsqueeze(1)), dim=1
        )

        # -- stride subsample: indices [stride-1, 2*stride-1, ...] --
        subsampled = self._buffer[:, amp_history_stride - 1 :: amp_history_stride, :]

        # -- store 3D buffer on env for AMP reward access --
        # shape: (num_envs, amp_num_frames, amp_obs_dim)
        # This mirrors the original AMPTrainEnv.amp_obs_history_buf.
        # The reward function (amp_reward) reads this in the NEXT step,
        # since observations are computed after rewards in step().
        self._env.amp_obs_history_buf = subsampled

        # -- return 3D tensor (num_envs, amp_num_frames, amp_obs_dim) --
        # The observation manager passes this through as-is (single term,
        # concatenate_terms=True just calls torch.cat on a single-element list).
        return subsampled

    def reset(self, env_ids: "Sequence[int] | None" = None) -> None:
        """Fill the buffer with the initial AMP observation for reset envs.

        Mirrors AMPTrainEnv._reset_idx lines 1239-1242:
            amp_obs_buf[env_ids] = _amp_observaton_vector_assemble()[env_ids]
            amp_obs_history[env_ids] = amp_obs_buf[env_ids].unsqueeze(1).repeat(
                1, amp_num_frames * amp_history_stride, 1)

        This ensures the sliding window starts with valid data after reset,
        instead of zeros which would produce NaN in the discriminator.
        """
        if self._buffer is None or self._amp_joint_cfg is None:
            return
        if isinstance(env_ids, slice):
            env_ids = None  # reset all
        if env_ids is not None and len(env_ids) == 0:
            return

        # compute single-frame AMP observation from current (post-reset) state
        robot = self._env.scene[self._amp_joint_cfg.name]
        amp_obs = torch.cat(
            [
                robot.data.projected_gravity_b,
                robot.data.root_lin_vel_b,
                robot.data.root_ang_vel_b,
                robot.data.joint_pos[:, self._amp_joint_cfg.joint_ids],
                robot.data.joint_vel[:, self._amp_joint_cfg.joint_ids],
                body_pos_in_base_frame(self._env, self._hand_cfg),
                body_pos_in_base_frame(self._env, self._feet_cfg),
            ],
            dim=-1,
        )
        amp_obs = torch.clamp(amp_obs, -100.0, 100.0)

        # fill the entire buffer with this observation (repeated for all frames)
        buffer_size = self._amp_num_frames * self._amp_history_stride
        if env_ids is None:
            self._buffer[:] = amp_obs.unsqueeze(1).repeat(1, buffer_size, 1)
        else:
            self._buffer[env_ids] = amp_obs[env_ids].unsqueeze(1).repeat(1, buffer_size, 1)
