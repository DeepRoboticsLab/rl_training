from __future__ import annotations

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from rl_training.assets.deeprobotics import DR02_AMP_DOF_ORDER, DR02_CFG, DR02_DOF_ORDER
from rl_training.tasks.manager_based.locomotion.amp.amp_env_cfg import OBS_HISTORY_LENGTH, AmpLocomotionEnvCfg


# Per-joint action scale
_ACTION_SCALE = {
    ".*hip_y.*": 0.25, ".*hip_x.*": 0.25, ".*hip_z.*": 0.25, ".*knee.*": 0.25,
    ".*ankle_y.*": 0.5, ".*ankle_x.*": 0.25, ".*waist_z.*": 0.25,
    ".*shoulder_y.*": 0.25, ".*shoulder_x.*": 0.25, ".*shoulder_z.*": 0.25, ".*elbow.*": 0.25,
}

_JERR_COEFFS = {
    "left_hip_y_joint": 0.0, "left_hip_x_joint": 1.0, "left_hip_z_joint": 0.0,
    "left_knee_joint": 0.0, "left_ankle_y_joint": 0.0, "left_ankle_x_joint": 0.0,
    "right_hip_y_joint": 0.0, "right_hip_x_joint": 1.0, "right_hip_z_joint": 0.0,
    "right_knee_joint": 0.0, "right_ankle_y_joint": 0.0, "right_ankle_x_joint": 0.0,
    "waist_z_joint": 7.5,
    "left_shoulder_y_joint": 0.01, "left_shoulder_x_joint": 0.25,
    "left_shoulder_z_joint": 0.25, "left_elbow_joint": 0.01,
    "right_shoulder_y_joint": 0.01, "right_shoulder_x_joint": 0.25,
    "right_shoulder_z_joint": 0.25, "right_elbow_joint": 0.01,
}

_DOF_TORQUE_COEFFS = {
    "left_hip_y_joint": 1.0, "left_hip_x_joint": 1.0, "left_hip_z_joint": 1.0,
    "left_knee_joint": 1.0, "left_ankle_y_joint": 1.0, "left_ankle_x_joint": 1.0,
    "right_hip_y_joint": 1.0, "right_hip_x_joint": 1.0, "right_hip_z_joint": 1.0,
    "right_knee_joint": 1.0, "right_ankle_y_joint": 1.0, "right_ankle_x_joint": 1.0,
    "waist_z_joint": 1.0,
    "left_shoulder_y_joint": 1.0, "left_shoulder_x_joint": 1.0,
    "left_shoulder_z_joint": 1.0, "left_elbow_joint": 1.0,
    "right_shoulder_y_joint": 1.0, "right_shoulder_x_joint": 1.0,
    "right_shoulder_z_joint": 1.0, "right_elbow_joint": 1.0,
}

_DOF_ACTION_COEFFS = {
    "left_hip_y_joint": 0.0, "left_hip_x_joint": 0.0, "left_hip_z_joint": 0.0,
    "left_knee_joint": 0.0, "left_ankle_y_joint": 0.0, "left_ankle_x_joint": 0.2,
    "right_hip_y_joint": 0.0, "right_hip_x_joint": 0.0, "right_hip_z_joint": 0.0,
    "right_knee_joint": 0.0, "right_ankle_y_joint": 0.0, "right_ankle_x_joint": 0.2,
    "waist_z_joint": 0.0,
    "left_shoulder_y_joint": 0.02, "left_shoulder_x_joint": 0.0,
    "left_shoulder_z_joint": 0.0, "left_elbow_joint": 0.01,
    "right_shoulder_y_joint": 0.02, "right_shoulder_x_joint": 0.0,
    "right_shoulder_z_joint": 0.0, "right_elbow_joint": 0.01,
}


@configclass
class DR02AmpFlatEnvCfg(AmpLocomotionEnvCfg):
    """DR02 AMP flat-terrain training configuration."""

    def __post_init__(self):
        super().__post_init__()

        # ── Robot asset ──────────────────────────────────────────
        self.scene.robot = DR02_CFG.replace(prim_path="/World/envs/env_.*/Robot")

        # CRITICAL: joint_names must match DR02_DOF_ORDER so that action[i]
        # corresponds to the same joint as observation[i].
        self.actions.joint_pos.joint_names = list(DR02_DOF_ORDER)
        self.actions.joint_pos.scale = _ACTION_SCALE

        # ── DR02-specific SceneEntityCfgs ─────────────────────────
        joint_cfg = SceneEntityCfg("robot", joint_names=list(DR02_DOF_ORDER), preserve_order=True)
        amp_joint_cfg = SceneEntityCfg("robot", joint_names=list(DR02_AMP_DOF_ORDER), preserve_order=True)
        feet_cfg = SceneEntityCfg("robot", body_names=".*ankle_x_link", preserve_order=True)
        hand_cfg = SceneEntityCfg("robot", body_names=".*hand_link", preserve_order=True)
        feet_sensor_cfg = SceneEntityCfg("contact_sensor", body_names=".*ankle_x_link", preserve_order=True)

        # AmpHelperManager configuration
        self.amp_helper.feet_cfg = feet_cfg.copy()
        self.amp_helper.feet_contact_cfg = feet_sensor_cfg.copy()
        self.amp_helper.torso_cfg = SceneEntityCfg("robot", body_names="body")
        self.amp_helper.obs_history_length = OBS_HISTORY_LENGTH

        # Observation overrides
        # Groups using _JOINT_CFG (all 21 joints in DR02 order)
        # Note: use .copy() for each assignment — the same SceneEntityCfg instance
        # cannot be shared across terms because resolve() mutates body_ids/joint_ids.
        for group_name in ("policy", "critic", "obs_future"):
            obs_group = getattr(self.observations, group_name)
            if hasattr(obs_group, "joint_pos"):
                obs_group.joint_pos.params["asset_cfg"] = joint_cfg.copy()
                obs_group.joint_pos.params["action_scale"] = _ACTION_SCALE
            if hasattr(obs_group, "joint_vel"):
                obs_group.joint_vel.params["asset_cfg"] = joint_cfg.copy()
            if hasattr(obs_group, "torques"):
                obs_group.torques.params["asset_cfg"] = joint_cfg.copy()

        # amp_obs_history
        amp_hist_group = self.observations.amp_obs_history
        amp_hist_group.amp_obs_history.params["amp_joint_cfg"] = amp_joint_cfg.copy()
        amp_hist_group.amp_obs_history.params["hand_cfg"] = hand_cfg.copy()
        amp_hist_group.amp_obs_history.params["feet_cfg"] = feet_cfg.copy()

        # Body-based observation terms
        for group_name in ("critic",):
            obs_group = getattr(self.observations, group_name)
            if hasattr(obs_group, "hand_pos"):
                obs_group.hand_pos.params["asset_cfg"] = hand_cfg.copy()
            if hasattr(obs_group, "foot_pos"):
                obs_group.foot_pos.params["asset_cfg"] = feet_cfg.copy()
            if hasattr(obs_group, "foot_force"):
                obs_group.foot_force.params["sensor_cfg"] = feet_sensor_cfg.copy()
            if hasattr(obs_group, "foot_vel"):
                obs_group.foot_vel.params["asset_cfg"] = feet_cfg.copy()

        # Reward overrides
        # Joint-based rewards with DR02 joint order
        for term_name in (
            "dof_pos_limits", "dof_vel_limits", "dof_torque_limits",
            "dof_vel_l2", "power", "dof_err", "stand_still",
        ):
            getattr(self.rewards, term_name).params["asset_cfg"] = joint_cfg.copy()

        # Rewards with per-joint coefficient dicts
        self.rewards.dof_torque_l2.params["asset_cfg"] = joint_cfg.copy()
        self.rewards.dof_torque_l2.params["torque_coeffs"] = _DOF_TORQUE_COEFFS
        self.rewards.action_l2.params["asset_cfg"] = joint_cfg.copy()
        self.rewards.action_l2.params["action_coeffs"] = _DOF_ACTION_COEFFS
        self.rewards.dof_err.params["jerr_coeffs"] = _JERR_COEFFS

        # Feet rewards
        feet_sensor_reward = SceneEntityCfg("contact_sensor", body_names=[".*ankle_x_link"])
        feet_reward = SceneEntityCfg("robot", body_names=[".*ankle_x_link"])
        self.rewards.feet_air_time.params["contact_sensor_cfg"] = feet_sensor_reward.copy()
        self.rewards.feet_contact.params["contact_sensor_cfg"] = feet_sensor_reward.copy()
        self.rewards.feet_distance.params["asset_cfg"] = feet_reward.copy()
        self.rewards.feet_slippage.params["contact_sensor_cfg"] = feet_sensor_reward.copy()
        self.rewards.feet_slippage.params["asset_cfg"] = feet_reward.copy()

        # Collision reward
        self.rewards.collision.params["contact_sensor_cfg"] = SceneEntityCfg(
            "contact_sensor", body_names=".*hand_link|.*wrist_*_link",
        )

        # hipz_deviation
        self.rewards.hipz_deviation.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=["left_hip_z_joint", "right_hip_z_joint"],
        )

        # Event overrides (DR02 body names)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base_link"
        self.events.add_torso_mass.params["asset_cfg"].body_names = "body"
        self.events.random_link_mass.params["asset_cfg"].body_names = "^(?!.*(base_link|body)).*$"
        self.events.random_base_com.params["asset_cfg"].body_names = "base_link"
        self.events.random_torso_com.params["asset_cfg"].body_names = "body"
        self.events.random_link_com.params["asset_cfg"].body_names = "^(?!.*(base_link|body)).*$"
        # reset_dof_pos may be commented out in base config
        if hasattr(self.events, "reset_dof_pos") and self.events.reset_dof_pos is not None:
            self.events.reset_dof_pos.params["asset_cfg"] = joint_cfg
        if hasattr(self.events, "reset_root_state") and self.events.reset_root_state is not None:
            self.events.reset_root_state.params["pose_range"] = {"x": (-1.5, 1.5), "y": (-1.5, 1.5)}
        # AMP reference
        if hasattr(self.events, "reset_amp_reference") and self.events.reset_amp_reference is not None:
            self.events.reset_amp_reference.params["amp_dof_cfg"] = amp_joint_cfg
        # self.events.push_robots.params["asset_cfg"].body_names = "base_link"

        # Termination overrides (DR02 body names)
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = ["base_link"]

        # Flat terrain
        self.scene.terrain.terrain_type = "plane"
        self.disable_zero_weight_rewards()
