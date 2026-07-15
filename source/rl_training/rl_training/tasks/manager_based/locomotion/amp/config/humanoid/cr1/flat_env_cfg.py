# CR1 AMP Flat-Terrain Environment Configuration
# Migrated from IsaacLabExtension cr1_amp_env_cfg.py (flat terrain mode)

from __future__ import annotations

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from rl_training.assets.deeprobotics import DR02_AMP_DOF_ORDER, DR02_CFG, DR02_DOF_ORDER
from rl_training.tasks.manager_based.locomotion.amp.amp_env_cfg import AmpLocomotionEnvCfg


# ────────────────────────────────────────────────────────────────────
# CR1-specific per-joint coefficient dictionaries
# ────────────────────────────────────────────────────────────────────

# Per-joint action scale (from cr1_amp_env_cfg.py action_scale dict)
# Keys are regex patterns matched against joint names
_ACTION_SCALE = {
    "hip_y": 0.25, "hip_x": 0.25, "hip_z": 0.25, "knee": 0.25,
    "ankle_y": 0.5, "ankle_x": 0.25, "waist_z": 0.25,
    "shoulder_y": 0.15, "shoulder_x": 0.25, "shoulder_z": 0.25, "elbow": 0.15,
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
    "left_knee_joint": 1.0, "left_ankle_y_joint": 4.0, "left_ankle_x_joint": 2.0,
    "right_hip_y_joint": 1.0, "right_hip_x_joint": 1.0, "right_hip_z_joint": 1.0,
    "right_knee_joint": 1.0, "right_ankle_y_joint": 4.0, "right_ankle_x_joint": 2.0,
    "waist_z_joint": 1.0,
    "left_shoulder_y_joint": 1.0, "left_shoulder_x_joint": 1.0,
    "left_shoulder_z_joint": 1.0, "left_elbow_joint": 1.0,
    "right_shoulder_y_joint": 1.0, "right_shoulder_x_joint": 1.0,
    "right_shoulder_z_joint": 1.0, "right_elbow_joint": 1.0,
}

_DOF_ACTION_COEFFS = {
    "left_hip_y_joint": 0.0, "left_hip_x_joint": 0.0, "left_hip_z_joint": 0.0,
    "left_knee_joint": 0.0, "left_ankle_y_joint": 1.0, "left_ankle_x_joint": 0.5,
    "right_hip_y_joint": 0.0, "right_hip_x_joint": 0.0, "right_hip_z_joint": 0.0,
    "right_knee_joint": 0.0, "right_ankle_y_joint": 1.0, "right_ankle_x_joint": 0.5,
    "waist_z_joint": 0.0,
    "left_shoulder_y_joint": 0.0, "left_shoulder_x_joint": 0.0,
    "left_shoulder_z_joint": 0.0, "left_elbow_joint": 0.0,
    "right_shoulder_y_joint": 0.0, "right_shoulder_x_joint": 0.0,
    "right_shoulder_z_joint": 0.0, "right_elbow_joint": 0.0,
}


@configclass
class CR1AmpFlatEnvCfg(AmpLocomotionEnvCfg):
    """CR1-B2-STD AMP flat-terrain training configuration.

    Inherits from :class:`AmpLocomotionEnvCfg` (generic base) and overrides
    with CR1-specific robot asset, joint/body configs, coefficient dicts,
    and body name patterns.
    """

    def __post_init__(self):
        super().__post_init__()

        # ── Robot asset ──────────────────────────────────────────
        self.scene.robot = DR02_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # ── Action scale (per-joint, CR1-specific) ───────────────
        self.actions.joint_pos.scale = _ACTION_SCALE

        # ── CR1-specific SceneEntityCfgs ─────────────────────────
        joint_cfg = SceneEntityCfg("robot", joint_names=list(DR02_DOF_ORDER), preserve_order=True)
        amp_joint_cfg = SceneEntityCfg("robot", joint_names=list(DR02_AMP_DOF_ORDER), preserve_order=True)
        feet_cfg = SceneEntityCfg("robot", body_names=".*ankle_x_link", preserve_order=True)
        hand_cfg = SceneEntityCfg("robot", body_names=".*hand_link", preserve_order=True)
        feet_sensor_cfg = SceneEntityCfg("contact_sensor", body_names=".*ankle_x_link", preserve_order=True)

        # ── Observation overrides ────────────────────────────────
        # Groups using _JOINT_CFG (all 21 joints in DR02 order)
        for group_name in ("policy", "critic", "obs_history", "obs_future"):
            obs_group = getattr(self.observations, group_name)
            if hasattr(obs_group, "joint_pos"):
                obs_group.joint_pos.params["asset_cfg"] = joint_cfg
                # joint_pos_action_scaled needs per-joint action_scale dict
                obs_group.joint_pos.params["action_scale"] = _ACTION_SCALE
            if hasattr(obs_group, "joint_vel"):
                obs_group.joint_vel.params["asset_cfg"] = joint_cfg
            if hasattr(obs_group, "torques"):
                obs_group.torques.params["asset_cfg"] = joint_cfg

        # Groups using _AMP_JOINT_CFG (20 joints, no waist)
        for group_name in ("amp_obs", "amp_obs_history"):
            obs_group = getattr(self.observations, group_name)
            if hasattr(obs_group, "joint_pos"):
                obs_group.joint_pos.params["asset_cfg"] = amp_joint_cfg
            if hasattr(obs_group, "joint_vel"):
                obs_group.joint_vel.params["asset_cfg"] = amp_joint_cfg

        # Body-based observation terms
        for group_name in ("critic", "amp_obs", "amp_obs_history"):
            obs_group = getattr(self.observations, group_name)
            if hasattr(obs_group, "hand_pos"):
                obs_group.hand_pos.params["asset_cfg"] = hand_cfg
            if hasattr(obs_group, "foot_pos"):
                obs_group.foot_pos.params["asset_cfg"] = feet_cfg
            if hasattr(obs_group, "foot_force"):
                obs_group.foot_force.params["sensor_cfg"] = feet_sensor_cfg
            if hasattr(obs_group, "foot_vel"):
                obs_group.foot_vel.params["asset_cfg"] = feet_cfg

        # ── Reward overrides ──────────────────────────────────────
        # Joint-based rewards with DR02 joint order
        for term_name in (
            "dof_pos_limits", "dof_vel_limits", "dof_torque_limits",
            "dof_vel_l2", "power", "dof_err", "stand_still",
        ):
            getattr(self.rewards, term_name).params["asset_cfg"] = joint_cfg

        # Rewards with per-joint coefficient dicts
        self.rewards.dof_torque_l2.params["asset_cfg"] = joint_cfg
        self.rewards.dof_torque_l2.params["torque_coeffs"] = _DOF_TORQUE_COEFFS
        self.rewards.action_l2.params["asset_cfg"] = joint_cfg
        self.rewards.action_l2.params["action_coeffs"] = _DOF_ACTION_COEFFS
        self.rewards.dof_err.params["jerr_coeffs"] = _JERR_COEFFS

        # Feet rewards
        feet_sensor_reward = SceneEntityCfg("contact_sensor", body_names=[".*ankle_x_link"])
        feet_reward = SceneEntityCfg("robot", body_names=[".*ankle_x_link"])
        self.rewards.feet_air_time.params["contact_sensor_cfg"] = feet_sensor_reward
        self.rewards.feet_contact.params["contact_sensor_cfg"] = feet_sensor_reward
        self.rewards.feet_distance.params["asset_cfg"] = feet_reward
        self.rewards.feet_slippage.params["contact_sensor_cfg"] = feet_sensor_reward
        self.rewards.feet_slippage.params["asset_cfg"] = feet_reward

        # Collision reward (CR1 body names)
        self.rewards.collision.params["contact_sensor_cfg"] = SceneEntityCfg(
            "contact_sensor", body_names=["base_link", "body"],
        )

        # hipz_deviation (CR1 hip_z joint names)
        self.rewards.hipz_deviation.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=["left_hip_z_joint", "right_hip_z_joint"],
        )

        # ── Event overrides (CR1 body names) ─────────────────────
        self.events.add_base_mass.params["asset_cfg"].body_names = "base_link"
        self.events.add_torso_mass.params["asset_cfg"].body_names = "body"
        self.events.random_link_mass.params["asset_cfg"].body_names = "^(?!.*(base_link|body)).*$"
        self.events.random_base_com.params["asset_cfg"].body_names = "base_link"
        self.events.random_torso_com.params["asset_cfg"].body_names = "body"
        self.events.random_link_com.params["asset_cfg"].body_names = "^(?!.*(base_link|body)).*$"
        self.events.reset_dof_pos.params["asset_cfg"] = joint_cfg
        self.events.push_robots.params["asset_cfg"].body_names = "base_link"

        # ── Termination overrides (CR1 body names) ───────────────
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = ["base_link", "body"]

        # ── Flat terrain ──────────────────────────────────────────
        self.scene.terrain.terrain_type = "plane"
        self.disable_zero_weight_rewards()
