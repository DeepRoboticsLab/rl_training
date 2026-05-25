# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import math

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .rough_env_cfg import DeeproboticsM20RoughEnvCfg_DF, DeeproboticsM20RewardsCfg
import rl_training.tasks.manager_based.locomotion.position.mdp as mdp
from rl_training.tasks.manager_based.locomotion.position.position_env_cfg import CommandsCfg


@configclass
class WaypointCommandsCfg(CommandsCfg):
    """CommandsCfg with base_velocity replaced by waypoint position command."""

    base_velocity = mdp.WaypointPositionCommandCfg(
        asset_name="robot",
        num_waypoints=12,
        max_waypoints=50,
        r_range=(1.0, 4.0),
        theta_min=-1.0,
        theta_max=1.0,
        # tighter switching improves tracking quality on moving targets
        reach_threshold=0.1,
        target_z=0.45,
        resampling_time_range=(1000.0, 1000.0),
        debug_vis=False,
        point_move_dt= 1.0,
        # easier moving-target curriculum: lower acceleration/base speed first
        point_move_a_max= 0.2,
        point_move_b_max = 0.5,
        point_move_vE_min = 0.3,
        point_move_vE_max = 1.0,
        continuous_point_motion=True,
        # slow down effective target motion per sim step for early-stage stability
        continuous_point_motion_dt_scale=1.0,
        # curriculum: auto-increase dt_scale as success improves
        continuous_point_motion_dt_scale_curriculum=True,
        continuous_point_motion_dt_scale_curriculum_threshold=0.9,
        continuous_point_motion_dt_scale_curriculum_increment=0.0005,
        continuous_point_motion_dt_scale_curriculum_ema_alpha=0.3,
    )


@configclass
class DeeproboticsM20WaypointRewardsCfg(DeeproboticsM20RewardsCfg):
    """RewardsCfg with waypoint-tracking rewards replacing velocity-tracking rewards."""

    # Waypoint position tracking reward
    track_waypoint_pos_exp = RewTerm(
        func=mdp.track_waypoint_pos_exp,
        weight=0.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.5)},
    )
    # Waypoint heading tracking reward
    track_waypoint_heading_exp = RewTerm(
        func=mdp.track_waypoint_heading_exp,
        weight=0.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.5)},
    )
    waypoint_progress_reward = RewTerm(
        func=mdp.waypoint_progress_reward,
        weight=0.0,
        params={"command_name": "base_velocity", "progress_scale": 10.0},
    )
    waypoint_reached_bonus = RewTerm(
        func=mdp.waypoint_reached_bonus,
        weight=0.0,
        params={"command_name": "base_velocity", "bonus_scale": 20.0},
    )
    waypoint_reach_speed_match_exp = RewTerm(
        func=mdp.waypoint_reach_speed_match_exp,
        weight=0.0,
        params={"command_name": "base_velocity", "std": 0.2, "near_ratio": 6.0, "decel_power": 1.0},
    )
    waypoint_near_overspeed_l2 = RewTerm(
        func=mdp.waypoint_near_overspeed_l2,
        weight=0.0,
        params={"command_name": "base_velocity", "near_ratio": 3.0, "speed_scale": 2.0},
    )
    bad_orientation_penalty = RewTerm(
        func=mdp.bad_orientation_penalty,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    front_wheels_air_sprint_penalty = RewTerm(
        func=mdp.front_wheels_air_sprint_penalty,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["fl_wheel", "fr_wheel"]),
            "forward_cmd_min": 0.3,
            "speed_threshold": 0.6,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    # Stand-still-at-waypoint penalty
    stand_still_without_waypoint_cmd = RewTerm(
        func=mdp.stand_still_without_waypoint_cmd,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.35,
            "asset_cfg": SceneEntityCfg("robot", joint_names=""),
        },
    )
    # ---- Skating-style turn rewards ----
    skate_turn_body_lean_reward = RewTerm(
        func=mdp.skate_turn_body_lean_reward,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "std": 0.5,
            "heading_threshold": 0.3,
            "speed_threshold": 0.3,
        },
    )
    skate_turn_leg_push_reward = RewTerm(
        func=mdp.skate_turn_leg_push_reward,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "heading_threshold": 0.3,
            "speed_threshold": 0.3,
            "push_scale": 1.0,
        },
    )
    skate_turn_wheel_drag_penalty = RewTerm(
        func=mdp.skate_turn_wheel_drag_penalty,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "heading_threshold": 0.5,
            "speed_threshold": 0.5,
            "drag_threshold": 2.0,
        },
    )


@configclass
class DeeproboticsM20FlatEnvCfg_DF(DeeproboticsM20RoughEnvCfg_DF):
    rewards: DeeproboticsM20WaypointRewardsCfg = DeeproboticsM20WaypointRewardsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards
        self.rewards.base_height_l2.params["sensor_cfg"] = None
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # ---- Replace velocity command with waypoint command ----
        # Command: [dx_b, dy_b, dz_b, dheading] (4 dim) replaces [vx, vy, omega_z] (3 dim)
        self.commands = WaypointCommandsCfg()

        # ---- Update observations for waypoint command (4 dim instead of 3) ----
        # The "velocity_commands" obs term automatically picks up the new command output
        # via generated_commands(command_name="base_velocity").
        # No structural change needed — the command_name still points to "base_velocity".

        # ---- Replace velocity-tracking rewards with waypoint-tracking rewards ----
        # Disable old velocity-tracking rewards
        self.rewards.track_lin_vel_xy_exp = None
        self.rewards.track_ang_vel_z_exp = None
        # Disable rewards that rely on velocity command semantics
        self.rewards.stand_still_without_cmd = None
        self.rewards.stand_still_without_cmd_wheel = None
        # Disable rewards that access velocity command indices (cmd[:, 0], cmd[:, 2] etc.)
        # which are incompatible with waypoint command [dx_b, dy_b, dz_b, dheading]
        self.rewards.feet_air_time_turn_in_place = None
        self.rewards.feet_slide_ang_z_cmd = None

        # Enable waypoint-tracking rewards with proper weights
        self.rewards.track_waypoint_pos_exp.weight = 1.0
        self.rewards.track_waypoint_heading_exp.weight = 1.0
        self.rewards.waypoint_progress_reward.weight = 2.0
        self.rewards.waypoint_reached_bonus.weight = 10.0
        self.rewards.waypoint_reach_speed_match_exp.weight = 0.0
        self.rewards.waypoint_near_overspeed_l2.weight = -0.0
        self.rewards.bad_orientation_penalty.weight = -0.0 #-1e+3
        self.rewards.front_wheels_air_sprint_penalty.weight = -1.0
        
        self.rewards.stand_still_without_waypoint_cmd.weight = -0.5
        self.rewards.stand_still_without_waypoint_cmd.params["asset_cfg"].joint_names = self.leg_joint_names

        # ---- Enable skating-style turn rewards ----
        self.rewards.skate_turn_body_lean_reward.weight = 0.05
        self.rewards.skate_turn_leg_push_reward.weight = 3.0
        self.rewards.skate_turn_wheel_drag_penalty.weight = -0.01

        # Reduce "near-default posture" attraction for waypoint tracking.
        # Loosen hipx penalty specifically to allow lateral leg push during skating turns.
        self.rewards.hipx_joint_pos_penalty.weight = -4.55
        self.rewards.hipy_joint_pos_penalty.weight = -1.0
        self.rewards.knee_joint_pos_penalty.weight = -0.5

        self.rewards.joint_acc_l2.weight = -2e-6
        self.rewards.joint_acc_wheel_l2.weight = -1e-8
        self.rewards.lin_vel_z_l2.weight = -0.5
        self.rewards.ang_vel_xy_l2.weight = -0.001

        self.rewards.feet_contact_without_cmd.weight = -0.0

        self.rewards.flat_orientation_l2.weight = -2.0

        self.rewards.is_terminated.weight = -500.0
        
        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "DeeproboticsM20FlatEnvCfg_DF":
            self.disable_zero_weight_rewards()
