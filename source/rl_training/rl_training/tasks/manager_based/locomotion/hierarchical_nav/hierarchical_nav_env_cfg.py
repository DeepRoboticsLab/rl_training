# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Hierarchical navigation environment configuration.

This environment wraps a low-level locomotion environment and provides
a high-level interface for goal-reaching navigation tasks.
"""

from __future__ import annotations

import math

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import rl_training.tasks.manager_based.locomotion.hierarchical_nav.mdp as mdp


@configclass
class CommandsCfg:
    """Command specifications for the high-level MDP."""

    goal_position = mdp.UniformGoalPositionCommandCfg(
        distance_range=(1.0, 5.0),
        resampling_time_range=(20.0, 20.0),
    )


@configclass
class ActionsCfg:
    """Action specifications for the high-level MDP.
    
    High-level actions are velocity commands [vx, vy, vyaw].
    These will be converted to joint actions by the low-level policy.
    
    Note: Velocity command actions will be handled by a custom action manager
    in the environment class that converts them to joint actions via the low-level policy.
    This is a placeholder configuration - actual action handling is done in the
    hierarchical navigation environment wrapper.
    """
    # Actions will be handled programmatically in the environment class
    # No action configuration needed here as we're using velocity commands directly


@configclass
class ObservationsCfg:
    """Observation specifications for the high-level MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for high-level policy group."""

        robot_position_2d = ObsTerm(
            func=mdp.robot_position_2d,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        robot_yaw = ObsTerm(
            func=mdp.robot_yaw,
            clip=(-math.pi, math.pi),
            scale=1.0,
        )
        goal_position_2d = ObsTerm(
            func=mdp.goal_position_2d,
            params={"command_name": "goal_position"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        distance_to_goal = ObsTerm(
            func=mdp.distance_to_goal,
            params={"command_name": "goal_position"},
            clip=(0.0, 20.0),
            scale=1.0,
        )
        direction_to_goal = ObsTerm(
            func=mdp.direction_to_goal,
            params={"command_name": "goal_position"},
            clip=(-1.0, 1.0),
            scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for high-level critic group."""

        robot_position_2d = ObsTerm(
            func=mdp.robot_position_2d,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        robot_yaw = ObsTerm(
            func=mdp.robot_yaw,
            clip=(-math.pi, math.pi),
            scale=1.0,
        )
        goal_position_2d = ObsTerm(
            func=mdp.goal_position_2d,
            params={"command_name": "goal_position"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        distance_to_goal = ObsTerm(
            func=mdp.distance_to_goal,
            params={"command_name": "goal_position"},
            clip=(0.0, 20.0),
            scale=1.0,
        )
        direction_to_goal = ObsTerm(
            func=mdp.direction_to_goal,
            params={"command_name": "goal_position"},
            clip=(-1.0, 1.0),
            scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Reward terms for the high-level MDP."""

    goal_reaching = RewTerm(
        func=mdp.goal_reaching_reward,
        weight=1.0,
        params={"std": 0.5, "command_name": "goal_position"},
    )
    goal_reaching_progress = RewTerm(
        func=mdp.goal_reaching_progress,
        weight=0.5,
        params={"command_name": "goal_position"},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the high-level MDP."""

    goal_reached = DoneTerm(
        func=mdp.goal_reached,
        params={"threshold": 0.5, "command_name": "goal_position"},
    )


@configclass
class HierarchicalNavEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for hierarchical navigation environment.
    
    This environment wraps a low-level locomotion environment and provides
    a high-level interface for goal-reaching navigation tasks.
    
    High-level actions are velocity commands [vx, vy, vyaw] that are
    converted to joint actions by a frozen low-level policy.
    """

    # Scene settings (will be inherited from low-level environment)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 10  # 10 low-level steps per 1 high-level step
        self.episode_length_s = 30.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = 1
        
        # Note: This configuration is designed to work with a hierarchical navigation
        # environment wrapper that converts high-level velocity commands to low-level
        # joint actions using a frozen low-level policy. The actual environment class
        # needs to be implemented separately to handle the velocity command -> joint
        # action conversion and decimation logic.

