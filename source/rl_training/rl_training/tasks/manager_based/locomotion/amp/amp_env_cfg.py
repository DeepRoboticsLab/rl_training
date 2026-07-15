# AMP Locomotion Environment Configuration (generic base)
# Migrated from IsaacLabExtension/exts/deeprobotics/deeprobotics/Env/cfg/cr1_amp_env_cfg.py
# Adapted from DirectRLEnvCfg to ManagerBasedRLEnvCfg

from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from rl_training.envs import AmpLocomotionEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import PhysxCfg, RenderCfg, SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

import rl_training.tasks.manager_based.locomotion.amp.mdp as mdp


##
# Observation scales (from cr1_amp_env_cfg.py normalization.obs_scales)
# Centralized default values — each ObsTerm references these.
# Override individual ObsTerm scales in robot-specific config if needed.
##


class obs_scales:
    """Default observation scale values.

    These values are referenced by ObsTerm definitions below.
    To customize for a specific robot, override the ObsTerm ``scale``
    in the robot-specific config's ``__post_init__``.
    """
    lin_vel = 2.0
    ang_vel = 0.2
    dof_vel = 0.1
    dof_tor = 5.0
    end_pos = 1.0
    foot_contact_forces = 0.002
    foot_velocity = 0.5


##
# Generic SceneEntityCfg defaults (referenced by ObsTerms)
# Override in robot-specific config (e.g. flat_env_cfg.py)
##

_JOINT_CFG = SceneEntityCfg("robot")
_AMP_JOINT_CFG = SceneEntityCfg("robot")
_FEET_CFG = SceneEntityCfg("robot", body_names=".*foot.*|.*ankle.*")
_HAND_CFG = SceneEntityCfg("robot", body_names=".*hand.*")
_FEET_SENSOR_CFG = SceneEntityCfg("contact_sensor", body_names=".*foot.*|.*ankle.*")


##
# Scene definition
##


@configclass
class AmpSceneCfg(InteractiveSceneCfg):
    """Scene configuraon for AMP flat-terrain training."""

    # ground terrain (flat plane)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # robot
    robot: ArticulationCfg = MISSING  # set by robot-specific config (e.g. flat_env_cfg.py)

    # contact sensor (for feet air time, contact forces)
    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        force_threshold=1.0,
    )

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class AmpCommandsCfg:
    """Command specifications for the AMP MDP."""

    base_velocity = mdp.AmpVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 8.0),
        heading_command=True,
        tracking_strength=0.5,
        zero_command_rate=0.2,
        inplace_turn_time=12.0,
        zero_cmd_threshold_xy=0.15,
        zero_cmd_threshold_z=0.15,
        pure_ang_vel_env_ratio=0.15,
        ranges=mdp.AmpVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.4),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_yaw=(-1.2, 1.2),
            heading=(-3.14, 3.14),
        ),
    )


@configclass
class AmpActionsCfg:
    """Action specifications for the AMP MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=MISSING,
        use_default_offset=True,
        clip=10.0,
        preserve_order=True,
    )


@configclass
class AmpObservationsCfg:
    """Observation specifications for the AMP MDP.

    All observation groups use individual ObsTerms with
    ``concatenate_terms=True``.  Noise, clip, scale, and history
    are handled natively by the ObservationManager.

    Observation groups:
    - **policy** (73 dims): ``[ang_vel(3), proj_grav(3), cmd(3), cmd_flag(1),
      dof_pos(21), dof_vel(21), actions(21)]``
    - **critic** (121 dims): ``[body_vel(3), policy terms(73), torques(21),
      foot_force(6), foot_vel(6), hand_pos(6), foot_pos(6)]``
    - **amp_obs** (61 dims): single-frame AMP discriminator observation
    - **obs_history** (730 dims): 10-frame × 73 policy obs history
    - **amp_obs_history** (305 dims): 5-frame × 61 AMP obs history
    - **obs_future** (69 dims): future observation for estimation
    - **vel_est** (3 dims): body velocity estimation target
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observation group (73 dims)."""

        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.35, n_max=0.35),
            scale=obs_scales.ang_vel,
            clip=(-100.0, 100.0),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.035, n_max=0.035),
            clip=(-100.0, 100.0),
        )
        velocity_command = ObsTerm(
            func=mdp.velocity_command,
            params={"command_name": "base_velocity"},
            scale=(obs_scales.lin_vel, obs_scales.lin_vel, obs_scales.ang_vel),
            clip=(-100.0, 100.0),
        )
        cmd_flag = ObsTerm(
            func=mdp.cmd_flag,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_action_scaled,
            params={"asset_cfg": _JOINT_CFG, "action_scale": MISSING},
            noise=Unoise(n_min=-0.0025, n_max=0.0025),
            clip=(-100.0, 100.0),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": _JOINT_CFG},
            noise=Unoise(n_min=-1.0, n_max=1.0),
            scale=obs_scales.dof_vel,
            clip=(-100.0, 100.0),
        )
        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Critic/privileged observation group (121 dims).

        Contains all policy terms (without noise) + privileged:
        body_vel, torques, foot_force, foot_vel, hand_pos, foot_pos.
        """

        body_vel = ObsTerm(func=mdp.base_lin_vel, scale=obs_scales.lin_vel, clip=(-100.0, 100.0))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=obs_scales.ang_vel, clip=(-100.0, 100.0))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100.0, 100.0))
        velocity_command = ObsTerm(
            func=mdp.velocity_command,
            params={"command_name": "base_velocity"},
            scale=(obs_scales.lin_vel, obs_scales.lin_vel, obs_scales.ang_vel),
            clip=(-100.0, 100.0),
        )
        cmd_flag = ObsTerm(
            func=mdp.cmd_flag,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_action_scaled,
            params={"asset_cfg": _JOINT_CFG, "action_scale": MISSING},
            clip=(-100.0, 100.0),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": _JOINT_CFG},
            scale=obs_scales.dof_vel,
            clip=(-100.0, 100.0),
        )
        actions = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0))
        torques = ObsTerm(
            func=mdp.torques_normalized,
            params={"asset_cfg": _JOINT_CFG, "torque_scale": obs_scales.dof_tor},
            clip=(-100.0, 100.0),
        )
        foot_force = ObsTerm(
            func=mdp.foot_contact_forces,
            params={"sensor_cfg": _FEET_SENSOR_CFG},
            scale=obs_scales.foot_contact_forces,
            clip=(-5000.0, 5000.0),
        )
        foot_vel = ObsTerm(
            func=mdp.foot_velocities,
            params={"asset_cfg": _FEET_CFG},
            scale=obs_scales.foot_velocity,
            clip=(-10.0, 10.0),
        )
        hand_pos = ObsTerm(
            func=mdp.body_pos_in_base_frame,
            params={"asset_cfg": _HAND_CFG},
            clip=(-1.5, 1.5),
        )
        foot_pos = ObsTerm(
            func=mdp.body_pos_in_base_frame,
            params={"asset_cfg": _FEET_CFG},
            clip=(-1.5, 1.5),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class AmpObsCfg(ObsGroup):
        """AMP discriminator observation group (61 dims).

        ``[proj_grav(3), lin_vel(3), ang_vel(3), joint_pos(20),
        joint_vel(20), hand_pos(6), foot_pos(6)]``
        """

        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100.0, 100.0))
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100.0, 100.0))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, clip=(-100.0, 100.0))
        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": _AMP_JOINT_CFG},
            clip=(-100.0, 100.0),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": _AMP_JOINT_CFG},
            clip=(-100.0, 100.0),
        )
        hand_pos = ObsTerm(
            func=mdp.body_pos_in_base_frame,
            params={"asset_cfg": _HAND_CFG},
            clip=(-100.0, 100.0),
        )
        foot_pos = ObsTerm(
            func=mdp.body_pos_in_base_frame,
            params={"asset_cfg": _FEET_CFG},
            clip=(-100.0, 100.0),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class ObsHistoryCfg(ObsGroup):
        """Policy observation history group (730 dims = 10 × 73).

        Uses ObservationManager's built-in ``history_length`` to maintain
        a sliding window — no manual buffer management needed.
        """

        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.35, n_max=0.35),
            scale=obs_scales.ang_vel,
            clip=(-100.0, 100.0),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.035, n_max=0.035),
            clip=(-100.0, 100.0),
        )
        velocity_command = ObsTerm(
            func=mdp.velocity_command,
            params={"command_name": "base_velocity"},
            scale=(obs_scales.lin_vel, obs_scales.lin_vel, obs_scales.ang_vel),
            clip=(-100.0, 100.0),
        )
        cmd_flag = ObsTerm(
            func=mdp.cmd_flag,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_action_scaled,
            params={"asset_cfg": _JOINT_CFG, "action_scale": MISSING},
            noise=Unoise(n_min=-0.0025, n_max=0.0025),
            clip=(-100.0, 100.0),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": _JOINT_CFG},
            noise=Unoise(n_min=-1.0, n_max=1.0),
            scale=obs_scales.dof_vel,
            clip=(-100.0, 100.0),
        )
        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 10

    @configclass
    class AmpObsHistoryCfg(ObsGroup):
        """AMP observation history group (305 dims = 5 × 61).

        Uses ObservationManager's built-in ``history_length`` to maintain
        a sliding window.  Original used stride=2 sampling (5 frames from
        10-frame buffer); simplified to 5 consecutive frames for training
        from scratch.
        """

        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100.0, 100.0))
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100.0, 100.0))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, clip=(-100.0, 100.0))
        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": _AMP_JOINT_CFG},
            clip=(-100.0, 100.0),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": _AMP_JOINT_CFG},
            clip=(-100.0, 100.0),
        )
        hand_pos = ObsTerm(
            func=mdp.body_pos_in_base_frame,
            params={"asset_cfg": _HAND_CFG},
            clip=(-100.0, 100.0),
        )
        foot_pos = ObsTerm(
            func=mdp.body_pos_in_base_frame,
            params={"asset_cfg": _FEET_CFG},
            clip=(-100.0, 100.0),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 5

    @configclass
    class ObsFutureCfg(ObsGroup):
        """Future observation group (69 dims).

        ``[ang_vel(3), proj_grav(3), dof_pos(21), dof_vel(21), actions(21)]``
        """

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=obs_scales.ang_vel, clip=(-100.0, 100.0))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100.0, 100.0))
        joint_pos = ObsTerm(
            func=mdp.joint_pos_action_scaled,
            params={"asset_cfg": _JOINT_CFG, "action_scale": MISSING},
            clip=(-100.0, 100.0),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": _JOINT_CFG},
            scale=obs_scales.dof_vel,
            clip=(-100.0, 100.0),
        )
        actions = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0))

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class VelEstCfg(ObsGroup):
        """Velocity estimation target group (3 dims)."""

        body_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=obs_scales.lin_vel)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    amp_obs: AmpObsCfg = AmpObsCfg()
    obs_history: ObsHistoryCfg = ObsHistoryCfg()
    amp_obs_history: AmpObsHistoryCfg = AmpObsHistoryCfg()
    obs_future: ObsFutureCfg = ObsFutureCfg()
    vel_est: VelEstCfg = VelEstCfg()


# ────────────────────────────────────────────────────────────────────
# Joint coefficient dictionaries (used by reward terms)
# ────────────────────────────────────────────────────────────────────

# Per-joint coefficient dictionaries are robot-specific.
# Define them in robot-specific config (e.g. flat_env_cfg.py) and pass
# via reward term params (torque_coeffs, action_coeffs, jerr_coeffs).


@configclass
class AmpRewardsCfg:
    """Reward terms for the AMP MDP.

    Generic defaults — override in robot-specific config (e.g. flat_env_cfg.py)
    with explicit joint names and per-joint coefficient dicts.
    """

    # Note: In the manager-based framework, ``only_positive_rewards`` is not
    # a recognized field on RewardsCfg (it was specific to DirectRLEnv).
    # Negative rewards are handled by their negative weights.

    # ── Task rewards (Gaussian) ──────────────────────────────────
    lin_vel_tracking = RewTerm(
        func=mdp.lin_vel_tracking,
        weight=0.8,
        params={"command_name": "base_velocity", "decay": 0.45},
    )
    ang_vel_tracking = RewTerm(
        func=mdp.ang_vel_tracking,
        weight=0.8,
        params={"command_name": "base_velocity", "decay": 0.3},
    )

    # ── Posture rewards (Gaussian) ───────────────────────────────
    base_height = RewTerm(
        func=mdp.base_height,
        weight=0.5,
        params={"target_height": 0.875, "decay": 0.02},
    )
    orientation = RewTerm(
        func=mdp.orientation,
        weight=0.5,
        params={"decay": 0.01},
    )
    ang_vel_xy = RewTerm(
        func=mdp.ang_vel_xy,
        weight=0.25,
        params={"decay": 0.2},
    )

    # ── Gait rewards ─────────────────────────────────────────────
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=-1.0,
        params={
            "contact_sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*foot.*|.*ankle.*"]),
            "threshold": 0.32,
        },
    )
    feet_contact = RewTerm(
        func=mdp.feet_contact,
        weight=0.5,
        params={
            "contact_sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*foot.*|.*ankle.*"]),
            "command_name": "base_velocity",
        },
    )
    feet_distance = RewTerm(
        func=mdp.feet_distance,
        weight=0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*foot.*|.*ankle.*"]),
            "target_distance": 0.28,
            "decay": 0.03,
        },
    )
    feet_slippage = RewTerm(
        func=mdp.feet_slippage,
        weight=-0.05,
        params={
            "contact_sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*foot.*|.*ankle.*"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*foot.*|.*ankle.*"]),
        },
    )

    # ── Penalty rewards ──────────────────────────────────────────
    dof_pos_limits = RewTerm(
        func=mdp.dof_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    dof_vel_limits = RewTerm(
        func=mdp.dof_vel_limits,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "soft_ratio": 0.8,
        },
    )
    dof_torque_limits = RewTerm(
        func=mdp.dof_torque_limits,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "soft_ratio": 0.95,
        },
    )

    # ── Regularization rewards ───────────────────────────────────
    dof_vel_l2 = RewTerm(
        func=mdp.dof_vel_l2,
        weight=-2e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    dof_torque_l2 = RewTerm(
        func=mdp.dof_torque_l2,
        weight=-5e-6,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    power = RewTerm(
        func=mdp.power,
        weight=-1e-5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    action_l2 = RewTerm(
        func=mdp.action_l2,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    smoothness_l2 = RewTerm(func=mdp.smoothness_l2, weight=-0.005)

    # ── Constraint rewards ───────────────────────────────────────
    hipz_deviation = RewTerm(
        func=mdp.hipz_deviation,
        weight=-2.5,
        params={"command_name": "base_velocity"},
    )
    dof_err = RewTerm(
        func=mdp.dof_err,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    stand_still = RewTerm(
        func=mdp.stand_still,
        weight=-0.1,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    collision = RewTerm(
        func=mdp.collision,
        weight=-0.1,
        params={
            "contact_sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*torso.*|.*body.*|.*base.*"]),
            "threshold": 1.0,
        },
    )

    # ── AMP reward (placeholder) ─────────────────────────────────
    amp_reward = RewTerm(func=mdp.amp_reward, weight=0.4)


@configclass
class AmpEventCfg:
    """Domain randomization events for AMP training.

    Uses IsaacLab native ``mdp.randomize_rigid_body_mass`` and
    ``mdp.randomize_rigid_body_com`` for mass/COM randomization,
    custom ``push_robots`` / reset functions.
    Override body names in robot-specific config (e.g. flat_env_cfg.py).
    """

    # Note: RewardComputeHelperManager handles its own initialization
    # (lazy init on first use) and reset (from AmpLocomotionEnv._reset_idx).
    # No startup/reset events needed for reward compute helper state.

    # ── Startup: physics material ────────────────────────────────
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.2, 1.25),
            "dynamic_friction_range": (0.2, 1.25),
            "restitution_range": (0.01, 0.3),
            "num_buckets": 1024,
            "make_consistent": False,
        },
    )

    # ── Startup: mass randomization ──────────────────────────────
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.5, 5.0),
            "operation": "add",
            "distribution": "uniform",
        },
    )
    add_torso_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso"),
            "mass_distribution_params": (-2.0, 7.5),
            "operation": "add",
            "distribution": "uniform",
        },
    )
    random_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.9, 1.15),
            "operation": "scale",
        },
    )

    # ── Startup: COM randomization ───────────────────────────────
    random_base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )
    random_torso_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso"),
            "com_range": {"x": (-0.075, 0.075), "y": (-0.075, 0.075), "z": (-0.075, 0.075)},
        },
    )
    random_link_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "com_range": {"x": (-0.015, 0.015), "y": (-0.015, 0.015), "z": (-0.015, 0.015)},
        },
    )

    # ── Reset: root state and DOF ────────────────────────────────
    reset_root_state = EventTerm(
        func=mdp.reset_root_state_randomized,
        mode="reset",
        params={
            "pose_range": {"x": (-1.5, 1.5), "y": (-1.5, 1.5)},
            "randomize": True,
        },
    )
    reset_dof_pos = EventTerm(
        func=mdp.reset_dof_pos_randomized,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pos_range": (0.5, 1.5),
            "pos_offset": (-0.25, 0.25),
            "vel_range": (0.0, 0.0),
            "randomize": True,
        },
    )
    # Note: reward compute helper state reset is handled by
    # RewardComputeHelperManager.reset() called from AmpLocomotionEnv._reset_idx()

    # ── Interval: push robots ────────────────────────────────────
    push_robots = EventTerm(
        func=mdp.push_robots,
        mode="interval",
        interval_range_s=(12.0, 12.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (200.0, 100.0, 50.0),
            "torque_range": (25.0, 50.0, 25.0),
        },
    )


@configclass
class AmpTerminationsCfg:
    """Termination terms for the AMP MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    illegal_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*torso.*|.*body.*|.*base.*"]),
            "threshold": 50.0,
        },
    )

    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 1.0},
    )

    base_height_too_low = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.0},
    )


##
# Environment configuration
##


@configclass
class AmpLocomotionEnvCfg(AmpLocomotionEnvCfg):
    """Base AMP locomotion environment configuration for humanoid robots.

    Uses flat terrain (plane).  No height scanner, no curriculum.
    Override robot-specific settings in child config (e.g. flat_env_cfg.py).
    """

    # environment
    decimation = 4
    episode_length_s = 20.0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=0.005,  # 0.02 / decimation
        render_interval=4,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=1.0,
        ),
        physx=PhysxCfg(
            enable_ccd=True,
            max_position_iteration_count=4,
            max_velocity_iteration_count=1,
            gpu_max_rigid_contact_count=2**25,
            gpu_max_rigid_patch_count=2**25,
            gpu_collision_stack_size=2**27,
        ),
        render=RenderCfg(dlss_mode="Off"),
    )

    # scene
    scene: AmpSceneCfg = AmpSceneCfg(num_envs=4096, env_spacing=3.0, replicate_physics=True)

    # MDP components
    observations: AmpObservationsCfg = AmpObservationsCfg()
    actions: AmpActionsCfg = AmpActionsCfg()
    commands: AmpCommandsCfg = AmpCommandsCfg()
    rewards: AmpRewardsCfg = AmpRewardsCfg()
    terminations: AmpTerminationsCfg = AmpTerminationsCfg()
    events: AmpEventCfg = AmpEventCfg()

    # no curriculum (user requirement: flat terrain only)
    curriculum: object | None = None

    def __post_init__(self):
        """Post-initialization: update sensor periods and physics material."""
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        # sync physics material from terrain
        self.sim.physics_material = self.scene.terrain.physics_material
        # update sensor update periods
        if self.scene.contact_sensor is not None:
            self.scene.contact_sensor.update_period = self.sim.dt

    def disable_zero_weight_rewards(self):
        """Set rewards with zero weight to None to skip computation."""
        for attr in dir(self.rewards):
            if not attr.startswith("__"):
                reward_attr = getattr(self.rewards, attr)
                if (
                    not callable(reward_attr)
                    and hasattr(reward_attr, "weight")
                    and reward_attr.weight == 0
                ):
                    setattr(self.rewards, attr, None)
