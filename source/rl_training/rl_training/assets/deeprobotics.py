# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import os
import re

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from rl_training.actuator import RandomPDActuatorCfg
from rl_training.assets import ISAACLAB_ASSETS_DATA_DIR

DEEPROBOTICS_LITE3_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Lite3/Lite3_usd/Lite3.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=1
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.375),
        joint_pos={
            ".*HipX_joint": 0.0,
            ".*HipY_joint": -0.65,
            ".*Knee_joint": 1.3,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.99,
    actuators={
        "Hip": DelayedPDActuatorCfg(
            joint_names_expr=[".*_Hip[X,Y]_joint"],
            effort_limit=24.0,
            velocity_limit=26.2,
            stiffness=30.0,
            damping=1.0,
            friction=0.0,
            armature=0.0,
            min_delay=0,
            max_delay=1,
        ),
        "Knee": DelayedPDActuatorCfg(
            joint_names_expr=[".*_Knee_joint"],
            effort_limit=36.0,
            velocity_limit=17.3,
            stiffness=30.0,
            damping=1.0,
            friction=0.0,
            armature=0.0,
            min_delay=0,
            max_delay=1,
        ),
    },
)

DEEPROBOTICS_M20_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/M20/M20_usd/M20.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=1
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.58),
        joint_pos={
            ".*hipx_joint": 0.0,
            "f[l,r]_hipy_joint": -0.3,
            "h[l,r]_hipy_joint": 0.3,
            "f[l,r]_knee_joint": 0.6,
            "h[l,r]_knee_joint": -0.6,
            ".*wheel_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "joint": DelayedPDActuatorCfg(
            joint_names_expr=[".*hipx_joint", ".*hipy_joint", ".*knee_joint"],
            effort_limit=76.4,
            velocity_limit=22.4,
            stiffness=80.0,
            damping=2.0,
            friction=0.0,
            armature=0.0,
            min_delay=0,
            max_delay=1,
        ),
        "wheel": DelayedPDActuatorCfg(
            joint_names_expr=[".*_wheel_joint"],
            effort_limit=21.6,
            velocity_limit=79.3,
            stiffness=0.0,
            damping=0.6,
            friction=0.0,
            armature=0.00243216,
            min_delay=0,
            max_delay=1,
        ),
    },
)


# ---------------------------------------------------------------------------
# DR02 (DR02-Pro)
# ---------------------------------------------------------------------------

def _convert_to_fix_joints_urdf(urdf_path: str) -> str:
    """Convert the original DR02-pro.urdf into a fix_joints version for training.
    """
    # Joints to fix (revolute → fixed) and all joints needing dont_collapse
    fixed_joints = {
        "waist_x_joint", "waist_y_joint",
        "neck_y_joint", "neck_z_joint",
        "left_wrist_x_joint", "left_wrist_y_joint", "left_wrist_z_joint",
        "right_wrist_x_joint", "right_wrist_y_joint", "right_wrist_z_joint",
    }
    dont_collapse_joints = fixed_joints | {
        "d435_head_joint", "d435_front_joint", "d435_rear_joint",
        "lidar_joint",
        "imu_head_joint", "imu_base_joint",
        "left_hand_joint", "right_hand_joint",
    }

    with open(urdf_path, "r") as f:
        content = f.read()

    # 1. Change revolute → fixed for the 10 training-restricted joints
    for joint_name in fixed_joints:
        content = re.sub(
            rf'(<joint name="{re.escape(joint_name)}"\s+type=")revolute(">)',
            r'\1fixed">',
            content,
        )

    # 2. Add dont_collapse="true" to all joints that need it
    for joint_name in dont_collapse_joints:
        pattern = rf'(<joint name="{re.escape(joint_name)}"\s+type="[^"]+")(>)'
        content = re.sub(pattern, r'\1 dont_collapse="true"\2', content)

    # Write to a stable path in the same directory so relative mesh paths work.
    src_dir = os.path.dirname(urdf_path)
    out_path = os.path.join(src_dir, "DR02-pro_fix_joints.urdf")
    if not os.path.exists(out_path):
        with open(out_path, "w") as f:
            f.write(content)
    return out_path

DR02_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=_convert_to_fix_joints_urdf(
            f"{ISAACLAB_ASSETS_DATA_DIR}/DR02/urdf/pro/DR02-pro.urdf"
        ),
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=100.0,
            max_angular_velocity=100 * 57.1,
            max_depenetration_velocity=1.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
            fix_root_link=False,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.92),
        joint_pos={
            # legs
            "left_hip_y_joint": -0.1,
            "left_hip_x_joint": 0.0,
            "left_hip_z_joint": 0.0,
            "left_knee_joint": 0.2,
            "left_ankle_y_joint": -0.1,
            "left_ankle_x_joint": 0.0,
            "right_hip_y_joint": -0.1,
            "right_hip_x_joint": 0.0,
            "right_hip_z_joint": 0.0,
            "right_knee_joint": 0.2,
            "right_ankle_y_joint": -0.1,
            "right_ankle_x_joint": 0.0,
            # waist
            "waist_z_joint": 0.0,
            # arms
            "left_shoulder_y_joint": 0.0,
            "left_shoulder_x_joint": 0.15,
            "left_shoulder_z_joint": 0.0,
            "left_elbow_joint": 1.35,
            "right_shoulder_y_joint": 0.0,
            "right_shoulder_x_joint": -0.15,
            "right_shoulder_z_joint": 0.0,
            "right_elbow_joint": 1.35,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.90,
    actuators={
        # All joints in a single cfg; per-joint parameters specified via dict (regex → value)
        "joints": RandomPDActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit={
                ".*_hip_y_joint|.*_hip_x_joint|.*_knee_joint": 330.0,
                ".*_hip_z_joint|.*_ankle_y_joint|waist_z_joint|.*_shoulder_y_joint|.*_shoulder_x_joint|.*_shoulder_z_joint|.*_elbow_joint": 105.0,
                ".*_ankle_x_joint": 35.0,
            },
            effort_limit_sim={
                ".*_hip_y_joint|.*_hip_x_joint|.*_knee_joint": 330.0,
                ".*_hip_z_joint|.*_ankle_y_joint|waist_z_joint|.*_shoulder_y_joint|.*_shoulder_x_joint|.*_shoulder_z_joint|.*_elbow_joint": 105.0,
                ".*_ankle_x_joint": 35.0,
            },
            velocity_limit={
                ".*_hip_y_joint|.*_hip_x_joint|.*_knee_joint": 18.0,
                ".*_hip_z_joint|.*_ankle_y_joint|waist_z_joint|.*_shoulder_y_joint|.*_shoulder_x_joint|.*_shoulder_z_joint|.*_elbow_joint": 17.38,
                ".*_ankle_x_joint": 20.76,
            },
            velocity_limit_sim=1e7,
            stiffness={
                ".*_hip_y_joint|.*_hip_x_joint|.*_knee_joint": 250.0,
                ".*_hip_z_joint": 180.0,
                ".*_ankle_y_joint|.*_shoulder_y_joint|.*_shoulder_x_joint|.*_shoulder_z_joint|.*_elbow_joint": 100.0,
                "waist_z_joint": 150.0,
                ".*_ankle_x_joint": 40.0,
            },
            damping={
                ".*_hip_y_joint|.*_hip_x_joint|.*_knee_joint": 6.0,
                ".*_hip_z_joint": 4.0,
                ".*_ankle_y_joint|.*_shoulder_y_joint|.*_shoulder_x_joint|.*_shoulder_z_joint|.*_elbow_joint": 2.5,
                "waist_z_joint": 3.0,
                ".*_ankle_x_joint": 1.0,
            },
            min_delay=0,
            max_delay=5,
            motor_strength=(0.6, 1.4),
            PD_random_range=(0.9, 1.1),
            pos_bias_range=(-0.04, 0.04),
            t_n_vel_range=(0.1, 0.99),
        ),
    },
)

DR02_DOF_ORDER = [
    "left_hip_y_joint", "left_hip_x_joint", "left_hip_z_joint",
    "left_knee_joint", "left_ankle_y_joint", "left_ankle_x_joint",
    "right_hip_y_joint", "right_hip_x_joint", "right_hip_z_joint",
    "right_knee_joint", "right_ankle_y_joint", "right_ankle_x_joint",
    "waist_z_joint",
    "left_shoulder_y_joint", "left_shoulder_x_joint", "left_shoulder_z_joint", "left_elbow_joint",
    "right_shoulder_y_joint", "right_shoulder_x_joint", "right_shoulder_z_joint", "right_elbow_joint",
]

# AMP discriminator DOF order (20 joints, no waist)
DR02_AMP_DOF_ORDER = [
    "left_hip_y_joint", "left_hip_x_joint", "left_hip_z_joint", "left_knee_joint", "left_ankle_y_joint", "left_ankle_x_joint",
    "right_hip_y_joint", "right_hip_x_joint", "right_hip_z_joint", "right_knee_joint", "right_ankle_y_joint", "right_ankle_x_joint",
    "left_shoulder_y_joint", "left_shoulder_x_joint", "left_shoulder_z_joint", "left_elbow_joint",
    "right_shoulder_y_joint", "right_shoulder_x_joint", "right_shoulder_z_joint", "right_elbow_joint",
]

