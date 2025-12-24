# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Configuration for hierarchical navigation with Deeprobotics M20 on rough terrain."""

from copy import deepcopy

from isaaclab.utils import configclass

from rl_training.tasks.manager_based.locomotion.hierarchical_nav.hierarchical_nav_env_cfg import (
    HierarchicalNavEnvCfg,
)
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.rough_env_cfg import (
    DeeproboticsM20RoughEnvCfg,
)


# Get scene from low-level config
_low_level_cfg_temp = DeeproboticsM20RoughEnvCfg()
_default_scene = _low_level_cfg_temp.scene


@configclass
class DeeproboticsM20RoughNavEnvCfg(HierarchicalNavEnvCfg):
    """Configuration for hierarchical navigation with Deeprobotics M20 on rough terrain."""

    # Scene settings - inherit from low-level environment
    scene = _default_scene
    
    # Low-level policy checkpoint path (should be set before environment creation)
    low_level_checkpoint: str = "logs/rsl_rl/deeprobotics_m20_rough/2025-12-15_16-08-31/model_19999.pt"

