# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from .rough_env_cfg import DeeproboticsLite3RoughEnvCfg
from rl_training.assets.terrains.terrain_generator_cfg import PERLIN_TERRAINS_CFG


@configclass
class DeeproboticsLite3PerlinEnvCfg(DeeproboticsLite3RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards
        # self.rewards.base_height_l2.params["sensor_cfg"] = None
        # change terrain to perlin
        self.scene.terrain.terrain_generator = PERLIN_TERRAINS_CFG
        # no height scan
        # self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # self.observations.critic.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "DeeproboticsLite3PerlinEnvCfg":
            self.disable_zero_weight_rewards()
