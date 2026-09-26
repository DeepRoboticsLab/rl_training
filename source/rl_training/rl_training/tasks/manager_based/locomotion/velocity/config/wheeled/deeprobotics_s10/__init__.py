# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

import gymnasium as gym

from . import agents


gym.register(
    id="Rough-Deeprobotics-S10-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:DeeproboticsS10RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10RoughPPORunnerCfg",
    },
)
