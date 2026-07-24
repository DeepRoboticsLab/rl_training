"""CR1 AMP environment registration."""

import gymnasium as gym

from . import agents  # noqa: F401


# Default: new rl_training rsl_rl (PPO_AMP with OnPolicyRunner)
gym.register(
    id="Amp-Flat-Deeprobotics-CR1-v0",
    entry_point="rl_training.envs:AmpLocomotionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:CR1AmpFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_cfg:CR1AmpRunnerCfg",
    },
)
