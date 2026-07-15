"""CR1 AMP environment registration."""

import gym

from . import agents  # noqa: F401


gym.register(
    id="Flat-CR1-Amp-v0",
    entry_point="rl_training.envs:AmpLocomotionEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:CR1AmpFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_cfg:CR1AmpRunnerCfg",
    },
)
