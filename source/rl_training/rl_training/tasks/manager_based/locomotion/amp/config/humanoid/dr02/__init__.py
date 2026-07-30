"""DR02 AMP environment registration."""

import gymnasium as gym

from . import agents  # noqa: F401


gym.register(
    id="Amp-Flat-Deeprobotics-DR02-v0",
    entry_point="rl_training.envs:AmpLocomotionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:DR02AmpFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_cfg:DR02AmpRunnerCfg",
    },
)
