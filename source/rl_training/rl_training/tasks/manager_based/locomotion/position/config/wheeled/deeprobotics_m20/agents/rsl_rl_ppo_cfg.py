# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class DeeproboticsM20RoughPPORunnerCfg_DF(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "deeprobotics_m20_rough_DF"
    empirical_normalization = False
    clip_actions = 100
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="log",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class DeeproboticsM20FlatPPORunnerCfg_DF(DeeproboticsM20RoughPPORunnerCfg_DF):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 5000
        self.experiment_name = "deeprobotics_m20_flat_DF"


# ---------------------------------------------------------------------------
# CENet configuration
# ---------------------------------------------------------------------------


@configclass
class CENetCfg:
    """Hyper-parameters for the CENet history encoder / decoder."""

    obs_dim: int = 0
    """Policy observation dimension.  Auto-detected from env at runtime; config value is ignored."""

    history_len: int = 6
    """Number of consecutive frames fed to the history encoder."""

    latent_dim: int = 32
    """VAE latent dimension (z).  Increased from 16 for better representation capacity."""

    vel_dim: int = 3
    """Predicted velocity dimension (v_pred).  Matches base_lin_vel in critic obs."""

    encoder_hidden_dims: list[int] = [256, 128]
    """Hidden layer sizes for the HistoryEncoder MLP."""

    decoder_hidden_dims: list[int] = [128, 256]
    """Hidden layer sizes for the FutureDecoder MLP."""

    aux_lr: float = 1e-3
    """Learning rate for the auxiliary optimizer (encoder + decoder)."""

    reward_window: int = 50
    """Size of the rolling reward buffer used to compute CV for AdaBoot."""

    kl_coef: float = 0.5
    """Weight for the KL-divergence term.  Reduced from 1.0 to avoid over-regularisation."""

    vel_coef: float = 5.0
    """Weight for the velocity-prediction MSE.  Increased from 1.0 to prioritise velocity accuracy."""

    obs_coef: float = 0.01
    """Weight for the future-obs reconstruction MSE.  Greatly reduced because raw obs MSE ~150
    dominates the total loss; lowering this lets vel and KL terms contribute meaningfully."""

    min_p_boot: float = 0.3
    """Floor for per-env AdaBoot probability.  Ensures the encoder receives sufficient
    training signal even when rewards are noisy (cv is high)."""

@configclass
class DeeproboticsM20FlatCENetRunnerCfg_DF(DeeproboticsM20FlatPPORunnerCfg_DF):
    """Runner config for M20 flat env with CENet history encoder + AdaBoot."""


    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "deeprobotics_m20_flat_DF_cenet"
        # obs_groups is set dynamically by CENetOnPolicyRunner; set to empty
        # here so to_dict() produces a valid (empty) dict rather than MISSING.
        self.obs_groups = {}
