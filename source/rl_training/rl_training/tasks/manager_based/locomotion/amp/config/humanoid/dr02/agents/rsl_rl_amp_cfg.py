from __future__ import annotations

import glob
import os
import time
from dataclasses import MISSING, field

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl.rl_cfg import (
    RslRlOnPolicyRunnerCfg,
    RslRlMLPModelCfg,
    RslRlPpoAlgorithmCfg,
)

from rl_training.tasks.manager_based.locomotion.amp.amp_env_cfg import (
    AMP_NUM_FRAMES,
    AMP_HISTORY_STRIDE,
)


_AMP_DATASETS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "datasets", "amp_dataset_ik"
)
_MOTION_FILES = glob.glob(os.path.join(_AMP_DATASETS_DIR, "*"))


@configclass
class AmpDiscriminatorCfg:
    """AMP discriminator network and training configuration."""

    hidden_dims: list[int] = [1024, 512]
    """Hidden layer dimensions of the discriminator MLP."""

    max_grad_norm: float = 5.0
    """Maximum gradient norm for discriminator optimizer."""

    learning_rate: float = 1e-4
    """Learning rate for the discriminator optimizer."""

    loss_coeff: float = 1.0
    """Coefficient for the discriminator (policy vs expert) loss."""

    grad_penalty_loss_coeff: float = 10.0
    """Coefficient for the gradient penalty loss."""


@configclass
class CENetCfg:
    """CE-Net (Context Encoding Network) configuration."""

    latent_dims: int = 19
    """Dimension of the latent encoding vector."""

    est_terms: dict = MISSING
    """Estimation terms dict. Keys are term names, values are dicts with
    'type' ('explicit' or 'implicit'), 'dim', and for explicit terms 'loss_coeff'."""

    encoder_hidden_dims: list[int] = [512, 256, 64]
    """Hidden dims of the CE-Net encoder."""

    decoder_hidden_dims: list[int] = [64, 128]
    """Hidden dims of the CE-Net decoder (for implicit / VAE path)."""

    activation: str = "elu"
    """Activation function for CE-Net layers."""

    obs_mse_coeff: float = 4.0
    """Coefficient for the observation prediction MSE loss (VAE decoder)."""

    vae_kl_coeff: float = 1.0
    """Coefficient for the VAE KL divergence loss."""


@configclass
class AmpDatasetCfg:
    """AMP motion dataset and replay buffer configuration."""

    num_frames: int = 2
    """Number of frames per AMP observation sequence."""

    history_stride: int = 2
    """Stride (in env steps) between AMP history frames."""

    num_preload_transitions: int = 100000
    """Number of transitions to preload from motion files."""

    replay_buffer_size: int = 1000000
    """Maximum size of the AMP replay buffer."""

    motion_files: list = MISSING
    """List of motion file paths for the expert dataset."""


@configclass
class SymmetryCfg:
    """Left-right symmetry configuration for data augmentation and mirror loss."""

    use_symmetry_data_augmentation: bool = False
    """If True, double each mini-batch with its left-right symmetrized version."""

    use_symmetry_mirror_loss: bool = False
    """If True, add a mirror loss that enforces actor(symmetrize(obs)) ≈ symmetrize(actor(obs))."""

    mirror_loss_coeff: float = 0.1
    """Coefficient for the symmetry mirror loss."""


@configclass
class AmpPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Algorithm configuration for PPO_AMP.

    Extends RslRlPpoAlgorithmCfg with AMP-specific sub-configs.
    """

    class_name: str = "rl_training.rsl_rl.algorithms:PPO_AMP"
    """Algorithm class name (resolve_callable format)."""

    amp_discriminator_cfg: AmpDiscriminatorCfg = MISSING
    """AMP discriminator network and training sub-configuration."""

    ce_net_cfg: CENetCfg = MISSING
    """CE-Net network and loss sub-configuration."""

    amp_dataset_cfg: AmpDatasetCfg = MISSING
    """AMP motion dataset and replay buffer sub-configuration."""

    symmetry_cfg: SymmetryCfg = field(default_factory=SymmetryCfg)
    """Left-right symmetry configuration."""


@configclass
class DR02AmpRunnerCfg(RslRlOnPolicyRunnerCfg):
    """DR02 AMP training configuration.

    """

    seed: int = int(time.time())
    device: str = "cuda:0"
    num_steps_per_env: int = 24
    max_iterations: int = 50000
    empirical_normalization: bool = False
    obs_groups: dict = field(
        default_factory=lambda: {
            "actor": ["policy", "latent"],
            "critic": ["critic"],
        }
    )
    save_interval: int = 500
    experiment_name: str = "dr02_amp"
    run_name: str = ""
    resume: bool = False
    load_run: str = ".*"
    load_checkpoint: str = "model_.*.pt"
    clip_actions: float = 10.0

    class_name: str = "rl_training.rsl_rl.runners:AMPOnPolicyRunner"

    # Actor
    actor: RslRlMLPModelCfg = RslRlMLPModelCfg(
        class_name="MLPModel",
        hidden_dims=[1024, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=1.0,
            std_type="scalar",
        ),
    )

    # Critic
    critic: RslRlMLPModelCfg = RslRlMLPModelCfg(
        class_name="MLPModel",
        hidden_dims=[1024, 512, 256, 128],
        activation="elu",
        obs_normalization=False,
    )

    # Algorithm
    algorithm: AmpPpoAlgorithmCfg = AmpPpoAlgorithmCfg(
        class_name="rl_training.rsl_rl.algorithms:PPO_AMP",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
        # AMP discriminator
        amp_discriminator_cfg=AmpDiscriminatorCfg(
            hidden_dims=[512, 256, 128],
            max_grad_norm=5.0,
            learning_rate=1e-4,
            loss_coeff=1,
            grad_penalty_loss_coeff=10,
        ),
        # CE-Net
        ce_net_cfg=CENetCfg(
            latent_dims=19,
            est_terms={
                "vel": {"type": "explicit", "dim": 3, "loss_coeff": 20.0},
                "implicit": {"type": "implicit", "dim": 16},
            },
            encoder_hidden_dims=[512, 256, 128],
            decoder_hidden_dims=[64, 128],
            activation="elu",
            obs_mse_coeff=2,
            vae_kl_coeff=0.2,
        ),
        # AMP dataset
        amp_dataset_cfg=AmpDatasetCfg(
            num_frames=AMP_NUM_FRAMES,
            history_stride=AMP_HISTORY_STRIDE,
            num_preload_transitions=2000000,
            replay_buffer_size=200000,
            motion_files=_MOTION_FILES,
        ),
        # Symmetry
        symmetry_cfg=SymmetryCfg(
            use_symmetry_data_augmentation=True,
            use_symmetry_mirror_loss=False,
            mirror_loss_coeff=0.1,
        ),
    )
