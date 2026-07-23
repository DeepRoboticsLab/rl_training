# DeepRobotics AMP Runner Configuration for CR1 humanoid
#
# This config uses the ORIGINAL deeprobotics rsl_rl (AMPOnPolicyRunner,
# AsymActorCritic, PPO_AMP) instead of the new rl_training rsl_rl.
# Purpose: isolate environment bugs from rsl_rl bugs by training the
# manager-based env with the proven deeprobotics training stack.
#
# Usage:
#   python train.py --task Amp-Flat-Deeprobotics-CR1-v0 \
#       --agent deeprobotics_amp_cfg_entry_point
#
# The to_train_cfg_dict() method produces the dict format expected by
# AMPOnPolicyRunner.__init__:
#   {"runner": {...}, "algorithm": {...}, "policy": {...}}

from __future__ import annotations

import glob
import os
import time
from dataclasses import MISSING, field

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl.rl_cfg import RslRlOnPolicyRunnerCfg

from rl_training.tasks.manager_based.locomotion.amp.amp_env_cfg import (
    AMP_NUM_FRAMES,
    AMP_HISTORY_STRIDE,
)

# Motion files for CR1-B2-STD (globbed from amp/datasets/amp_dataset_ik/)
_AMP_DATASETS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "datasets", "amp_dataset_ik"
)
_MOTION_FILES = glob.glob(os.path.join(_AMP_DATASETS_DIR, "*"))


@configclass
class CR1DeepRoboticsAmpRunnerCfg(RslRlOnPolicyRunnerCfg):
    """CR1 AMP config using original deeprobotics rsl_rl.

    All fields are flat (no nested configclass) so that ``to_train_cfg_dict()``
    can restructure them into the runner/algorithm/policy dict format
    expected by ``AMPOnPolicyRunner``.
    """

    # ---- Runner ----
    class_name: str = "deeprobotics.rsl_rl.runners:AMPOnPolicyRunner"
    seed: int = int(time.time())
    device: str = "cuda:0"
    num_steps_per_env: int = 24
    max_iterations: int = 50000
    save_interval: int = 500
    experiment_name: str = "cr1_amp_dr"
    run_name: str = ""
    resume: bool = False
    load_run: str = ""
    load_checkpoint: str = ""
    clip_actions: float = 10.0

    # ---- PPO params ----
    learning_rate: float = 5e-4
    clip_param: float = 0.2
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    gamma: float = 0.99
    lam: float = 0.95
    value_loss_coef: float = 1.0
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0
    use_clipped_value_loss: bool = True
    schedule: str = "adaptive"
    desired_kl: float = 0.01
    normalize_advantage_per_mini_batch: bool = False

    # ---- AMP discriminator params ----
    discriminate_learning_rate: float = 1e-4
    discriminate_loss_coeff: float = 1.0
    grad_penalty_loss_coeff: float = 10.0
    discriminate_max_grad_norm: float = 5.0
    amp_discriminator_hidden_dims: list = field(
        default_factory=lambda: [512, 256, 128]
    )

    # ---- CE-Net params ----
    obs_mse_coeff: float = 2.0
    vae_kl_coeff: float = 0.2
    latent_dims: int = 19
    encoder_hidden_dims: list = field(default_factory=lambda: [512, 256, 128])
    decoder_hidden_dims: list = field(default_factory=lambda: [64, 128])
    init_noise_std: float = 1.0
    activation: str = "elu"
    est_terms: dict = field(
        default_factory=lambda: {
            "vel": {"type": "explicit", "dim": 3, "shape": [3], "loss_coeff": 20.0},
            "implicit": {"type": "implicit", "dim": 16},
        }
    )

    # ---- AMP dataset params ----
    amp_motion_files: list = field(default_factory=lambda: _MOTION_FILES)
    amp_history_stride: int = AMP_HISTORY_STRIDE
    amp_num_frames: int = AMP_NUM_FRAMES
    amp_num_preload_transitions: int = 2000000
    amp_replay_buffer_size: int = 200000

    # ---- Symmetry ----
    symmetry_cfg: dict = field(
        default_factory=lambda: {
            "use_symmetry_data_augmentation": False,
            "use_symmetry_mirror_loss": False,
            "mirror_loss_coeff": 0.1,
            "data_augmentation_func": (
                "rl_training.tasks.manager_based.locomotion.amp.symmetry.cr1:compute_symmetric_states"
            ),
        }
    )

    # ---- Actor / Critic hidden dims (used as MLPModel, ignored by deeprobotics runner) ----
    actor_hidden_dims: list = field(default_factory=lambda: [1024, 256, 128])
    critic_hidden_dims: list = field(default_factory=lambda: [1024, 512, 256, 128])

    def to_train_cfg_dict(self):
        """Produce dict format expected by deeprobotics AMPOnPolicyRunner.

        This is a SEPARATE method from ``to_dict()`` (which is used by hydra
        for config serialization). ``train.py`` calls this when
        ``is_custom_runner`` is True.

        AMPOnPolicyRunner.__init__ expects:
            train_cfg["runner"]   -> {num_steps_per_env, save_interval, policy_class_name}
            train_cfg["algorithm"] -> PPO + AMP params + amp_motion_files
            train_cfg["policy"]   -> AsymActorCritic network params (flat dict)
        """
        return {
            # ---- Runner config ----
            "runner": {
                "num_steps_per_env": self.num_steps_per_env,
                "save_interval": self.save_interval,
                "policy_class_name": "AsymActorCritic",
            },
            # ---- Algorithm config (PPO_AMP params) ----
            "algorithm": {
                # PPO standard
                "learning_rate": self.learning_rate,
                "clip_param": self.clip_param,
                "num_learning_epochs": self.num_learning_epochs,
                "num_mini_batches": self.num_mini_batches,
                "gamma": self.gamma,
                "lam": self.lam,
                "value_loss_coef": self.value_loss_coef,
                "entropy_coef": self.entropy_coef,
                "max_grad_norm": self.max_grad_norm,
                "use_clipped_value_loss": self.use_clipped_value_loss,
                "schedule": self.schedule,
                "desired_kl": self.desired_kl,
                "normalize_advantage_per_mini_batch": self.normalize_advantage_per_mini_batch,
                # AMP discriminator
                "discriminate_learning_rate": self.discriminate_learning_rate,
                "discriminate_loss_coeff": self.discriminate_loss_coeff,
                "grad_penalty_loss_coeff": self.grad_penalty_loss_coeff,
                "discriminate_max_grad_norm": self.discriminate_max_grad_norm,
                # CE-Net
                "obs_mse_coeff": self.obs_mse_coeff,
                "vae_kl_coeff": self.vae_kl_coeff,
                # AMP dataset
                "amp_motion_files": self.amp_motion_files,
                "amp_history_stride": self.amp_history_stride,
                "amp_num_frames": self.amp_num_frames,
                "amp_num_preload_transitions": self.amp_num_preload_transitions,
                "amp_replay_buffer_size": self.amp_replay_buffer_size,
                # Symmetry
                "symmetry_cfg": self.symmetry_cfg,
            },
            # ---- Policy config (AsymActorCritic network params) ----
            "policy": {
                # Network dims
                "actor_hidden_dims": self.actor_hidden_dims,
                "critic_hidden_dims": self.critic_hidden_dims,
                "encoder_hidden_dims": self.encoder_hidden_dims,
                "decoder_hidden_dims": self.decoder_hidden_dims,
                "init_noise_std": self.init_noise_std,
                "activation": self.activation,
                # CE-Net
                "latent_dims": self.latent_dims,
                "est_terms": self.est_terms,
                # AMP discriminator
                "amp_discriminator_hidden_dims": self.amp_discriminator_hidden_dims,
            },
        }

    def __post_init__(self):
        super().__post_init__()


# Entry point for gym registration
deeprobotics_amp_cfg_entry_point = CR1DeepRoboticsAmpRunnerCfg
