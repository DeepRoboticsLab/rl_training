# AMP Runner Configuration for CR1 humanoid
# Migrated from IsaacLabExtension/exts/deeprobotics/deeprobotics/Env/cfg/cr1_amp_cfg.py
# Adapted to configclass-based configuration compatible with to_dict() serialization

from __future__ import annotations

import glob
import os
import time
from dataclasses import MISSING, field

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl.rl_cfg import RslRlBaseRunnerCfg


# Motion files for CR1-B2-STD (globbed from amp/datasets/)
_AMP_DATASETS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "datasets"
)
_MOTION_FILES = glob.glob(os.path.join(_AMP_DATASETS_DIR, "*"))


@configclass
class AmpRunnerCfg:
    """Sub-configuration for the ``runner`` section consumed by AMPOnPolicyRunner."""

    class_name: str = "rl_training.amp_rsl_rl.runners:AMPOnPolicyRunner"
    """Runner class name (resolve_callable format)."""

    policy_class_name: str = "AsymActorCritic"
    """Policy class name."""

    algorithm_class_name: str = "rl_training.amp_rsl_rl.algorithms:PPO_AMP"
    """Algorithm class name (resolve_callable format)."""

    num_steps_per_env: int = MISSING
    """Steps per environment per update."""

    save_interval: int = MISSING
    """Iterations between saves."""

    save_items: list = MISSING
    """List of files to copy to log directory for reproducibility."""


@configclass
class AmpPolicyCfg:
    """Policy configuration for AMP AsymActorCritic network."""

    class_name: str = "AsymActorCritic"
    """Policy class name."""

    init_noise_std: float = 1.0
    """Initial noise standard deviation. Not used by AsymActorCritic but kept for cfg compatibility."""

    actor_hidden_dims: list[int] = MISSING
    """Actor network hidden layer dimensions."""

    critic_hidden_dims: list[int] = MISSING
    """Critic network hidden layer dimensions."""

    encoder_hidden_dims: list[int] = MISSING
    """Encoder network hidden layer dimensions (for CE-Net)."""

    activation: str = MISSING
    """Activation function name."""

    amp_discriminator_hidden_dims: list[int] = MISSING
    """AMP discriminator hidden layer dimensions."""

    latent_dims: int = MISSING
    """Latent dimension for estimation terms."""

    est_terms: dict = MISSING
    """Estimation terms configuration dict."""


@configclass
class AmpPpoAlgorithmCfg:
    """Algorithm configuration for PPO_AMP."""

    class_name: str = "rl_training.amp_rsl_rl.algorithms:PPO_AMP"
    """Algorithm class name (resolve_callable format)."""

    # PPO standard parameters
    value_loss_coef: float = MISSING
    use_clipped_value_loss: bool = MISSING
    clip_param: float = MISSING
    entropy_coef: float = MISSING
    num_learning_epochs: int = MISSING
    num_mini_batches: int = MISSING
    learning_rate: float = MISSING
    schedule: str = MISSING
    gamma: float = MISSING
    lam: float = MISSING
    desired_kl: float = MISSING
    max_grad_norm: float = MISSING
    normalize_advantage_per_mini_batch: bool = False

    # AMP-specific parameters
    discriminate_max_grad_norm: float = MISSING
    discriminate_learning_rate: float = MISSING
    discriminate_loss_coeff: float = MISSING
    grad_penalty_loss_coeff: float = MISSING
    obs_mse_coeff: float = MISSING
    vae_kl_coeff: float = MISSING
    amp_num_frames: int = MISSING
    amp_history_stride: int = MISSING
    amp_num_preload_transitions: int = MISSING
    amp_replay_buffer_size: int = MISSING
    amp_motion_files: list = MISSING
    symmetry_cfg: dict = None
    """Symmetry configuration dict with keys: use_symmetry_data_augmentation, use_symmetry_mirror_loss, mirror_loss_coeff."""


@configclass
class AmpOnPolicyRunnerCfg(RslRlBaseRunnerCfg):
    """Configuration for the AMP On-Policy Runner.

    Inherits common runner fields (seed, device, num_steps_per_env, etc.)
    from :class:`RslRlBaseRunnerCfg` and adds a top-level ``class_name``
    plus AMP-specific ``runner``, ``policy``, and ``algorithm`` sub-configurations.

    The ``class_name`` field enables ``resolve_callable`` in the training
    script to dynamically resolve ``AMPOnPolicyRunner`` instead of the
    default ``OnPolicyRunner``.  This also signals to the training script
    to skip ``handle_deprecated_rsl_rl_cfg`` (which would otherwise clear
    the AMP-specific ``policy`` sub-configuration).

    When ``to_dict()`` is called, the resulting dict has the structure::

        {
            "seed": ..., "device": ..., "num_steps_per_env": ...,
            "class_name": "rl_training.amp_rsl_rl.runners:AMPOnPolicyRunner",
            "runner": {"class_name": "rl_training.amp_rsl_rl.runners:AMPOnPolicyRunner", ...},
            "policy": {"class_name": "AsymActorCritic", ...},
            "algorithm": {"class_name": "rl_training.amp_rsl_rl.algorithms:PPO_AMP", ...},
            ...
        }

    This matches the dict format expected by ``AMPOnPolicyRunner.__init__``
    and ``PPO_AMP.construct_algorithm``.
    """

    class_name: str = "rl_training.amp_rsl_rl.runners:AMPOnPolicyRunner"
    """Runner class name (resolve_callable format).  This field enables the
    training script to dynamically resolve ``AMPOnPolicyRunner`` and signals
    that the deprecated cfg handler should be skipped."""

    runner: AmpRunnerCfg = MISSING
    """Runner sub-configuration."""

    policy: AmpPolicyCfg = MISSING
    """Policy sub-configuration."""

    algorithm: AmpPpoAlgorithmCfg = MISSING
    """Algorithm sub-configuration."""


@configclass
class CR1AmpRunnerCfg(AmpOnPolicyRunnerCfg):
    """CR1-B2-STD AMP training configuration.

    Migrated from ``cr1_amp_cfg.py``. All values match the original
    configuration used for CR1 humanoid AMP training.
    """

    # ---- Top-level runner fields (from RslRlBaseRunnerCfg) ----
    seed: int = int(time.time())
    device: str = "cuda:0"
    num_steps_per_env: int = 24
    max_iterations: int = 50000
    empirical_normalization: bool = False
    obs_groups: dict = field(
        default_factory=lambda: {
            "actor": ["policy"],
            "critic": ["critic"],
        }
    )
    save_interval: int = 500
    experiment_name: str = "cr1_amp"
    run_name: str = ""
    logger: str = "tensorboard"
    resume: bool = False
    load_run: str = ".*"
    load_checkpoint: str = "model_.*.pt"
    clip_actions: float = 10.0

    # ---- Runner sub-config ----
    runner: AmpRunnerCfg = AmpRunnerCfg(
        class_name="rl_training.amp_rsl_rl.runners:AMPOnPolicyRunner",
        policy_class_name="AsymActorCritic",
        algorithm_class_name="rl_training.amp_rsl_rl.algorithms:PPO_AMP",
        num_steps_per_env=24,
        save_interval=500,
        save_items=[
            os.path.join(os.path.dirname(__file__), "rsl_rl_amp_cfg.py"),
            os.path.join(os.path.dirname(__file__), "..", "flat_env_cfg.py"),
        ],
    )

    # ---- Policy config ----
    policy: AmpPolicyCfg = AmpPolicyCfg(
        class_name="AsymActorCritic",
        init_noise_std=1.0,
        actor_hidden_dims=[1024, 256, 128],
        critic_hidden_dims=[1024, 512, 256, 128],
        encoder_hidden_dims=[512, 256, 128],
        activation="elu",
        amp_discriminator_hidden_dims=[512, 256, 128],
        latent_dims=19,
        est_terms={
            "vel_est": {"type": "explicit", "dim": 3, "shape": [3], "loss_coeff": 20.0},
            "implicit": {"type": "implicit", "dim": 16},
        },
    )

    # ---- Algorithm config ----
    algorithm: AmpPpoAlgorithmCfg = AmpPpoAlgorithmCfg(
        class_name="rl_training.amp_rsl_rl.algorithms:PPO_AMP",
        # PPO params
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
        # AMP-specific params
        discriminate_max_grad_norm=5.0,
        discriminate_learning_rate=1e-4,
        discriminate_loss_coeff=1,
        grad_penalty_loss_coeff=10,
        obs_mse_coeff=2,
        vae_kl_coeff=0.2,
        amp_num_frames=5,
        amp_history_stride=2,
        amp_num_preload_transitions=2000000,
        amp_replay_buffer_size=200000,
        amp_motion_files=_MOTION_FILES,
        symmetry_cfg={
            "use_symmetry_data_augmentation": True,
            "use_symmetry_mirror_loss": False,
            "mirror_loss_coeff": 0.1,
            # Path to the symmetry function (module:function format).
            # The AMP runner must be adapted to import and call this function
            # instead of accessing env.symmetrize_* methods directly.
            "data_augmentation_func": (
                "rl_training.tasks.manager_based.locomotion.amp.symmetry.cr1:compute_symmetric_states"
            ),
        },
    )
