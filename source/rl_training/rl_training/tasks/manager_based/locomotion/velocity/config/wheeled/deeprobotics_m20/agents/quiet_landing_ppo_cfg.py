"""Independent quiet V1 PPO configuration; baseline architecture and optimizer."""
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class QuietLandingAlgorithmCfg(RslRlPpoAlgorithmCfg):
    # Reuse the proven preservation algorithm, without inheriting teacher config.
    class_name = "rl_training.stair_guarded_ppo:StairGuardedPPO"
    anchor_coefficient: float = 0.1
    critic_warmup_iterations: int = 50
    reference_std_multiplier: float = 1.05
    anchor_uneven_weight: float = 0.1
    actor_update_mask_key: str | None = None


@configclass
class DeeproboticsM20QuietLandingPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = "rl_training.quiet_landing_runner:QuietLandingRunner"
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 100
    experiment_name = "deeprobotics_m20_quiet_landing_v1"
    run_name = "quiet_landing_v1"
    empirical_normalization = False
    clip_actions = 100
    logger = "tensorboard"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0, noise_std_type="log", actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128], activation="elu",
    )
    algorithm = QuietLandingAlgorithmCfg(
        value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.1,
        entropy_coef=0.0, num_learning_epochs=5, num_mini_batches=4,
        learning_rate=1.0e-5, schedule="fixed", gamma=0.99, lam=0.95,
        desired_kl=0.01, max_grad_norm=1.0,
    )


@configclass
class DeeproboticsM20QuietLandingV15PPORunnerCfg(DeeproboticsM20QuietLandingPPORunnerCfg):
    experiment_name = "deeprobotics_m20_quiet_landing_v1_5"
    run_name = "quiet_landing_v1_5"
