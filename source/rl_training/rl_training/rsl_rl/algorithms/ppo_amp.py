# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO-AMP: Proximal Policy Optimization with Adversarial Motion Priors.

Inherits from the rsl_rl PPO base class. Actor and critic are now separate
MLPModel instances (created via construct_algorithm), while the CE-Net encoder
and AMP discriminator remain as standalone nn.Module subclasses.
"""

from __future__ import annotations

import inspect

import torch
import torch.nn as nn
import torch.optim as optim
from itertools import chain
from tensordict import TensorDict


from rsl_rl.algorithms.ppo import PPO
from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups
from rsl_rl.extensions import resolve_symmetry_config

from ..modules.ce_net import CENet
from ..modules.amp_discriminator import AMP_Discriminator
from ..storage.amp_storage import ReplayBuffer
from ..utils.utils import Normalizer
from ..datasets.motion_loader import Dataset_Loader


class AMPInferencePolicy(nn.Module):
    """Inference wrapper combining CE-Net encoder + Actor.

    Accepts a TensorDict (from env wrapper) containing "policy" and
    "obs_history" keys.  Encodes obs_history through CE-Net (uses mu
    for implicit, not sampled z), concatenates obs + latent, and
    forwards through the actor MLP to return the deterministic action mean.
    """

    def __init__(self, ce_net: CENet, actor: MLPModel):
        super().__init__()
        self.ce_net = ce_net
        self.actor = actor

    def forward(self, obs: TensorDict) -> torch.Tensor:
        """Forward pass for play mode: accepts TensorDict from env wrapper."""
        return self._forward_impl(obs["policy"], obs["obs_history"])

    def _forward_impl(self, obs: torch.Tensor, obs_history: torch.Tensor) -> torch.Tensor:
        """Core logic shared by forward() and export wrappers."""
        # CE-Net encoding (use mu for implicit, matching old AsymActorCritic.forward)
        if self.ce_net.has_implicit:
            encodings, mu, _ = self.ce_net.encode(obs_history)
            encodings_list = [
                v for k, v in encodings.items()
                if k != "implicit" and k in self.ce_net.est_terms.keys()
            ]
            encodings_list.append(mu)
            latent = torch.cat(encodings_list, dim=-1)
        else:
            encodings = self.ce_net.encode(obs_history)
            latent = torch.cat(
                [encodings[k] for k in self.ce_net.est_terms.keys()], dim=-1
            )

        # Concatenate obs + latent
        actor_input = torch.cat([obs, latent], dim=-1)

        # Normalize and forward through actor MLP
        normalized = self.actor.obs_normalizer(actor_input)
        mlp_output = self.actor.mlp(normalized)

        # Return deterministic output (mean)
        if self.actor.distribution is not None:
            return self.actor.distribution.deterministic_output(mlp_output)
        return mlp_output


class AMPExportWrapper(nn.Module):
    """Single-input export wrapper: obs_history -> action.

    The deployed model only receives ``obs_history`` as input.  This wrapper
    extracts the latest ``num_obs`` slice as the current observation and calls
    the underlying ``AMPInferencePolicy._forward_impl(obs, obs_history)``.

    All dimensions are derived from the network itself (CENet.num_actor_obs /
    CENet.num_history_frames), so no env reference is needed.
    """

    def __init__(self, policy: AMPInferencePolicy):
        super().__init__()
        self.policy = policy
        self.num_obs = policy.ce_net.num_actor_obs
        self.input_size = policy.ce_net.num_actor_obs * policy.ce_net.num_history_frames

    def forward(self, obs_history: torch.Tensor) -> torch.Tensor:
        obs = obs_history[:, -self.num_obs:]
        return self.policy._forward_impl(obs, obs_history)

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        """Return representative dummy inputs for ONNX/JIT tracing."""
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["observation"]

    @property
    def output_names(self) -> list[str]:
        return ["action"]


class PPO_AMP(PPO):
    """PPO with Adversarial Motion Priors and CE-Net asymmetric actor-critic."""

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        ce_net: CENet,
        amp_discriminator: AMP_Discriminator,
        amp_dataset: Dataset_Loader,
        amp_normalizer: Normalizer | None = None,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.996,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.0,
        learning_rate: float = 1e-3,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "fixed",
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        optimizer: str = "adam",
        # AMP / CE-Net sub-configs (passed as dicts from construct_algorithm)
        ce_net_cfg: dict | None = None,
        amp_disc_cfg: dict | None = None,
        # Estimation terms (extracted from ce_net_cfg by construct_algorithm)
        est_terms: dict | None = None,
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
        # Environment handle (for symmetry augmentation)
        env: VecEnv | None = None,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        # ------------------------------------------------------------------ #
        # Call base PPO __init__ to set up actor, critic, storage, optimizer
        # ------------------------------------------------------------------ #
        # We pass symmetry_cfg=None to base PPO to avoid its symmetry setup
        # (PPO_AMP handles symmetry separately). The base PPO stores
        # self.actor, self.critic, self.optimizer, self.storage, etc.
        super().__init__(
            actor=actor,
            critic=critic,
            storage=storage,
            num_learning_epochs=num_learning_epochs,
            num_mini_batches=num_mini_batches,
            clip_param=clip_param,
            gamma=gamma,
            lam=lam,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            optimizer=optimizer,
            use_clipped_value_loss=use_clipped_value_loss,
            schedule=schedule,
            desired_kl=desired_kl,
            normalize_advantage_per_mini_batch=normalize_advantage_per_mini_batch,
            device=device,
            rnd_cfg=rnd_cfg,
            symmetry_cfg=None,  # PPO_AMP handles symmetry itself
            multi_gpu_cfg=multi_gpu_cfg,
        )

        # Override base PPO's symmetry with AMP-specific handling
        self.env = env

        if symmetry_cfg is not None:
            self.symmetry = symmetry_cfg  # store the raw dict
            self.use_symmetry_data_augmentation = symmetry_cfg.get("use_data_augmentation", False)
            self.use_symmetry_mirror_loss = symmetry_cfg.get("use_mirror_loss", False)
            self.mirror_loss_coeff = symmetry_cfg.get("mirror_loss_coeff", 0.0)
            self.use_symmetry = (
                self.use_symmetry_data_augmentation or self.use_symmetry_mirror_loss
            )
            # Resolve the data augmentation function
            func_ref = symmetry_cfg.get("data_augmentation_func", None)
            self.compute_symmetric_states = resolve_callable(func_ref) if func_ref is not None else None
        else:
            self.symmetry = None
            self.use_symmetry_data_augmentation = False
            self.use_symmetry_mirror_loss = False
            self.mirror_loss_coeff = 0.0
            self.use_symmetry = False
            self.compute_symmetric_states = None

        # ------------------------------------------------------------------ #
        # CE-Net module
        # ------------------------------------------------------------------ #
        self.ce_net = ce_net.to(self.device)

        # Extract CE-Net training params from sub-config dict
        _ce_cfg = ce_net_cfg or {}
        self.est_terms = est_terms if est_terms is not None else _ce_cfg.get("est_terms", {})
        self.est_explicit_key = [k for k, d in self.est_terms.items() if d["type"] == "explicit"]
        self.est_explicit_loss_coeff = [
            d["loss_coeff"] for d in self.est_terms.values() if d["type"] == "explicit"
        ]
        self.obs_mse_coeff = _ce_cfg.get("obs_mse_coeff", 4.0)
        self.vae_kl_coeff = _ce_cfg.get("vae_kl_coeff", 1.0)

        # ------------------------------------------------------------------ #
        # AMP discriminator & dataset
        # ------------------------------------------------------------------ #
        self.amp_discriminator = amp_discriminator.to(self.device)
        self.amp_dataset = amp_dataset
        self.amp_normalizer = amp_normalizer

        # Extract AMP discriminator training params from sub-config dict
        _disc_cfg = amp_disc_cfg or {}
        self.discriminate_loss_coeff = _disc_cfg.get("loss_coeff", 1.0)
        self.grad_penalty_loss_coeff = _disc_cfg.get("grad_penalty_loss_coeff", 10.0)
        self.discriminate_max_grad_norm = _disc_cfg.get("max_grad_norm", 5.0)
        _discriminate_learning_rate = _disc_cfg.get("learning_rate", 1e-4)

        # ------------------------------------------------------------------ #
        # AMP replay buffer (initialised in construct_algorithm)
        # ------------------------------------------------------------------ #
        self.amp_storage: ReplayBuffer | None = None
        self.amp_transition: ReplayBuffer.Transition | None = None

        # ------------------------------------------------------------------ #
        # Additional optimizers (CE-Net and AMP discriminator)
        # ------------------------------------------------------------------ #
        self.ce_optimizer = optim.Adam(self.ce_net.parameters(), lr=learning_rate)
        self.amp_optimizer = optim.Adam(
            self.amp_discriminator.parameters(),
            lr=_discriminate_learning_rate,
            weight_decay=1e-4,
        )

        self.batch_index = 0

    # ------------------------------------------------------------------ #
    # act – sample actions and store transition
    # ------------------------------------------------------------------ #
    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample actions from the policy and store transition data."""
        policy_obs = obs["policy"]
        critic_obs = obs["critic"]
        obs_history = obs["obs_history"]

        # CE-Net encoding: compute latent and inject into obs for actor
        encodings_cat = self.ce_net.get_encodings_cat(obs_history)
        # Create a modified obs dict with latent injected.
        # amp_obs_history is excluded — it's stored separately in the AMP
        # replay buffer, not needed in PPO storage.
        obs_with_latent = obs.clone()
        obs_with_latent["latent"] = encodings_cat.detach()
        del obs_with_latent["amp_obs_history"]

        # Sample action using the MLPModel actor
        self.transition.hidden_states = (self.actor.get_hidden_state(), self.critic.get_hidden_state())
        self.transition.actions = self.actor(obs_with_latent, stochastic_output=True).detach()
        self.transition.values = self.critic(obs_with_latent).detach()
        self.transition.actions_log_prob = self.actor.get_output_log_prob(self.transition.actions).detach()
        self.transition.distribution_params = tuple(p.detach() for p in self.actor.output_distribution_params)
        self.transition.observations = obs_with_latent.detach()
        return self.transition.actions

    # ------------------------------------------------------------------ #
    # process_env_step – record reward/done and AMP obs
    # ------------------------------------------------------------------ #
    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict,
    ) -> None:
        """Record one environment step and update AMP replay buffer."""
        # Update normalizers (base PPO interface)
        self.actor.update_normalization(obs)
        self.critic.update_normalization(obs)

        # observation from the new obs TensorDict
        obs_future = obs["obs_future"]
        amp_obs = obs["amp_obs_history"]  # already 3D: (num_envs, amp_num_frames, amp_obs_dim)

        self.transition.observations["obs_future"] = obs_future.detach()
        self.amp_transition.obs = amp_obs.detach()

        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Bootstrapping on timeouts
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # Record the transition
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.actor.reset(dones)
        self.critic.reset(dones)

        # AMP replay buffer
        self.amp_storage.add_transitions(self.amp_transition)
        self.amp_transition.clear()

    # ------------------------------------------------------------------ #
    # compute_returns – TD-lambda advantage estimation
    # ------------------------------------------------------------------ #
    def compute_returns(self, obs: TensorDict) -> None:
        """Compute return and advantage targets."""
        st = self.storage
        last_values = self.critic(obs).detach()

        advantage = 0
        for step in reversed(range(st.num_transitions_per_env)):
            next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            next_is_not_terminal = 1.0 - st.dones[step].float()
            delta = st.rewards[step] + next_is_not_terminal * self.gamma * next_values - st.values[step]
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            st.returns[step] = advantage + st.values[step]
        st.advantages = st.returns - st.values
        if not self.normalize_advantage_per_mini_batch:
            st.advantages = (st.advantages - st.advantages.mean()) / (st.advantages.std() + 1e-8)

    # ------------------------------------------------------------------ #
    # update – main training loop
    # ------------------------------------------------------------------ #
    def update(self) -> dict[str, float]:
        """Run optimization epochs and return mean losses."""
        mean_loss: dict[str, float] = {}
        for key in self.est_explicit_key:
            mean_loss[f"{key}_est_mse"] = 0.0
        mean_loss.update({
            "obs_pred": 0.0, "discriminate": 0.0, "grad_penalty": 0.0,
            "policy_pred": 0.0, "expert_pred": 0.0,
            "value": 0.0, "surrogate": 0.0, "entropy": 0.0,
        })
        if self.mirror_loss_coeff:
            mean_loss["symmetry"] = 0.0

        # Mini-batch generators
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        amp_policy_gen = self.amp_storage.feed_forward_generator(self.num_mini_batches, self.num_learning_epochs)
        amp_expert_gen = self.amp_dataset.feed_forward_generator(self.num_mini_batches, self.num_learning_epochs)

        self.batch_index = 0
        for batch, amp_policy_batch, amp_expert_batch in zip(generator, amp_policy_gen, amp_expert_gen):
            self._learning_mini_epoch(batch, amp_policy_batch, amp_expert_batch, mean_loss)
            self.batch_index += 1

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_loss = {k: v / num_updates for k, v in mean_loss.items()}
        self.storage.clear()
        return mean_loss

    # ------------------------------------------------------------------ #
    # learning_mini_epoch – per-mini-batch gradient updates
    # ------------------------------------------------------------------ #
    def _learning_mini_epoch(self, batch, amp_policy_batch, amp_expert_batch, mean_loss: dict) -> None:
        original_batch_size = batch.observations.batch_size[0]

        # Extract observations from TensorDict
        policy_obs = batch.observations["policy"]
        critic_obs = batch.observations["critic"]
        obs_history = batch.observations["obs_history"]
        obs_future = batch.observations["obs_future"]
        latent = batch.observations["latent"]
        est_explicit_batch = [
            batch.observations[k] for k in self.est_explicit_key if k in batch.observations
        ]

        # Normalize advantages per mini-batch if configured
        if self.normalize_advantage_per_mini_batch:
            with torch.no_grad():
                batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)

        # -- Symmetry data augmentation -------------------------------- #
        if self.use_symmetry_data_augmentation and self.compute_symmetric_states is not None:
            obs_aug, actions_aug = self.compute_symmetric_states(
                env=self.env, obs=batch.observations, actions=batch.actions
            )
            policy_obs = obs_aug["policy"]
            obs_history = obs_aug.get("obs_history", policy_obs)
            critic_obs = obs_aug["critic"]
            if obs_future is not None:
                obs_future = obs_aug.get("obs_future", obs_future)
            est_explicit_batch = [obs_aug[k] for k in self.est_explicit_key if k in obs_aug]
            actions_batch = actions_aug
            num_aug = int(policy_obs.shape[0] / original_batch_size)
            old_actions_log_prob = batch.old_actions_log_prob.repeat(num_aug, 1)
            target_values = batch.values.repeat(num_aug, 1)
            advantages = batch.advantages.repeat(num_aug, 1)
            returns = batch.returns.repeat(num_aug, 1)
            old_mu = batch.old_distribution_params[0].repeat(num_aug, 1)
            old_sigma = batch.old_distribution_params[1].repeat(num_aug, 1)
        else:
            actions_batch = batch.actions
            old_actions_log_prob = batch.old_actions_log_prob
            target_values = batch.values
            advantages = batch.advantages
            returns = batch.returns
            old_mu = batch.old_distribution_params[0]
            old_sigma = batch.old_distribution_params[1]

        # -- PPO loss ------------------------------------------------- #
        # Forward through actor (MLPModel) with stochastic output
        self.actor(batch.observations, stochastic_output=True)
        actions_log_prob = self.actor.get_output_log_prob(actions_batch)
        value_batch = self.critic(batch.observations)
        mu_batch = self.actor.output_mean[:original_batch_size]
        sigma_batch = self.actor.output_std[:original_batch_size]
        entropy_batch = self.actor.output_entropy[:original_batch_size]

        # Adaptive learning rate (matched to old deeprobotics PPO)
        if self.desired_kl is not None and self.schedule == "adaptive":
            with torch.inference_mode():
                # Manual KL formula for numerical stability (matches old PPO.adjust_learning_rate)
                dist_diff = torch.log(sigma_batch / old_sigma + 1e-5)
                dist_diff += (torch.square(old_sigma) + torch.square(old_mu - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5
                kl = torch.sum(dist_diff, dim=-1).mean()
                if kl > self.desired_kl * 2.0:
                    self.learning_rate = max(1e-6, self.learning_rate / 1.2)
                elif self.desired_kl / 2.0 > kl > 0.0:
                    self.learning_rate = min(1e-3, self.learning_rate * 1.2)
                for pg in self.optimizer.param_groups:
                    pg["lr"] = self.learning_rate

        # Surrogate loss
        ratio = torch.exp(actions_log_prob - torch.squeeze(old_actions_log_prob))
        surrogate = -torch.squeeze(advantages) * ratio
        surrogate_clipped = -torch.squeeze(advantages) * torch.clamp(
            ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
        )
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        # Value loss
        if self.use_clipped_value_loss:
            value_clipped = target_values + (value_batch - target_values).clamp(-self.clip_param, self.clip_param)
            value_losses = (value_batch - returns).pow(2)
            value_losses_clipped = (value_clipped - returns).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = (returns - value_batch).pow(2).mean()

        loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()
        mean_loss["value"] += value_loss.item()
        mean_loss["surrogate"] += surrogate_loss.item()
        mean_loss["entropy"] += self.entropy_coef * entropy_batch.mean().item()

        # -- Symmetry mirror loss ------------------------------------- #
        if self.use_symmetry_mirror_loss and self.compute_symmetric_states is not None:
            if self.use_symmetry_data_augmentation:
                mean_actions = self.actor.deterministic_output(
                    self.actor.get_latent(actor_input_obs)
                )
            else:
                encodings_orig = self.ce_net.get_encodings_cat(
                    batch.observations.get("obs_history", batch.observations["policy"])
                )
                actor_input_orig = batch.observations.clone()
                actor_input_orig["latent"] = encodings_orig
                mean_actions = self.actor.deterministic_output(
                    self.actor.get_latent(actor_input_orig)
                )

            action_mean_orig = mean_actions[:original_batch_size]
            _, actions_mean_symm = self.compute_symmetric_states(
                env=self.env, obs=None, actions=action_mean_orig
            )
            if not self.use_symmetry_data_augmentation:
                obs_aug, _ = self.compute_symmetric_states(
                    env=self.env, obs=batch.observations, actions=None
                )
                enc_aug = self.ce_net.get_encodings_cat(obs_aug.get("obs_history", obs_aug["policy"]))
                actor_input_aug = obs_aug.clone()
                actor_input_aug["latent"] = enc_aug
                mean_actions_aug = self.actor.deterministic_output(
                    self.actor.get_latent(actor_input_aug)
                )
            else:
                mean_actions_aug = mean_actions

            symmetry_loss = nn.functional.l1_loss(
                mean_actions_aug[original_batch_size:],
                actions_mean_symm.detach()[original_batch_size:] if actions_mean_symm.shape[0] > original_batch_size else actions_mean_symm.detach(),
            )
            loss += self.mirror_loss_coeff * symmetry_loss
            mean_loss["symmetry"] += self.mirror_loss_coeff * symmetry_loss.item()

        # -- AMP discriminator loss ----------------------------------- #
        batch_size = amp_policy_batch.shape[0]
        num_frames = amp_policy_batch.shape[1]
        feature_dim = amp_policy_batch.shape[2]
        if self.amp_normalizer is not None:
            if self.batch_index < self.num_mini_batches:
                i = torch.randint(low=0, high=num_frames, size=(1,)).item()
                self.amp_normalizer.update(
                    torch.cat((amp_policy_batch[:, -1], amp_expert_batch[:, i]), dim=0)
                )
            with torch.no_grad():
                policy_state = self.amp_normalizer.normalize(
                    amp_policy_batch.reshape(-1, feature_dim)
                ).reshape(batch_size, -1)
                expert_state = self.amp_normalizer.normalize(
                    amp_expert_batch.reshape(-1, feature_dim)
                ).reshape(batch_size, -1)
        else:
            policy_state = amp_policy_batch.reshape(batch_size, -1)
            expert_state = amp_expert_batch.reshape(batch_size, -1)

        policy_pred = self.amp_discriminator(policy_state)
        expert_pred = self.amp_discriminator(expert_state)
        policy_loss = nn.MSELoss()(policy_pred, -1 * torch.ones_like(policy_pred))
        expert_loss = nn.MSELoss()(expert_pred, torch.ones_like(expert_pred))
        discriminate_loss = 0.5 * (expert_loss + policy_loss)
        grad_penalty_loss = self.amp_discriminator.compute_grad_pen(expert_state)
        amp_loss = (
            self.discriminate_loss_coeff * discriminate_loss
            + self.grad_penalty_loss_coeff * grad_penalty_loss
        )
        mean_loss["discriminate"] += self.discriminate_loss_coeff * discriminate_loss.item()
        mean_loss["grad_penalty"] += self.grad_penalty_loss_coeff * grad_penalty_loss.item()
        mean_loss["policy_pred"] += policy_pred.mean().item()
        mean_loss["expert_pred"] += expert_pred.mean().item()

        # -- CE-Net loss --------------------------------------------- #
        ce_loss = torch.zeros(1, requires_grad=False, dtype=torch.float, device=self.device)
        if self.ce_net.has_implicit and obs_future is not None:
            encodings, decodings, mu, log_var = self.ce_net.ce_net(obs_history)
            explicit_est = [encodings[k] for k in self.est_explicit_key]
            obs_pred = decodings["obs_pred"]
            kld_loss = torch.mean(
                -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0
            )
            obs_mse_loss = nn.functional.mse_loss(obs_pred, obs_future)
            ce_loss = ce_loss + self.vae_kl_coeff * kld_loss + self.obs_mse_coeff * obs_mse_loss
            mean_loss["obs_pred"] += obs_mse_loss.item()
        else:
            encodings = self.ce_net.ce_net(obs_history)
            explicit_est = [encodings[k] for k in self.est_explicit_key]
        ce_loss = ce_loss + self._get_explicit_est_loss(explicit_est, est_explicit_batch, mean_loss)

        # PPO gradient
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # CE-Net gradient
        self.ce_optimizer.zero_grad()
        ce_loss.backward()
        nn.utils.clip_grad_norm_(self.ce_net.parameters(), self.max_grad_norm)
        self.ce_optimizer.step()

        # AMP discriminator gradient
        self.amp_optimizer.zero_grad()
        amp_loss.backward()
        nn.utils.clip_grad_norm_(
            self.amp_discriminator.parameters(), self.discriminate_max_grad_norm
        )
        self.amp_optimizer.step()

    def _get_explicit_est_loss(self, estimations, nominals, mean_loss: dict) -> torch.Tensor:
        est_loss = torch.zeros(1, dtype=torch.float, requires_grad=False, device=self.device)
        for est, nom, key, coeff in zip(
            estimations, nominals, self.est_explicit_key, self.est_explicit_loss_coeff
        ):
            est_err = nn.functional.mse_loss(est, nom)
            mean_loss[f"{key}_est_mse"] += est_err.item()
            est_loss += coeff * est_err
        return est_loss

    # ------------------------------------------------------------------ #
    # Mode switching
    # ------------------------------------------------------------------ #
    def train_mode(self) -> None:
        self.actor.train()
        self.critic.train()
        self.ce_net.train()
        self.amp_discriminator.train()

    def eval_mode(self) -> None:
        self.actor.eval()
        self.critic.eval()
        self.ce_net.eval()
        self.amp_discriminator.eval()

    # ------------------------------------------------------------------ #
    # Save / Load
    # ------------------------------------------------------------------ #
    def save(self) -> dict:
        saved_dict = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "ce_net_state_dict": self.ce_net.state_dict(),
            "ce_optimizer_state_dict": self.ce_optimizer.state_dict(),
            "amp_discriminator_state_dict": self.amp_discriminator.state_dict(),
            "amp_optimizer_state_dict": self.amp_optimizer.state_dict(),
        }
        if self.amp_normalizer is not None:
            norm = self.amp_normalizer
            saved_dict["amp_normalizer_state_dict"] = {
                "mean": norm.mean.clone(),
                "var": norm.var.clone(),
                "mean_history": norm.mean_history.clone(),
                "var_history": norm.var_history.clone(),
                "current_index": norm.current_index,
                "total_updates": norm.total_updates,
            }
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        if load_cfg is None:
            load_cfg = {
                "actor": True, "critic": True, "optimizer": True,
                "ce_net": True, "amp": True, "normalizer": True,
                "iteration": True,
            }
        if load_cfg.get("actor", True):
            self.actor.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
        if load_cfg.get("critic", True):
            self.critic.load_state_dict(loaded_dict["critic_state_dict"], strict=strict)
        if load_cfg.get("optimizer", True):
            self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        if load_cfg.get("ce_net", True):
            self.ce_net.load_state_dict(loaded_dict["ce_net_state_dict"], strict=strict)
            if "ce_optimizer_state_dict" in loaded_dict:
                self.ce_optimizer.load_state_dict(loaded_dict["ce_optimizer_state_dict"])
        if load_cfg.get("amp", True):
            if "amp_discriminator_state_dict" in loaded_dict:
                self.amp_discriminator.load_state_dict(loaded_dict["amp_discriminator_state_dict"])
            if "amp_optimizer_state_dict" in loaded_dict:
                self.amp_optimizer.load_state_dict(loaded_dict["amp_optimizer_state_dict"])
        if load_cfg.get("normalizer", True) and self.amp_normalizer is not None:
            if "amp_normalizer_state_dict" in loaded_dict:
                ns = loaded_dict["amp_normalizer_state_dict"]
                norm = self.amp_normalizer
                norm.mean.copy_(ns["mean"])
                norm.var.copy_(ns["var"])
                norm.mean_history.copy_(ns["mean_history"])
                norm.var_history.copy_(ns["var_history"])
                norm.current_index = ns["current_index"]
                norm.total_updates = ns["total_updates"]
        return load_cfg.get("iteration", False)

    # ------------------------------------------------------------------ #
    # Get policy for inference / training logging
    # ------------------------------------------------------------------ #
    def get_policy(self) -> MLPModel:
        """Return the bare actor model (used by runner for logging)."""
        return self.actor

    def get_inference_policy(self) -> "AMPInferencePolicy":
        """Return a combined CE-Net + Actor wrapper for inference.

        This mirrors the old AsymActorCritic.forward(obs, obs_h) interface:
        - Encodes obs_history through CE-Net (uses mu for implicit, not sampled z)
        - Concatenates obs + latent and forwards through actor MLP
        - Returns deterministic action mean
        """
        self.eval_mode()
        return AMPInferencePolicy(self.ce_net, self.actor)

    # ------------------------------------------------------------------ #
    # construct_algorithm – static factory used by OnPolicyRunner
    # ------------------------------------------------------------------ #
    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> "PPO_AMP":
        """Construct PPO_AMP from configuration.

        This mirrors the base PPO.construct_algorithm pattern but adds:
        - Dummy "latent" observation in obs for actor dimension calculation
        - CE-Net (CENet) standalone module
        - AMP discriminator, dataset, and normalizer
        - AMP replay buffer (amp_storage)
        """
        alg_cfg = cfg["algorithm"]

        # Resolve class callables
        alg_class = resolve_callable(alg_cfg.pop("class_name"))
        actor_class = resolve_callable(cfg["actor"].pop("class_name"))
        critic_class = resolve_callable(cfg["critic"].pop("class_name"))

        # Get environment info
        _env = env.unwrapped if hasattr(env, "unwrapped") else env

        # ------------------------------------------------------------------ #
        # Extract sub-configs before auto-filter
        # ------------------------------------------------------------------ #
        ce_net_cfg = alg_cfg.pop("ce_net_cfg", {})
        amp_disc_cfg = alg_cfg.pop("amp_discriminator_cfg", {})
        amp_dataset_cfg = alg_cfg.pop("amp_dataset_cfg", {})

        # Add dummy "latent" to obs for dimension calculation
        latent_dims = ce_net_cfg.get("latent_dims", 19)
        num_envs = _env.num_envs
        obs_with_latent = obs.clone()
        obs_with_latent["latent"] = torch.zeros(num_envs, latent_dims, device=device)

        # Resolve observation groups (validate that "latent" is in actor set)
        default_sets = ["actor", "critic"]
        cfg["obs_groups"] = resolve_obs_groups(obs_with_latent, cfg["obs_groups"], default_sets)

        # Resolve symmetry config
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        # Initialize actor (MLPModel) - takes ["policy", "latent"] as obs groups
        actor: MLPModel = actor_class(
            obs_with_latent, cfg["obs_groups"], "actor", _env.action_manager.total_action_dim, **cfg["actor"]
        ).to(device)
        print(f"[AMP] Actor Model: {actor}")

        # Initialize critic (MLPModel) - takes ["critic"] as obs groups
        critic: MLPModel = critic_class(
            obs_with_latent, cfg["obs_groups"], "critic", 1, **cfg["critic"]
        ).to(device)
        print(f"[AMP] Critic Model: {critic}")

        # Initialize CE-Net (standalone nn.Module) from ce_net_cfg
        est_terms = ce_net_cfg.get("est_terms", {})
        num_actor_obs = obs_with_latent["policy"].shape[-1]
        num_obs_future = obs_with_latent["obs_future"].shape[-1]
        len_prio_history = obs_with_latent["obs_history"].shape[-1] // num_actor_obs

        ce_net = CENet(
            num_actor_obs=num_actor_obs,
            num_future_obs=num_obs_future,
            len_prio_history=len_prio_history,
            latent_dims=latent_dims,
            est_terms=est_terms,
            encoder_hidden_dims=ce_net_cfg.get("encoder_hidden_dims", [512, 256, 64]),
            decoder_hidden_dims=ce_net_cfg.get("decoder_hidden_dims", [64, 128]),
            activation=ce_net_cfg.get("activation", "elu"),
        ).to(device)

        # AMP components from amp_dataset_cfg and amp_disc_cfg
        amp_num_frames = int(amp_dataset_cfg.get("num_frames", 2))
        num_amp_discriminator_obs = obs_with_latent["amp_obs_history"].shape[-1]  # 3D: (num_envs, amp_num_frames, amp_obs_dim)

        amp_motion_files = amp_dataset_cfg.get("motion_files", None)
        if amp_motion_files is None:
            raise ValueError("'amp_motion_files' must be specified for AMP training")
        amp_dataset = Dataset_Loader(
            device=device,
            time_between_frames=_env.step_dt * amp_dataset_cfg.get("history_stride", 2),
            num_envs=_env.num_envs,
            num_transitions_per_env=cfg["num_steps_per_env"],
            num_frames=amp_num_frames,
            preload_transitions=True,
            num_preload_transitions=amp_dataset_cfg.get("num_preload_transitions", 100000),
            motion_files=amp_motion_files,
        )
        amp_normalizer = Normalizer(num_amp_discriminator_obs, device)
        amp_discriminator = AMP_Discriminator(
            obs_with_latent["amp_obs_history"].shape[-1] * amp_num_frames,
            amp_disc_cfg.get("hidden_dims", [1024, 512]),
        ).to(device)

        storage_obs = obs_with_latent.clone()
        del storage_obs["amp_obs_history"]
        storage = RolloutStorage(
            "rl", _env.num_envs, cfg["num_steps_per_env"], storage_obs, [_env.action_manager.total_action_dim], device
        )

        # Auto-filter: pass only params that __init__ accepts.
        # Sub-configs (ce_net_cfg, amp_disc_cfg, amp_dataset_cfg) were popped
        # above, so only PPO-level params remain in alg_cfg and flow through
        # the auto-filter. Params extracted for explicit module construction
        # are passed directly below.
        _sig = inspect.signature(alg_class)
        _constructed = {"ce_net_cfg", "amp_disc_cfg", "est_terms"}
        init_alg_cfg = {
            k: v for k, v in alg_cfg.items()
            if k in _sig.parameters and k not in _constructed
        }

        # Construct algorithm
        alg: PPO_AMP = alg_class(
            actor=actor,
            critic=critic,
            storage=storage,
            ce_net=ce_net,
            amp_discriminator=amp_discriminator,
            amp_dataset=amp_dataset,
            amp_normalizer=amp_normalizer,
            ce_net_cfg=ce_net_cfg,
            amp_disc_cfg=amp_disc_cfg,
            est_terms=est_terms,
            env=env,
            device=device,
            **init_alg_cfg,
            multi_gpu_cfg=cfg.get("multi_gpu"),
        )

        # Init AMP replay buffer directly (no separate init_amp_storage method)
        alg.amp_storage = ReplayBuffer(
            _env.num_envs,
            cfg["num_steps_per_env"],
            amp_num_frames,
            num_amp_discriminator_obs,
            amp_dataset_cfg.get("replay_buffer_size", 1000000),
            device=device,
        )
        alg.amp_transition = ReplayBuffer.Transition()

        # Pass AMP components back to environment
        if hasattr(_env, "_set_amp_discriminator"):
            _env._set_amp_discriminator(alg.amp_discriminator, alg.amp_normalizer, alg.amp_dataset)

        return alg
