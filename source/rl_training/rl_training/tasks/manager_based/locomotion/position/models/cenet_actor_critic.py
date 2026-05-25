# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""CENet (Causal Estimation Network) components for history-based VAE encoder.

Architecture overview:
  HistoryEncoder  : 6 × obs_dim  →  [128, 64]  →  mu(16) + log_var(16) + v_pred(3)
                    Reparameterisation → z(16)
  FutureDecoder   : z(16) + v_pred(3) = 19  →  [64, 128]  →  obs_pred(obs_dim)

Actor input (assembled by CENetOnPolicyRunner):
  [o_t(obs_dim), z(16), actor_vel(3)]  →  Actor MLP [512, 256, 128]  →  action
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_mlp(
    input_dim: int,
    hidden_dims: list[int],
    output_dim: int,
    activation: str = "elu",
) -> nn.Sequential:
    """Build a sequential MLP: input → hidden[0] → ... → hidden[-1] → output."""
    act_map: dict[str, type[nn.Module]] = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "selu": nn.SELU,
        "lrelu": nn.LeakyReLU,
    }
    Act: type[nn.Module] = act_map.get(activation.lower()) or nn.ELU

    layers: list[nn.Module] = []
    in_dim = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(in_dim, h))
        layers.append(Act())
        in_dim = h
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# HistoryEncoder  (VAE encoder)
# ---------------------------------------------------------------------------

class HistoryEncoder(nn.Module):
    """VAE encoder that maps a window of historical observations to a latent
    variable and a predicted next-frame linear velocity.

    Input shape  : ``[batch, history_len * obs_dim]``
    Output shapes:
        - ``z``        : ``[batch, latent_dim]``  (reparameterised sample)
        - ``v_pred``   : ``[batch, vel_dim]``     (predicted next-frame velocity)
        - ``mu``       : ``[batch, latent_dim]``  (VAE mean, used for KL loss)
        - ``log_var``  : ``[batch, latent_dim]``  (VAE log-variance, used for KL)

    MLP architecture: ``history_len * obs_dim → hidden_dims → 2 * latent_dim + vel_dim``
    """

    def __init__(
        self,
        obs_dim: int,
        history_len: int = 6,
        latent_dim: int = 32,
        vel_dim: int = 3,
        activation: str = "elu",
        hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.vel_dim = vel_dim
        input_dim = history_len * obs_dim
        output_dim = 2 * latent_dim + vel_dim
        if hidden_dims is None:
            hidden_dims = [256, 128]
        self.net = _build_mlp(input_dim, hidden_dims, output_dim, activation)

    def forward(
        self, obs_history: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            obs_history: Flattened history tensor ``[batch, history_len * obs_dim]``.

        Returns:
            Tuple ``(z, v_pred, mu, log_var)`` where:
              - ``z``       : reparameterised latent  ``[batch, latent_dim]``
              - ``v_pred``  : predicted velocity      ``[batch, vel_dim]``
              - ``mu``      : distribution mean       ``[batch, latent_dim]``
              - ``log_var`` : distribution log-var    ``[batch, latent_dim]``
        """
        out = self.net(obs_history)  # [batch, 2*latent_dim + vel_dim]
        mu = out[:, : self.latent_dim]
        log_var = out[:, self.latent_dim : 2 * self.latent_dim]
        v_pred = out[:, 2 * self.latent_dim :]

        # Reparameterisation trick (only during training to keep inference deterministic)
        if self.training:
            std = (0.5 * log_var).exp().clamp(max=2.0)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu  # deterministic at inference

        return z, v_pred, mu, log_var


# ---------------------------------------------------------------------------
# FutureDecoder
# ---------------------------------------------------------------------------

class FutureDecoder(nn.Module):
    """Decoder that reconstructs the next observation from the latent variable
    and the predicted velocity.

    Input shape : ``[batch, latent_dim + vel_dim]``
    Output shape: ``[batch, obs_dim]``

    MLP architecture: ``latent_dim + vel_dim → hidden_dims → obs_dim``
    """

    def __init__(
        self,
        latent_dim: int = 32,
        vel_dim: int = 3,
        obs_dim: int = 54,
        activation: str = "elu",
        hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        input_dim = latent_dim + vel_dim
        if hidden_dims is None:
            hidden_dims = [128, 256]
        self.net = _build_mlp(input_dim, hidden_dims, obs_dim, activation)

    def forward(self, z: torch.Tensor, v_pred: torch.Tensor) -> torch.Tensor:
        """Predict the next observation.

        Args:
            z:      Latent variable  ``[batch, latent_dim]``.
            v_pred: Predicted velocity ``[batch, vel_dim]``.

        Returns:
            Predicted future observation ``[batch, obs_dim]``.
        """
        x = torch.cat([z, v_pred], dim=-1)
        return self.net(x)
