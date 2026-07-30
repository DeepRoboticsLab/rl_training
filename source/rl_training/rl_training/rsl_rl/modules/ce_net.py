# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CE-Net: Context-Encoder Network with VAE for asymmetric actor-critic."""

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.modules import MLP


class CENet(nn.Module):
    """Context-Encoder Network with VAE for asymmetric actor-critic.

    Encodes history observations into latent representations (explicit + implicit),
    and decodes implicit latents into future observation predictions.
    Used as a standalone module alongside separate actor/critic MLPModels.
    """

    def __init__(
        self,
        num_actor_obs: int,
        num_future_obs: int,
        len_prio_history: int = 1,
        latent_dims: int | None = None,
        est_terms: dict | None = None,
        encoder_hidden_dims: list | None = None,
        decoder_hidden_dims: list | None = None,
        activation: str = "elu",
        **kwargs,
    ):
        super().__init__()

        if latent_dims is None:
            if est_terms is not None:
                latent_dims = sum(term["dim"] for term in est_terms.values())
            else:
                latent_dims = 19
        if est_terms is None:
            est_terms = {}
        if decoder_hidden_dims is None:
            decoder_hidden_dims = [64, 128]
        if encoder_hidden_dims is None:
            encoder_hidden_dims = [512, 256, 64]

        if kwargs:
            print(f"[CENet] Unexpected args, ignoring: {list(kwargs.keys())}")

        self.est_terms = est_terms
        self.est_explicit_key = [k for k, d in est_terms.items() if d["type"] == "explicit"]
        self.num_actor_obs = num_actor_obs
        self.num_history_frames = len_prio_history
        self.latent_dims = latent_dims

        encoder_input_dim = num_actor_obs * self.num_history_frames
        self.encoder = MLP(
            input_dim=encoder_input_dim,
            output_dim=latent_dims,
            hidden_dims=encoder_hidden_dims,
            activation=activation,
        )

        # Explicit estimation layers
        est_explicit_dict = {
            k: nn.Linear(latent_dims, d["dim"])
            for k, d in est_terms.items()
            if d["type"] == "explicit"
        }
        self.est_explicit_layers = nn.ModuleDict(est_explicit_dict)

        # Implicit estimation (VAE)
        self.fc_mu = nn.Linear(latent_dims, est_terms["implicit"]["dim"])
        self.fc_var = nn.Linear(latent_dims, est_terms["implicit"]["dim"])
        self.decoder = MLP(
            input_dim=est_terms["implicit"]["dim"],
            output_dim=num_future_obs,
            hidden_dims=decoder_hidden_dims,
            activation=activation,
        )

        print(f"[CENet] Encoder MLP: {self.encoder}")
        print(f"[CENet] Decoder MLP: {self.decoder}")

    def encode(self, obs_h: torch.Tensor, **kwargs):
        """Encode history observations into explicit and implicit latents.

        Returns:
            encodings: Dict mapping term names to latent tensors.
            mu: Mean of the implicit latent distribution.
            log_var: Log-variance of the implicit latent distribution.
        """
        encodings = {}
        latent = self.encoder(obs_h)

        # Explicit estimation
        est_explicit = [torch.clip(layer(latent), -10, 10) for layer in self.est_explicit_layers.values()]
        encodings.update({k: v for k, v in zip(self.est_explicit_key, est_explicit)})

        # Implicit estimation (VAE)
        mu = torch.clip(self.fc_mu(latent), -10, 10)
        log_var = torch.clip(self.fc_var(latent), -10, 10)
        z = torch.clip(self.reparameterize(mu, log_var), -10, 10)
        encodings["implicit"] = z

        return encodings, mu, log_var

    def decode(self, encodings: dict) -> dict:
        """Decode implicit latents into future observation predictions."""
        return {"obs_pred": self.decoder(encodings["implicit"])}

    def forward(self, obs_h: torch.Tensor, **kwargs):
        """Full forward pass: encode + decode.

        Returns:
            encodings, decodings, mu, log_var
        """
        encodings, mu, log_var = self.encode(obs_h)
        return encodings, self.decode(encodings), mu, log_var

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """VAE reparameterization trick: z = mu + std * eps.

        Args:
            mu: Mean of the latent Gaussian.
            logvar: Log-variance of the latent Gaussian.

        Returns:
            Sampled latent tensor.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
