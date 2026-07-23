# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.modules import MLP

class CENet(nn.Module):
    """CE-Net: Context-Encoder Network with VAE for asymmetric actor-critic.

    Encodes history observations into latent representations (explicit + implicit),
    and decodes implicit latents into future observation predictions.
    Used as a standalone module alongside separate actor/critic MLPModels.
    """

    def __init__(self,
                 num_actor_obs,
                 num_future_obs,
                 len_prio_history=1,
                 latent_dims=None,
                 est_terms=None,
                 encoder_hidden_dims=None,
                 decoder_hidden_dims=None,
                 activation='elu',
                 latent_to_obs_pred=False,
                 **kwargs):
        super(CENet, self).__init__()

        if latent_dims is None:
            if est_terms is not None:
                latent_dims = sum([term["dim"] for term in est_terms.values()])
            else:
                latent_dims = 19
        if est_terms is None:
            est_terms = {}
        if decoder_hidden_dims is None:
            decoder_hidden_dims = [64, 128]
        if encoder_hidden_dims is None:
            encoder_hidden_dims = [512, 256, 64]

        if kwargs:
            print("CENet.__init__ got unexpected args, ignoring: " + str([k for k in kwargs.keys()]))

        self.est_terms = est_terms
        self.est_explicit_key = [k for k, d in est_terms.items() if d["type"] == "explicit"]
        self.num_actor_obs = num_actor_obs
        self.num_history_frames = len_prio_history
        encoder_input_dim = num_actor_obs * self.num_history_frames
        self.latent_to_obs_pred = latent_to_obs_pred
        self.latent_dims = latent_dims

        self.encoder = MLP(input_dim=encoder_input_dim, output_dim=latent_dims,
                          hidden_dims=encoder_hidden_dims, activation=activation)

        # Explicit estimation layers
        est_explicit_dict = {k: nn.Linear(latent_dims, d["dim"]) for k, d in est_terms.items() if d["type"] == "explicit"}
        self.est_explicit_layers = nn.ModuleDict(est_explicit_dict)

        # Implicit estimation (VAE)
        if est_terms['implicit']["dim"] > 0:
            self.fc_mu = nn.Linear(latent_dims, est_terms['implicit']["dim"])
            self.fc_var = nn.Linear(latent_dims, est_terms['implicit']["dim"])
            if self.latent_to_obs_pred:
                self.decoder = MLP(input_dim=latent_dims, output_dim=num_future_obs,
                                  hidden_dims=decoder_hidden_dims, activation=activation)
            else:
                self.decoder = MLP(input_dim=est_terms['implicit']["dim"], output_dim=num_future_obs,
                                  hidden_dims=decoder_hidden_dims, activation=activation)
            self.has_implicit = True
        else:
            self.has_implicit = False
            raise RuntimeError("Implicit latent dimensions less than 0")

        print(f"[CE-NET] Encoder MLP: {self.encoder}")
        print(f"[CE-NET] Decoder MLP: {self.decoder}")

    def encode(self, obs_h, **kwargs):
        encodings = {}
        latent = self.encoder(obs_h)
        # Explicit estimation
        est_explicit = [torch.clip(layer(latent), -10, 10) for layer in self.est_explicit_layers.values()]
        encodings.update({k: v for k, v in zip(self.est_explicit_key, est_explicit)})
        # Implicit estimation (VAE)
        if self.has_implicit:
            mu = torch.clip(self.fc_mu(latent), -10, 10)
            log_var = torch.clip(self.fc_var(latent), -10, 10)
            z = torch.clip(self.reparameterize(mu, log_var), -10, 10)
            encodings["implicit"] = z
            return encodings, mu, log_var
        else:
            return encodings

    def decode(self, encodings):
        decodings = {}
        if self.latent_to_obs_pred:
            obs_pred = self.decoder(torch.cat([encodings[k] for k in self.est_terms.keys()], dim=-1))
            decodings["obs_pred"] = self.decoder(obs_pred)
        else:
            decodings["obs_pred"] = self.decoder(encodings["implicit"])
        return decodings

    def ce_net(self, obs_h, **kwargs):
        if self.has_implicit:
            encodings, mu, log_var = self.encode(obs_h)
            return encodings, self.decode(encodings), mu, log_var
        else:
            return self.encode(obs_h)

    def get_encodings_cat(self, obs_h):
        """Encode and return concatenated encodings (for injection into actor input)."""
        if self.has_implicit:
            encodings, _, _ = self.encode(obs_h)
            encodings_list = [v for k, v in encodings.items() if (k != "implicit") and (k in self.est_terms.keys())]
            encodings_list.append(encodings["implicit"])
            return torch.cat(encodings_list, dim=-1)
        else:
            encodings = self.encode(obs_h)
            return torch.cat([encodings[k] for k in self.est_terms.keys()], dim=-1)

    def reparameterize(self, mu, logvar):
        """

        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
