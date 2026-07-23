import torch
import torch.nn as nn
from torch import autograd


class AMP_Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_layer_sizes, print_net_info=True):
        super(AMP_Discriminator, self).__init__()
        amp_layers = []
        curr_in_dim = input_dim

        for hidden_dim in hidden_layer_sizes:
            amp_layers.append(nn.Linear(curr_in_dim, hidden_dim))
            # amp_layers.append(nn.SELU())
            amp_layers.append(nn.LeakyReLU(0.1))
            curr_in_dim = hidden_dim
        self.trunk = nn.Sequential(*amp_layers)
        self.amp_linear = nn.Linear(hidden_layer_sizes[-1], 1)

        self.trunk.train()
        self.amp_linear.train()
        if print_net_info:
            print(f"[AMP]Trunk MLP: {self.trunk}")
            print(f"[AMP]Linear MLP: {self.amp_linear}")

    def forward(self, x):
        h = self.trunk(x)
        d = self.amp_linear(h)
        return d

    def compute_grad_pen(self, expert_state):
        expert_data = expert_state
        expert_data.requires_grad =True

        disc = self.amp_linear(self.trunk(expert_data))
        ones = torch.ones(disc.size(), device=disc.device)
        grad = autograd.grad(
            outputs=disc, inputs=expert_data,
            grad_outputs=ones, create_graph=True,
            retain_graph=True, only_inputs=True)[0]

        # Enforce that the grad norm approaches 0.
        grad_pen = (grad.norm(2, dim=1) - 0).pow(2).mean()
        return grad_pen

    def compute_amp_reward(self, state, normalizer=None):
        with torch.no_grad():
            self.eval()

            if normalizer is not None:
                batch_size = state.shape[0]
                feature_dim = state.shape[2]
                state = normalizer.normalize(state.reshape(-1, feature_dim)).reshape(batch_size, -1)
            else:
                batch_size = state.shape[0]
                state = state.reshape(batch_size, -1)

            d = self.amp_linear(self.trunk(state))

            reward = torch.clamp(1 - (1/4) * torch.square(d - 1), min=0)

            self.train()
        return reward.squeeze()
