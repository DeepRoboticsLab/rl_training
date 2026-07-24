# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AMP replay buffer for discriminator training."""

from __future__ import annotations

import torch


class ReplayBuffer:
    """Circular replay buffer for storing AMP discriminator observations.

    Overwrites the oldest data when the buffer is full, and provides
    random mini-batch sampling for discriminator training.
    """

    class Transition:
        """Single transition container for AMP observations."""

        def __init__(self):
            self.obs = None

        def clear(self):
            self.obs = None

    def __init__(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        amp_num_frames: int,
        amp_discriminator_obs_shape: int,
        amp_replay_buffer_size: int,
        device: str = "cpu",
    ):
        self.device = device
        self.obs = torch.zeros(
            amp_replay_buffer_size, amp_num_frames, amp_discriminator_obs_shape, device=self.device
        )
        self.buffer_size = amp_replay_buffer_size
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.step = 0
        self.num_samples = 0
        self.mini_batch_size = self.num_envs * self.num_transitions_per_env

    def add_transitions(self, transition: Transition):
        """Add a batch of transitions to the circular buffer."""
        num_states = transition.obs.shape[0]
        start_idx = self.step
        end_idx = self.step + num_states

        if end_idx > self.buffer_size:
            # Overwrite oldest data when buffer is full
            self.obs[self.step:self.buffer_size].copy_(transition.obs[:self.buffer_size - self.step])
            self.obs[:end_idx - self.buffer_size].copy_(transition.obs[self.buffer_size - self.step:])
        else:
            self.obs[start_idx:end_idx].copy_(transition.obs)

        self.num_samples = min(self.buffer_size, max(end_idx, self.num_samples))
        self.step = (self.step + num_states) % self.buffer_size

    def feed_forward_generator(self, num_mini_batches: int, num_epochs: int = 5):
        """Yield random mini-batches of observations for discriminator training."""
        batch_size = self.num_envs * self.num_transitions_per_env
        self.mini_batch_size = batch_size // num_mini_batches

        for _ in range(num_epochs):
            for _ in range(num_mini_batches):
                sample_idxs = torch.randint(0, self.num_samples, (self.mini_batch_size,))
                yield self.obs[sample_idxs].to(self.device)
