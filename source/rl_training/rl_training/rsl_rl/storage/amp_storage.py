# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AMP replay buffer storage for discriminator training."""

from __future__ import annotations

import torch


class ReplayBuffer:
    """环形缓冲区，用于存储 AMP discriminator 的观测数据。"""

    class Transition:
        def __init__(self, **kwargs):
            self.obs = None

        def clear(self):
            self.obs = None

    def __init__(self,
                 num_envs,
                 num_transitions_per_env,
                 amp_num_frames,
                 amp_discriminator_obs_shape,
                 amp_replay_buffer_size,
                 device='cpu',
                 **kwargs):
        self.device = device

        # Core
        self.obs = torch.zeros(amp_replay_buffer_size, amp_num_frames, amp_discriminator_obs_shape, device=self.device)
        self.buffer_size = amp_replay_buffer_size
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.step = 0
        self.num_samples = 0
        self.mini_batch_size = self.num_envs * self.num_transitions_per_env

    def add_transitions(self, transition: Transition):
        num_states = transition.obs.shape[0]
        start_idx = self.step
        end_idx = self.step + num_states
        if end_idx > self.buffer_size:
            # replay buffer need not to clear, but overwrite the oldest data when buffer is full
            self.obs[self.step:self.buffer_size].copy_(transition.obs[:self.buffer_size - self.step])
            self.obs[:end_idx - self.buffer_size].copy_(transition.obs[self.buffer_size - self.step:])
        else:
            self.obs[start_idx:end_idx].copy_(transition.obs)

        self.num_samples = min(self.buffer_size, max(end_idx, self.num_samples))
        self.step = (self.step + num_states) % self.buffer_size

    def feed_forward_generator(self, num_mini_batches, num_epochs=5):
        batch_size = self.num_envs * self.num_transitions_per_env
        self.mini_batch_size = batch_size // num_mini_batches

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                sample_idxs = torch.randint(0, self.num_samples, (self.mini_batch_size,))
                yield self.obs[sample_idxs].to(self.device)

