# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for AMP training."""

from __future__ import annotations

import copy
import numpy as np
import torch

_EPS = np.finfo(float).eps * 4.0


class RunningMeanStd:
    """Running mean and standard deviation with a sliding window."""

    def __init__(self, device="cuda:0", shape: tuple = (), window_size: int = 200):
        """Calculate the running mean and std of the last ``window_size`` updates.

        Args:
            shape: The shape of the data stream's output.
            window_size: The number of recent updates to consider.
        """
        self.device = device
        self.window_size = window_size
        self.mean = torch.zeros(shape, dtype=torch.float64, device=device)
        self.var = torch.ones(shape, dtype=torch.float64, device=device)
        self.mean_history = torch.zeros((window_size, shape), dtype=torch.float64, device=device)
        self.var_history = torch.zeros((window_size, shape), dtype=torch.float64, device=device)
        self.current_index = 0
        self.total_updates = 0

    def update(self, arr: torch.Tensor) -> None:
        batch_mean = arr.mean(dim=0)
        batch_var = arr.var(dim=0, unbiased=True)

        # Store the current batch's mean, var, and count in the history
        self.mean_history[self.current_index] = batch_mean
        self.var_history[self.current_index] = batch_var

        # Update the index for the next update
        self.current_index = (self.current_index + 1) % self.window_size
        self.total_updates += 1

        # Calculate the effective window size
        effective_window_size = min(self.total_updates, self.window_size)

        # Calculate the running mean over the last ``effective_window_size`` updates
        self.mean = self.mean_history[:effective_window_size].mean(dim=0)

        # Calculate the correct running variance
        if effective_window_size == 1:
            self.var = batch_var
        else:
            # Within-batch variance (average of batch variances)
            within_var = self.var_history[:effective_window_size].mean(dim=0)
            # Between-batch variance (variance of batch means)
            between_var = self.mean_history[:effective_window_size].var(dim=0, unbiased=False)
            # Total variance is sum of within and between variances
            self.var = within_var + between_var


class Normalizer(RunningMeanStd):
    """Normalize input tensors using running mean and std statistics."""

    def __init__(self, input_dim, device="cuda:0", epsilon=1e-4, clip_obs=10.0, window_size=100):
        super().__init__(device=device, shape=input_dim, window_size=window_size)
        self.epsilon = torch.tensor(epsilon, dtype=torch.float64, device=device)
        self.clip_obs = torch.tensor(clip_obs, dtype=torch.float64, device=device)

    def normalize(self, input: torch.Tensor) -> torch.Tensor:
        return torch.clamp(
            (input - self.mean) / torch.sqrt(self.var + self.epsilon),
            -self.clip_obs, self.clip_obs
        ).to(torch.float32)


def quaternion_slerp(q_start, q_end, fraction, spin=0, shortestpath=True):
    """Batch quaternion spherical linear interpolation.

    Args:
        q_start: Start quaternions, shape ``[..., 4]`` (x, y, z, w).
        q_end: End quaternions, shape ``[..., 4]``.
        fraction: Interpolation fraction, shape ``[..., 1]``.
        spin: Additional spin in multiples of pi.
        shortestpath: Whether to take the shortest path on the quaternion sphere.

    Returns:
        Interpolated quaternions, shape ``[..., 4]``.
    """
    q0 = copy.deepcopy(q_start)
    q1 = copy.deepcopy(q_end)

    out = torch.zeros_like(q0)

    zero_mask = torch.isclose(fraction, torch.zeros_like(fraction)).squeeze()
    ones_mask = torch.isclose(fraction, torch.ones_like(fraction)).squeeze()
    out[zero_mask] = q0[zero_mask]
    out[ones_mask] = q1[ones_mask]

    # Clip dot product to [-1, 1] to avoid NaN in acos
    d = torch.clip(torch.sum(q0 * q1, dim=-1, keepdim=True), min=-1.0, max=1.0)
    dist_mask = (torch.abs(torch.abs(d) - 1.0) < _EPS).squeeze()
    out[dist_mask] = q0[dist_mask]

    if shortestpath:
        d_old = torch.clone(d)
        d = torch.where(d_old < 0, -d, d)
        q1 = torch.where(d_old < 0, -q1, q1)

    angle = torch.acos(d) + spin * torch.pi

    angle_mask = (torch.abs(angle) < _EPS).squeeze()
    out[angle_mask] = q0[angle_mask]

    final_mask = torch.logical_or(zero_mask, ones_mask)
    final_mask = torch.logical_or(final_mask, dist_mask)
    final_mask = torch.logical_or(final_mask, angle_mask)
    final_mask = torch.logical_not(final_mask)

    isin = 1.0 / angle
    q0 *= torch.sin((1.0 - fraction) * angle) * isin
    q1 *= torch.sin(fraction * angle) * isin
    q0 += q1
    out[final_mask] = q0[final_mask]
    return out
