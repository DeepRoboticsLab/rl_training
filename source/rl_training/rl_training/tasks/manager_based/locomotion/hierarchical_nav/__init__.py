# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Hierarchical navigation environments."""

# Note: Hierarchical navigation environments are created programmatically
# by wrapping low-level environments with HierarchicalNavEnv wrapper.
# They are not registered with gym.register() because they require
# a pre-loaded frozen policy wrapper.

