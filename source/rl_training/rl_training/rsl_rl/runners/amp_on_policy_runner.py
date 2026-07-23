# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AMP On-Policy Runner.

Extends the base ``OnPolicyRunner`` with AMP-specific policy export:
- ``get_inference_policy``: returns ``AMPInferencePolicy`` (CE-Net + Actor)
- ``export_policy_to_jit``: exports combined CE-Net + Actor to TorchScript
- ``export_policy_to_onnx``: exports combined CE-Net + Actor to ONNX

Training flow is identical to the base runner — ``PPO_AMP.construct_algorithm``
handles all AMP-specific initialisation.
"""

from __future__ import annotations

import os

import torch

from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from ..algorithms.ppo_amp import AMPExportWrapper


class AMPOnPolicyRunner(OnPolicyRunner):
    """On-policy runner for AMP training with custom policy export."""

    def get_inference_policy(self, device: str | None = None):
        """Return combined CE-Net + Actor for inference.

        Overrides the base runner which returns the bare actor MLPModel.
        """
        self.alg.eval_mode()
        policy = self.alg.get_inference_policy()
        if device is not None:
            policy = policy.to(device)
        return policy

    def export_policy_to_jit(self, path: str, filename: str = "policy.pt") -> None:
        """Export CE-Net + Actor to a TorchScript file."""
        import copy
        policy = copy.deepcopy(self.alg.get_inference_policy())
        wrapped = AMPExportWrapper(policy)
        wrapped.to("cpu").eval()

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, filename)

        with torch.inference_mode():
            traced = torch.jit.trace(wrapped, wrapped.get_dummy_inputs()[0])
            traced.save(save_path)
        print(f"[INFO] TorchScript saved to: {save_path}")
        print(f"  input:  {wrapped.input_names[0]}  shape (1, {wrapped.input_size})")

    def export_policy_to_onnx(self, path: str, filename: str = "policy.onnx", verbose: bool = False) -> None:
        """Export CE-Net + Actor to an ONNX file."""
        import copy
        policy = copy.deepcopy(self.alg.get_inference_policy())
        wrapped = AMPExportWrapper(policy)
        wrapped.to("cpu").eval()

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, filename)

        with torch.inference_mode():
            torch.onnx.export(
                wrapped,
                wrapped.get_dummy_inputs(),
                save_path,
                export_params=True,
                opset_version=17,
                verbose=verbose,
                input_names=wrapped.input_names,
                output_names=wrapped.output_names,
            )
        print(f"[INFO] ONNX saved to: {save_path}")
        print(f"  input:  {wrapped.input_names[0]}  shape (1, {wrapped.input_size})")


__all__ = ["AMPOnPolicyRunner"]
