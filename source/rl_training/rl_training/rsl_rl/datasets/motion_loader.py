
"""Motion dataset loader for AMP training.

Loads JSON motion files, standardizes quaternions, and provides
batch interpolation for sampling expert motion frames during training.
All operations are performed on CUDA tensors for efficiency.
"""

import glob
import json
import os

import numpy as np
import torch
from pybullet_utils import transformations

from ..utils import utils
from . import motion_util, pose3d

# Datasets are located under this directory
_AMP_DATASETS_DIR = os.path.abspath(os.path.dirname(__file__))
DEFAULT_MOTION_GLOB = os.path.join(_AMP_DATASETS_DIR, "amp_dataset_ik", "*")


class Dataset_Loader:
    """Motion dataset loader with weighted sampling and frame interpolation.

    Loads motion clips from JSON files, stores them as CUDA tensors,
    and provides batch sampling with linear interpolation for
    expert motion data during AMP discriminator training.
    """

    POS_SIZE = 3
    ROT_SIZE = 4
    PROJECTED_GRAVITY_SIZE = 3
    LINEAR_VEL_SIZE = 3
    ANGULAR_VEL_SIZE = 3
    JOINT_POS_SIZE = 20
    JOINT_VEL_SIZE = 20
    HAND_AND_FOOT_POS_SIZE = 12

    ROOT_POS_START_IDX = 0
    ROOT_POS_END_IDX = ROOT_POS_START_IDX + POS_SIZE  # [0: 3]

    ROOT_ROT_START_IDX = ROOT_POS_END_IDX
    ROOT_ROT_END_IDX = ROOT_ROT_START_IDX + ROT_SIZE  # [3: 7]

    PROJECTED_GRAVITY_START_IDX = ROOT_ROT_END_IDX
    PROJECTED_GRAVITY_END_IDX = PROJECTED_GRAVITY_START_IDX + PROJECTED_GRAVITY_SIZE  # [7: 10]

    LINEAR_VEL_START_IDX = PROJECTED_GRAVITY_END_IDX
    LINEAR_VEL_END_IDX = LINEAR_VEL_START_IDX + LINEAR_VEL_SIZE  # [10: 13]

    ANGULAR_VEL_START_IDX = LINEAR_VEL_END_IDX
    ANGULAR_VEL_END_IDX = ANGULAR_VEL_START_IDX + ANGULAR_VEL_SIZE  # [13: 16]

    JOINT_POSE_START_IDX = ANGULAR_VEL_END_IDX
    JOINT_POSE_END_IDX = JOINT_POSE_START_IDX + JOINT_POS_SIZE  # [16: 36]

    JOINT_VEL_START_IDX = JOINT_POSE_END_IDX
    JOINT_VEL_END_IDX = JOINT_VEL_START_IDX + JOINT_VEL_SIZE  # [36: 56]

    HAND_AND_FOOT_POS_START_IDX = JOINT_VEL_END_IDX
    HAND_AND_FOOT_POS_END_IDX = HAND_AND_FOOT_POS_START_IDX + HAND_AND_FOOT_POS_SIZE  # [56: 68]

    def __init__(
        self,
        device: str,
        time_between_frames: float,
        num_envs: int | None = None,
        num_transitions_per_env: int | None = None,
        num_frames: int = 2,
        preload_transitions: bool = False,
        num_preload_transitions: int = 100000,
        motion_files: list | None = None,
    ):
        self.device = device
        self.time_between_frames = time_between_frames
        self.num_frames = num_frames

        if num_envs is not None and num_transitions_per_env is not None:
            self.num_transitions_per_env = num_transitions_per_env
            self.num_envs = num_envs

        # All metadata stored as CUDA tensors
        self.trajectories_full = []
        self.trajectory_names = []
        self.trajectory_idxs = []
        self.trajectory_lens = torch.tensor([], device=device, dtype=torch.float32)
        self.trajectory_weights = torch.tensor([], device=device, dtype=torch.float32)
        self.trajectory_frame_durations = torch.tensor([], device=device, dtype=torch.float32)
        self.trajectory_num_frames = torch.tensor([], device=device, dtype=torch.float32)
        self.trajectory_lens_all = 0.0

        if motion_files is None:
            motion_files = glob.glob(DEFAULT_MOTION_GLOB)
        if not motion_files:
            raise ValueError(
                f"No motion files found. Pass motion_files explicitly "
                f"or populate {DEFAULT_MOTION_GLOB}"
            )

        for i, motion_file in enumerate(motion_files):
            self.trajectory_names.append(motion_file.split(".")[0])
            with open(motion_file, "r") as f:
                motion_json = json.load(f)
                motion_data = np.array(motion_json["Frames"])

                if motion_data.shape[1] != self.HAND_AND_FOOT_POS_END_IDX:
                    raise ValueError(
                        f"Motion data length mismatch: {motion_data.shape[1]} "
                        f"vs {self.HAND_AND_FOOT_POS_END_IDX}"
                    )

                # Standardize quaternions (numpy only during loading)
                for f_i in range(motion_data.shape[0]):
                    root_rot = self.get_root_rot(motion_data[f_i])
                    root_rot = pose3d.QuaternionNormalize(root_rot)
                    root_rot = motion_util.standardize_quaternion(root_rot)
                    motion_data[f_i, self.ROOT_ROT_START_IDX:self.ROOT_ROT_END_IDX] = root_rot

                # Convert to CUDA tensor
                traj_full = torch.tensor(
                    motion_data[:, :self.HAND_AND_FOOT_POS_END_IDX],
                    dtype=torch.float32,
                    device=device,
                )
                self.trajectories_full.append(traj_full)
                self.trajectory_idxs.append(i)

                # Metadata as CUDA tensors
                self.trajectory_weights = torch.cat([
                    self.trajectory_weights,
                    torch.tensor([float(motion_json["MotionWeight"])], device=device),
                ])
                fps = float(motion_json["fps"])
                self.trajectory_frame_durations = torch.cat([
                    self.trajectory_frame_durations,
                    torch.tensor([1.0 / fps], device=device),
                ])
                traj_len = (motion_data.shape[0] - 1) / fps
                self.trajectory_lens = torch.cat([
                    self.trajectory_lens,
                    torch.tensor([traj_len], device=device),
                ])
                self.trajectory_lens_all += traj_len
                self.trajectory_num_frames = torch.cat([
                    self.trajectory_num_frames,
                    torch.tensor([motion_data.shape[0]], device=device),
                ])

        # Normalize sampling weights
        print(f"[AMP Dataset] Total trajectory length: {self.trajectory_lens_all:.2f}s")
        self.trajectory_weights /= self.trajectory_weights.sum()

        # Preload transitions for faster sampling
        self.preload_transitions = preload_transitions
        if self.preload_transitions:
            print(f"[AMP Dataset] Preloading {num_preload_transitions} transitions")
            traj_idxs = self.weighted_traj_idx_sample_batch(num_preload_transitions)
            times = self.traj_time_sample_batch(traj_idxs, self.num_frames)
            self.preloaded_s = []
            for i in range(self.num_frames):
                self.preloaded_s.append(
                    self.get_full_frame_at_time_batch(traj_idxs, times + i * self.time_between_frames)
                )

    def weighted_traj_idx_sample_batch(self, size: int) -> torch.Tensor:
        """Weighted sampling of trajectory indices using multinomial."""
        return torch.multinomial(self.trajectory_weights, num_samples=size, replacement=True)

    def traj_time_sample_batch(self, traj_idxs: torch.Tensor, num_frame: int = 1) -> torch.Tensor:
        """Sample random time points within each trajectory."""
        offset = self.time_between_frames * num_frame + self.trajectory_frame_durations[traj_idxs]
        time_samples = self.trajectory_lens[traj_idxs] * torch.rand(
            len(traj_idxs), device=self.device
        ) - offset
        return torch.maximum(torch.zeros_like(time_samples), time_samples)

    def slerp(self, val0, val1, blend):
        """Linear interpolation between two tensors."""
        return (1.0 - blend) * val0 + blend * val1

    def get_full_frame_at_time_batch(self, traj_idxs: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """Batch interpolate motion frames at given times across trajectories."""
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]

        idx_low = torch.floor(p * n).long()
        idx_high = torch.ceil(p * n).long()

        batch_size = len(traj_idxs)
        all_frame_pos_starts = torch.zeros(batch_size, self.POS_SIZE, device=self.device)
        all_frame_pos_ends = torch.zeros_like(all_frame_pos_starts)
        all_frame_rot_starts = torch.zeros(batch_size, self.ROT_SIZE, device=self.device)
        all_frame_rot_ends = torch.zeros_like(all_frame_rot_starts)
        amp_dim = self.HAND_AND_FOOT_POS_END_IDX - self.PROJECTED_GRAVITY_START_IDX
        all_frame_amp_starts = torch.zeros(batch_size, amp_dim, device=self.device)
        all_frame_amp_ends = torch.zeros_like(all_frame_amp_starts)

        # Process each unique trajectory index
        for traj_idx in traj_idxs.unique():
            traj_mask = traj_idxs == traj_idx
            trajectory = self.trajectories_full[traj_idx]

            all_frame_pos_starts[traj_mask] = self.get_root_pos_batch(trajectory[idx_low[traj_mask]])
            all_frame_pos_ends[traj_mask] = self.get_root_pos_batch(trajectory[idx_high[traj_mask]])
            all_frame_rot_starts[traj_mask] = self.get_root_rot_batch(trajectory[idx_low[traj_mask]])
            all_frame_rot_ends[traj_mask] = self.get_root_rot_batch(trajectory[idx_high[traj_mask]])
            all_frame_amp_starts[traj_mask] = trajectory[idx_low[traj_mask]][
                :, self.PROJECTED_GRAVITY_START_IDX:self.HAND_AND_FOOT_POS_END_IDX
            ]
            all_frame_amp_ends[traj_mask] = trajectory[idx_high[traj_mask]][
                :, self.PROJECTED_GRAVITY_START_IDX:self.HAND_AND_FOOT_POS_END_IDX
            ]

        blend = (p * n - idx_low).unsqueeze(-1).to(dtype=torch.float32)

        pos_blend = self.slerp(all_frame_pos_starts, all_frame_pos_ends, blend)
        rot_blend = utils.quaternion_slerp(all_frame_rot_starts, all_frame_rot_ends, blend)
        amp_blend = self.slerp(all_frame_amp_starts, all_frame_amp_ends, blend)

        return torch.cat([pos_blend, rot_blend, amp_blend], dim=-1)

    def get_full_frame_batch(self, batch_size: int) -> torch.Tensor:
        """Sample a batch of full motion frames."""
        if self.preload_transitions:
            idxs = torch.randint(0, self.preloaded_s[0].shape[0], (batch_size,), device=self.device)
            return self.preloaded_s[0][idxs]
        else:
            traj_idxs = self.weighted_traj_idx_sample_batch(batch_size)
            times = self.traj_time_sample_batch(traj_idxs, self.num_frames)
            return self.get_full_frame_at_time_batch(traj_idxs, times)

    def feed_forward_generator(self, num_mini_batches: int, num_epochs: int = 5):
        """Yield mini-batches of expert motion frames for discriminator training."""
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches

        for _ in range(num_epochs):
            for _ in range(num_mini_batches):
                if self.preload_transitions:
                    idxs = torch.randint(
                        0, self.preloaded_s[0].shape[0], (mini_batch_size,), device=self.device
                    )
                    frames = [
                        self.preloaded_s[i][idxs][
                            :, self.PROJECTED_GRAVITY_START_IDX:self.HAND_AND_FOOT_POS_END_IDX
                        ]
                        for i in range(self.num_frames)
                    ]
                else:
                    traj_idx = self.weighted_traj_idx_sample_batch(mini_batch_size)
                    start_time = self.traj_time_sample_batch(traj_idx, self.num_frames)
                    frames = []
                    for i in range(self.num_frames):
                        frames.append(
                            self.get_full_frame_at_time_batch(
                                traj_idx, start_time + i * self.time_between_frames
                            )[:, self.PROJECTED_GRAVITY_START_IDX:self.HAND_AND_FOOT_POS_END_IDX]
                        )
                yield torch.stack(frames).transpose(0, 1)

    @property
    def num_motions(self) -> int:
        return len(self.trajectory_names)

    def get_root_pos(self, pose):
        return pose[self.ROOT_POS_START_IDX:self.ROOT_POS_END_IDX]

    def get_root_pos_batch(self, poses):
        return poses[:, self.ROOT_POS_START_IDX:self.ROOT_POS_END_IDX]

    def get_root_rot(self, pose):
        return pose[self.ROOT_ROT_START_IDX:self.ROOT_ROT_END_IDX]

    def get_root_rot_batch(self, poses):
        return poses[:, self.ROOT_ROT_START_IDX:self.ROOT_ROT_END_IDX]

    def get_projected_gravity(self, pose):
        return pose[self.PROJECTED_GRAVITY_START_IDX:self.PROJECTED_GRAVITY_END_IDX]

    def get_projected_gravity_batch(self, poses):
        return poses[:, self.PROJECTED_GRAVITY_START_IDX:self.PROJECTED_GRAVITY_END_IDX]

    def get_linear_vel(self, pose):
        return pose[self.LINEAR_VEL_START_IDX:self.LINEAR_VEL_END_IDX]

    def get_linear_vel_batch(self, poses):
        return poses[:, self.LINEAR_VEL_START_IDX:self.LINEAR_VEL_END_IDX]

    def get_angular_vel(self, pose):
        return pose[self.ANGULAR_VEL_START_IDX:self.ANGULAR_VEL_END_IDX]

    def get_angular_vel_batch(self, poses):
        return poses[:, self.ANGULAR_VEL_START_IDX:self.ANGULAR_VEL_END_IDX]

    def get_joint_pose(self, pose):
        return pose[self.JOINT_POSE_START_IDX:self.JOINT_POSE_END_IDX]

    def get_joint_pose_batch(self, poses):
        return poses[:, self.JOINT_POSE_START_IDX:self.JOINT_POSE_END_IDX]

    def get_joint_vel(self, pose):
        return pose[self.JOINT_VEL_START_IDX:self.JOINT_VEL_END_IDX]

    def get_joint_vel_batch(self, poses):
        return poses[:, self.JOINT_VEL_START_IDX:self.JOINT_VEL_END_IDX]

    def get_hand_and_foot_pos_local(self, pose):
        return pose[self.HAND_AND_FOOT_POS_START_IDX:self.HAND_AND_FOOT_POS_END_IDX]

    def get_hand_and_foot_local_batch(self, poses):
        return poses[:, self.HAND_AND_FOOT_POS_START_IDX:self.HAND_AND_FOOT_POS_END_IDX]
