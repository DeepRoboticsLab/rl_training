import os
import glob
import json
import logging

import torch
import numpy as np
from pybullet_utils import transformations

from ..utils import utils
from . import pose3d
from . import motion_util

# Datasets are located under rl_training/rl_training/rsl_rl/datasets/
_AMP_DATASETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
DEFAULT_MOTION_GLOB = os.path.join(_AMP_DATASETS_DIR, 'amp_dataset_ik', '*')


class Dataset_Loader:
    # 常量定义保持不变
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
    JOINT_VEL_END_IDX = JOINT_VEL_START_IDX + JOINT_VEL_SIZE  # [36： 56]

    HAND_AND_FOOT_POS_START_IDX = JOINT_VEL_END_IDX
    HAND_AND_FOOT_POS_END_IDX = HAND_AND_FOOT_POS_START_IDX + HAND_AND_FOOT_POS_SIZE  # [56: ]

    def __init__(
            self,
            device,
            time_between_frames,
            num_envs=None,
            num_transitions_per_env=None,
            num_frames=2,
            preload_transitions=False,
            num_preload_transitions=100000,
            motion_files=None,
    ):
        self.device = device
        self.time_between_frames = time_between_frames
        self.num_frames = num_frames

        if num_envs is not None and num_transitions_per_env is not None:
            self.num_transitions_per_env = num_transitions_per_env
            self.num_envs = num_envs

        # 所有元数据均存储为CUDA张量（移除numpy依赖）
        self.trajectories_full = []  # 存储完整轨迹（CUDA张量）
        self.trajectory_names = []
        self.trajectory_idxs = []
        self.trajectory_lens = torch.tensor([], device=device, dtype=torch.float32)  # 轨迹时长（CUDA）
        self.trajectory_weights = torch.tensor([], device=device, dtype=torch.float32)  # 采样权重（CUDA）
        self.trajectory_frame_durations = torch.tensor([], device=device, dtype=torch.float32)  # 每帧时长（CUDA）
        self.trajectory_num_frames = torch.tensor([], device=device, dtype=torch.float32)  # 总帧数（CUDA）
        self.trajectory_lens_all = 0.0

        if motion_files is None:
            motion_files = glob.glob(DEFAULT_MOTION_GLOB)
        if not motion_files:
            raise ValueError(f"No motion files found. Pass motion_files explicitly or populate {DEFAULT_MOTION_GLOB}")

        for i, motion_file in enumerate(motion_files):
            self.trajectory_names.append(motion_file.split('.')[0])
            with open(motion_file, "r") as f:
                motion_json = json.load(f)
                motion_data = np.array(motion_json["Frames"])  # 临时用numpy读入，随后转为CUDA

                if motion_data.shape[1] != self.HAND_AND_FOOT_POS_END_IDX:
                    raise ValueError(
                        f"Motion Data length mismatch: {motion_data.shape[1]} vs {self.HAND_AND_FOOT_POS_END_IDX}")

                # 标准化四元数（仅在数据加载时用numpy，随后转为CUDA）
                for f_i in range(motion_data.shape[0]):
                    root_rot = self.get_root_rot(motion_data[f_i])
                    root_rot = pose3d.QuaternionNormalize(root_rot)
                    root_rot = motion_util.standardize_quaternion(root_rot)
                    motion_data[f_i, self.ROOT_ROT_START_IDX:self.ROOT_ROT_END_IDX] = root_rot

                # 转为CUDA张量并存储
                traj_full = torch.tensor(motion_data[:, :self.HAND_AND_FOOT_POS_END_IDX], dtype=torch.float32,
                                         device=device)
                self.trajectories_full.append(traj_full)
                self.trajectory_idxs.append(i)

                # 元数据转为CUDA张量并拼接
                self.trajectory_weights = torch.cat(
                    [self.trajectory_weights, torch.tensor([float(motion_json["MotionWeight"])], device=device)])
                fps = float(motion_json["fps"])
                self.trajectory_frame_durations = torch.cat(
                    [self.trajectory_frame_durations, torch.tensor([1.0 / fps], device=device)])
                traj_len = (motion_data.shape[0] - 1) / fps
                self.trajectory_lens = torch.cat([self.trajectory_lens, torch.tensor([traj_len], device=device)])
                self.trajectory_lens_all += traj_len
                self.trajectory_num_frames = torch.cat(
                    [self.trajectory_num_frames, torch.tensor([motion_data.shape[0]], device=device)])

        # 归一化权重（CUDA上操作）
        print("总轨迹长度(s):", self.trajectory_lens_all)
        self.trajectory_weights /= self.trajectory_weights.sum()

        # 预加载（全CUDA操作）
        self.preload_transitions = preload_transitions
        if self.preload_transitions:
            print(f'Preloading {num_preload_transitions} transitions to discriminator')
            traj_idxs = self.weighted_traj_idx_sample_batch(num_preload_transitions)
            times = self.traj_time_sample_batch(traj_idxs, self.num_frames)
            self.preloaded_s = []
            for i in range(self.num_frames):
                self.preloaded_s.append(
                    self.get_full_frame_at_time_batch(traj_idxs, times + i * self.time_between_frames))

    def weighted_traj_idx_sample_batch(self, size):
        """CUDA上加权采样轨迹索引（替代np.random.choice）"""
        return torch.multinomial(self.trajectory_weights, num_samples=size, replacement=True)

    def traj_time_sample_batch(self, traj_idxs, num_frame=1):
        """CUDA上采样时间点（替代numpy操作）"""
        subst = self.time_between_frames * num_frame + self.trajectory_frame_durations[traj_idxs]
        time_samples = self.trajectory_lens[traj_idxs] * torch.rand(len(traj_idxs), device=self.device) - subst
        return torch.maximum(torch.zeros_like(time_samples), time_samples)

    def slerp(self, val0, val1, blend):
        return (1.0 - blend) * val0 + blend * val1

    def get_full_frame_at_time_batch(self, traj_idxs, times):
        """全CUDA操作：批量获取插值帧"""
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]

        # 替代np.floor和np.ceil（CUDA上操作）
        idx_low = torch.floor(p * n).long()
        idx_high = torch.ceil(p * n).long()

        # 初始化插值张量（直接在CUDA上分配）
        batch_size = len(traj_idxs)
        all_frame_pos_starts = torch.zeros(batch_size, self.POS_SIZE, device=self.device)
        all_frame_pos_ends = torch.zeros_like(all_frame_pos_starts)
        all_frame_rot_starts = torch.zeros(batch_size, self.ROT_SIZE, device=self.device)
        all_frame_rot_ends = torch.zeros_like(all_frame_rot_starts)
        amp_dim = self.HAND_AND_FOOT_POS_END_IDX - self.PROJECTED_GRAVITY_START_IDX
        all_frame_amp_starts = torch.zeros(batch_size, amp_dim, device=self.device)
        all_frame_amp_ends = torch.zeros_like(all_frame_amp_starts)

        # 遍历唯一轨迹索引（CUDA张量直接操作）
        for traj_idx in traj_idxs.unique():  # 用torch.unique替代set()，避免CPU转换
            traj_mask = traj_idxs == traj_idx
            trajectory = self.trajectories_full[traj_idx]

            # 批量填充（CUDA上索引操作）
            all_frame_pos_starts[traj_mask] = self.get_root_pos_batch(trajectory[idx_low[traj_mask]])
            all_frame_pos_ends[traj_mask] = self.get_root_pos_batch(trajectory[idx_high[traj_mask]])
            all_frame_rot_starts[traj_mask] = self.get_root_rot_batch(trajectory[idx_low[traj_mask]])
            all_frame_rot_ends[traj_mask] = self.get_root_rot_batch(trajectory[idx_high[traj_mask]])
            all_frame_amp_starts[traj_mask] = trajectory[idx_low[traj_mask]][:,
                                              self.PROJECTED_GRAVITY_START_IDX:self.HAND_AND_FOOT_POS_END_IDX]
            all_frame_amp_ends[traj_mask] = trajectory[idx_high[traj_mask]][:,
                                            self.PROJECTED_GRAVITY_START_IDX:self.HAND_AND_FOOT_POS_END_IDX]

        # 插值系数计算（全CUDA）
        blend = (p * n - idx_low).unsqueeze(-1).to(dtype=torch.float32)

        # 插值操作
        pos_blend = self.slerp(all_frame_pos_starts, all_frame_pos_ends, blend)
        rot_blend = utils.quaternion_slerp(all_frame_rot_starts, all_frame_rot_ends, blend)
        amp_blend = self.slerp(all_frame_amp_starts, all_frame_amp_ends, blend)

        return torch.cat([pos_blend, rot_blend, amp_blend], dim=-1)

    def get_full_frame_batch(self, batch_size):
        """全CUDA批量获取帧（无numpy操作）"""
        if self.preload_transitions:
            idxs = torch.randint(0, self.preloaded_s[0].shape[0], (batch_size,), device=self.device)
            return self.preloaded_s[0][idxs]
        else:
            traj_idxs = self.weighted_traj_idx_sample_batch(batch_size)
            times = self.traj_time_sample_batch(traj_idxs, self.num_frames)
            return self.get_full_frame_at_time_batch(traj_idxs, times)

    def feed_forward_generator(self, num_mini_batches, num_epochs=5):
        """生成器：全CUDA操作，无CPU交互"""
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                if self.preload_transitions:
                    # CUDA随机索引
                    start_idx = torch.randint(0, self.preloaded_s[0].shape[0], (mini_batch_size,), device=self.device)
                    frames = [
                        self.preloaded_s[i][start_idx][:,
                        self.PROJECTED_GRAVITY_START_IDX:self.HAND_AND_FOOT_POS_END_IDX]
                        for i in range(self.num_frames)
                    ]
                else:
                    traj_idx = self.weighted_traj_idx_sample_batch(mini_batch_size)
                    start_time = self.traj_time_sample_batch(traj_idx, self.num_frames)
                    frames = []
                    for i in range(self.num_frames):
                        frames.append(
                            self.get_full_frame_at_time_batch(traj_idx, start_time + i * self.time_between_frames)
                            [:, self.PROJECTED_GRAVITY_START_IDX:self.HAND_AND_FOOT_POS_END_IDX]
                        )
                yield torch.stack(frames).transpose(0, 1)

    # 工具方法保持不变（均操作CUDA张量）
    @property
    def num_motions(self):
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
