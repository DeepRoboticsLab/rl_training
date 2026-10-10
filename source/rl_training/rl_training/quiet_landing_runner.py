"""Quiet task runner: log the union of sparse event metrics, not first-step keys."""
import torch
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils.logger import Logger


class QuietLandingLogger(Logger):
    def log(self, *args, **kwargs):
        iteration = kwargs.get("it", args[0] if args else 0)
        if self.writer is not None:
            keys = sorted({key for info in self.ep_extras for key in info})
            for key in keys:
                samples = [torch.as_tensor(info[key], device=self.device, dtype=torch.float32).reshape(-1)
                           for info in self.ep_extras if key in info]
                values = torch.cat(samples)
                finite = values[torch.isfinite(values)]
                if finite.numel():
                    self.writer.add_scalar(key, finite.mean(), iteration)
        # Parent handles PPO/train/performance cards. It must not re-log extras
        # using ep_extras[0], which silently misses conditional touchdown keys.
        self.ep_extras.clear()
        return super().log(*args, **kwargs)


class QuietLandingRunner(OnPolicyRunner):
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        super().__init__(env, train_cfg, log_dir, device)
        self.logger = QuietLandingLogger(
            log_dir=log_dir, cfg=self.cfg, env_cfg=self.env.cfg, num_envs=self.env.num_envs,
            is_distributed=self.is_distributed, gpu_world_size=self.gpu_world_size,
            gpu_global_rank=self.gpu_global_rank, device=self.device,
        )
