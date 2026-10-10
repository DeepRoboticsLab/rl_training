# M20 Quiet Landing V1.5

V1.5 合并连续力峰尾部和稳定四足支撑负载平衡。独立任务 `Rough-Deeprobotics-M20-QuietLanding-V1_5-v0`，使用现有 `train.py`，不新增训练入口。V1 保留；V1.1/V1.2 的独立任务注册、env_config 和 ppo_config 类已移除，历史 checkpoint、训练日志与评测记录保留。

Actor 57+187=244D、Critic285D、Action16D 和固定 PD 保持。接触、力及平衡信号只用于奖励和诊断，不加入策略观测。环境不继承 pre_teacher/stair_teacher；任务通过能力奖励采用独立快照。V2 留作奖励课程学习拓展，本次没有启用权重课程。

## 融合奖励

触地后40ms窗口内法向力峰值 F，x=max(F-100,0)/200，k=2。V1 的 min(x²,4) 替换为 x²（x≤k），以及 k²+2k(x-k)（x>k）。权重沿用 -0.005，仅在峰值增长时增量计费，跨控制步保留窗口。这样严重冲击也有继续减小的梯度，不重复惩罚同一峰值。

四轮顺序 FL/FR/HL/HR，法向力使用0.5s时间常数的指数均值。D=(FL+HR-FR-HL)/(FL+FR+HL+HR)，平衡代价 min((max(|D|-0.05,0)/0.2)²,4)，权重 -0.05，按有效支撑时间积分。这针对持续对角负载偏差，不要求动态步态每个瞬间四轮受力相同。

平衡仅在有效高程图高度差≤0.025m、四轮支撑且每轮法向力>5N、总法向力≥100N、没有新触地、|机身竖直速度|≤0.1m/s、机身角速度模长≤0.3rad/s、|偏航指令|<0.1rad/s、机身重力投影z<-0.98，且持续满足0.15s时启用。任一条件失效即清空稳定统计并重新等待。因此楼梯、摆腿、转弯等阶段不会被强制平均受力。

V1 的触地法向速度、窗口超额冲量、力加载率、腿轮 Action Rate、关节加速度、轮加加速度、机身角/竖直加速度、扭矩变化率、轮滑移、摆腿净空、身体碰撞和任务奖励沿用。两项融合后的长期效果尚未验证，不能根据冒烟检查宣称更安静或通过率改善。

## 训练

依赖 Isaac Lab、Isaac Sim 及 rsl-rl-lib==5.0.1；CPU检查另需 torch、numpy、mujoco。M20模型使用与baseline一致的资产；换机器时设置 `RL_TRAINING_M20_USD_PATH=/absolute/path/M20.usd`。Checkpoint 不上传代码仓库，需要自行准备。

从仓库根目录运行。先核验资源；4096环境是之前V1运行规模，不是本轮测得的V1.5资源峰值。新训练从V1初始化Actor/Critic/参考策略，优化器与迭代计数重新开始：

```bash
PYTHON=/home/ubuntu/miniconda3/envs/m20_wzh/bin/python
V1_CHECKPOINT=logs/rsl_rl/deeprobotics_m20_quiet_landing_v1/2026-10-10_00-48-58_quiet_landing_v1/model_230300.pt
"$PYTHON" scripts/reinforcement_learning/rsl_rl/train.py \
  --task Rough-Deeprobotics-M20-QuietLanding-V1_5-v0 \
  --headless --device cuda:0 --num_envs 4096 --seed 42 --max_iterations 10000 \
  --init_actor_from "$V1_CHECKPOINT" --init_critic_from "$V1_CHECKPOINT" \
  --preserve_actor_from "$V1_CHECKPOINT"
```

继续V1.5训练用 `--resume --checkpoint /absolute/path/model_x.pt`，去掉上述三项初始化参数，恢复优化器、参考策略及预热计数。`--max_iterations` 是追加更新数。

## 诊断与验证边界

TensorBoard 使用 `QuietReward/*`、`QuietLanding/*`、`QuietLoad/*`；不带入此前步态调试cards。稳定支撑样本为空时，只记录覆盖率/has_samples，不以零值伪装负载平衡。MuJoCo共享采样模块 `deploy/deploy_mujoco/quiet_landing_metrics.py` 仍用于速度、比动能/轮质量平动能及实际接触力诊断，其默认扩展奖励参数是V1，不能把它当作V1.5平衡门控复现。

CPU反例检查：
```bash
"$PYTHON" scripts/tools/check_quiet_landing.py
"$PYTHON" scripts/tools/check_quiet_extended.py
```

当前工作区已通过两项CPU检查和8环境、2次PPO更新：第1次Critic预热、第2次Actor更新，保存checkpoint；组合反例验证尾部惩罚与负载平衡同时存在、新触地会关闭平衡、reset清空状态。0.6s短回合不能证明正式训练会获得足够稳定支撑样本。没有进行正式V1.5长训练或all_terrain效果验收。

发布采用从最新origin/main创建的隔离worktree，仅提交quiet任务及必要依赖，保留上游AMP训练分支；不带入当前工作区其他机器人删除或teacher/student改动。复审由同family Codex完成，未进行跨family复审。

发布快照同样通过CPU检查及原train.py的8环境、2次PPO更新，确认上游保留AMP代码后仍可启动本任务。工作区checkpoint检查：Actor首层244/输出16、Critic首层285；Critic预热后Actor与V1差为0，下一次Actor更新最大权重差0.0001887083，参考Actor仍由V1初始化。TensorBoard未出现gait/ascent/descent调试卡；短程稳定支撑覆盖率为0，因此本次不证明负载平衡在真实长回合中有效。
