# 계층적 강화학습 (Hierarchical Reinforcement Learning) 구조

## 📐 전체 구조 개요

```
┌─────────────────────────────────────────────────────────────┐
│                    High-Level Policy                        │
│              (학습 대상 - 새로 학습하는 정책)                │
│                                                             │
│  Observation: 8D (로봇 위치, 목표 위치, 거리, 방향)         │
│  Action: 3D (vx, vy, vyaw) - 속도 명령                    │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      │ Velocity Command [vx, vy, vyaw]
                      ▼
┌─────────────────────────────────────────────────────────────┐
│         FrozenLocomotionPolicy (Frozen Policy)              │
│        (고정된 정책 - 이미 학습된 Low-Level Policy)          │
│                                                             │
│  Input: Velocity Command [vx, vy, vyaw]                    │
│  Output: Joint Actions [num_joints]                        │
│                                                             │
│  ※ 이 정책은 frozen 상태로, 학습되지 않음                   │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      │ Joint Actions
                      ▼
┌─────────────────────────────────────────────────────────────┐
│          Low-Level Environment                              │
│    (Isaac Lab - Rough-Deeprobotics-M20-v0)                  │
│                                                             │
│  - 물리 시뮬레이션                                          │
│  - 로봇 관절 제어                                          │
│  - 센서 데이터                                              │
└─────────────────────────────────────────────────────────────┘
```

## 🔄 데이터 흐름 (Step 과정)

### 1. High-Level Step

```python
# High-Level Policy가 velocity command 생성
high_level_action = [vx, vy, vyaw]  # 예: [1.0, 0.5, 0.2]
```

### 2. Frozen Policy 변환

```python
# FrozenLocomotionPolicy가 velocity command를 joint actions로 변환
joint_actions = frozen_policy(high_level_action)
# joint_actions: [num_envs, num_joints] 형태
```

### 3. Low-Level Environment 실행

```python
# Low-level environment를 decimation 횟수만큼 실행
for _ in range(decimation):  # 기본값: 10회
    obs, rewards, dones, extras = low_level_env.step(joint_actions)
```

### 4. High-Level Observation & Reward 계산

```python
# High-level observation 계산 (8D)
high_level_obs = [
    robot_pos_x,      # 로봇 x 위치
    robot_pos_y,      # 로봇 y 위치
    robot_yaw,        # 로봇 방향 (yaw)
    goal_pos_x,       # 목표 x 위치
    goal_pos_y,       # 목표 y 위치
    distance,         # 목표까지 거리
    direction_x,      # 목표 방향 (로봇 프레임 x)
    direction_y,      # 목표 방향 (로봇 프레임 y)
]

# High-level reward 계산
high_level_reward = goal_reaching_reward + 0.5 * progress_reward
```

## 📊 High-Level MDP (Markov Decision Process)

### Observation Space (8D)

```python
observation_space = Box(
    low=-inf,
    high=inf,
    shape=(8,),
    dtype=float32
)

# 구성 요소:
# [0:2]   robot_position_2d: 로봇의 x, y 위치
# [2]     robot_yaw: 로봇의 방향 (yaw 각도)
# [3:5]   goal_position_2d: 목표의 x, y 위치
# [5]     distance: 목표까지의 거리
# [6:8]   direction: 목표 방향 (로봇 프레임 기준)
```

### Action Space (3D)

```python
action_space = Box(
    low=[-2.0, -2.0, -2.0],
    high=[2.0, 2.0, 2.0],
    dtype=float32
)

# 구성 요소:
# [0] vx:   앞/뒤 속도 (m/s)
# [1] vy:   좌/우 속도 (m/s)
# [2] vyaw: 회전 속도 (rad/s)
```

### Reward Function

```python
# Goal Reaching Reward (Exponential Kernel)
goal_reward = exp(-distance / std^2)  # std = 0.5

# Progress Reward
progress = prev_distance - current_distance

# Total Reward
total_reward = goal_reward + 0.5 * progress
```

### Termination Conditions

```python
# 1. Goal Reached
goal_reached = distance < 0.5  # 0.5m 이내 도달

# 2. Low-Level Done
# Low-level environment의 termination (시간 초과, 낙상 등)

# High-Level Termination
terminated = goal_reached OR low_level_done
```

## ⏱️ 시간 스케일 (Decimation)

### Decimation = 10

- **High-Level Step**: 1 step
- **Low-Level Steps**: 10 steps
- **의미**: High-level policy가 1번의 결정을 내리면, 그 결정이 low-level에서 10번 실행됨

```
High-Level: [Step 1] ──────────────> [Step 2] ──────────────> [Step 3]
              │                        │                        │
              │ (decimation=10)        │ (decimation=10)        │
              ▼                        ▼                        ▼
Low-Level:  [1,2,3,...,10]         [11,12,13,...,20]       [21,22,23,...,30]
```

**이유**:
- High-level은 장기적인 목표(목표 도달)에 집중
- Low-level은 단기적인 제어(균형, 보행)에 집중
- Decimation을 통해 시간 스케일을 분리

## 🔐 Frozen Policy의 역할

### Frozen Policy란?

```python
# Low-level policy를 frozen (고정) 상태로 설정
policy_nn.eval()  # Evaluation 모드
for param in policy_nn.parameters():
    param.requires_grad = False  # Gradient 계산 비활성화
```

### 왜 Frozen인가?

1. **Low-level policy는 이미 학습 완료**: 로봇의 기본 보행 능력은 이미 학습됨
2. **High-level만 학습**: Navigation 전략만 학습하기 위해 low-level은 고정
3. **계산 효율성**: Low-level policy의 gradient 계산 불필요
4. **안정성**: Low-level policy가 변경되지 않아 학습이 더 안정적

### Frozen Policy의 동작

```python
class FrozenLocomotionPolicy:
    def __call__(self, velocity_command):
        # 1. Low-level environment에 velocity command 설정
        env.command_manager.set_command("base_velocity", velocity_command)
        
        # 2. Observation 가져오기
        obs = env.observation_manager.compute()
        
        # 3. Frozen policy로 joint actions 생성 (gradient 계산 없음)
        with torch.no_grad():
            joint_actions = self.inference_policy(obs)
        
        # 4. Original command 복원
        return joint_actions
```

## 🎯 학습 목표

### Low-Level (이미 학습 완료)
- **목표**: 주어진 velocity command에 따라 로봇을 움직이는 것
- **학습 완료**: Rough terrain에서 안정적으로 보행 가능

### High-Level (현재 학습 중)
- **목표**: 목표 위치까지 효율적으로 도달하는 navigation 전략 학습
- **입력**: 로봇 상태 + 목표 정보
- **출력**: Velocity command (vx, vy, vyaw)
- **Reward**: Goal reaching + Progress

## 📈 학습 과정

### 1. Low-Level Policy 로드

```python
# 사전 학습된 low-level policy 체크포인트 로드
low_level_checkpoint = "logs/rsl_rl/deeprobotics_m20_rough/.../model_19999.pt"
ppo_runner.load(low_level_checkpoint)
inference_policy = ppo_runner.get_inference_policy()
freeze_policy(ppo_runner)  # Frozen 상태로 설정
```

### 2. High-Level Environment 생성

```python
# Low-level environment 생성
low_env = gym.make("Rough-Deeprobotics-M20-v0")

# Frozen policy wrapper 생성
frozen_policy = FrozenLocomotionPolicy(inference_policy, low_env)

# High-level environment 생성
hierarchical_env = HierarchicalNavEnv(
    env=low_env,
    frozen_policy_wrapper=frozen_policy,
    decimation=10
)
```

### 3. High-Level Policy 학습

```python
# High-level policy 학습 (PPO)
runner = OnPolicyRunner(hierarchical_env, ...)
runner.learn(num_learning_iterations=20000)
```

## 🔑 주요 개념 정리

### 1. 계층 구조
- **High-Level**: 장기 목표 (Navigation)
- **Low-Level**: 단기 제어 (Locomotion)

### 2. Action Space 분리
- **High-Level**: Abstract actions (velocity commands)
- **Low-Level**: Primitive actions (joint torques)

### 3. Time Scale 분리
- **Decimation**: High-level 1 step = Low-level 10 steps

### 4. Policy 분리
- **High-Level Policy**: 학습 대상
- **Low-Level Policy**: Frozen (고정)

## 📝 요약

1. **High-Level Policy**가 **velocity command**를 생성
2. **Frozen Low-Level Policy**가 이를 **joint actions**로 변환
3. **Low-Level Environment**가 물리 시뮬레이션 실행 (10회 반복)
4. **High-Level**은 목표 도달에 대한 **reward**를 받고 학습

이 구조를 통해:
- ✅ 로봇의 기본 보행 능력은 유지하면서
- ✅ Navigation 전략만 별도로 학습 가능
- ✅ 더 빠르고 안정적인 학습 가능

