# Decimation (10회 반복) 설명

## 🤔 왜 Low-Level Environment를 10회 반복하나요?

### 1. 시간 스케일 분리 (Temporal Abstraction)

**High-Level과 Low-Level의 목표가 다릅니다:**

- **High-Level**: 장기 목표 (Navigation)
  - "목표 위치까지 어떻게 가야 할까?"
  - "어느 방향으로 이동해야 할까?"
  - 결정 주기: 느림 (약 0.05초 = 10 steps × 0.005초)

- **Low-Level**: 단기 제어 (Locomotion)
  - "지금 이 순간 균형을 잡기 위해 어떤 관절을 움직여야 할까?"
  - "속도를 맞추기 위해 어떻게 관절을 조절해야 할까?"
  - 결정 주기: 빠름 (0.005초 = 1 step)

**Decimation = 10의 의미:**
```
High-Level 1 step (0.05초) = Low-Level 10 steps (0.05초)
```

### 2. Action Frequency 제어

High-level이 너무 자주 결정을 바꾸면:

**❌ 문제점:**
- 로봇이 불안정해질 수 있음
- High-level policy가 의미 있는 행동을 할 시간이 부족
- Low-level policy가 새로운 velocity command에 적응할 시간이 부족

**✅ Decimation = 10의 장점:**
- High-level action이 충분한 시간 동안 유지됨
- Low-level policy가 velocity command를 안정적으로 추종할 시간 확보
- 로봇이 더 자연스럽고 안정적으로 움직임

### 3. 계산 효율성

```python
# Decimation = 10인 경우
High-Level Step: [계산] ─────────────────────> [계산]
                     │                           │
                     │ (10번 실행)              │ (10번 실행)
                     ▼                           ▼
Low-Level Steps: [1][2][3]...[10]          [11][12][13]...[20]
```

- High-level policy가 1번 계산되면 10번 사용됨
- 계산 비용 감소
- High-level policy 학습이 더 효율적

### 4. 물리 시뮬레이션 시간 스케일과의 조화

Low-level environment의 설정:
```python
# velocity_env_cfg.py
self.sim.dt = 0.005  # 물리 시뮬레이션 time step = 0.005초
self.decimation = 4  # Low-level의 decimation (관찰 업데이트 주기)
```

**실제 시간 스케일:**
- 물리 시뮬레이션: 0.005초/step (200 Hz)
- Low-level step: 0.005초 × 4 = 0.02초/step (50 Hz)
- High-level step: 0.005초 × 4 × 10 = 0.2초/step (5 Hz)

High-level은 **0.2초마다** velocity command를 새로 결정합니다.

### 5. 실제 동작 예시

**Decimation = 10인 경우:**

```
시간:  0.0s    0.05s   0.1s    0.15s   0.2s
       │       │       │       │       │
HL:    [v=1.0] ────────┼───────┼───> [v=0.8] ────> ...
       │               │       │       │
       │ (같은 명령)    │       │       │
       ▼               ▼       ▼       ▼
LL:    [1][2]...[10] [1][2]...[10] [1][2]...[10] [1][2]...[10]
```

High-level이 velocity command를 1.0에서 0.8로 바꾸면, 그 다음 10번의 low-level step 동안 0.8로 유지됩니다.

## 📊 Decimation 값 선택

### 일반적인 값들

- **Decimation = 1**: High-level과 low-level이 같은 주기
  - ❌ 너무 자주 바뀌어 불안정할 수 있음
  - ✅ 빠른 반응이 필요한 경우

- **Decimation = 4-10**: 일반적으로 사용되는 범위
  - ✅ 균형잡힌 선택
  - ✅ 대부분의 navigation 작업에 적합

- **Decimation = 20+**: 매우 느린 high-level 결정
  - ✅ 장거리 navigation에 적합
  - ❌ 빠른 반응이 필요한 경우 부적합

### 현재 구현에서 Decimation = 10인 이유

1. **Low-level environment의 특성**
   - Low-level도 내부적으로 decimation=4를 사용
   - High-level이 10배 느리게 결정하는 것이 적절

2. **Navigation 작업의 특성**
   - 목표까지 가는 것은 장기적인 작업
   - 0.2초마다 경로를 재계산하는 것이 충분

3. **실험적 검증**
   - 일반적으로 사용되는 값
   - 다른 hierarchical RL 연구에서도 유사한 값 사용

## 🔧 Decimation 값을 바꾸고 싶다면?

```python
# hierarchical_nav_env.py 또는 train_hierarchical_nav.py에서
hierarchical_env = HierarchicalNavEnv(
    env=low_env,
    frozen_policy_wrapper=frozen_policy,
    decimation=10,  # 이 값을 변경
)
```

**값 변경 시 고려사항:**
- 작게 (예: 5): High-level이 더 자주 결정 → 빠른 반응, 불안정할 수 있음
- 크게 (예: 20): High-level이 덜 자주 결정 → 느린 반응, 안정적일 수 있음

## ✅ 요약

**왜 10회 반복인가?**

1. ✅ **시간 스케일 분리**: High-level은 장기, Low-level은 단기
2. ✅ **안정성**: High-level action이 충분한 시간 동안 유지
3. ✅ **효율성**: High-level 계산 비용 감소
4. ✅ **물리적 의미**: 0.2초마다 경로 재계산 (실용적)
5. ✅ **실험적 검증**: 일반적으로 사용되는 값

**결론**: 10은 "마법의 숫자"가 아니라, navigation 작업에 적합한 균형잡힌 선택입니다! 🎯

