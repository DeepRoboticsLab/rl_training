# 빠른 시작 가이드

## 서버로 코드 전송하기

### 1단계: 서버 정보 확인

다음 정보를 준비하세요:
- **사용자명**: 서버 SSH 계정 (예: `ubuntu`, `root`, `dohyun`)
- **서버 주소**: IP 주소 또는 도메인 (예: `192.168.1.100` 또는 `server.company.com`)
- **대상 경로**: 서버의 rl_training 디렉토리 경로 (예: `/workspace/rl_training`)

### 2단계: SSH 접속 테스트

먼저 SSH 접속이 되는지 확인하세요:

```bash
ssh <사용자명>@<서버주소>
```

예시:
```bash
ssh ubuntu@192.168.1.100
# 또는
ssh dohyun@server.company.com
```

접속이 되면 `exit`로 나오세요.

### 3단계: 코드 전송

```bash
cd /home/dohyun/Documents/rl_training
bash docker/sync_all_code.sh <사용자명>@<서버주소>:<대상경로>
```

**실제 예시:**

```bash
# IP 주소 사용
bash docker/sync_all_code.sh ubuntu@192.168.1.100:/workspace/rl_training

# 도메인 사용
bash docker/sync_all_code.sh dohyun@server.company.com:/workspace/rl_training

# 다른 경로
bash docker/sync_all_code.sh user@server:/home/user/rl_training
```

### 4단계: 전송 확인

스크립트가 실행되면:
1. 전송할 파일 목록이 표시됩니다
2. `계속하시겠습니까? (y/N):` 질문에 `y` 입력
3. 파일 전송이 시작됩니다

### 5단계: 서버에서 확인

```bash
# 서버에 SSH 접속
ssh <사용자명>@<서버주소>

# 파일 확인
ls -la /workspace/rl_training/source/
ls -la /workspace/rl_training/scripts/reinforcement_learning/rsl_rl/train_hierarchical_nav.py
```

## 체크포인트 전송 (필요시)

학습에 필요한 low-level policy 체크포인트도 전송해야 합니다:

```bash
# 로컬에서
scp -r logs/rsl_rl/deeprobotics_m20_rough/2025-12-15_16-08-31/ \
    <사용자명>@<서버주소>:/workspace/rl_training/logs/rsl_rl/deeprobotics_m20_rough/
```

## 문제 해결

### SSH 접속이 안 되는 경우
- 서버 주소 확인
- 네트워크 연결 확인
- 방화벽 설정 확인

### 권한 문제
- 서버 경로에 쓰기 권한이 있는지 확인
- `sudo` 사용 필요할 수도 있음

### 파일이 전송되지 않는 경우
- 서버 경로가 올바른지 확인
- 디스크 공간 확인 (`df -h`)

