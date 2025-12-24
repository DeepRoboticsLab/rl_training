# Docker 관련 스크립트

서버로 코드 전송을 위한 스크립트입니다.

## 📤 서버/컨테이너 전송 스크립트

### `sync_to_remote_container.sh` (서버의 Docker 컨테이너 직접 전송 ⭐⭐⭐)
로컬 컴퓨터에서 서버의 Docker 컨테이너 내부로 직접 코드를 전송합니다. 권한 문제 없이 컨테이너에 바로 전송 가능합니다.

```bash
# 사용법
bash docker/sync_to_remote_container.sh <user>@<server> <container_name> [container_path]

# 예시
bash docker/sync_to_remote_container.sh dhk@30.30.99.10 my_container
bash docker/sync_to_remote_container.sh dhk@30.30.99.10 my_container /workspaces/rl_training
```

### `sync_to_container.sh` (로컬 Docker 컨테이너 전송)
로컬의 Docker 컨테이너 내부로 직접 코드를 전송합니다.

```bash
# 사용법
bash docker/sync_to_container.sh <container_name> [container_path]

# 예시
bash docker/sync_to_container.sh my_container /workspaces/rl_training
```

### `sync_all_code.sh` (서버 호스트 전송)
모든 코드를 서버 호스트로 전송합니다. 나중에 코드를 추가해도 이 스크립트 하나로 모든 변경사항을 전송할 수 있습니다.

```bash
# 사용법
bash docker/sync_all_code.sh <user>@<server>:<remote_path>

# 예시
bash docker/sync_all_code.sh ubuntu@192.168.1.100:/workspace/rl_training
```

**전송하는 항목:**
- ✅ 모든 소스 코드 (`source/`)
- ✅ 모든 스크립트 (`scripts/`)
- ✅ 로봇 모델 파일 (`deep_robotics_model/`)
- ✅ Docker 스크립트 (`docker/`)
- ✅ 설정 파일들 (`pyproject.toml`, `README.md` 등)

**제외되는 항목:**
- ❌ `logs/` (체크포인트는 별도 전송 필요)
- ❌ `outputs/`
- ❌ 캐시 파일들 (`__pycache__/`, `*.pyc` 등)

### `sync_hierarchical_nav.sh`
Hierarchical navigation 코드만 전송합니다 (특정 코드만 업데이트할 때 사용).

```bash
bash docker/sync_hierarchical_nav.sh <user>@<server>:<remote_path>
```

### `sync_to_server.sh`
전체 프로젝트를 서버로 동기화합니다 (기존 스크립트, `sync_all_code.sh` 사용 권장).

```bash
bash docker/sync_to_server.sh <user>@<server>:<remote_path>
```

## 📥 서버에서 모델 다운로드 스크립트

### `download_checkpoints.sh`
서버에서 학습된 체크포인트/모델을 로컬로 다운로드합니다.

```bash
# 사용법
bash docker/download_checkpoints.sh <user>@<server>:<remote_path> [local_path]

# 예시: 전체 logs 디렉토리 다운로드
bash docker/download_checkpoints.sh ubuntu@192.168.1.100:/workspace/rl_training/logs

# 예시: 특정 실험 결과만 다운로드
bash docker/download_checkpoints.sh user@server:/workspace/rl_training/logs/rsl_rl/hierarchical_nav/2025-12-24_10-00-00 ./checkpoints
```

### `download_latest_checkpoint.sh`
서버에서 가장 최신 체크포인트만 빠르게 다운로드합니다.

```bash
# 사용법
bash docker/download_latest_checkpoint.sh <user>@<server>:<experiment_path> [local_path]

# 예시: hierarchical_nav 실험의 최신 체크포인트
bash docker/download_latest_checkpoint.sh ubuntu@192.168.1.100:/workspace/rl_training/logs/rsl_rl/hierarchical_nav
```

## 🐳 Docker 컨테이너로 코드 전송

### ⭐⭐⭐ 볼륨 마운트 사용 (가장 권장!)

볼륨 마운트를 사용하면 서버 호스트에 파일만 전송하면 컨테이너 내부에서 자동으로 접근 가능합니다.

1. 서버 호스트로 코드 전송:
   ```bash
   bash docker/sync_all_code.sh dhk@30.30.99.10:/home/dhk/workspaces/rl_training
   ```

2. Docker 컨테이너 실행 시 볼륨 마운트:
   ```bash
   docker run -it --gpus all \
       -v /home/dhk/workspaces/rl_training:/workspaces/rl_training \
       <image_name> bash
   ```

자세한 내용은 `docker/VOLUME_MOUNT_GUIDE.md` 참고

### 직접 전송 (볼륨 마운트 없이)

볼륨 마운트 문제가 있거나 컨테이너에 직접 전송하고 싶다면:

```bash
# 1. 실행 중인 컨테이너 이름 확인
docker ps

# 2. 컨테이너로 직접 전송
bash docker/sync_to_container.sh <container_name> /workspaces/rl_training
```

## 📝 SSH 접속 정보 (서버 호스트 전송 시)

`user@server` 형식:
- `user`: SSH 로그인 사용자명 (예: `ubuntu`, `root`, `dohyun`)
- `server`: 서버 주소 (IP 또는 도메인, 예: `192.168.1.100` 또는 `server.company.com`)

## ✅ 전송 후 확인

```bash
# 서버에서 파일 확인
ssh user@server 'ls -la /workspace/rl_training/source/rl_training/rl_training/tasks/manager_based/locomotion/hierarchical_nav/'

# Docker 컨테이너 내부에서 확인
docker exec -it <container_name> ls /workspace/rl_training/source/rl_training/rl_training/tasks/manager_based/locomotion/hierarchical_nav/

# 의존성 설치 (컨테이너 내부에서)
docker exec -it <container_name> bash -c 'cd /workspace/rl_training && python -m pip install -e source/rl_training'
```

