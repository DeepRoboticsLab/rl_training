# 볼륨 마운트 사용 가이드 (가장 편한 방법! ⭐⭐⭐)

## 🎯 왜 볼륨 마운트가 제일 편한가?

볼륨 마운트를 사용하면:
- ✅ 서버 호스트에 파일만 전송하면 됨
- ✅ 컨테이너 내부에서 자동으로 접근 가능
- ✅ 파일 수정 시 컨테이너 내부에서도 즉시 반영
- ✅ 권한 문제 없음
- ✅ docker cp 불필요

## 📋 사용 방법

### 1단계: 서버 호스트에 코드 전송

```bash
# 로컬에서 서버 호스트로 전송
bash docker/sync_all_code.sh dhk@30.30.99.10:/home/dhk/workspaces/rl_training
```

### 2단계: Docker 컨테이너 실행 시 볼륨 마운트

```bash
# 서버에서 Docker 컨테이너 실행
docker run -it --gpus all \
    -v /home/dhk/workspaces/rl_training:/workspaces/rl_training \
    <your_image> bash
```

또는 `docker-compose.yml` 사용:

```yaml
version: '3'
services:
  training:
    image: <your_image>
    volumes:
      - /home/dhk/workspaces/rl_training:/workspaces/rl_training
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

## 🔄 워크플로우

### 첫 전송 및 설정

```bash
# 1. 로컬에서 서버 호스트로 코드 전송
bash docker/sync_all_code.sh dhk@30.30.99.10:/home/dhk/workspaces/rl_training

# 2. 서버에서 컨테이너 실행 (볼륨 마운트 포함)
ssh dhk@30.30.99.10
docker run -it --gpus all \
    -v /home/dhk/workspaces/rl_training:/workspaces/rl_training \
    <image_name> bash

# 3. 컨테이너 내부에서 확인
cd /workspaces/rl_training
ls -la source/
```

### 코드 업데이트 시

```bash
# 로컬에서 서버 호스트로 코드 전송 (동일한 경로)
bash docker/sync_all_code.sh dhk@30.30.99.10:/home/dhk/workspaces/rl_training

# 컨테이너 내부에서 자동으로 새 코드 사용 가능!
# (컨테이너 재시작 불필요, 파일 변경만으로 반영)
```

## 📝 Docker 컨테이너 실행 예시

### 이미 실행 중인 컨테이너에 볼륨 추가

```bash
# 기존 컨테이너가 있다면, 새로 실행할 때 볼륨 마운트 추가
docker stop <container_name>
docker rm <container_name>
docker run -it --gpus all \
    -v /home/dhk/workspaces/rl_training:/workspaces/rl_training \
    --name <container_name> \
    <image_name> bash
```

### Docker Compose 사용 (권장)

```yaml
# docker-compose.yml
version: '3.8'
services:
  training:
    image: isaac-sim:5.1.0
    volumes:
      - /home/dhk/workspaces/rl_training:/workspaces/rl_training
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    stdin_open: true
    tty: true
```

```bash
# 실행
docker-compose up -d
docker-compose exec training bash
```

## ✅ 장점 요약

| 방법 | 장점 | 단점 |
|------|------|------|
| **볼륨 마운트** ⭐ | - 가장 편리<br>- 자동 동기화<br>- 권한 문제 없음 | - 컨테이너 실행 시 설정 필요 |
| docker cp | - 컨테이너 실행 후에도 가능 | - 매번 명령 실행 필요<br>- 번거로움 |
| 서버 호스트만 | - 간단 | - 컨테이너 접근 시 문제 가능 |

## 🎯 추천 워크플로우

1. **처음 한 번만**: Docker 컨테이너 실행 시 볼륨 마운트 설정
2. **코드 전송**: `sync_all_code.sh`로 서버 호스트에 전송
3. **자동 반영**: 컨테이너 내부에서 바로 사용 가능!

## 💡 팁

- 볼륨 마운트 경로는 컨테이너 내부 경로와 일치시키는 것이 좋습니다
- 예: `/home/dhk/workspaces/rl_training` (호스트) → `/workspaces/rl_training` (컨테이너)
- 컨테이너가 이미 실행 중이라면, 볼륨 마운트 없이 실행된 것이므로 새로 실행해야 합니다

