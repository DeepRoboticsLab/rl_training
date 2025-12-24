#!/bin/bash
# Docker 컨테이너로 직접 코드를 전송하는 스크립트
#
# 사용법:
#   bash docker/sync_to_container.sh <container_name> [container_path]
#
# 예시:
#   bash docker/sync_to_container.sh my_container
#   bash docker/sync_to_container.sh my_container /workspaces/rl_training

set -e

if [ $# -lt 1 ]; then
    echo "Error: Container name is required"
    echo "Usage: bash docker/sync_to_container.sh <container_name> [container_path]"
    echo ""
    echo "예시:"
    echo "  bash docker/sync_to_container.sh my_container"
    echo "  bash docker/sync_to_container.sh my_container /workspaces/rl_training"
    echo ""
    echo "실행 중인 컨테이너 목록:"
    docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}"
    exit 1
fi

CONTAINER_NAME="$1"
CONTAINER_PATH=${2:-"/workspaces/rl_training"}
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "========================================="
echo "Docker 컨테이너로 코드 전송"
echo "========================================="
echo "컨테이너: ${CONTAINER_NAME}"
echo "대상 경로: ${CONTAINER_PATH}"
echo "소스: ${PROJECT_DIR}"
echo "========================================="
echo ""

# 컨테이너가 실행 중인지 확인
if ! docker ps --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
    echo "❌ 오류: 컨테이너 '${CONTAINER_NAME}'가 실행 중이 아닙니다"
    echo ""
    echo "실행 중인 컨테이너 목록:"
    docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}"
    exit 1
fi

echo "✅ 컨테이너 실행 중"
echo ""

# 전송할 파일/디렉토리 목록
SYNC_ITEMS=(
    "source/"
    "scripts/"
    "deep_robotics_model/"
    "docker/"
    "docs/"
    "pyproject.toml"
    "README.md"
    "LICENSE"
    "LICENSE-robot_lab"
    "VERSION"
    "CONTRIBUTORS.md"
)

echo "전송할 항목:"
for item in "${SYNC_ITEMS[@]}"; do
    if [ -e "${PROJECT_DIR}/${item}" ]; then
        if [ -d "${PROJECT_DIR}/${item}" ]; then
            echo "  ✅ ${item}/ (디렉토리)"
        else
            echo "  ✅ ${item} (파일)"
        fi
    else
        echo "  ⚠️  ${item} (존재하지 않음)"
    fi
done
echo ""

# 전송 확인
read -p "계속하시겠습니까? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "취소되었습니다."
    exit 0
fi

echo ""
echo "전송 시작..."
echo ""

# 컨테이너 내부에 대상 디렉토리 생성
echo "컨테이너 내부 디렉토리 생성 중..."
docker exec "${CONTAINER_NAME}" mkdir -p "${CONTAINER_PATH}" || {
    echo "❌ 오류: 컨테이너 내부 디렉토리 생성 실패"
    exit 1
}
echo "✅ 디렉토리 준비 완료"
echo ""

# 각 항목 전송
for item in "${SYNC_ITEMS[@]}"; do
    if [ ! -e "${PROJECT_DIR}/${item}" ]; then
        echo "⚠️  건너뜀: ${item} (파일/디렉토리 없음)"
        continue
    fi
    
    SOURCE="${PROJECT_DIR}/${item}"
    TARGET="${CONTAINER_NAME}:${CONTAINER_PATH}/${item}"
    
    echo "📦 전송: ${item}"
    
    # docker cp 사용
    if [ -d "$SOURCE" ]; then
        # 디렉토리: 컨테이너 내부에 디렉토리 생성 후 복사
        docker exec "${CONTAINER_NAME}" mkdir -p "${CONTAINER_PATH}/$(dirname ${item})" 2>/dev/null || true
        docker cp "${SOURCE}" "${TARGET}" 2>/dev/null || docker cp "${SOURCE}/." "${TARGET}/" 2>/dev/null || {
            echo "  ⚠️  디렉토리 전송 실패 (무시하고 계속)"
        }
    else
        # 파일: 부모 디렉토리 생성 후 파일 복사
        PARENT_DIR="$(dirname ${item})"
        if [ "$PARENT_DIR" != "." ]; then
            docker exec "${CONTAINER_NAME}" mkdir -p "${CONTAINER_PATH}/${PARENT_DIR}" 2>/dev/null || true
        fi
        docker cp "${SOURCE}" "${TARGET}" || {
            echo "  ⚠️  파일 전송 실패 (무시하고 계속)"
        }
    fi
    echo ""
done

echo "========================================="
echo "✅ 파일 전송 완료!"
echo "========================================="
echo ""
echo "컨테이너 내부에서 확인:"
echo "  docker exec -it ${CONTAINER_NAME} ls -la ${CONTAINER_PATH}/source/"
echo ""
echo "의존성 설치:"
echo "  docker exec -it ${CONTAINER_NAME} bash -c 'cd ${CONTAINER_PATH} && python -m pip install -e source/rl_training'"
echo ""

