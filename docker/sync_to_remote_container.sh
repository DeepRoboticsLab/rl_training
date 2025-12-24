#!/bin/bash
# 로컬 컴퓨터에서 서버의 Docker 컨테이너로 직접 코드 전송
#
# 사용법:
#   bash docker/sync_to_remote_container.sh <user>@<server> <container_name> [container_path]
#
# 예시:
#   bash docker/sync_to_remote_container.sh dhk@30.30.99.10 my_container
#   bash docker/sync_to_remote_container.sh dhk@30.30.99.10 my_container /workspaces/rl_training

set -e

if [ $# -lt 2 ]; then
    echo "Error: Server and container name are required"
    echo "Usage: bash docker/sync_to_remote_container.sh <user>@<server> <container_name> [container_path]"
    echo ""
    echo "예시:"
    echo "  bash docker/sync_to_remote_container.sh dhk@30.30.99.10 my_container"
    echo "  bash docker/sync_to_remote_container.sh dhk@30.30.99.10 my_container /workspaces/rl_training"
    exit 1
fi

SERVER="$1"
CONTAINER_NAME="$2"
CONTAINER_PATH=${3:-"/workspaces/rl_training"}
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "========================================="
echo "서버의 Docker 컨테이너로 코드 전송"
echo "========================================="
echo "서버: ${SERVER}"
echo "컨테이너: ${CONTAINER_NAME}"
echo "컨테이너 경로: ${CONTAINER_PATH}"
echo "소스: ${PROJECT_DIR}"
echo "========================================="
echo ""

# 서버에서 컨테이너가 실행 중인지 확인
echo "서버의 컨테이너 확인 중..."
if ! ssh "${SERVER}" "docker ps --format '{{.Names}}' | grep -q '^${CONTAINER_NAME}$'"; then
    echo "❌ 오류: 서버의 컨테이너 '${CONTAINER_NAME}'가 실행 중이 아닙니다"
    echo ""
    echo "서버의 실행 중인 컨테이너:"
    ssh "${SERVER}" "docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}'"
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
ssh "${SERVER}" "docker exec ${CONTAINER_NAME} mkdir -p ${CONTAINER_PATH}" || {
    echo "❌ 오류: 컨테이너 내부 디렉토리 생성 실패"
    exit 1
}
echo "✅ 디렉토리 준비 완료"
echo ""

# 임시 디렉토리 생성 (로컬)
TEMP_DIR=$(mktemp -d)
trap "rm -rf ${TEMP_DIR}" EXIT

echo "임시 압축 파일 생성 중..."

# 각 항목을 임시 디렉토리로 복사
for item in "${SYNC_ITEMS[@]}"; do
    if [ -e "${PROJECT_DIR}/${item}" ]; then
        if [ -d "${PROJECT_DIR}/${item}" ]; then
            mkdir -p "${TEMP_DIR}/$(dirname ${item})"
            cp -r "${PROJECT_DIR}/${item}" "${TEMP_DIR}/${item}"
        else
            mkdir -p "${TEMP_DIR}/$(dirname ${item})"
            cp "${PROJECT_DIR}/${item}" "${TEMP_DIR}/${item}"
        fi
    fi
done

# tar로 압축
TAR_FILE="${TEMP_DIR}/rl_training_code.tar.gz"
cd "${TEMP_DIR}"
tar -czf "${TAR_FILE}" .
cd - > /dev/null

echo "✅ 압축 완료"
echo ""

# 서버의 임시 디렉토리에 전송
SERVER_TEMP_DIR="/tmp/rl_training_sync_$$"
echo "서버 임시 디렉토리로 전송 중..."
ssh "${SERVER}" "mkdir -p ${SERVER_TEMP_DIR}"
scp "${TAR_FILE}" "${SERVER}:${SERVER_TEMP_DIR}/rl_training_code.tar.gz"

echo "✅ 서버로 전송 완료"
echo ""

# 서버에서 압축 해제 후 컨테이너로 복사
echo "컨테이너로 복사 중..."
ssh "${SERVER}" <<REMOTE_SCRIPT
    set -e
    cd ${SERVER_TEMP_DIR}
    tar -xzf rl_training_code.tar.gz
    
    # 각 항목을 컨테이너로 복사
    for item in source scripts deep_robotics_model docker docs pyproject.toml README.md LICENSE LICENSE-robot_lab VERSION CONTRIBUTORS.md; do
        if [ -e "\${item}" ]; then
            if [ -d "\${item}" ]; then
                docker exec ${CONTAINER_NAME} mkdir -p ${CONTAINER_PATH}/\$(dirname \${item}) 2>/dev/null || true
                docker cp "\${item}/." ${CONTAINER_NAME}:${CONTAINER_PATH}/\${item}/ 2>/dev/null || true
                echo "  📁 복사 완료: \${item}/"
            else
                docker exec ${CONTAINER_NAME} mkdir -p ${CONTAINER_PATH}/\$(dirname \${item}) 2>/dev/null || true
                docker cp "\${item}" ${CONTAINER_NAME}:${CONTAINER_PATH}/\${item} 2>/dev/null || true
                echo "  📄 복사 완료: \${item}"
            fi
        fi
    done
    
    # 임시 파일 정리
    rm -rf ${SERVER_TEMP_DIR}
REMOTE_SCRIPT

echo ""
echo "========================================="
echo "✅ 파일 전송 완료!"
echo "========================================="
echo ""
echo "컨테이너 내부에서 확인:"
echo "  ssh ${SERVER} 'docker exec -it ${CONTAINER_NAME} ls -la ${CONTAINER_PATH}/source/'"
echo ""
echo "의존성 설치:"
echo "  ssh ${SERVER} 'docker exec -it ${CONTAINER_NAME} bash -c \"cd ${CONTAINER_PATH} && python -m pip install -e source/rl_training\"'"
echo ""
