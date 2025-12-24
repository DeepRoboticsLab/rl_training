#!/bin/bash
# 회사 서버로 프로젝트 파일 동기화 스크립트
#
# 사용법:
#   bash docker/sync_to_server.sh <server_user>@<server_host>:<remote_path>
#
# 예시:
#   bash docker/sync_to_server.sh user@server.company.com:/workspace/rl_training
#   bash docker/sync_to_server.sh user@192.168.1.100:/home/user/rl_training

set -e

if [ $# -lt 1 ]; then
    echo "Error: Server destination is required"
    echo "Usage: bash docker/sync_to_server.sh <user>@<host>:<remote_path>"
    echo ""
    echo "예시:"
    echo "  bash docker/sync_to_server.sh user@server.company.com:/workspace/rl_training"
    exit 1
fi

DESTINATION="$1"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "========================================="
echo "회사 서버로 파일 동기화"
echo "========================================="
echo "소스: ${PROJECT_DIR}"
echo "대상: ${DESTINATION}"
echo "========================================="
echo ""

# rsync 옵션
RSYNC_OPTS=(
    -avz                    # archive, verbose, compress
    --progress              # 진행 상황 표시
    --exclude='.git'        # Git 저장소 제외 (이미 서버에 있을 경우)
    --exclude='__pycache__'
    --exclude='*.pyc'
    --exclude='*.pyo'
    --exclude='.pytest_cache'
    --exclude='**/*.egg-info'
    --exclude='.DS_Store'
    --exclude='*.dmp'
    --exclude='**/.thumbs'
    --exclude='**/.idea'
    --exclude='**/.vscode'
    --exclude='logs/**'     # 로그 파일 제외 (필요하면 별도로)
    --exclude='outputs/**'  # 출력 파일 제외 (필요하면 별도로)
    --exclude='wandb/**'
    --exclude='.neptune/**'
    --exclude='**/runs/**'
    --exclude='**/recordings/**'
    --exclude='**/output/**'
    --exclude='**/videos/**'
    --exclude='_isaac_sim*'
    --exclude='_repo'
    --exclude='_build'
    --exclude='.lastformat'
    --exclude='**/usd/*'
    --exclude='*.tmp'
)

# 체크포인트 파일도 전송할지 확인
read -p "체크포인트 파일(logs/)도 전송하시겠습니까? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "✅ 체크포인트 파일 포함하여 전송합니다"
    # logs/ 제외 항목 제거
    RSYNC_OPTS=(${RSYNC_OPTS[@]//--exclude='logs\/**'//})
else
    echo "⏭️  체크포인트 파일 제외하고 전송합니다"
fi

echo ""
echo "전송 시작..."
echo ""

# rsync 실행
rsync "${RSYNC_OPTS[@]}" "${PROJECT_DIR}/" "${DESTINATION}/"

echo ""
echo "========================================="
echo "✅ 파일 동기화 완료!"
echo "========================================="
echo ""
echo "서버에서 확인 사항:"
echo "  1. 파일이 올바르게 전송되었는지 확인:"
echo "     ssh ${DESTINATION%%:*} 'ls -la ${DESTINATION##*:}'"
echo ""
echo "  2. Docker 컨테이너에서 확인:"
echo "     docker exec -it <container_name> ls /workspace/rl_training"
echo ""

