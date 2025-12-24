#!/bin/bash
# 서버에서 학습된 체크포인트/모델을 로컬로 다운로드하는 스크립트
#
# 사용법:
#   bash docker/download_checkpoints.sh <user>@<server>:<remote_path> [local_path]
#
# 예시:
#   bash docker/download_checkpoints.sh ubuntu@192.168.1.100:/workspace/rl_training/logs
#   bash docker/download_checkpoints.sh user@server:/workspace/rl_training/logs ./downloaded_logs
#   bash docker/download_checkpoints.sh user@server:/workspace/rl_training/logs/rsl_rl/hierarchical_nav/2025-12-24_10-00-00 ./checkpoints

set -e

if [ $# -lt 1 ]; then
    echo "Error: Server source path is required"
    echo "Usage: bash docker/download_checkpoints.sh <user>@<server>:<remote_path> [local_path]"
    echo ""
    echo "예시:"
    echo "  # 전체 logs 디렉토리 다운로드"
    echo "  bash docker/download_checkpoints.sh ubuntu@192.168.1.100:/workspace/rl_training/logs"
    echo ""
    echo "  # 특정 실험 결과만 다운로드"
    echo "  bash docker/download_checkpoints.sh user@server:/workspace/rl_training/logs/rsl_rl/hierarchical_nav/2025-12-24_10-00-00 ./checkpoints"
    echo ""
    echo "  # 최신 체크포인트만 다운로드"
    echo "  bash docker/download_checkpoints.sh user@server:/workspace/rl_training/logs/rsl_rl/hierarchical_nav ./latest_checkpoints"
    exit 1
fi

SOURCE="$1"
LOCAL_DEST=${2:-"./downloaded_logs"}

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FULL_LOCAL_DEST="${PROJECT_DIR}/${LOCAL_DEST}"

echo "========================================="
echo "서버에서 체크포인트 다운로드"
echo "========================================="
echo "소스: ${SOURCE}"
echo "대상: ${FULL_LOCAL_DEST}"
echo "========================================="
echo ""

# rsync 옵션
RSYNC_OPTS=(
    -avz                    # archive, verbose, compress
    --progress              # 진행 상황 표시
    --include='*/'          # 디렉토리 포함
    --include='*.pt'        # 체크포인트 파일 (*.pt)
    --include='*.pth'       # PyTorch 모델 파일
    --include='*.ckpt'      # 체크포인트 파일
    --include='*.yaml'      # 설정 파일
    --include='*.yml'       # 설정 파일
    --include='*.json'      # JSON 파일 (로그 등)
    --include='*.csv'       # CSV 파일 (메트릭 등)
    --include='*.txt'       # 텍스트 파일 (로그 등)
    --exclude='*'           # 나머지 제외
)

# 또는 모든 파일 다운로드 옵션
DOWNLOAD_ALL=false

read -p "모든 파일을 다운로드하시겠습니까? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    DOWNLOAD_ALL=true
    RSYNC_OPTS=(
        -avz
        --progress
        --exclude='__pycache__'
        --exclude='*.pyc'
        --exclude='*.pyo'
        --exclude='.DS_Store'
    )
fi

# 로컬 디렉토리 생성
mkdir -p "${FULL_LOCAL_DEST}"

echo ""
if [ "$DOWNLOAD_ALL" = true ]; then
    echo "모든 파일 다운로드 모드"
else
    echo "체크포인트 및 로그 파일만 다운로드 모드"
    echo "  포함: *.pt, *.pth, *.ckpt, *.yaml, *.yml, *.json, *.csv, *.txt"
fi
echo ""

echo "다운로드 시작..."
echo ""

# rsync 실행
rsync "${RSYNC_OPTS[@]}" "${SOURCE}/" "${FULL_LOCAL_DEST}/"

echo ""
echo "========================================="
echo "✅ 다운로드 완료!"
echo "========================================="
echo ""
echo "다운로드된 위치: ${FULL_LOCAL_DEST}"
echo ""

# 다운로드된 파일 확인
if [ "$DOWNLOAD_ALL" = false ]; then
    echo "다운로드된 체크포인트 파일:"
    find "${FULL_LOCAL_DEST}" -name "*.pt" -o -name "*.pth" -o -name "*.ckpt" | head -20
    echo ""
fi

echo "파일 목록:"
ls -lh "${FULL_LOCAL_DEST}" | head -20
echo ""

