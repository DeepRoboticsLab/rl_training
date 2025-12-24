#!/bin/bash
# 서버에서 가장 최신 체크포인트만 다운로드하는 스크립트
#
# 사용법:
#   bash docker/download_latest_checkpoint.sh <user>@<server>:<experiment_path> [local_path]
#
# 예시:
#   # hierarchical_nav 실험의 최신 체크포인트
#   bash docker/download_latest_checkpoint.sh ubuntu@192.168.1.100:/workspace/rl_training/logs/rsl_rl/hierarchical_nav
#
#   # 모든 실험 중 최신 체크포인트 찾기
#   bash docker/download_latest_checkpoint.sh ubuntu@192.168.1.100:/workspace/rl_training/logs/rsl_rl

set -e

if [ $# -lt 1 ]; then
    echo "Error: Server experiment path is required"
    echo "Usage: bash docker/download_latest_checkpoint.sh <user>@<server>:<experiment_path> [local_path]"
    echo ""
    echo "예시:"
    echo "  bash docker/download_latest_checkpoint.sh ubuntu@192.168.1.100:/workspace/rl_training/logs/rsl_rl/hierarchical_nav"
    exit 1
fi

SOURCE="$1"
LOCAL_DEST=${2:-"./latest_checkpoint"}

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FULL_LOCAL_DEST="${PROJECT_DIR}/${LOCAL_DEST}"

echo "========================================="
echo "서버에서 최신 체크포인트 다운로드"
echo "========================================="
echo "소스: ${SOURCE}"
echo "대상: ${FULL_LOCAL_DEST}"
echo "========================================="
echo ""

# 임시 디렉토리 생성
TEMP_DIR=$(mktemp -d)
trap "rm -rf ${TEMP_DIR}" EXIT

echo "서버에서 최신 체크포인트 찾는 중..."

# 서버에서 최신 체크포인트 찾기
LATEST_CHECKPOINT=$(ssh "${SOURCE%%:*}" "find ${SOURCE##*:} -name 'model_*.pt' -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-")

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "❌ 오류: 체크포인트 파일을 찾을 수 없습니다."
    echo "   경로를 확인해주세요: ${SOURCE##*:}"
    exit 1
fi

echo "✅ 최신 체크포인트 발견: ${LATEST_CHECKPOINT}"
echo ""

# 체크포인트 파일명 추출
CHECKPOINT_NAME=$(basename "$LATEST_CHECKPOINT")
CHECKPOINT_DIR=$(dirname "$LATEST_CHECKPOINT")

# 관련 파일들도 함께 다운로드 (설정 파일 등)
echo "체크포인트 및 관련 파일 다운로드 중..."
echo ""

# 로컬 디렉토리 생성
mkdir -p "${FULL_LOCAL_DEST}"

# 체크포인트 파일 및 설정 파일 다운로드
rsync -avz --progress \
    "${SOURCE%%:*}:${CHECKPOINT_DIR}/model_*.pt" \
    "${SOURCE%%:*}:${CHECKPOINT_DIR}/*.yaml" \
    "${SOURCE%%:*}:${CHECKPOINT_DIR}/*.yml" \
    "${SOURCE%%:*}:${CHECKPOINT_DIR}/*.json" \
    "${FULL_LOCAL_DEST}/" 2>/dev/null || true

# 체크포인트 파일 확인
if [ -f "${FULL_LOCAL_DEST}/${CHECKPOINT_NAME}" ]; then
    FILE_SIZE=$(du -h "${FULL_LOCAL_DEST}/${CHECKPOINT_NAME}" | cut -f1)
    echo ""
    echo "========================================="
    echo "✅ 다운로드 완료!"
    echo "========================================="
    echo ""
    echo "체크포인트 파일: ${FULL_LOCAL_DEST}/${CHECKPOINT_NAME}"
    echo "파일 크기: ${FILE_SIZE}"
    echo ""
    echo "다운로드된 파일들:"
    ls -lh "${FULL_LOCAL_DEST}"
    echo ""
else
    echo "❌ 오류: 체크포인트 파일 다운로드 실패"
    exit 1
fi

