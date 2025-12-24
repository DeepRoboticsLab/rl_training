#!/bin/bash
# 서버 디렉토리 생성 스크립트
# 사용법: bash docker/fix_server_dir.sh <user>@<server>:<remote_path>

if [ $# -lt 1 ]; then
    echo "Usage: bash docker/fix_server_dir.sh <user>@<server>:<remote_path>"
    echo "예시: bash docker/fix_server_dir.sh dhk@30.30.99.10:/workspaces/rl_training"
    exit 1
fi

DESTINATION="$1"
SERVER_USER_HOST="${DESTINATION%%:*}"
SERVER_PATH="${DESTINATION##*:}"

echo "서버 디렉토리 생성 중..."
echo "서버: ${SERVER_USER_HOST}"
echo "경로: ${SERVER_PATH}"
echo ""

ssh "${SERVER_USER_HOST}" "mkdir -p ${SERVER_PATH}" && {
    echo "✅ 디렉토리 생성 완료!"
    echo ""
    echo "이제 파일 전송을 다시 시도하세요:"
    echo "bash docker/sync_all_code.sh ${DESTINATION}"
} || {
    echo "❌ 디렉토리 생성 실패"
    echo "수동으로 생성해주세요:"
    echo "ssh ${SERVER_USER_HOST} 'mkdir -p ${SERVER_PATH}'"
    exit 1
}

