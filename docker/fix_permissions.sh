#!/bin/bash
# 서버 디렉토리 권한 수정 스크립트
# 사용법: bash docker/fix_permissions.sh <user>@<server>:<remote_path>

if [ $# -lt 1 ]; then
    echo "Usage: bash docker/fix_permissions.sh <user>@<server>:<remote_path>"
    echo "예시: bash docker/fix_permissions.sh dhk@30.30.99.10:/home/dhk/workspaces/rl_training"
    exit 1
fi

DESTINATION="$1"
SERVER_USER_HOST="${DESTINATION%%:*}"
SERVER_PATH="${DESTINATION##*:}"

echo "서버 디렉토리 권한 수정 중..."
echo "서버: ${SERVER_USER_HOST}"
echo "경로: ${SERVER_PATH}"
echo ""

ssh "${SERVER_USER_HOST}" "chmod -R u+w ${SERVER_PATH} 2>/dev/null && chown -R \$(whoami) ${SERVER_PATH} 2>/dev/null || echo '일부 권한 수정 실패 (무시 가능)'" && {
    echo "✅ 권한 수정 완료!"
    echo ""
    echo "이제 파일 전송을 다시 시도하세요:"
    echo "bash docker/sync_all_code.sh ${DESTINATION}"
} || {
    echo "⚠️  권한 수정 중 일부 오류 발생 (계속 진행 가능)"
    echo "수동으로 권한 수정:"
    echo "ssh ${SERVER_USER_HOST}"
    echo "chmod -R u+w ${SERVER_PATH}"
    echo "chown -R \$(whoami) ${SERVER_PATH}"
}

