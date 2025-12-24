#!/bin/bash
# Hierarchical Navigation 코드를 서버로 전송하는 스크립트
#
# 사용법:
#   bash docker/sync_hierarchical_nav.sh <user>@<server>:<remote_path>
#
# 예시:
#   bash docker/sync_hierarchical_nav.sh user@server.company.com:/workspace/rl_training
#   bash docker/sync_hierarchical_nav.sh user@192.168.1.100:/home/user/rl_training

set -e

if [ $# -lt 1 ]; then
    echo "Error: Server destination is required"
    echo "Usage: bash docker/sync_hierarchical_nav.sh <user>@<host>:<remote_path>"
    echo ""
    echo "예시:"
    echo "  bash docker/sync_hierarchical_nav.sh user@server.company.com:/workspace/rl_training"
    exit 1
fi

DESTINATION="$1"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "========================================="
echo "Hierarchical Navigation 코드 서버 전송"
echo "========================================="
echo "소스: ${PROJECT_DIR}"
echo "대상: ${DESTINATION}"
echo "========================================="
echo ""

# rsync 옵션
RSYNC_OPTS=(
    -avz                    # archive, verbose, compress
    --progress              # 진행 상황 표시
)

# 전송할 파일/디렉토리 목록
SYNC_ITEMS=(
    "source/rl_training/rl_training/tasks/manager_based/locomotion/hierarchical_nav/"
    "source/rl_training/rl_training/utils/frozen_policy.py"
    "scripts/reinforcement_learning/rsl_rl/train_hierarchical_nav.py"
    "scripts/reinforcement_learning/rsl_rl/test_hierarchical_nav.py"
    "scripts/reinforcement_learning/rsl_rl/test_frozen_policy_wrapper.py"
    "scripts/reinforcement_learning/rsl_rl/validate_frozen_policy.py"
    "docker/"
)

echo "전송할 항목:"
for item in "${SYNC_ITEMS[@]}"; do
    if [ -e "${PROJECT_DIR}/${item}" ]; then
        echo "  ✅ ${item}"
    else
        echo "  ⚠️  ${item} (존재하지 않음)"
    fi
done
echo ""

# 전송 실행
echo "전송 시작..."
echo ""

for item in "${SYNC_ITEMS[@]}"; do
    if [ ! -e "${PROJECT_DIR}/${item}" ]; then
        echo "⚠️  건너뜀: ${item} (파일/디렉토리 없음)"
        continue
    fi
    
    SOURCE="${PROJECT_DIR}/${item}"
    
    # 디렉토리인지 파일인지 확인
    if [ -d "$SOURCE" ]; then
        # 디렉토리: 끝에 슬래시 제거하고 대상 경로에 디렉토리명 포함
        DEST="${DESTINATION}/$(dirname ${item})"
        echo "📁 디렉토리 전송: ${item} -> ${DEST}/"
        rsync "${RSYNC_OPTS[@]}" "$SOURCE" "$DEST/"
    else
        # 파일: 부모 디렉토리로 전송
        DEST="${DESTINATION}/$(dirname ${item})"
        echo "📄 파일 전송: ${item} -> ${DEST}/"
        rsync "${RSYNC_OPTS[@]}" "$SOURCE" "$DEST/"
    fi
    echo ""
done

echo "========================================="
echo "✅ 파일 전송 완료!"
echo "========================================="
echo ""
echo "다음 단계:"
echo ""
echo "1. 서버에서 파일 확인:"
echo "   ssh ${DESTINATION%%:*} 'ls -la ${DESTINATION##*:}/source/rl_training/rl_training/tasks/manager_based/locomotion/hierarchical_nav/'"
echo ""
echo "2. Docker 컨테이너 내부에서 확인:"
echo "   docker exec -it <container_name> ls /workspace/rl_training/source/rl_training/rl_training/tasks/manager_based/locomotion/hierarchical_nav/"
echo ""
echo "3. 의존성 설치 (컨테이너 내부에서):"
echo "   docker exec -it <container_name> bash -c 'cd /workspace/rl_training && python -m pip install -e source/rl_training'"
echo ""
echo "4. 환경 확인:"
echo "   docker exec -it <container_name> bash -c 'cd /workspace/rl_training && python scripts/tools/list_envs.py | grep -i hierarchical'"
echo ""
echo "5. 체크포인트 전송 (필요시):"
echo "   scp -r logs/rsl_rl/deeprobotics_m20_rough/2025-12-15_16-08-31/ \\"
echo "       ${DESTINATION}/logs/rsl_rl/deeprobotics_m20_rough/"
echo ""

