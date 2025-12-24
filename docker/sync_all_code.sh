#!/bin/bash
# 모든 코드를 서버로 전송하는 스크립트
#
# 사용법:
#   bash docker/sync_all_code.sh <user>@<server>:<remote_path>
#
# 예시:
#   bash docker/sync_all_code.sh ubuntu@192.168.1.100:/workspace/rl_training
#   bash docker/sync_all_code.sh user@server.company.com:/workspace/rl_training
#
# 이 스크립트는 다음을 전송합니다:
# - 모든 소스 코드 (source/)
# - 모든 스크립트 (scripts/)
# - 설정 파일들 (pyproject.toml, README.md 등)
# - 로봇 모델 파일들 (deep_robotics_model/)
# - Docker 스크립트 (docker/)
# 
# 제외되는 것:
# - logs/ (체크포인트 파일은 별도 전송 필요)
# - outputs/
# - 캐시 파일들 (__pycache__, *.pyc 등)

set -e

if [ $# -lt 1 ]; then
    echo "Error: Server destination is required"
    echo "Usage: bash docker/sync_all_code.sh <user>@<host>:<remote_path>"
    echo ""
    echo "예시:"
    echo "  bash docker/sync_all_code.sh ubuntu@192.168.1.100:/workspace/rl_training"
    exit 1
fi

DESTINATION="$1"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "========================================="
echo "모든 코드를 서버로 전송"
echo "========================================="
echo "소스: ${PROJECT_DIR}"
echo "대상: ${DESTINATION}"
echo "========================================="
echo ""

# rsync 옵션
RSYNC_OPTS=(
    -avz                    # archive, verbose, compress
    --progress              # 진행 상황 표시
    -O                      # 디렉토리 타임스탬프 보존하지 않음 (권한 문제 회피)
    --no-times              # 파일 타임스탬프 보존하지 않음 (권한 문제 회피)
    --delete                # 대상에 있지만 소스에 없는 파일 삭제 (선택사항, 주석 처리 가능)
    --exclude='.git'        # Git 저장소 제외 (서버에 이미 있을 경우)
    --exclude='__pycache__' # Python 캐시
    --exclude='*.pyc'
    --exclude='*.pyo'
    --exclude='.pytest_cache'
    --exclude='**/*.egg-info'
    --exclude='.DS_Store'
    --exclude='*.dmp'
    --exclude='**/.thumbs'
    --exclude='**/.idea'
    --exclude='**/.vscode'
    --exclude='logs/**'     # 로그 파일 제외 (체크포인트는 별도 전송)
    --exclude='outputs/**'  # 출력 파일 제외
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
    --exclude='tree.txt'
)

# 전송할 주요 디렉토리/파일 목록
SYNC_ITEMS=(
    "source/"                    # 모든 소스 코드
    "scripts/"                   # 모든 스크립트
    "deep_robotics_model/"       # 로봇 모델 파일
    "docker/"                    # Docker 스크립트
    "docs/"                      # 문서
    "pyproject.toml"            # 프로젝트 설정
    "README.md"                 # README
    "LICENSE"                   # 라이선스
    "LICENSE-robot_lab"         # 라이선스
    "VERSION"                   # 버전 정보
    "CONTRIBUTORS.md"           # 기여자 정보
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
echo "제외되는 항목:"
echo "  ❌ logs/ (체크포인트는 별도 전송 필요)"
echo "  ❌ outputs/"
echo "  ❌ __pycache__/, *.pyc (캐시 파일)"
echo "  ❌ .git/ (Git 저장소)"
echo ""

# 학습 중 전송 안내
echo "⚠️  참고: 서버에서 학습이 진행 중이어도 전송 가능합니다."
echo "   - 현재 실행 중인 학습에는 영향 없음"
echo "   - 다음 실행 시 새 코드가 사용됨"
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

# 서버 정보 분리
SERVER_USER_HOST="${DESTINATION%%:*}"
SERVER_PATH="${DESTINATION##*:}"

# 서버에 대상 디렉토리가 없으면 생성
echo "서버 디렉토리 확인 및 생성 중..."
ssh "${SERVER_USER_HOST}" "mkdir -p ${SERVER_PATH} && chmod -R u+w ${SERVER_PATH} 2>/dev/null || true" || {
    echo "⚠️  경고: 서버 디렉토리 권한 설정 실패 (계속 진행)"
}
echo "✅ 서버 디렉토리 준비 완료: ${SERVER_PATH}"
echo ""

# 각 항목을 개별적으로 전송 (더 안정적)
for item in "${SYNC_ITEMS[@]}"; do
    if [ ! -e "${PROJECT_DIR}/${item}" ]; then
        echo "⚠️  건너뜀: ${item} (파일/디렉토리 없음)"
        continue
    fi
    
    SOURCE="${PROJECT_DIR}/${item}"
    TARGET="${SERVER_USER_HOST}:${SERVER_PATH}/${item}"
    
    echo "📦 전송: ${item}"
    
    # 디렉토리인 경우
    if [ -d "$SOURCE" ]; then
        # 대상 디렉토리 생성 및 권한 설정
        ssh "${SERVER_USER_HOST}" "mkdir -p ${SERVER_PATH}/${item} && chmod -R u+w ${SERVER_PATH}/${item} 2>/dev/null || true" 2>/dev/null || true
        rsync "${RSYNC_OPTS[@]}" "${SOURCE}/" "${TARGET}/"
    else
        # 파일인 경우 부모 디렉토리 생성 및 권한 설정
        PARENT_DIR="$(dirname ${item})"
        ssh "${SERVER_USER_HOST}" "mkdir -p ${SERVER_PATH}/${PARENT_DIR} && chmod -R u+w ${SERVER_PATH}/${PARENT_DIR} 2>/dev/null || true" 2>/dev/null || true
        rsync "${RSYNC_OPTS[@]}" "${SOURCE}" "${TARGET}"
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
echo "   ssh ${DESTINATION%%:*} 'ls -la ${DESTINATION##*:}/source/'"
echo ""
echo "2. Docker 컨테이너 내부에서 확인:"
echo "   docker exec -it <container_name> ls /workspace/rl_training/source/"
echo ""
echo "3. 의존성 설치 (컨테이너 내부에서, 최초 1회):"
echo "   docker exec -it <container_name> bash -c 'cd /workspace/rl_training && python -m pip install -e source/rl_training'"
echo ""
echo "4. 환경 확인:"
echo "   docker exec -it <container_name> bash -c 'cd /workspace/rl_training && python scripts/tools/list_envs.py'"
echo ""
echo "5. 체크포인트 전송 (필요시):"
echo "   scp -r logs/rsl_rl/deeprobotics_m20_rough/2025-12-15_16-08-31/ \\"
echo "       ${DESTINATION}/logs/rsl_rl/deeprobotics_m20_rough/"
echo ""

