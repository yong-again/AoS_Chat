#!/usr/bin/env bash
# AoS_Chat 앱을 Cloudflare Tunnel로 외부에 공유.
#
# 사용법 (프로젝트 루트에서):
#   ./scripts/share.sh          # 앱이 안 떠 있으면 함께 실행 + 터널 시작
#
# - 발급되는 https://<랜덤>.trycloudflare.com URL을 상대에게 전달하면 됨
# - URL은 터널을 켤 때마다 바뀌고, Ctrl+C로 종료하면 즉시 무효화됨
# - 링크를 아는 누구나 접속 가능 → 외부 공유 시 접근 코드 설정 권장:
#     APP_ACCESS_CODE=원하는코드 ./scripts/share.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PORT="${PORT:-8501}"

if ! command -v cloudflared >/dev/null 2>&1; then
    echo "cloudflared가 없습니다: brew install cloudflared" >&2
    exit 1
fi

# 앱이 안 떠 있으면 백그라운드로 시작
if ! lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "▶ Streamlit 앱 시작 (포트 $PORT)..."
    nohup uv run streamlit run app.py --server.port "$PORT" --server.headless true \
        > runtime/logs/streamlit_share.log 2>&1 &
    # 앱이 뜰 때까지 대기 (임베딩 모델 로딩 포함 최대 90초)
    for _ in $(seq 1 90); do
        lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1 && break
        sleep 1
    done
    if ! lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
        echo "앱 시작 실패 — runtime/logs/streamlit_share.log 확인" >&2
        exit 1
    fi
else
    echo "▶ 포트 $PORT 에서 실행 중인 앱을 사용합니다."
fi

if [ -n "${APP_ACCESS_CODE:-}" ]; then
    echo "▶ 접근 코드 게이트 활성화됨"
else
    echo "⚠️  접근 코드 없이 공유합니다 — 링크를 아는 누구나 사용 가능 (Gemini 쿼터 소진 주의)"
fi

echo "▶ Cloudflare Tunnel 시작 — 아래 https://*.trycloudflare.com URL을 공유하세요. 종료: Ctrl+C"
exec cloudflared tunnel --url "http://localhost:$PORT"
