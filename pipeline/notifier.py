"""알림 모듈 — 파이프라인 완료/실패 시 외부 채널로 메시지 전송."""

import json
import urllib.request

from core.logging_config import get_logger

log = get_logger(__name__)

TELEGRAM_CHAT_ID = "7657071197"


def send_telegram(bot_token: str, message: str) -> None:
    """Telegram Bot API로 메시지 전송."""
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    data = json.dumps({"chat_id": TELEGRAM_CHAT_ID, "text": message}).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=10)
        log.debug("Telegram 알림 전송 완료")
    except Exception as e:
        log.warning("Telegram 알림 전송 실패: %s", e)


def notify_pipeline_result(bot_token: str, total: int, errors: list[str]) -> None:
    """파이프라인 완료 결과를 Telegram으로 알림."""
    if not bot_token:
        log.warning("TELEGRAM_BOT_TOKEN이 설정되지 않아 알림을 건너뜁니다.")
        return

    summary = f"[AoS Parser] 파싱 완료\n성공: {total - len(errors)}개 / 실패: {len(errors)}개 / 전체: {total}개"
    if errors:
        summary += "\n\n실패 목록:\n" + "\n".join(f"  - {e}" for e in errors)

    send_telegram(bot_token, summary)
