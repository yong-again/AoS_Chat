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


def notify_pipeline_progress(
    bot_token: str,
    current: int,
    total: int,
    success: int,
    errors: list[str],
    elapsed_sec: float,
) -> None:
    """파이프라인 진행 상황을 Telegram으로 알림."""
    if not bot_token:
        return

    pct = int(current / total * 100) if total else 0
    elapsed = f"{int(elapsed_sec // 60)}분 {int(elapsed_sec % 60)}초"
    msg = (
        f"[AoS Parser] 진행 중 {pct}% ({current}/{total})\n"
        f"✅ 성공: {success}개 | ❌ 실패: {len(errors)}개\n"
        f"⏱ 경과: {elapsed}"
    )
    if errors:
        recent = errors[-3:]
        msg += "\n\n최근 실패:\n" + "\n".join(f"  - {e}" for e in recent)

    send_telegram(bot_token, msg)


def notify_pipeline_result(
    bot_token: str,
    total: int,
    errors: list[str],
    elapsed_sec: float = 0.0,
) -> None:
    """파이프라인 완료 결과를 Telegram으로 알림."""
    if not bot_token:
        log.warning("TELEGRAM_BOT_TOKEN이 설정되지 않아 알림을 건너뜁니다.")
        return

    success = total - len(errors)
    elapsed = f"{int(elapsed_sec // 60)}분 {int(elapsed_sec % 60)}초"
    summary = (
        f"[AoS Parser] 파싱 완료 🎉\n"
        f"✅ 성공: {success}개 / ❌ 실패: {len(errors)}개 / 전체: {total}개\n"
        f"⏱ 총 소요 시간: {elapsed}"
    )
    if errors:
        summary += "\n\n실패 목록:\n" + "\n".join(f"  - {e}" for e in errors)

    send_telegram(bot_token, summary)
