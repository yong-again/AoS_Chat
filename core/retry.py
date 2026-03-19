"""
재시도/백오프 공통 모듈.
서버 과부하(429/503 등) 시 즉시 재요청하지 않고 지수 백오프로 재시도합니다.
"""

from __future__ import annotations

import random
import time
from typing import Optional

import requests

from core.logging_config import get_logger

log = get_logger(__name__)


def extract_status_code(exc: Exception) -> Optional[int]:
    """
    다양한 예외에서 status code를 최대한 추출.
    - google.genai.errors.ServerError: code/status_code 속성이 있는 경우
    - requests.HTTPError: exc.response.status_code
    """
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if isinstance(code, int):
        return code
    if isinstance(exc, requests.HTTPError):
        resp = getattr(exc, "response", None)
        return getattr(resp, "status_code", None)
    return None


def is_retryable_status(code: Optional[int]) -> bool:
    # 과부하/일시 장애에 대한 최소 방어
    return code in {429, 500, 502, 503, 504}


def retry_with_exponential_backoff(
    fn,
    *,
    max_attempts: int = 6,
    base_delay_seconds: float = 1.0,
    max_delay_seconds: float = 60.0,
    jitter_ratio: float = 0.2,
    label: str = "operation",
    retry_if=None,
):
    """
    서버 과부하 시 즉시 재요청하지 않도록 지수 백오프로 재시도.
    delay = min(max_delay, base * 2^(attempt-1)) + jitter(±jitter_ratio)
    """
    if retry_if is None:
        retry_if = lambda exc: True  # noqa: E731

    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt >= max_attempts or not retry_if(exc):
                raise

            delay = min(max_delay_seconds, base_delay_seconds * (2 ** (attempt - 1)))
            jitter = delay * jitter_ratio * (random.random() * 2 - 1)  # [-, +]
            sleep_for = max(0.0, delay + jitter)

            status = extract_status_code(exc)
            if status is not None:
                log.warning(
                    "%s 실패 (attempt %s/%s, status=%s). %.2fs 후 재시도. error=%s",
                    label,
                    attempt,
                    max_attempts,
                    status,
                    sleep_for,
                    exc,
                )
            else:
                log.warning(
                    "%s 실패 (attempt %s/%s). %.2fs 후 재시도. error=%s",
                    label,
                    attempt,
                    max_attempts,
                    sleep_for,
                    exc,
                )

            time.sleep(sleep_for)

    if last_exc is not None:
        raise last_exc
