"""
PDF 다운로드 + Gemini 업로드/추출 IO 모듈.
"""

from __future__ import annotations

import json
import tempfile
import time
from json import JSONDecodeError

import requests
from google import genai
from google.genai import types

import config as cfg
from logging_config import get_logger
from retry import extract_status_code, is_retryable_status, retry_with_exponential_backoff

log = get_logger(__name__)


def download_pdf(url: str) -> bytes:
    """PDF URL을 GET으로 다운로드해 바이트 반환."""

    def _once() -> bytes:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        return resp.content

    def _retry_if(exc: Exception) -> bool:
        if isinstance(exc, requests.HTTPError):
            return is_retryable_status(extract_status_code(exc))
        return isinstance(
            exc,
            (
                requests.Timeout,
                requests.ConnectionError,
                requests.ChunkedEncodingError,
            ),
        )

    return retry_with_exponential_backoff(
        _once,
        label=f"download_pdf({url})",
        retry_if=_retry_if,
    )


def upload_pdf_to_gemini(client: genai.Client, pdf_bytes: bytes) -> types.File:
    """PDF 바이트를 임시 파일로 저장 후 Gemini에 업로드, 처리 완료까지 대기."""

    def _upload_once() -> types.File:
        with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp.flush()
            return client.files.upload(file=tmp.name)

    uploaded = retry_with_exponential_backoff(
        _upload_once,
        label="gemini_file_upload",
        retry_if=lambda exc: is_retryable_status(extract_status_code(exc)),
    )

    while uploaded.state.name == "PROCESSING":
        time.sleep(cfg.FILE_POLL_INTERVAL_SECONDS)
        uploaded = retry_with_exponential_backoff(
            lambda: client.files.get(name=uploaded.name),
            label="gemini_file_get",
            retry_if=lambda exc: is_retryable_status(extract_status_code(exc)),
        )

    return uploaded


def extract_json_with_gemini(client: genai.Client, file: types.File, prompt: str) -> dict:
    """Gemini로 PDF + 프롬프트 전달 후 JSON 파싱된 dict 반환."""

    def _once() -> dict:
        response = client.models.generate_content(
            model=cfg.GEMINI_MODEL,
            contents=[file, prompt],
            config=types.GenerateContentConfig(
                response_mime_type=cfg.GEMINI_JSON_MIME,
                temperature=cfg.GEMINI_TEMPERATURE,
            ),
        )
        return json.loads(response.text)

    def _retry_if(exc: Exception) -> bool:
        # google.genai.errors.ServerError: 503 UNAVAILABLE (high demand) 등
        if is_retryable_status(extract_status_code(exc)):
            return True
        # 응답이 간헐적으로 JSON이 아닐 때도 재시도
        if isinstance(exc, JSONDecodeError):
            return True
        return False

    return retry_with_exponential_backoff(
        _once,
        label="gemini_generate_content",
        retry_if=_retry_if,
    )


def delete_gemini_file(client: genai.Client, name: str) -> None:
    retry_with_exponential_backoff(
        lambda: client.files.delete(name=name),
        label="gemini_file_delete",
        retry_if=lambda exc: is_retryable_status(extract_status_code(exc)),
    )

