"""
PDF 다운로드 + Gemini 업로드/추출 IO 모듈.
"""

from __future__ import annotations

import io
import json
import tempfile
import time
from json import JSONDecodeError
from typing import Any, Type

import requests
from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError
from pypdf import PdfReader, PdfWriter

from core import config as cfg
from core.logging_config import get_logger
from core.retry import extract_status_code, is_retryable_status, retry_with_exponential_backoff

log = get_logger(__name__)


# -----------------------------------------------------------------------------
# PDF 청킹 유틸
# -----------------------------------------------------------------------------


def split_pdf_bytes(pdf_bytes: bytes, chunk_size: int) -> list[bytes]:
    """PDF를 chunk_size 페이지 단위로 분할해 bytes 리스트로 반환."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    total_pages = len(reader.pages)
    chunks: list[bytes] = []

    for start in range(0, total_pages, chunk_size):
        writer = PdfWriter()
        for page in reader.pages[start: start + chunk_size]:
            writer.add_page(page)
        buf = io.BytesIO()
        writer.write(buf)
        chunks.append(buf.getvalue())

    log.debug("PDF 청킹: 총 %d페이지 → %d청크 (chunk_size=%d)", total_pages, len(chunks), chunk_size)
    return chunks


def _merge_lists(a: list, b: list) -> list:
    return a + b


def _merge_dicts(base: dict, extra: dict) -> dict:
    """두 dict를 재귀적으로 병합. list 값은 이어붙이고, dict 값은 재귀 병합."""
    result = dict(base)
    for key, val in extra.items():
        if key not in result:
            result[key] = val
        elif isinstance(result[key], list) and isinstance(val, list):
            result[key] = _merge_lists(result[key], val)
        elif isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _merge_dicts(result[key], val)
        # 스칼라(str 등)는 첫 청크 값 유지 (spearhead_name 등)
    return result


def merge_chunk_results(chunks: list[Any]) -> Any:
    """청크별 파싱 결과를 하나로 병합."""
    if not chunks:
        return {}
    result = chunks[0]
    for chunk in chunks[1:]:
        if isinstance(result, list) and isinstance(chunk, list):
            result = _merge_lists(result, chunk)
        elif isinstance(result, dict) and isinstance(chunk, dict):
            result = _merge_dicts(result, chunk)
    return result


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


def extract_json_with_gemini(
    client: genai.Client,
    file: types.File,
    prompt: str,
    schema_cls: Type[BaseModel],
) -> dict:
    """Gemini로 PDF + 프롬프트 전달 후 Pydantic 스키마로 검증된 dict 반환.

    response_schema에 Pydantic 모델을 전달해 Structured Output을 적용합니다.
    응답은 schema_cls.model_validate_json() 후 .model_dump()로 변환됩니다.
    """

    def _once() -> dict:
        response = client.models.generate_content(
            model=cfg.GEMINI_MODEL,
            contents=[file, prompt],
            config=types.GenerateContentConfig(
                response_mime_type=cfg.GEMINI_JSON_MIME,
                response_schema=schema_cls,
                temperature=cfg.GEMINI_TEMPERATURE,
            ),
        )
        return schema_cls.model_validate_json(response.text).model_dump()

    def _retry_if(exc: Exception) -> bool:
        if is_retryable_status(extract_status_code(exc)):
            return True
        if isinstance(exc, (JSONDecodeError, ValidationError)):
            return True
        return False

    return retry_with_exponential_backoff(
        _once,
        label="gemini_generate_content",
        retry_if=_retry_if,
    )


def merge_faction_chunk_results(chunk_results: list[dict]) -> dict:
    """faction_db 청크 결과 병합 (FactionPackResult + SpearheadFactionResult 혼합 지원).

    FactionPackResult 형태(aos_matched_play 키 보유): aos_matched_play 합산,
    경계 청크의 spearhead도 수집.
    SpearheadFactionResult 형태(spearhead 키만 보유): spearhead 합산.
    """
    merged_amp: dict = {}
    merged_sp: dict = {}
    current_spearhead_name: str | None = None

    for r in chunk_results:
        amp = r.get("aos_matched_play") or {}
        sp = r.get("spearhead") or {}

        if amp:
            merged_amp = _merge_dicts(merged_amp, amp) if merged_amp else dict(amp)

        if _has_spearhead_data(r):
            returned_name = sp.get("spearhead_name")
            if returned_name and returned_name != current_spearhead_name:
                current_spearhead_name = returned_name
            elif not returned_name and current_spearhead_name:
                sp = dict(sp)
                sp["spearhead_name"] = current_spearhead_name
            merged_sp = _merge_dicts(merged_sp, sp) if merged_sp else dict(sp)

    return {"aos_matched_play": merged_amp, "spearhead": merged_sp}


def _has_spearhead_data(parsed: dict) -> bool:
    """FactionPackResult dict에 실질적인 스피어헤드 데이터가 있는지 확인."""
    sp = parsed.get("spearhead") or {}
    return bool(sp.get("spearhead_name") or sp.get("warscrolls") or sp.get("spearhead_rules"))


def process_faction_chunks(
    client: genai.Client,
    chunks: list[bytes],
    faction_prompt: str,
    spearhead_prompt: str,
    doc_name: str = "",
) -> dict:
    """팩션 팩 청크를 스피어헤드 감지 기반 적응형 스키마로 처리.

    - 스피어헤드 감지 전: FactionPackResult 스키마로 처리
    - 스피어헤드 감지 후: SpearheadFactionResult 스키마로 전환
      (스피어헤드 첫 페이지가 청크 경계에 걸려 이후 청크가 통째로 스피어헤드인 경우 대응)
    - 반환: {"aos_matched_play": ..., "spearhead": ...}
    """
    from pipeline.schemas import FactionPackResult, SpearheadFactionResult

    spearhead_mode = False
    current_spearhead_name: str | None = None  # 가장 최근에 확정된 spearhead_name
    aos_results: list[dict] = []
    spearhead_results: list[dict] = []
    total = len(chunks)

    for c_idx, chunk_bytes in enumerate(chunks, start=1):
        log.debug("  [%s] 청크 [%d/%d] Gemini 업로드 중", doc_name, c_idx, total)
        aos_file = upload_pdf_to_gemini(client, chunk_bytes)

        if not spearhead_mode:
            log.debug("  [%s] 청크 [%d/%d] JSON 추출 중 (FactionPackResult)", doc_name, c_idx, total)
            parsed = extract_json_with_gemini(client, aos_file, faction_prompt, FactionPackResult)
            delete_gemini_file(client, aos_file.name)
            aos_results.append(parsed)
            if _has_spearhead_data(parsed):
                current_spearhead_name = (parsed.get("spearhead") or {}).get("spearhead_name")
                log.info(
                    "  [%s] 청크 %d/%d: 스피어헤드 감지 (name=%s) → 이후 청크 SpearheadFactionResult 전환",
                    doc_name, c_idx, total, current_spearhead_name,
                )
                spearhead_mode = True
        else:
            log.debug("  [%s] 청크 [%d/%d] JSON 추출 중 (SpearheadFactionResult)", doc_name, c_idx, total)
            parsed = extract_json_with_gemini(client, aos_file, spearhead_prompt, SpearheadFactionResult)
            delete_gemini_file(client, aos_file.name)

            # spearhead_name 전파: 새 이름이 나오면 갱신, 없으면 현재 이름 주입
            sp = parsed.get("spearhead") or {}
            returned_name = sp.get("spearhead_name")
            if returned_name and returned_name != current_spearhead_name:
                log.info(
                    "  [%s] 청크 %d/%d: 새 spearhead_name 감지 → %s",
                    doc_name, c_idx, total, returned_name,
                )
                current_spearhead_name = returned_name
            elif not returned_name and current_spearhead_name:
                sp["spearhead_name"] = current_spearhead_name
                parsed["spearhead"] = sp

            spearhead_results.append(parsed)

        time.sleep(cfg.API_DELAY_SECONDS)

    # ── 병합 ─────────────────────────────────────────────────────────────────
    merged_amp: dict = {}
    merged_sp: dict = {}

    # FactionPackResult 청크: aos_matched_play 합산, 경계 청크의 spearhead도 수집
    for r in aos_results:
        amp = r.get("aos_matched_play") or {}
        if amp:
            merged_amp = _merge_dicts(merged_amp, amp) if merged_amp else dict(amp)
        if _has_spearhead_data(r):
            sp = r.get("spearhead") or {}
            merged_sp = _merge_dicts(merged_sp, sp) if merged_sp else dict(sp)

    # SpearheadFactionResult 청크: spearhead 합산 (각 청크에 이미 올바른 name이 주입됨)
    for r in spearhead_results:
        sp = r.get("spearhead") or {}
        if sp:
            merged_sp = _merge_dicts(merged_sp, sp) if merged_sp else dict(sp)

    return {"aos_matched_play": merged_amp, "spearhead": merged_sp}


def delete_gemini_file(client: genai.Client, name: str) -> None:
    retry_with_exponential_backoff(
        lambda: client.files.delete(name=name),
        label="gemini_file_delete",
        retry_if=lambda exc: is_retryable_status(extract_status_code(exc)),
    )
