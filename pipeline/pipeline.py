"""AoS (Age of Sigmar) PDF 파이프라인(오케스트레이션).

이 파일은 오케스트레이션만 담당합니다.
- 분류/태스크: `pipeline/classifier.py`
- PDF 다운로드 & Gemini 처리(재시도 포함): `pipeline/gemini_io.py`
- 저장 경로/파일명 규칙: `core/utils.py` (디렉터리 자동 생성)
"""

import json
import time

from dotenv import dotenv_values
from google import genai

from core import config as cfg
from core.logging_config import get_logger, setup_logging
from core.utils import build_output_path, save_json
from pipeline.classifier import build_db_tasks, print_db_tasks_summary
from pipeline.gemini_io import (
    delete_gemini_file,
    download_pdf,
    extract_json_with_gemini,
    merge_chunk_results,
    split_pdf_bytes,
    upload_pdf_to_gemini,
)

log = get_logger(__name__)


# -----------------------------------------------------------------------------
# 저장
# -----------------------------------------------------------------------------


def save_parsed_json(
    parsed: dict,
    db_target: str,
    doc_name: str,
    out_dir: str = ".",
) -> list[str]:
    """파싱 결과를 JSON 파일로 저장. 분리 저장된 파일들의 경로 리스트 반환."""
    saved_paths = []

    # 팩션 팩의 경우 본편과 스피어헤드를 물리적으로 분리
    if db_target == "faction_db" and isinstance(parsed, dict) and "aos_matched_play" in parsed and "spearhead" in parsed:
        # 본편 저장
        path_faction = build_output_path(db_target="faction_db", doc_name=doc_name, outputs_dir=out_dir)
        save_json(path_faction, parsed["aos_matched_play"])
        saved_paths.append(str(path_faction))

        # 스피어헤드 저장
        path_spearhead = build_output_path(db_target="spearhead_db", doc_name=doc_name, outputs_dir=out_dir)
        save_json(path_spearhead, parsed["spearhead"])
        saved_paths.append(str(path_spearhead))
    else:
        # 일반 문서 저장
        path = build_output_path(db_target=db_target, doc_name=doc_name, outputs_dir=out_dir)
        save_json(path, parsed)
        saved_paths.append(str(path))

    return saved_paths


# -----------------------------------------------------------------------------
# 파이프라인 진입점
# -----------------------------------------------------------------------------


def process_aos_pipeline(
    pdf_data_dict: dict[str, dict[str, str]],
    *,
    config_path: str = ".env",
    output_dir: str = ".",
    dry_run: bool = False,
) -> None:
    """
    스크래핑된 data 딕셔너리를 기준으로 PDF 다운로드 → Gemini 업로드 → JSON 추출 → 저장.
    dry_run=True면 분류 요약만 출력하고 실제 API 호출은 하지 않음.
    """
    config = dotenv_values(config_path)
    client = genai.Client(api_key=config["GEMINI_API_KEY"])

    db_tasks = build_db_tasks(pdf_data_dict)
    print_db_tasks_summary(db_tasks)

    if dry_run:
        log.info("(dry_run: API 호출 없음)")
        return

    # 전체 문서 수 집계
    all_tasks = [(db, task) for db, tasks in db_tasks.items() for task in tasks]
    total = len(all_tasks)
    log.info("=" * 60)
    log.info("파싱 시작: 총 %d개 문서", total)
    log.info("=" * 60)

    errors: list[str] = []

    for idx, (db_target, task) in enumerate(all_tasks, start=1):
        doc_name = task["name"]
        url = task["url"]
        prompt = task["prompt"]

        log.info("[%d/%d] (%s) %s", idx, total, db_target, doc_name)

        try:
            log.debug("  → PDF 다운로드 중: %s", url)
            pdf_bytes = download_pdf(url)

            chunk_size = cfg.CHUNK_SIZES.get(db_target, 8)
            chunks = split_pdf_bytes(pdf_bytes, chunk_size)
            total_chunks = len(chunks)

            if total_chunks > 1:
                log.info("  → 청킹 파싱: %d페이지 단위 / 총 %d청크", chunk_size, total_chunks)

            chunk_results = []
            for c_idx, chunk_bytes in enumerate(chunks, start=1):
                if total_chunks > 1:
                    log.debug("  → 청크 [%d/%d] Gemini 업로드 중", c_idx, total_chunks)
                else:
                    log.debug("  → Gemini 업로드 중")

                aos_file = upload_pdf_to_gemini(client, chunk_bytes)
                log.debug("  → 청크 [%d/%d] JSON 추출 중", c_idx, total_chunks)
                parsed_chunk = extract_json_with_gemini(client, aos_file, prompt)
                client.files.delete(name=aos_file.name)
                chunk_results.append(parsed_chunk)
                time.sleep(cfg.API_DELAY_SECONDS)

            parsed = merge_chunk_results(chunk_results)

            paths = save_parsed_json(parsed, db_target, doc_name, output_dir)
            for p in paths:
                log.info("  ✓ 저장: %s", p)

        except Exception as e:
            msg = f"[{idx}/{total}] {db_target} / {doc_name}"
            errors.append(msg)
            log.error("  ✗ 에러 발생 (%s)", msg)
            log.exception("    원인: %s", e)

    # 최종 요약
    log.info("=" * 60)
    log.info("파싱 완료: 성공 %d개 / 실패 %d개 / 전체 %d개", total - len(errors), len(errors), total)
    if errors:
        log.warning("실패한 문서 목록:")
        for err in errors:
            log.warning("  - %s", err)
    log.info("=" * 60)


# -----------------------------------------------------------------------------
# 스크립트 실행 시: 스크래핑 → 파싱 → (선택) 파이프라인 실행
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    setup_logging()
    _config = dotenv_values("/workspace/AoS_Chat/.env")

    # === 여기부터 디버그용: 특정 문서 1개만 실행 ===
    DEBUG_DOC_NAME = "Lumineth realm-load"  # 원하는 문서 이름으로 수정

    with open("/workspace/AoS_Chat/data.json", "r") as f:
        data = json.load(f)

    debug_data = {}
    for section, items in data.items():
        for doc_name, url in items.items():
            if DEBUG_DOC_NAME in doc_name:
                log.info("DEBUG match: %s", doc_name)
                debug_data.setdefault(section, {})[doc_name] = url

    print("=== DEBUG: 선택된 문서 목록 ===")
    for section, items in debug_data.items():
        for name, url in items.items():
            print(f"[섹션] {section} / [문서] {name}")
            print(f"URL: {url}")

    process_aos_pipeline(pdf_data_dict=debug_data, config_path="/workspace/AoS_Chat/.env", dry_run=False)
