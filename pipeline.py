"""AoS (Age of Sigmar) PDF 파이프라인(오케스트레이션).

이 파일은 오케스트레이션만 담당합니다.
- 분류/태스크: `classifier.py`
- PDF 다운로드 & Gemini 처리(재시도 포함): `gemini_io.py`
- 저장 경로/파일명 규칙: `utils.py` (디렉터리 자동 생성)
"""

import json
import time

from dotenv import dotenv_values
from google import genai

import config as cfg
from classifier import build_db_tasks, print_db_tasks_summary
from gemini_io import (
    delete_gemini_file,
    download_pdf,
    extract_json_with_gemini,
    upload_pdf_to_gemini,
)
from logging_config import get_logger, setup_logging
from utils import build_output_path, save_json

log = get_logger(__name__)


# -----------------------------------------------------------------------------
# 저장
# -----------------------------------------------------------------------------


def save_parsed_json(
    parsed: dict,
    db_target: str,
    doc_name: str,
    out_dir: str = ".",
) -> str:
    """파싱 결과를 JSON 파일로 저장. 파일 경로 반환."""
    path = build_output_path(db_target=db_target, doc_name=doc_name, outputs_dir=out_dir)
    save_json(path, parsed)
    return str(path)


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

    for db_target, task_list in db_tasks.items():
        for task in task_list:
            doc_name = task["name"]
            url = task["url"]
            prompt = task["prompt"]
            log.info("[%s] 파싱 시작: %s", db_target, doc_name)

            try:
                pdf_bytes = download_pdf(url)
                aos_file = upload_pdf_to_gemini(client, pdf_bytes)
                parsed = extract_json_with_gemini(client, aos_file, prompt)
                client.files.delete(name=aos_file.name)

                path = save_parsed_json(parsed, db_target, doc_name, output_dir)
                log.info("저장 완료: %s", path)
                time.sleep(cfg.API_DELAY_SECONDS)
            except Exception as e:
                log.exception("에러 (%s): %s", doc_name, e)


# -----------------------------------------------------------------------------
# 스크립트 실행 시: 스크래핑 → 파싱 → (선택) 파이프라인 실행
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    setup_logging()
    _config = dotenv_values("/workspace/AoS_Chat/.env")

    # === 여기부터 디버그용: 특정 문서 1개만 실행 ===
    DEBUG_DOC_NAME = "Lumineth realm-load"  # 원하는 문서 이름으로 수정

    # load data from json file
    with open("/workspace/AoS_Chat/data.json", "r") as f:
        data = json.load(f)

    # data 구조에서 해당 문서만 골라낸 dict 생성
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

    # 실제 파이프라인 실행 (dry_run=False 로 설정!)
    process_aos_pipeline(pdf_data_dict=debug_data, config_path="/workspace/AoS_Chat/.env", dry_run=False)
