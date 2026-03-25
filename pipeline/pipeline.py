"""AoS (Age of Sigmar) PDF 파이프라인(오케스트레이션).

이 파일은 오케스트레이션만 담당합니다.
- 분류/태스크: `pipeline/classifier.py`
- PDF 다운로드 & Gemini 처리(재시도 포함): `pipeline/gemini_io.py`
- 저장 경로/파일명 규칙: `core/utils.py` (디렉터리 자동 생성)
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import dotenv_values
from google import genai

from core import config as cfg
from core.logging_config import get_logger, setup_logging
from core.utils import build_output_path, save_json
from pipeline.checkpoint import ask_resume, filter_pending, find_completed, print_checkpoint_status
from pipeline.classifier import build_db_tasks, print_db_tasks_summary
from pipeline.notifier import notify_pipeline_progress, notify_pipeline_result
from pipeline.gemini_io import (
    delete_gemini_file,
    download_pdf,
    extract_json_with_gemini,
    merge_chunk_results,
    process_faction_chunks,
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
    """파싱 결과를 JSON 파일로 저장. 분리 저장된 파일들의 경로 리스트 반환.

    BalanceResult / OtherResult는 Pydantic 래퍼 필드를 unwrap하여 기존 포맷(배열)으로 저장합니다.
    - balance_db: {"units": [...]} → [...]
    - other_db:   {"entries": [...]} → [...]
    """
    saved_paths = []

    # 팩션 팩: 스피어헤드 유무에 따라 분기
    if db_target == "faction_db":
        spearhead_data = parsed.get("spearhead") or {}
        has_spearhead = bool(
            spearhead_data.get("spearhead_name")
            or spearhead_data.get("warscrolls")
            or spearhead_data.get("spearhead_rules")
        )

        if has_spearhead:
            # 스피어헤드 있음: faction_db에는 본편만, spearhead_db에는 스피어헤드만 분리 저장
            path_faction = build_output_path(db_target="faction_db", doc_name=doc_name, outputs_dir=out_dir)
            save_json(path_faction, parsed["aos_matched_play"])
            saved_paths.append(str(path_faction))

            # 팩션 이름 추출 (예: "Faction Pack: Lumineth Realm-lords" → "Lumineth Realm-lords")
            faction_name = doc_name.replace("Faction Pack:", "").replace("Faction Pack", "").strip()

            raw_name = spearhead_data.get("spearhead_name")
            clean_name = str(raw_name).strip() if raw_name is not None else ""
            if not clean_name or clean_name.lower() in ["none", "null", "unknown", ""]:
                spearhead_name = "unknown"
            else:
                spearhead_name = clean_name

            spearhead_doc_name = f"spearhead_{faction_name}_-_{spearhead_name}"
            path_spearhead = build_output_path(db_target="spearhead_db", doc_name=spearhead_doc_name, outputs_dir=out_dir)
            save_json(path_spearhead, spearhead_data)
            saved_paths.append(str(path_spearhead))
        else:
            # 스피어헤드 없음: faction_db에만 저장
            path_faction = build_output_path(db_target="faction_db", doc_name=doc_name, outputs_dir=out_dir)
            save_json(path_faction, parsed["aos_matched_play"])
            saved_paths.append(str(path_faction))

    # balance_db: BalanceResult 래퍼 unwrap → 배열로 저장
    elif db_target == "balance_db" and "units" in parsed:
        path = build_output_path(db_target=db_target, doc_name=doc_name, outputs_dir=out_dir)
        save_json(path, parsed["units"])
        saved_paths.append(str(path))

    # other_db: OtherResult 래퍼 unwrap → 배열로 저장
    elif db_target == "other_db" and "entries" in parsed:
        path = build_output_path(db_target=db_target, doc_name=doc_name, outputs_dir=out_dir)
        save_json(path, parsed["entries"])
        saved_paths.append(str(path))

    else:
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

    # 전체 태스크 목록
    all_tasks = [(db, task) for db, tasks in db_tasks.items() for task in tasks]
    total_all = len(all_tasks)

    # ── 체크포인트 확인 ──────────────────────────────────────────
    completed = find_completed(all_tasks, output_dir)
    if completed:
        print_checkpoint_status(all_tasks, completed)
        resume = ask_resume(len(completed), total_all)
        if resume:
            all_tasks = filter_pending(all_tasks, completed)
            log.info("체크포인트 복원: %d개 건너뜀, %d개 남음", len(completed), len(all_tasks))
        else:
            log.info("처음부터 재시작합니다.")
    # ────────────────────────────────────────────────────────────

    total = len(all_tasks)
    log.info("=" * 60)
    log.info("파싱 시작: %d개 문서 (전체 %d개, workers=%d)", total, total_all, cfg.PIPELINE_MAX_WORKERS)
    log.info("=" * 60)

    errors: list[str] = []
    bot_token = config.get("TELEGRAM_BOT_TOKEN", "")
    NOTIFY_INTERVAL = 10  # N개 완료마다 진행 상황 알림
    start_time = time.time()

    def _process_single(db_target: str, task: dict) -> list[str]:
        """단일 문서를 처리하고 저장된 파일 경로를 반환 (워커 스레드에서 실행)."""
        doc_name = task["name"]
        url = task["url"]
        prompt = task["prompt"]
        schema_cls = task["schema"]

        log.debug("  → PDF 다운로드 중: %s", url)
        pdf_bytes = download_pdf(url)

        chunk_size = cfg.CHUNK_SIZES.get(db_target, 8)
        chunks = split_pdf_bytes(pdf_bytes, chunk_size)
        total_chunks = len(chunks)

        if total_chunks > 1:
            log.info("  [%s] 청킹 파싱: %d페이지 단위 / 총 %d청크", doc_name, chunk_size, total_chunks)

        if db_target == "faction_db":
            # 팩션 팩: 스피어헤드 감지 시 이후 청크 스키마 자동 전환
            parsed = process_faction_chunks(
                client, chunks, prompt, cfg.SPEARHEAD_FACTION_PROMPT, doc_name=doc_name
            )
        else:
            chunk_results = []
            for c_idx, chunk_bytes in enumerate(chunks, start=1):
                log.debug("  [%s] 청크 [%d/%d] Gemini 업로드 중", doc_name, c_idx, total_chunks)
                aos_file = upload_pdf_to_gemini(client, chunk_bytes)
                log.debug("  [%s] 청크 [%d/%d] JSON 추출 중", doc_name, c_idx, total_chunks)
                parsed_chunk = extract_json_with_gemini(client, aos_file, prompt, schema_cls)
                delete_gemini_file(client, aos_file.name)
                chunk_results.append(parsed_chunk)
                time.sleep(cfg.API_DELAY_SECONDS)
            parsed = merge_chunk_results(chunk_results)
        return save_parsed_json(parsed, db_target, doc_name, output_dir)

    # ── 병렬 처리 (ThreadPoolExecutor) ──────────────────────────────────────
    with ThreadPoolExecutor(max_workers=cfg.PIPELINE_MAX_WORKERS) as executor:
        future_to_task = {
            executor.submit(_process_single, db_target, task): (idx, db_target, task)
            for idx, (db_target, task) in enumerate(all_tasks, start=1)
        }

        completed_count = 0
        for future in as_completed(future_to_task):
            idx, db_target, task = future_to_task[future]
            doc_name = task["name"]
            completed_count += 1

            try:
                paths = future.result()
                for p in paths:
                    log.info("[%d/%d] ✓ (%s) %s → %s", completed_count, total, db_target, doc_name, p)
            except Exception as e:
                msg = f"({db_target}) {doc_name}"
                errors.append(msg)
                log.error("[%d/%d] ✗ 에러: %s", completed_count, total, msg)
                log.exception("    원인: %s", e)

            # 진행 상황 알림 (N개 단위)
            if completed_count % NOTIFY_INTERVAL == 0:
                elapsed = time.time() - start_time
                notify_pipeline_progress(
                    bot_token,
                    current=completed_count,
                    total=total,
                    success=completed_count - len(errors),
                    errors=errors,
                    elapsed_sec=elapsed,
                )
                log.info("  → 진행 알림 전송 (%d/%d)", completed_count, total)
    # ────────────────────────────────────────────────────────────────────────

    # 최종 요약
    elapsed_total = time.time() - start_time
    success_count = total - len(errors)
    elapsed_str = f"{int(elapsed_total // 60)}분 {int(elapsed_total % 60)}초"
    log.info("=" * 60)
    log.info("파싱 완료: 성공 %d개 / 실패 %d개 / 전체 %d개 (소요: %s)", success_count, len(errors), total, elapsed_str)
    if errors:
        log.warning("실패한 문서 목록:")
        for err in errors:
            log.warning("  - %s", err)
    log.info("=" * 60)

    # Telegram 알림
    notify_pipeline_result(bot_token, total, errors, elapsed_sec=elapsed_total)


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
