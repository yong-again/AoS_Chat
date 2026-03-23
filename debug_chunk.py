"""
청킹 디버그 스크립트 — data.json의 모든 문서 처리

사용법:
  python debug_chunk.py                          # 10개 샘플 문서 처리 (기본값)
  python debug_chunk.py --sample 5              # 5개 샘플
  python debug_chunk.py --sample 0              # 전체 문서 처리
  python debug_chunk.py --section "Spearhead"   # 특정 섹션만
  python debug_chunk.py --doc "Lumineth"        # 문서명에 키워드 포함
  python debug_chunk.py --db faction_db         # 특정 DB 타겟만
  python debug_chunk.py --dry-run               # 청킹만 확인 (Gemini 호출 없음)
  python debug_chunk.py --seed 42               # 샘플 재현을 위한 시드 지정

체크 항목:
  1. 청킹 분할: PDF 페이지 범위 확인 + 청크 PDF 저장
  2. 청크별 파싱 결과: 청크마다 Gemini 응답 저장  [--dry-run 시 건너뜀]
  3. 병합 결과: merge_chunk_results() 결과 vs 기존 저장 파일 비교
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
import time
from pathlib import Path

from dotenv import dotenv_values
from google import genai
from pypdf import PdfReader

sys.path.insert(0, str(Path(__file__).parent))

from core import config as cfg
from core.logging_config import get_logger, setup_logging
from core.utils import safe_filename
from pipeline.classifier import build_db_tasks
from pipeline.gemini_io import (
    delete_gemini_file,
    download_pdf,
    extract_json_with_gemini,
    merge_chunk_results,
    split_pdf_bytes,
    upload_pdf_to_gemini,
)

setup_logging()
log = get_logger("debug_chunk")

DEBUG_DIR = Path("debug_output")


# ── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="청킹 디버그 — data.json 기반 전체 문서 처리")
    p.add_argument("--section", help="섹션명 필터 (부분 일치)", default=None)
    p.add_argument("--doc", help="문서명 키워드 필터 (부분 일치)", default=None)
    p.add_argument("--db", help="DB 타겟 필터 (예: faction_db)", default=None)
    p.add_argument("--sample", type=int, default=10, metavar="N",
                   help="처리할 문서 수 (기본값: 10, 0=전체)")
    p.add_argument("--seed", type=int, default=None, help="샘플 랜덤 시드 (재현용)")
    p.add_argument("--dry-run", action="store_true", help="청킹만 확인, Gemini 호출 생략")
    return p.parse_args()


# ── 유틸 ────────────────────────────────────────────────────────────────────

def section_header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def summarize_json(data: dict | list, db_target: str, label: str) -> None:
    """DB 타입에 따라 파싱 결과 요약 출력."""
    print(f"\n[{label}]")

    if db_target == "faction_db":
        _summarize_faction(data)
    elif db_target == "balance_db":
        units = data if isinstance(data, list) else []
        print(f"  유닛 수: {len(units)}")
        for u in units[:5]:
            print(f"    - {u.get('unit_name', '?')} / {u.get('points', '?')}pts")
        if len(units) > 5:
            print(f"    ... 외 {len(units) - 5}개")
    elif db_target in ("spearhead_db", "rule_db", "other_db"):
        if isinstance(data, dict):
            for k, v in data.items():
                count = len(v) if isinstance(v, list) else ("dict" if isinstance(v, dict) else type(v).__name__)
                print(f"  {k}: {count}")
        elif isinstance(data, list):
            print(f"  항목 수: {len(data)}")
    else:
        print(f"  타입: {type(data).__name__}")


def _summarize_faction(data: dict) -> None:
    for section_key in ("aos_matched_play", "spearhead"):
        if section_key not in data:
            continue
        sec = data[section_key]
        print(f"  [{section_key}]")
        if "army_rules" in sec:
            ar = sec["army_rules"]
            for field in ("battle_traits", "battle_formations", "heroic_traits", "artefacts_of_power", "lores"):
                val = ar.get(field)
                count = len(val) if isinstance(val, list) else ("null" if val is None else "?")
                print(f"    {field}: {count}")
        ws = sec.get("warscrolls", [])
        print(f"    warscrolls: {len(ws) if isinstance(ws, list) else '?'}")
        if isinstance(ws, list):
            for w in ws:
                name = w.get("unit_name", "?")
                abilities = len(w.get("abilities") or [])
                weapons = len(w.get("weapons") or [])
                print(f"      - {name} (weapons={weapons}, abilities={abilities})")


# ── 단일 문서 처리 ───────────────────────────────────────────────────────────

def process_doc(task: dict, client: genai.Client | None, dry_run: bool) -> None:
    doc_name: str = task["name"]
    url: str = task["url"]
    prompt: str = task["prompt"]
    db_target: str = task["db_target"]
    chunk_size: int = cfg.CHUNK_SIZES.get(db_target, cfg.CHUNK_SIZES["faction_db"])

    safe_name = safe_filename(doc_name)
    doc_dir = DEBUG_DIR / safe_name
    chunks_dir = doc_dir / "chunk_pdfs"
    results_dir = doc_dir / "chunk_results"
    doc_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    existing_output = Path("outputs") / db_target / f"{safe_name}.json"

    # ── Step 1: 청킹 분할 확인 ──────────────────────────────────────────────
    section_header(f"STEP 1 — 청킹 분할 확인 [{doc_name}]")

    log.info("PDF 다운로드 중: %s", url)
    pdf_bytes = download_pdf(url)

    reader = PdfReader(io.BytesIO(pdf_bytes))
    total_pages = len(reader.pages)
    print(f"\n  문서명:    {doc_name}")
    print(f"  DB 타겟:   {db_target}")
    print(f"  총 페이지: {total_pages}")
    print(f"  청크 크기: {chunk_size} 페이지")

    chunks = split_pdf_bytes(pdf_bytes, chunk_size)
    print(f"  청크 수:   {len(chunks)}")
    print()

    for i, chunk_bytes in enumerate(chunks, start=1):
        start_page = (i - 1) * chunk_size + 1
        end_page = min(i * chunk_size, total_pages)
        actual_pages = len(PdfReader(io.BytesIO(chunk_bytes)).pages)

        chunk_path = chunks_dir / f"chunk_{i:02d}_p{start_page:03d}-{end_page:03d}.pdf"
        chunk_path.write_bytes(chunk_bytes)
        print(f"  청크 {i:02d}: 페이지 {start_page:3d}~{end_page:3d} ({actual_pages}페이지) → {chunk_path.name}")

    print(f"\n  ✓ 청크 PDF 저장 완료: {chunks_dir}/")

    if dry_run:
        print("\n  [--dry-run] Gemini 파싱 건너뜀.")
        return

    # ── Step 2: 청크별 Gemini 파싱 ──────────────────────────────────────────
    section_header(f"STEP 2 — 청크별 Gemini 파싱 [{doc_name}]")

    chunk_results: list = []
    for i, chunk_bytes in enumerate(chunks, start=1):
        start_page = (i - 1) * chunk_size + 1
        end_page = min(i * chunk_size, total_pages)
        print(f"\n  [청크 {i}/{len(chunks)}] 페이지 {start_page}~{end_page} 처리 중...")

        aos_file = upload_pdf_to_gemini(client, chunk_bytes)
        log.debug("  Gemini 파일 업로드 완료: %s", aos_file.name)

        parsed = extract_json_with_gemini(client, aos_file, prompt)
        delete_gemini_file(client, aos_file.name)

        result_path = results_dir / f"chunk_{i:02d}_result.json"
        result_path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  ✓ 청크 {i} 결과 저장: {result_path.name}")

        summarize_json(parsed, db_target, f"청크 {i} — 페이지 {start_page}~{end_page}")
        chunk_results.append(parsed)
        time.sleep(cfg.API_DELAY_SECONDS)

    # ── Step 3: 병합 결과 비교 ──────────────────────────────────────────────
    section_header(f"STEP 3 — 병합 결과 vs 기존 저장 파일 [{doc_name}]")

    merged = merge_chunk_results(chunk_results)
    merged_path = doc_dir / "merged_result.json"
    merged_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  ✓ 병합 결과 저장: {merged_path}")

    summarize_json(merged, db_target, "병합 결과 (새로 파싱)")

    if existing_output.exists():
        existing = json.loads(existing_output.read_text(encoding="utf-8"))
        summarize_json(existing, db_target, f"기존 저장 파일 ({existing_output})")
    else:
        print(f"\n  (기존 출력 파일 없음: {existing_output})")

    print(f"\n  결과물 위치: {doc_dir}/")


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    DEBUG_DIR.mkdir(exist_ok=True)

    # data.json 로드
    data_path = Path("data.json")
    if not data_path.exists():
        log.error("data.json을 찾을 수 없습니다. python main.py --force-scrape 로 먼저 생성하세요.")
        sys.exit(1)

    data: dict = json.loads(data_path.read_text(encoding="utf-8"))

    # 섹션 필터
    if args.section:
        data = {k: v for k, v in data.items() if args.section.lower() in k.lower()}
        if not data:
            log.error("'%s' 섹션을 찾을 수 없습니다.", args.section)
            sys.exit(1)

    # classifier로 DB 태스크 빌드
    db_tasks = build_db_tasks(data)

    # db_target 필드 주입 + 전체 태스크 리스트 구성
    all_tasks: list[dict] = []
    for db_target, task_list in db_tasks.items():
        for task in task_list:
            all_tasks.append({**task, "db_target": db_target})

    # 문서명 키워드 필터
    if args.doc:
        all_tasks = [t for t in all_tasks if args.doc.lower() in t["name"].lower()]
    # DB 타겟 필터
    if args.db:
        all_tasks = [t for t in all_tasks if t["db_target"] == args.db]

    if not all_tasks:
        log.error("처리할 문서가 없습니다. 필터 조건을 확인하세요.")
        sys.exit(1)

    # 샘플링
    if args.sample and args.sample < len(all_tasks):
        total = len(all_tasks)
        rng = random.Random(args.seed)
        all_tasks = rng.sample(all_tasks, args.sample)
        seed_info = f"  시드: {args.seed}" if args.seed is not None else ""
        print(f"\n샘플링: {args.sample}개 선택 (전체 {total}개 중){seed_info}")

    print(f"\n처리 대상 문서: {len(all_tasks)}개")
    for t in all_tasks:
        print(f"  [{t['db_target']}] {t['name']}")

    # Gemini 클라이언트 (dry-run이면 불필요)
    client = None
    if not args.dry_run:
        env = dotenv_values(".env")
        if not env.get("GEMINI_API_KEY"):
            log.error("GEMINI_API_KEY가 .env에 없습니다.")
            sys.exit(1)
        client = genai.Client(api_key=env["GEMINI_API_KEY"])

    # 문서별 처리
    succeeded, failed = 0, 0
    for i, task in enumerate(all_tasks, start=1):
        print(f"\n\n{'#' * 60}")
        print(f"  문서 {i}/{len(all_tasks)}: {task['name']}")
        print(f"{'#' * 60}")
        try:
            process_doc(task, client, dry_run=args.dry_run)
            succeeded += 1
        except Exception as e:
            log.error("처리 실패 [%s]: %s", task["name"], e, exc_info=True)
            failed += 1

    section_header("디버그 완료")
    print(f"\n  성공: {succeeded}개  실패: {failed}개")
    print(f"  결과물 위치: {DEBUG_DIR}/")


if __name__ == "__main__":
    main()
