"""
전체 파이프라인 실행 스크립트.

흐름: 스크래핑(또는 캐시 로드) → PDF 파싱(Gemini) → JSON 저장

사용법:
    python run_all.py                        # data.json 전체 처리
    python run_all.py --dry-run              # 분류 요약만 출력, API 호출 없음
    python run_all.py --force-scrape         # 웹에서 PDF 목록 재스크래핑 후 처리
    python run_all.py --section "Spearhead"  # 특정 섹션만 처리
"""

from __future__ import annotations

import argparse
import sys

from dotenv import dotenv_values

from core.logging_config import get_logger, setup_logging
from pipeline import process_aos_pipeline
from pipeline.scraper import get_or_scrape_pdf_index
from core.utils import default_cache_path, default_outputs_dir

log = get_logger(__name__)

ENV_PATH = "/workspace/AoS_Chat/.env"
OUTPUT_DIR = str(default_outputs_dir())


def count_docs(data: dict[str, dict[str, str]]) -> int:
    return sum(len(items) for items in data.values())


def filter_by_section(
    data: dict[str, dict[str, str]], section: str
) -> dict[str, dict[str, str]]:
    matched = {k: v for k, v in data.items() if section.lower() in k.lower()}
    if not matched:
        log.warning("섹션 '%s' 에 해당하는 항목이 없습니다.", section)
    return matched


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(description="AoS PDF 전체 파이프라인")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="분류 요약만 출력하고 API 호출은 하지 않음",
    )
    parser.add_argument(
        "--force-scrape",
        action="store_true",
        help="캐시를 무시하고 웹에서 PDF 목록 재스크래핑",
    )
    parser.add_argument(
        "--section",
        type=str,
        default=None,
        help="특정 섹션 이름(부분 일치)만 처리 (예: 'Spearhead', 'Faction Packs')",
    )
    args = parser.parse_args()

    dotenv_values(ENV_PATH)  # .env 파일 존재 확인

    # 1단계: 스크래핑 또는 캐시 로드
    log.info("PDF 인덱스 로드 (force=%s) ...", args.force_scrape)
    data = get_or_scrape_pdf_index(
        cache_path=default_cache_path(),
        env_path=ENV_PATH,
        force=args.force_scrape,
    )

    # 2단계: 섹션 필터 (옵션)
    if args.section:
        data = filter_by_section(data, args.section)
        if not data:
            sys.exit(1)

    total = count_docs(data)
    log.info("처리 대상: 섹션 %d개 / 문서 %d개", len(data), total)
    for section, items in data.items():
        log.info("  [%s] %d개", section, len(items))

    # 3단계: 전체 파이프라인 실행 (PDF 파싱 → 저장)
    process_aos_pipeline(
        pdf_data_dict=data,
        config_path=ENV_PATH,
        output_dir=OUTPUT_DIR,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
