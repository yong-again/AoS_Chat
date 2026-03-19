"""
실행 엔트리포인트(개발/디버그용).
- pdf index 캐시 사용/스크래핑
- 1개 문서 선택 실행
"""

from __future__ import annotations

from dotenv import dotenv_values

from logging_config import get_logger, setup_logging
from scraper import get_or_scrape_pdf_index
from utils import default_cache_path
from pipeline import process_aos_pipeline

log = get_logger(__name__)


def build_single_doc_data(
    data: dict[str, dict[str, str]], *, keyword: str
) -> dict[str, dict[str, str]]:
    debug_data: dict[str, dict[str, str]] = {}
    for section, items in data.items():
        for doc_name, url in items.items():
            if keyword in doc_name:
                log.info("DEBUG match: %s", doc_name)
                debug_data.setdefault(section, {})[doc_name] = url
    return debug_data


if __name__ == "__main__":
    setup_logging()

    env_path = "/workspace/AoS_Chat/.env"
    _ = dotenv_values(env_path)  # env 파일 존재 확인용 (키는 pipeline 내부에서 다시 로드)

    data = get_or_scrape_pdf_index(cache_path=default_cache_path(), env_path=env_path, force=False)

    # 한 개 문서만 실행하고 싶을 때 keyword만 바꾸면 됨
    keyword = "Kharadron Overlords"
    single = build_single_doc_data(data, keyword=keyword)

    process_aos_pipeline(pdf_data_dict=single, config_path=env_path, output_dir="/workspace/AoS_Chat/outputs", dry_run=False)

