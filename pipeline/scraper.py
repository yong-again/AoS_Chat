"""
PDF URL 인덱스 스크래핑/캐시 모듈.

요구사항:
- URL은 한 번만 스크래핑(캐시 재사용)해서 불필요한 API 호출 방지
- api 호출 -> data 가공 -> 데이터 저장 흐름 분리
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from dotenv import dotenv_values
from firecrawl import FirecrawlApp

from core import config as cfg
from core.logging_config import get_logger
from core.utils import load_json, save_json

log = get_logger(__name__)


def fetch_markdown(url: str, *, firecrawl_api_key: str) -> str:
    app = FirecrawlApp(api_key=firecrawl_api_key)
    doc = app.scrape(url, formats=["markdown", "html"])
    return doc.markdown if doc.markdown else ""


def parse_pdf_index(markdown_content: str) -> dict[str, dict[str, str]]:
    """
    마크다운에서 #### 섹션과 [텍스트](.pdf) 링크 파싱.
    반환: { "섹션명": { "문서명": "pdf_url", ... }, ... }
    """
    data: dict[str, dict[str, str]] = {}
    current_key: Optional[str] = None

    for line in markdown_content.split("\n"):
        title_match = re.search(r"^####\s+(.+)$", line)
        if title_match:
            raw_title = title_match.group(1).strip()
            # 섹션명 끝에 붙는 숫자 제거(다운로드 개수)
            current_key = re.sub(r"\s*\d+\s*$", "", raw_title)
            if current_key and current_key not in data:
                data[current_key] = {}
            continue

        pdf_match = re.search(r"\[([^\]]+)\]\((https?://[^\s)]+\.pdf)", line)
        if pdf_match and current_key:
            doc_name = pdf_match.group(1).strip()
            pdf_url = pdf_match.group(2).strip()
            data[current_key][doc_name] = pdf_url

    return data


def get_or_scrape_pdf_index(
    *,
    cache_path: str | Path,
    env_path: str = ".env",
    force: bool = False,
) -> dict[str, dict[str, str]]:
    """
    캐시가 있으면 로드, 없으면 스크래핑 후 저장.
    """
    cache_path = Path(cache_path)
    if cache_path.exists() and not force:
        log.info("pdf index 캐시 사용: %s", cache_path)
        return load_json(cache_path)

    config = dotenv_values(env_path)
    markdown = fetch_markdown(cfg.AOS_DOWNLOADS_URL, firecrawl_api_key=config["FIRECRAWL_API_KEY"])
    data = parse_pdf_index(markdown)

    save_json(cache_path, data)
    log.info("pdf index 저장: %s", cache_path)
    return data
