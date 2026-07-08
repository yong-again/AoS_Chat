"""
Wahapedia AoS4 "The Rules" 페이지 스크래퍼.

URL 구조
--------
룰 문서는 페이지당 정적 HTML 하나입니다 (끝에 슬래시 필수, .html 아님):

    https://wahapedia.ru/aos4/the-rules/{slug}/

본문은 ``#wrapper`` 안에 있고 h1(문서 제목) → h2 → h3 → h4 헤딩 계층으로
구성됩니다. FAQ 항목은 ``.faq`` div(Q/A 쌍)로 본문에 섞여 있습니다.
노이즈: <script>(Books 테이블 토글 등), ``.tooltip_templates``,
``.noprint``, ``.NavBtn``, 광고 래퍼(``.page_ads_wrapper`` 등), "Books" 섹션.

파싱 결과는 헤딩 경로(section) 단위 청크 리스트로,
``wahapedia_rules/<slug>.json`` 에 캐시됩니다:

    [{"page": "The Core Rules", "category": "core_rules",
      "section": "1.0 Core Concepts > 1.1 ...", "text": "..."}, ...]

`--load-db` 옵션으로 전체 재빌드 없이 rule_db 컬렉션에 증분 적재할 수
있습니다(기존 wahapedia_* 소스 삭제 후 재적재).
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup, NavigableString, Tag

from core.logging_config import get_logger
from core.utils import ensure_dir, load_json, project_dir, save_json

log = get_logger(__name__)

BASE_URL = "https://wahapedia.ru/aos4/the-rules/{slug}/"
from core.scrape_config import (
    CHUNK_OVERLAP_CHARS,
    HEADING_LEVELS,
    MAX_CHUNK_CHARS,
    MIN_CHUNK_CHARS,
    NOISE_SELECTORS,
    REQUEST_DELAY_S,
    REQUEST_TIMEOUT_S,
    RE_WHITESPACE,
    SKIP_SECTIONS,
    STRIP_TAGS,
    TOC_HEADER_SELECTOR,
    USER_AGENT,
    WRAPPER_SELECTORS,
)

DATA_DIR = project_dir() / "data" / "wahapedia_rules"

# default=True 인 페이지가 기본 스크래핑 대상.
# (First Blood / Path to Glory는 요청 범위 밖이라 기본 제외, slug 지정 시 사용 가능)
RULES_PAGES: dict[str, dict] = {
    "quick-start-guide": {"name": "Quick Start Guide", "category": "quick_start", "default": True},
    "the-core-rules": {"name": "The Core Rules", "category": "core_rules", "default": True},
    "faqs": {"name": "FAQs", "category": "faq", "default": True},
    # General's Handbooks (season 메타데이터로 앱에서 현행/과거 시즌 필터)
    "general-s-handbook-2024-25": {"name": "General's Handbook 2024-25", "category": "generals_handbook", "season": "2024-25", "default": True},
    "general-s-handbook-2025-26": {"name": "General's Handbook 2025-26", "category": "generals_handbook", "season": "2025-26", "default": True},
    "general-s-handbook-2026-27": {"name": "General's Handbook 2026-27", "category": "generals_handbook", "season": "2026-27", "default": True},
    # Spearhead Battlepack — 룰 질문(rule_db)과 스피어헤드 질문(spearhead_db)
    # 어느 쪽으로 라우팅되어도 검색되도록 두 DB에 모두 적재
    "fire-and-jade": {"name": "Fire and Jade", "category": "spearhead_battlepack", "default": True, "db": ("rule_db", "spearhead_db")},
    "sand-and-bone": {"name": "Sand and Bone", "category": "spearhead_battlepack", "default": True, "db": ("rule_db", "spearhead_db")},
    "city-of-ash": {"name": "City of Ash", "category": "spearhead_battlepack", "default": True, "db": ("rule_db", "spearhead_db")},
    "spearhead-doubles": {"name": "Spearhead Doubles", "category": "spearhead_battlepack", "default": True, "db": ("rule_db", "spearhead_db")},
    # Matched Play Battlepack
    "first-blood": {"name": "First Blood", "category": "matched_play_battlepack", "default": False},
    # Path to Glory Battlepack
    "ascension": {"name": "Ascension", "category": "path_to_glory", "default": False},
    "ravaged-coast": {"name": "Ravaged Coast", "category": "path_to_glory", "default": False},
    "blighted-wilds": {"name": "Blighted Wilds", "category": "path_to_glory", "default": False},
}

def _norm(text: str) -> str:
    return RE_WHITESPACE.sub(" ", text).strip()


def _overlap_tail(piece: str, overlap_chars: int) -> str:
    """청크 끝에서 오버랩으로 쓸 꼬리(마지막 1~2문장, ~overlap_chars자)를 추출.

    꼬리가 문장 중간에서 시작하지 않도록 첫 문장 경계('. ') 이후부터,
    문장 경계가 없으면 첫 단어 경계 이후부터 시작한다."""
    if overlap_chars <= 0 or len(piece) <= overlap_chars:
        return ""
    tail = piece[-overlap_chars:]
    sent = tail.find(". ")
    if sent != -1 and sent + 2 < len(tail):
        return tail[sent + 2:].strip()
    space = tail.find(" ")
    if space != -1:
        return tail[space + 1:].strip()
    return tail.strip()


def _split_long_text(
    text: str,
    max_chars: int = MAX_CHUNK_CHARS,
    overlap_chars: int = CHUNK_OVERLAP_CHARS,
) -> list[str]:
    """긴 섹션을 문장 경계 근처에서 max_chars 이하 조각으로 분할.

    슬라이딩 윈도우: 직전 청크의 마지막 1~2문장(~overlap_chars자)을 다음
    청크 시작 부분에 겹쳐, 청크 경계에서 문맥이 끊기지 않게 한다.
    (cut ≥ max_chars//2 > overlap_chars 이므로 항상 전진 — 무한 루프 없음)"""
    if len(text) <= max_chars:
        return [text]
    parts = []
    rest = text
    while len(rest) > max_chars:
        cut = rest.rfind(". ", max_chars // 2, max_chars)
        if cut == -1:
            cut = rest.rfind(" ", max_chars // 2, max_chars)
        if cut == -1:
            cut = max_chars
        piece = rest[: cut + 1].strip()
        parts.append(piece)
        tail = _overlap_tail(piece, overlap_chars)
        remainder = rest[cut + 1:].strip()
        rest = f"{tail} {remainder}".strip() if tail else remainder
    if rest:
        parts.append(rest)
    return parts


def parse_rules_page(html: str, page_slug: str) -> list[dict]:
    """룰 페이지 HTML 하나를 헤딩 경로 단위 청크 리스트로 파싱."""
    page = RULES_PAGES[page_slug]
    soup = BeautifulSoup(html, "html.parser")
    # 잘못 중첩된 HTML 때문에 #wrapper에는 본문 일부만 담긴다 → 상위 #wrapper2 사용
    wrapper = next(
        (w for sel in WRAPPER_SELECTORS if (w := soup.select_one(sel)) is not None),
        soup,
    )

    for tag in wrapper.find_all(list(STRIP_TAGS)):
        tag.decompose()
    for sel in NOISE_SELECTORS:
        for tag in wrapper.select(sel):
            tag.decompose()
    # 목차: .contents_header를 감싼 부모 span이 목차 전체
    for toc_header in wrapper.select(TOC_HEADER_SELECTOR):
        target = toc_header.parent if toc_header.parent and toc_header.parent.name == "span" else toc_header
        target.decompose()

    chunks: list[dict] = []
    path: dict[int, str] = {}   # 헤딩 레벨 → 제목 (현재 섹션 경로)
    buf: list[str] = []
    skipping = False

    def flush():
        # 키워드 span 등 인라인 요소 사이에 공백이 없는 경우가 많아
        # 텍스트 노드 사이에 공백을 넣는다 (드롭캡은 목차에만 있어 제거됨)
        text = _norm(" ".join(buf))
        buf.clear()
        if skipping or len(text) < MIN_CHUNK_CHARS:
            return
        # h1(문서 제목)은 page 필드로 이미 있으므로 섹션 경로에서 제외
        titles = [path[lv] for lv in sorted(path) if lv > 0]
        section = " > ".join(titles)
        for piece in _split_long_text(text):
            chunks.append(
                {
                    "page": page["name"],
                    "category": page["category"],
                    "section": section,
                    "text": piece,
                }
            )

    for el in wrapper.descendants:
        if isinstance(el, Tag) and el.name in HEADING_LEVELS:
            flush()
            level = HEADING_LEVELS[el.name]
            title = _norm(el.get_text(" ", strip=True))
            # 스킵 섹션(Books 등)은 헤딩 직후의 텍스트 블록만 건너뛴다.
            # (FAQ 페이지처럼 이후 h3 전부가 실제 내용인 경우가 있어
            # 다음 헤딩이 나오면 무조건 스킵 해제)
            skipping = title in SKIP_SECTIONS
            if skipping:
                # 스킵된 헤딩은 이후 섹션 경로에 남기지 않는다
                for lv in list(path):
                    if lv >= level:
                        del path[lv]
                continue
            path[level] = title
            for lv in list(path):
                if lv > level:
                    del path[lv]
        elif isinstance(el, NavigableString):
            if skipping:
                continue
            # 헤딩 내부 텍스트는 섹션 제목으로만 사용
            if el.find_parent(list(HEADING_LEVELS)) is not None:
                continue
            s = str(el)
            if s.strip():
                buf.append(s)
    flush()
    return chunks


def cache_path(page_slug: str) -> Path:
    return DATA_DIR / f"{page_slug}.json"


def fetch_rules_page(page_slug: str, *, force: bool = False) -> list[dict]:
    """룰 페이지 하나를 fetch + 파싱. 결과는 wahapedia_rules/<slug>.json에 캐시."""
    if page_slug not in RULES_PAGES:
        raise ValueError(f"Unknown rules page slug: {page_slug}")
    cache_file = cache_path(page_slug)

    if cache_file.exists() and not force:
        log.info("rules 캐시 사용: %s", cache_file)
        return load_json(cache_file)

    url = BASE_URL.format(slug=page_slug)
    log.info("fetch: %s", url)
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT_S)
    resp.raise_for_status()
    chunks = parse_rules_page(resp.text, page_slug)

    ensure_dir(DATA_DIR)
    save_json(cache_file, chunks, indent=1)
    log.info("저장: %s (%d chunks)", cache_file, len(chunks))
    return chunks


def fetch_all_rules(*, force: bool = False, include_optional: bool = False) -> dict[str, list[dict]]:
    """기본 대상 페이지 전체 fetch. 개별 실패(404 등)는 로그만 남기고 계속."""
    db: dict[str, list[dict]] = {}
    for slug, page in RULES_PAGES.items():
        if not page["default"] and not include_optional:
            continue
        cached = cache_path(slug).exists()
        try:
            db[slug] = fetch_rules_page(slug, force=force)
        except requests.RequestException as e:
            log.warning("룰 페이지 스크래핑 실패, 건너뜀: %s (%s)", slug, e)
            continue
        if force or not cached:
            time.sleep(REQUEST_DELAY_S)
    return db


def chunk_payload(filepath: Path) -> tuple[list[str], list[dict], list[str]]:
    """캐시 파일 하나를 (documents, metadatas, ids)로 변환.

    build_db.py(전체 재빌드)와 load_rules_to_chromadb(증분 적재)가 공용.
    """
    chunks = load_json(filepath)
    source_file = f"wahapedia_{filepath.stem}.json"
    season = (RULES_PAGES.get(filepath.stem) or {}).get("season")
    docs = [
        f"[{c['page']} | {c['section']}] {c['text']}" if c["section"] else f"[{c['page']}] {c['text']}"
        for c in chunks
    ]
    metadatas = []
    for c in chunks:
        meta = {
            "source": source_file,
            "type": "rule",
            "category": c["category"],
            "page": c["page"],
            "section": c["section"] or c["page"],
        }
        if season:
            meta["season"] = season
        metadatas.append(meta)
    ids = [f"{source_file}_rule_{i}" for i in range(len(chunks))]
    return docs, metadatas, ids


def page_db_targets(page_slug: str) -> tuple[str, ...]:
    """페이지가 적재될 컬렉션 목록 (기본 rule_db, 배틀팩은 rule_db+spearhead_db)."""
    return (RULES_PAGES.get(page_slug) or {}).get("db", ("rule_db",))


def load_rules_to_chromadb(db_path: str = "./my_warhammer_db") -> int:
    """캐시된 룰 청크를 페이지별 대상 DB(rule_db/spearhead_db)에 증분 적재.

    전체 재빌드(build_db.py) 없이 실행 가능: 기존 wahapedia_* 소스 청크를
    지우고 캐시 파일 내용으로 다시 채운다. 반환값은 적재한 청크 수.
    """
    import chromadb
    from sentence_transformers import SentenceTransformer

    from build_db import EMBEDDING_MODEL_NAME

    files = sorted(DATA_DIR.glob("*.json"))
    if not files:
        log.warning("wahapedia_rules 캐시가 없습니다 (python -m pipeline.wahapedia_rules 먼저 실행)")
        return 0

    client = chromadb.PersistentClient(path=db_path)
    collections = {name: client.get_collection(name) for name in ("rule_db", "spearhead_db")}
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    total = 0
    for filepath in files:
        docs, metadatas, ids = chunk_payload(filepath)
        source_file = f"wahapedia_{filepath.stem}.json"
        # 대상 DB가 바뀌었을 수 있으므로 양쪽 모두에서 기존 청크 제거
        for col in collections.values():
            col.delete(where={"source": source_file})
        if not docs:
            continue
        embeddings = embed_model.encode(["passage: " + d for d in docs]).tolist()
        targets = page_db_targets(filepath.stem)
        for name in targets:
            collections[name].add(documents=docs, embeddings=embeddings, metadatas=metadatas, ids=ids)
        total += len(docs)
        log.info("적재: %s → %s (%d chunks)", source_file, "/".join(targets), len(docs))

    log.info("룰 페이지 적재 완료: 총 %d chunks", total)
    return total


if __name__ == "__main__":
    import argparse

    from core.logging_config import setup_logging

    setup_logging()

    parser = argparse.ArgumentParser(description="Wahapedia AoS4 The Rules 스크래퍼")
    parser.add_argument(
        "pages",
        nargs="*",
        help="페이지 slug (미지정 시 기본 대상 전체). 예: the-core-rules faqs",
    )
    parser.add_argument("--force", action="store_true", help="캐시 무시하고 다시 스크래핑")
    parser.add_argument("--all", action="store_true", help="기본 제외 페이지(First Blood, Path to Glory)까지 포함")
    parser.add_argument("--load-db", action="store_true", help="스크래핑 후 rule_db에 증분 적재")
    parser.add_argument("--list", action="store_true", help="지원 페이지 목록 출력")
    args = parser.parse_args()

    if args.list:
        for slug, p in RULES_PAGES.items():
            mark = "*" if p["default"] else " "
            print(f"{mark} {slug:30s} {p['category']:24s} {p['name']}")
        raise SystemExit(0)

    if args.pages:
        for slug in args.pages:
            if slug not in RULES_PAGES:
                parser.error(f"알 수 없는 페이지: {slug!r} (--list로 목록 확인)")
            chunks = fetch_rules_page(slug, force=args.force)
            print(f"{slug}: {len(chunks)} chunks")
    else:
        db = fetch_all_rules(force=args.force, include_optional=args.all)
        for slug, chunks in db.items():
            print(f"{slug}: {len(chunks)} chunks")

    if args.load_db:
        load_rules_to_chromadb()
