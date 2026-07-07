"""
Wahapedia AoS4 팩션 룰 페이지 스크래퍼.

URL 구조
--------
팩션 루트 페이지에 팩션 룰 전체가 정적 HTML로 모여 있습니다:

    https://wahapedia.ru/aos4/factions/{slug}/

h2 섹션 흐름 (팩션에 따라 일부만 존재):

    Books                    책 구매 테이블 — 스킵
    Designers' Commentary    팩션 FAQ
    Faction Rules / Battle Traits / Battle Formations / Heroic Traits /
    Artefacts of Power / Spell·Prayer·Manifestation Lore / (고유 섹션들)
                             → faction_db (category=army_rules)
    PATH TO GLORY            → 기본 스킵 (앱 DB 분류에 없음)
    SPEARHEAD                → 이후 전부 spearhead_db (category=spearhead_rules)
    <스피어헤드 세트 이름>    SPEARHEAD 뒤에 오는 h2 = 세트명 (spearhead_name
                             메타데이터로 저장; SPEARHEAD 섹션 자체가 없는
                             팩션도 있음)

파싱 결과는 wahapedia_rules와 같은 헤딩 경로 단위 청크이며 target 필드로
적재 대상 DB를 구분해 ``wahapedia_factions/<slug>.json`` 에 캐시됩니다.
`--load-db` 옵션으로 전체 재빌드 없이 증분 적재할 수 있습니다.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup, NavigableString, Tag

from core.logging_config import get_logger
from core.utils import ensure_dir, load_json, project_dir, save_json
from pipeline.factions import FACTIONS
from pipeline.wahapedia_rules import (
    HEADING_LEVELS,
    MIN_CHUNK_CHARS,
    NOISE_SELECTORS,
    SKIP_SECTIONS,
    USER_AGENT,
    _norm,
    _split_long_text,
)

log = get_logger(__name__)

BASE_URL = "https://wahapedia.ru/aos4/factions/{slug}/"
DATA_DIR = project_dir() / "wahapedia_factions"
REQUEST_DELAY_S = 1.0
REQUEST_TIMEOUT_S = 60

# h2 제목으로 구간(모드)을 전환하는 마커
SPEARHEAD_MARKER = "SPEARHEAD"
SKIP_MODE_MARKERS = {"PATH TO GLORY"}  # 앱 DB 분류에 없는 모드 — 스킵


def parse_faction_page(html: str, faction_slug: str) -> list[dict]:
    """팩션 루트 페이지 HTML을 target(DB) 태그가 붙은 청크 리스트로 파싱."""
    faction = FACTIONS[faction_slug]
    soup = BeautifulSoup(html, "html.parser")
    wrapper = soup.select_one("#wrapper2") or soup.select_one("#wrapper") or soup

    for tag in wrapper.find_all(["script", "style", "noscript"]):
        tag.decompose()
    for sel in NOISE_SELECTORS:
        for tag in wrapper.select(sel):
            tag.decompose()
    for toc_header in wrapper.select(".contents_header"):
        target = toc_header.parent if toc_header.parent and toc_header.parent.name == "span" else toc_header
        target.decompose()

    chunks: list[dict] = []
    path: dict[int, str] = {}
    buf: list[str] = []
    skipping = False          # Books 등 헤딩 직후 블록 스킵
    mode = "faction"          # faction | spearhead | skip(PtG 등)
    spearhead_name = ""       # SPEARHEAD 이후 h2 = 세트 이름

    def flush():
        text = _norm(" ".join(buf))
        buf.clear()
        if skipping or mode == "skip" or len(text) < MIN_CHUNK_CHARS:
            return
        titles = [path[lv] for lv in sorted(path) if lv > 0]
        section = " > ".join(titles)
        for piece in _split_long_text(text):
            chunk = {
                "faction": faction["name"],
                "faction_slug": faction_slug,
                "target": "spearhead_db" if mode == "spearhead" else "faction_db",
                "category": "spearhead_rules" if mode == "spearhead" else "army_rules",
                "section": section,
                "text": piece,
            }
            if mode == "spearhead" and spearhead_name:
                chunk["spearhead_name"] = spearhead_name
            chunks.append(chunk)

    for el in wrapper.descendants:
        if isinstance(el, Tag) and el.name in HEADING_LEVELS:
            flush()
            level = HEADING_LEVELS[el.name]
            title = _norm(el.get_text(" ", strip=True))

            # 구간 전환은 h2에서만 판단 (PtG 내부의 h1 등은 모드 유지)
            if level == 1:
                upper = title.upper()
                if upper == SPEARHEAD_MARKER:
                    mode = "spearhead"
                    spearhead_name = ""
                elif upper in SKIP_MODE_MARKERS:
                    mode = "skip"
                elif mode == "spearhead":
                    # SPEARHEAD 이후의 h2는 스피어헤드 세트 이름
                    spearhead_name = title
                elif mode == "skip" and upper not in SKIP_MODE_MARKERS:
                    # PtG 뒤에 SPEARHEAD 외 일반 섹션이 다시 오는 경우는
                    # 관찰되지 않았지만, 오면 팩션 룰로 복귀
                    mode = "faction"

            skipping = title in SKIP_SECTIONS
            if skipping:
                for lv in list(path):
                    if lv >= level:
                        del path[lv]
                continue
            path[level] = title
            for lv in list(path):
                if lv > level:
                    del path[lv]
        elif isinstance(el, NavigableString):
            if skipping or mode == "skip":
                continue
            if el.find_parent(list(HEADING_LEVELS)) is not None:
                continue
            s = str(el)
            if s.strip():
                buf.append(s)
    flush()
    return chunks


def cache_path(faction_slug: str) -> Path:
    return DATA_DIR / f"{faction_slug}.json"


def fetch_faction_rules(faction_slug: str, *, force: bool = False) -> list[dict]:
    """팩션 하나의 룰 페이지를 fetch + 파싱. wahapedia_factions/<slug>.json에 캐시."""
    if faction_slug not in FACTIONS:
        raise ValueError(f"Unknown faction slug: {faction_slug}")
    cache_file = cache_path(faction_slug)

    if cache_file.exists() and not force:
        log.info("faction rules 캐시 사용: %s", cache_file)
        return load_json(cache_file)

    url = BASE_URL.format(slug=faction_slug)
    log.info("fetch: %s", url)
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT_S)
    resp.raise_for_status()
    chunks = parse_faction_page(resp.text, faction_slug)

    ensure_dir(DATA_DIR)
    save_json(cache_file, chunks, indent=1)
    n_sp = sum(1 for c in chunks if c["target"] == "spearhead_db")
    log.info("저장: %s (faction %d + spearhead %d chunks)", cache_file, len(chunks) - n_sp, n_sp)
    return chunks


def fetch_all_faction_rules(*, force: bool = False) -> dict[str, list[dict]]:
    """전 팩션 룰 페이지 fetch. 개별 실패(404 등)는 로그만 남기고 계속."""
    db: dict[str, list[dict]] = {}
    for slug in FACTIONS:
        cached = cache_path(slug).exists()
        try:
            db[slug] = fetch_faction_rules(slug, force=force)
        except requests.RequestException as e:
            log.warning("팩션 룰 스크래핑 실패, 건너뜀: %s (%s)", slug, e)
            continue
        if force or not cached:
            time.sleep(REQUEST_DELAY_S)
    return db


def chunk_payload(filepath: Path) -> dict[str, tuple[list[str], list[dict], list[str]]]:
    """캐시 파일 하나를 target(DB)별 (documents, metadatas, ids)로 변환.

    build_db.py(전체 재빌드)와 load_factions_to_chromadb(증분 적재)가 공용.
    """
    chunks = load_json(filepath)
    source_file = f"wahapedia_faction_{filepath.stem}.json"
    out: dict[str, tuple[list, list, list]] = {}
    counters: dict[str, int] = {}
    for c in chunks:
        target = c["target"]
        docs, metadatas, ids = out.setdefault(target, ([], [], []))
        header = f"{c['faction']} | {c['section']}" if c["section"] else c["faction"]
        docs.append(f"[{header}] {c['text']}")
        meta = {
            "source": source_file,
            # 기존 청크와 같은 표기("lumineth realm lords")로 팩션 필터에 걸리게 함
            "faction": c["faction_slug"].replace("-", " ").strip(),
            "type": "rule",
            "category": c["category"],
            "section": c["section"] or c["faction"],
        }
        if c.get("spearhead_name"):
            meta["spearhead_name"] = c["spearhead_name"]
        metadatas.append(meta)
        idx = counters.get(target, 0)
        counters[target] = idx + 1
        ids.append(f"{source_file}_{target}_rule_{idx}")
    return out


def load_factions_to_chromadb(db_path: str = "./my_warhammer_db") -> dict[str, int]:
    """캐시된 팩션 룰 청크를 faction_db/spearhead_db에 증분 적재.

    전체 재빌드 없이 실행 가능: 기존 wahapedia_faction_* 소스 청크를
    지우고 캐시 내용으로 다시 채운다. 반환값은 DB별 적재 청크 수.
    """
    import chromadb
    from sentence_transformers import SentenceTransformer

    from build_db import EMBEDDING_MODEL_NAME

    files = sorted(DATA_DIR.glob("*.json"))
    if not files:
        log.warning("wahapedia_factions 캐시가 없습니다 (python -m pipeline.wahapedia_factions 먼저 실행)")
        return {}

    client = chromadb.PersistentClient(path=db_path)
    collections = {
        name: client.get_collection(name)
        for name in ("faction_db", "spearhead_db")
    }
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    totals: dict[str, int] = {name: 0 for name in collections}
    for filepath in files:
        source_file = f"wahapedia_faction_{filepath.stem}.json"
        payload = chunk_payload(filepath)
        for name, collection in collections.items():
            collection.delete(where={"source": source_file})
            docs, metadatas, ids = payload.get(name, ([], [], []))
            if not docs:
                continue
            embeddings = embed_model.encode(["passage: " + d for d in docs]).tolist()
            collection.add(documents=docs, embeddings=embeddings, metadatas=metadatas, ids=ids)
            totals[name] += len(docs)
        log.info(
            "적재: %s (faction_db %d, spearhead_db %d)",
            source_file,
            len(payload.get("faction_db", ([],))[0]),
            len(payload.get("spearhead_db", ([],))[0]),
        )

    log.info("팩션 룰 적재 완료: %s", totals)
    return totals


if __name__ == "__main__":
    import argparse

    from core.logging_config import setup_logging

    from pipeline.factions import resolve_faction_slug

    setup_logging()

    parser = argparse.ArgumentParser(description="Wahapedia AoS4 팩션 룰 스크래퍼")
    parser.add_argument(
        "factions",
        nargs="*",
        help="팩션 slug 또는 이름 (미지정 시 전체). 예: nighthaunt 'Sons of Behemat'",
    )
    parser.add_argument("--force", action="store_true", help="캐시 무시하고 다시 스크래핑")
    parser.add_argument("--load-db", action="store_true", help="스크래핑 후 faction_db/spearhead_db에 증분 적재")
    args = parser.parse_args()

    if args.factions:
        for name in args.factions:
            slug = resolve_faction_slug(name)
            if slug is None:
                parser.error(f"알 수 없는 팩션: {name!r}")
            chunks = fetch_faction_rules(slug, force=args.force)
            n_sp = sum(1 for c in chunks if c["target"] == "spearhead_db")
            print(f"{slug}: faction {len(chunks) - n_sp} + spearhead {n_sp} chunks")
    else:
        db = fetch_all_faction_rules(force=args.force)
        for slug, chunks in db.items():
            n_sp = sum(1 for c in chunks if c["target"] == "spearhead_db")
            print(f"{slug}: faction {len(chunks) - n_sp} + spearhead {n_sp} chunks")

    if args.load_db:
        load_factions_to_chromadb()
