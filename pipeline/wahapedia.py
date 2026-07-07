"""
Wahapedia AoS4 warscroll(유닛 카드) + abilities 스크래퍼.

URL 구조
--------
팩션별 워스크롤은 정적 HTML 한 페이지에 모여 있습니다:

    https://wahapedia.ru/aos4/factions/{slug}/warscrolls.html

각 워스크롤은 ``div.datasheet`` 블록 하나이며 내부 레이아웃은 팩션과
무관하게 동일합니다:

    a[name=<Unit-Anchor>]              유닛 앵커(연대 옵션 링크에도 사용)
    .wsHeaderIn                        유닛 이름 (+ 선택적 .wsAddName 부제)
    .AoS_profile                       코어 스탯:
        .wsMove / .wsWounds / .wsSave / .wsBravery   (bravery == Control)
    table.wTable                       무기 테이블; 섹션 헤더 행(tr.wsHeaderRow)이
                                       RANGED/MELEE WEAPONS를 구분하고, 데이터 행은
                                       tr.wsDataRow, 셀 순서는 Rng Atk Hit Wnd Rnd Dmg
    .PitchedBattleProfile              Unit Size / Points / Base size /
                                       Can be reinforced / Regiment Options
    .abHeader + .abBody                어빌리티(헤더=발동 타이밍,
                                       바디=이름/Declare/Effect)
    .wsKeywordLine1 / .wsKeywordLine2  유닛 키워드 / 팩션 키워드

결과는 팩션당 JSON 파일(warscolls/<slug>.json)로 캐시되며,
빌드 파이프라인(build_db 등)이 그대로 읽어 청킹할 수 있는 구조입니다.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup, Tag

from core.logging_config import get_logger
from core.utils import ensure_dir, load_json, project_dir, save_json
from pipeline.factions import FACTIONS

log = get_logger(__name__)

BASE_URL = "https://wahapedia.ru/aos4/factions/{slug}/warscrolls.html"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
DATA_DIR = project_dir() / "warscolls"
REQUEST_DELAY_S = 1.0  # 팩션 페이지 요청 사이 대기(예의상)
REQUEST_TIMEOUT_S = 60


def _text(node: Tag | None) -> str:
    if node is None:
        return ""
    return re.sub(r"\s+", " ", node.get_text(" ", strip=True)).strip()


def _parse_stat(value: str) -> str:
    """스탯은 표시 문자열('14\"', '3+', '8') 그대로 유지.
    수치 변환('-'/'D6' 처리 포함)은 사용하는 쪽에서 담당."""
    return value.replace("”", '"').strip()


def _parse_weapons(datasheet: Tag) -> tuple[list[dict], list[dict]]:
    ranged: list[dict] = []
    melee: list[dict] = []
    table = datasheet.select_one("table.wTable")
    if table is None:
        return ranged, melee

    current = None
    for tr in table.find_all("tr"):
        classes = tr.get("class") or []
        if "wsHeaderRow" in classes:
            header = _text(tr).upper()
            if "RANGED" in header:
                current = ranged
            elif "MELEE" in header:
                current = melee
            continue
        if "wsDataRow" in classes and "wsDataRow_short" not in classes:
            cells = [td for td in tr.find_all("td") if "wsCell" in (td.get("class") or [])]
            # 행마다 td.wsDataCell_long이 여러 개 있으며, 무기 이름은 텍스트가
            # 있는 첫 셀에 있음(나머지는 레이아웃용 빈 칸)
            name = ""
            abilities = []
            for name_td in tr.select("td.wsDataCell_long"):
                for ab in name_td.select(".wsWeaponAbility"):
                    ab_text = _text(ab)
                    if ab_text:
                        abilities.append(ab_text)
                    ab.extract()
                name = _text(name_td)
                if name:
                    break
            if current is None or not name:
                continue
            values = [_parse_stat(_text(td)) for td in cells]
            # ranged 행: [Rng, Atk, Hit, Wnd, Rnd, Dmg]
            # melee 행은 선두 range 셀이 비어 있음
            if current is ranged and len(values) >= 6:
                rng, atk, hit, wnd, rnd, dmg = values[:6]
            elif len(values) >= 6:
                _, atk, hit, wnd, rnd, dmg = values[:6]
                rng = ""
            elif len(values) == 5:
                atk, hit, wnd, rnd, dmg = values
                rng = ""
            else:
                continue
            current.append(
                {
                    "name": name,
                    "range": rng,
                    "attacks": atk,
                    "hit": hit,
                    "wound": wnd,
                    "rend": rnd,
                    "damage": dmg,
                    "abilities": abilities,
                }
            )
    return ranged, melee


def _parse_battle_profile(datasheet: Tag) -> dict:
    profile = {
        "unit_size": None,
        "points": None,
        "base_size": "",
        "can_be_reinforced": False,
        "regiment_options": "",
    }
    box = datasheet.select_one(".PitchedBattleProfile")
    if box is None:
        return profile
    text = _text(box)
    m = re.search(r"Unit Size\s*:?\s*(\d+)", text)
    if m:
        profile["unit_size"] = int(m.group(1))
    m = re.search(r"Points\s*:?\s*(\d+)", text)
    if m:
        profile["points"] = int(m.group(1))
    m = re.search(r"Base size\s*:?\s*([\d×x.\s]+mm)", text)
    if m:
        profile["base_size"] = m.group(1).strip()
    m = re.search(r"Can be reinforced\s*:?\s*(Yes|No)", text, re.I)
    if m:
        profile["can_be_reinforced"] = m.group(1).lower() == "yes"
    m = re.search(r"Regiment Options\s*:?\s*(.+)$", text)
    if m:
        profile["regiment_options"] = m.group(1).strip()
    return profile


def _parse_abilities(datasheet: Tag) -> list[dict]:
    abilities = []
    for body in datasheet.select(".abBody"):
        # 발동 타이밍 헤더는 바디 바로 앞 테이블의 td.abHeader에 있음
        header_td = body.find_previous("td", class_="abHeader")
        name_tag = body.find("b")
        name = _text(name_tag).rstrip(":") if name_tag else ""
        full = _text(body)
        declare = ""
        effect = ""
        m = re.search(r"Declare\s*:\s*(.*?)(?:Effect\s*:|$)", full, re.S)
        if m:
            declare = m.group(1).strip()
        m = re.search(r"Effect\s*:\s*(.*)$", full, re.S)
        if m:
            effect = m.group(1).strip()
        abilities.append(
            {
                "name": name,
                "timing": _text(header_td),
                "declare": declare,
                "effect": effect or full,
            }
        )
    return abilities


def _parse_keywords(datasheet: Tag) -> list[str]:
    keywords: list[str] = []
    for line in datasheet.select(".wsKeywordLine1, .wsKeywordLine2"):
        # 키워드 문구는 쉼표로 구분되며, 한 문구는 인접한 span.kwb 여러 개로
        # 이루어질 수 있음("STORMCAST ETERNALS"는 span 두 개)
        for phrase in _text(line).split(","):
            phrase = phrase.strip()
            if phrase:
                keywords.append(phrase)
    return keywords


def parse_warscrolls(html: str, faction_slug: str) -> list[dict]:
    """팩션 warscrolls 페이지 HTML 하나를 워스크롤 dict 리스트로 파싱."""
    soup = BeautifulSoup(html, "html.parser")
    faction = FACTIONS[faction_slug]
    warscrolls = []

    for ds in soup.select("div.datasheet"):
        header = ds.select_one(".wsHeaderIn")
        if header is None:
            continue
        # 이름에 부제 div(.wsAddName)가 붙을 수 있음. 예: "on Dracoth"
        add = header.select_one(".wsAddName")
        add_name = _text(add) if add else ""
        if add:
            add.extract()
        # 이미지 검색 아이콘 제거
        for icon in header.select(".picSearch"):
            icon.extract()
        name = _text(header)
        if add_name:
            name = f"{name} {add_name}"

        # 워드 세이브가 있는 유닛은 .AoS_profile_Ward(.wsWard div 추가) 사용
        stats = ds.select_one(".AoS_profile, .AoS_profile_Ward")
        anchor = ds.find("a", attrs={"name": True})
        ranged, melee = _parse_weapons(ds)
        ws = {
            "id": anchor["name"] if anchor else re.sub(r"\W+", "-", name),
            "name": name,
            "faction": faction["name"],
            "faction_slug": faction_slug,
            "grand_alliance": faction["alliance"],
            "move": _parse_stat(_text(stats.select_one(".wsMove")) if stats else ""),
            "health": _parse_stat(_text(stats.select_one(".wsWounds")) if stats else ""),
            "save": _parse_stat(_text(stats.select_one(".wsSave")) if stats else ""),
            "control": _parse_stat(_text(stats.select_one(".wsBravery")) if stats else ""),
            "ward": _parse_stat(_text(stats.select_one(".wsWard")) if stats else ""),
            "ranged_weapons": ranged,
            "melee_weapons": melee,
            "abilities": _parse_abilities(ds),
            "keywords": _parse_keywords(ds),
        }
        ws.update(_parse_battle_profile(ds))
        warscrolls.append(ws)

    return warscrolls


def cache_path(faction_slug: str) -> Path:
    return DATA_DIR / f"{faction_slug}.json"


def fetch_faction_warscrolls(faction_slug: str, *, force: bool = False) -> list[dict]:
    """팩션 하나를 fetch + 파싱. 결과는 warscolls/<slug>.json에 캐시."""
    if faction_slug not in FACTIONS:
        raise ValueError(f"Unknown faction slug: {faction_slug}")
    cache_file = cache_path(faction_slug)

    if cache_file.exists() and not force:
        log.info("warscroll 캐시 사용: %s", cache_file)
        return load_json(cache_file)

    url = BASE_URL.format(slug=faction_slug)
    log.info("fetch: %s", url)
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT_S)
    resp.raise_for_status()
    warscrolls = parse_warscrolls(resp.text, faction_slug)

    ensure_dir(DATA_DIR)
    save_json(cache_file, warscrolls, indent=1)
    log.info("저장: %s (%d warscrolls)", cache_file, len(warscrolls))
    return warscrolls


def fetch_all_factions(*, force: bool = False) -> dict[str, list[dict]]:
    """전 팩션 fetch. 개별 팩션 실패(404 등)는 로그만 남기고 계속 진행."""
    db: dict[str, list[dict]] = {}
    for slug in FACTIONS:
        cached = cache_path(slug).exists()
        try:
            db[slug] = fetch_faction_warscrolls(slug, force=force)
        except requests.RequestException as e:
            log.warning("팩션 스크래핑 실패, 건너뜀: %s (%s)", slug, e)
            continue
        if force or not cached:
            time.sleep(REQUEST_DELAY_S)
    return db


if __name__ == "__main__":
    import argparse

    from core.logging_config import setup_logging

    setup_logging()

    parser = argparse.ArgumentParser(description="Wahapedia AoS4 warscroll 스크래퍼")
    parser.add_argument(
        "factions",
        nargs="*",
        help="팩션 slug 또는 이름 (미지정 시 전체). 예: skaven 'Stormcast Eternals'",
    )
    parser.add_argument("--force", action="store_true", help="캐시 무시하고 다시 스크래핑")
    parser.add_argument("--list", action="store_true", help="지원 팩션 slug 목록 출력")
    args = parser.parse_args()

    if args.list:
        for slug, f in FACTIONS.items():
            print(f"{slug:26s} {f['alliance']:12s} {f['name']}")
        raise SystemExit(0)

    from pipeline.factions import resolve_faction_slug

    if args.factions:
        slugs = []
        for name in args.factions:
            slug = resolve_faction_slug(name)
            if slug is None:
                parser.error(f"알 수 없는 팩션: {name!r} (--list로 목록 확인)")
            slugs.append(slug)
        for slug in slugs:
            scrolls = fetch_faction_warscrolls(slug, force=args.force)
            print(f"{slug}: {len(scrolls)} warscrolls")
    else:
        db = fetch_all_factions(force=args.force)
        for slug, scrolls in db.items():
            print(f"{slug}: {len(scrolls)} warscrolls")
