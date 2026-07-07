"""
Wahapedia AoS4 팩션 레지스트리.

`csv_file/Factions.csv`의 wahapedia URL slug 기준 목록으로,
warscroll 스크래퍼(`pipeline.wahapedia`)의 URL 구성과
자유 형식 팩션명 → slug 매핑에 사용합니다.
"""

from __future__ import annotations

import re

GRAND_ALLIANCES = ("Order", "Chaos", "Death", "Destruction", "Universal")

FACTIONS: dict[str, dict[str, str]] = {
    # Order
    "stormcast-eternals": {"id": "SE", "name": "Stormcast Eternals", "alliance": "Order"},
    "cities-of-sigmar": {"id": "CoS", "name": "Cities of Sigmar", "alliance": "Order"},
    "sylvaneth": {"id": "SY", "name": "Sylvaneth", "alliance": "Order"},
    "lumineth-realm-lords": {"id": "LRL", "name": "Lumineth Realm-lords", "alliance": "Order"},
    "daughters-of-khaine": {"id": "DoK", "name": "Daughters of Khaine", "alliance": "Order"},
    "idoneth-deepkin": {"id": "ID", "name": "Idoneth Deepkin", "alliance": "Order"},
    "kharadron-overlords": {"id": "KO", "name": "Kharadron Overlords", "alliance": "Order"},
    "fyreslayers": {"id": "FY", "name": "Fyreslayers", "alliance": "Order"},
    "seraphon": {"id": "SN", "name": "Seraphon", "alliance": "Order"},
    # Chaos
    "slaves-to-darkness": {"id": "StD", "name": "Slaves to Darkness", "alliance": "Chaos"},
    "blades-of-khorne": {"id": "BoK", "name": "Blades of Khorne", "alliance": "Chaos"},
    "disciples-of-tzeentch": {"id": "DoT", "name": "Disciples of Tzeentch", "alliance": "Chaos"},
    "maggotkin-of-nurgle": {"id": "MoN", "name": "Maggotkin of Nurgle", "alliance": "Chaos"},
    "hedonites-of-slaanesh": {"id": "HS", "name": "Hedonites of Slaanesh", "alliance": "Chaos"},
    "skaven": {"id": "ST", "name": "Skaven", "alliance": "Chaos"},
    "beasts-of-chaos": {"id": "BoC", "name": "Beasts of Chaos", "alliance": "Chaos"},
    "helsmiths-of-hashut": {"id": "HoH", "name": "Helsmiths of Hashut", "alliance": "Chaos"},
    # Death
    "soulblight-gravelords": {"id": "SG", "name": "Soulblight Gravelords", "alliance": "Death"},
    "flesh-eater-courts": {"id": "FE", "name": "Flesh-eater Courts", "alliance": "Death"},
    "nighthaunt": {"id": "NT", "name": "Nighthaunt", "alliance": "Death"},
    "ossiarch-bonereapers": {"id": "OB", "name": "Ossiarch Bonereapers", "alliance": "Death"},
    # Destruction
    "ironjawz": {"id": "IJ", "name": "Ironjawz", "alliance": "Destruction"},
    "kruleboyz": {"id": "KB", "name": "Kruleboyz", "alliance": "Destruction"},
    "gloomspite-gitz": {"id": "GG", "name": "Gloomspite Gitz", "alliance": "Destruction"},
    "ogor-mawtribes": {"id": "OM", "name": "Ogor Mawtribes", "alliance": "Destruction"},
    "sons-of-behemat": {"id": "SoB", "name": "Sons of Behemat", "alliance": "Destruction"},
    "bonesplitterz": {"id": "BS", "name": "Bonesplitterz", "alliance": "Destruction"},
    # 공용 (팩션 무관 엔들리스 스펠)
    "endless-spells": {"id": "EnS", "name": "Endless Spells", "alliance": "Universal"},
}

NAME_TO_SLUG = {v["name"].lower(): k for k, v in FACTIONS.items()}
ID_TO_SLUG = {v["id"]: k for k, v in FACTIONS.items()}


def _norm_name(name: str) -> str:
    """소문자화 + 구두점 제거: 'Flesh-eater Courts', 'flesh eater courts',
    slug 표기가 모두 같은 값으로 비교되도록 정규화."""
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def resolve_faction_slug(name: str) -> str | None:
    """자유 형식 팩션명(질의문/로스터 등)을 wahapedia slug로 매핑."""
    if not name:
        return None
    if name.strip().lower() in FACTIONS:
        return name.strip().lower()
    if name.strip() in ID_TO_SLUG:
        return ID_TO_SLUG[name.strip()]
    n = _norm_name(name)
    for slug, f in FACTIONS.items():
        fn = _norm_name(f["name"])
        if n == fn or n in fn or fn in n:
            return slug
    return None
