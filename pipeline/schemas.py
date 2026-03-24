"""
AoS 파이프라인 Pydantic 스키마 정의.

각 DB 타입별 Gemini Structured Output 스키마를 정의합니다.
- FactionPackResult  : faction_db  (FACTION_PROMPT)
- RuleResult         : rule_db / spearhead_db(코어룰 성격)
- BalanceResult      : balance_db  (BALANCE_PROMPT)
- SpearheadFactionResult : spearhead_db (SPEARHEAD_FACTION_PROMPT)
- OtherResult        : other_db    (OTHER_PROMPT)
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


# ── 공통 빌딩 블록 ─────────────────────────────────────────────────────────────


class NameEffect(BaseModel):
    name: str
    effect: Optional[str] = None


class LoreEntry(BaseModel):
    name: str
    type: Optional[str] = None  # Spell / Prayer / Manifestation
    effect: Optional[str] = None


class Stats(BaseModel):
    M: Optional[str] = None
    S: Optional[str] = None
    C: Optional[str] = None
    H: Optional[str] = None


class Weapon(BaseModel):
    name: str
    type: Optional[str] = None   # Melee / Ranged
    range: Optional[str] = None
    attacks: Optional[str] = None
    hit: Optional[str] = None
    wound: Optional[str] = None
    rend: Optional[str] = None
    damage: Optional[str] = None
    ability: Optional[str] = None


class Ability(BaseModel):
    title: str
    timing: Optional[str] = None
    effect: Optional[str] = None


class Warscroll(BaseModel):
    unit_name: str
    stats: Optional[Stats] = None
    weapons: Optional[list[Weapon]] = None
    abilities: Optional[list[Ability]] = None
    keywords: Optional[list[str]] = None


# ── FACTION_PROMPT → FactionPackResult ────────────────────────────────────────


class ArmyRules(BaseModel):
    battle_traits: Optional[list[NameEffect]] = None
    battle_formations: Optional[list[NameEffect]] = None
    heroic_traits: Optional[list[NameEffect]] = None
    artefacts_of_power: Optional[list[NameEffect]] = None
    lores: Optional[list[LoreEntry]] = None


class AosMatchedPlay(BaseModel):
    army_rules: Optional[ArmyRules] = None
    warscrolls: Optional[list[Warscroll]] = None


class SpearheadData(BaseModel):
    spearhead_name: Optional[str] = None
    spearhead_rules: Optional[list[NameEffect]] = None
    warscrolls: Optional[list[Warscroll]] = None


class FactionPackResult(BaseModel):
    aos_matched_play: AosMatchedPlay
    spearhead: SpearheadData


# ── RULE_PROMPT → RuleResult ───────────────────────────────────────────────────


class RuleEntry(BaseModel):
    name: Optional[str] = None
    effect: Optional[str] = None


class CoreRules(BaseModel):
    mechanics: Optional[list[RuleEntry]] = None
    terrain: Optional[list[RuleEntry]] = None
    glossary: Optional[list[RuleEntry]] = None


class SpearheadCoreRules(BaseModel):
    mechanics: Optional[list[RuleEntry]] = None


class RuleResult(BaseModel):
    core_rules: CoreRules
    spearhead_rules: SpearheadCoreRules


# ── BALANCE_PROMPT → BalanceResult ────────────────────────────────────────────
# 저장 시 units 필드를 unwrap하여 기존과 동일하게 배열로 저장됩니다.


class BalanceEntry(BaseModel):
    unit_name: str
    points: Optional[str] = None
    unit_size: Optional[str] = None
    regiment_options: Optional[str] = None


class BalanceResult(BaseModel):
    units: list[BalanceEntry]


# ── SPEARHEAD_FACTION_PROMPT → SpearheadFactionResult ─────────────────────────


class SpearheadFactionResult(BaseModel):
    spearhead: SpearheadData


# ── OTHER_PROMPT → OtherResult ────────────────────────────────────────────────
# 저장 시 entries 필드를 unwrap하여 기존과 동일하게 배열로 저장됩니다.


class OtherEntry(BaseModel):
    name: str
    army_rules: Optional[list[NameEffect]] = None
    warscrolls: Optional[list[Warscroll]] = None


class OtherResult(BaseModel):
    entries: list[OtherEntry]
