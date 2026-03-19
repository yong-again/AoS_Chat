"""
AoS_Chat 설정 모음: 상수/키워드/프롬프트.

`pipeline.py` 등에서 `import config as cfg` 형태로 불러 사용합니다.
"""

# -----------------------------------------------------------------------------
# 스크래핑 대상
# -----------------------------------------------------------------------------

AOS_DOWNLOADS_URL = (
    "https://www.warhammer-community.com/en-gb/downloads/warhammer-age-of-sigmar/"
)

# -----------------------------------------------------------------------------
# 분류/필터링
# -----------------------------------------------------------------------------

EXCLUDE_KEYWORDS = ["Tournament Organiser", "Supplement", "Battletome:"]

DB_NAMES = ("rule_db", "faction_db", "balance_db", "spearhead_db", "other_db")

# -----------------------------------------------------------------------------
# Gemini 설정
# -----------------------------------------------------------------------------

GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_JSON_MIME = "application/json"
GEMINI_TEMPERATURE = 0.1
API_DELAY_SECONDS = 5
FILE_POLL_INTERVAL_SECONDS = 2

# PDF 청크 크기 (페이지 단위) — 문서 밀도에 따라 조정
# 포인트 표는 밀도가 높아 4~5페이지, 팩션/스피어헤드는 8페이지, 코어룰은 15페이지
CHUNK_SIZES: dict[str, int] = {
    "balance_db":   4,
    "faction_db":   8,
    "spearhead_db": 8,
    "other_db":     8,
    "rule_db":      15,
}

# -----------------------------------------------------------------------------
# 프롬프트 (카테고리별)
# -----------------------------------------------------------------------------

FACTION_PROMPT = """
이 문서는 워해머 에이지 오브 지그마 팩션 팩입니다.
본편 매치드 플레이용 규칙과 스피어헤드(Spearhead) 전용 규칙을 완벽하게 분리하여 다음 구조의 JSON 형식으로 추출하세요:

{
  "aos_matched_play": {
    "army_rules": {
      "battle_traits": [{"name": "규칙명", "effect": "설명"}],
      "battle_formations": [{"name": "포메이션명", "effect": "설명"}],
      "heroic_traits": [{"name": "트레잇명", "effect": "설명"}],
      "artefacts_of_power": [{"name": "아티팩트명", "effect": "설명"}],
      "lores": [{"name": "마법/기도명", "type": "Spell/Prayer/Manifestation", "effect": "설명"}]
    },
    "warscrolls": [
      {
        "unit_name": "유닛 이름",
        "stats": {"M": "이동", "S": "세이브", "C": "컨트롤", "H": "체력"},
        "weapons": [
          {"name": "무기명", "type": "Melee/Ranged", "range": "거리", "attacks": "횟수", "hit": "명중", "wound": "운드", "rend": "관통", "damage": "피해", "ability": "특수 능력"}
        ],
        "abilities": [
          {"title": "능력 이름", "timing": "사용 단계(예: Hero Phase)", "effect": "능력 효과 설명"}
        ],
        "keywords": ["키워드1", "키워드2"]
      }
    ]
  },
  "spearhead": {
    "spearhead_name": "스피어헤드 팩트 이름",
    "spearhead_rules": [{"name": "규칙명", "effect": "설명"}],
    "warscrolls": [
      {
        "unit_name": "유닛 이름",
        "stats": {"M": "이동", "S": "세이브", "C": "컨트롤", "H": "체력"},
        "weapons": [
          {"name": "무기명", "type": "Melee/Ranged", "range": "거리", "attacks": "횟수", "hit": "명중", "wound": "운드", "rend": "관통", "damage": "피해", "ability": "특수 능력"}
        ],
        "abilities": [
          {"title": "능력 이름", "timing": "사용 단계", "effect": "능력 효과 설명"}
        ],
        "keywords": ["키워드1", "키워드2"]
      }
    ]
  }
}

[매우 중요한 지시사항]
1. 데이터가 없는 항목은 null로 표시하세요. 
2. 스피어헤드 로고가 있는 마지막 섹션의 데이터가 본편과 섞이지 않도록 각별히 주의하세요.
3. 본편(aos_matched_play)에 등장한 유닛과 스피어헤드(spearhead)에 등장하는 유닛의 이름이 같더라도, 스피어헤드에서는 수치(Stats, Weapons)와 특수 능력(Abilities)이 완전히 다르게 재설정되어 있습니다.
4. 따라서 스피어헤드의 워스크롤을 추출할 때 "본편과 동일" 등의 이유로 절대 생략하지 말고, 스피어헤드 페이지에 적힌 고유의 능력치와 무기를 빠짐없이 독립적으로 끝까지 추출해야 합니다.
"""

RULE_PROMPT = """
이 문서는 워해머 에이지 오브 지그마의 코어 룰 및 용어집입니다.
일반 코어 룰 메커니즘과 스피어헤드 전용 코어 룰을 분리하여 JSON으로 요약 추출하세요.
{"core_rules": {"mechanics": [], "terrain": [], "glossary": []}, "spearhead_rules": {"mechanics": []}}
"""

BALANCE_PROMPT = """
이 문서는 배틀 프로필(포인트)입니다.
유닛 이름, 포인트, 유닛 사이즈, 연대(Regiment) 옵션을 정확한 테이블 형태의 JSON 리스트로 추출하세요.
"""

DEFAULT_PROMPT = "문서의 핵심 내용을 구조화된 JSON으로 추출하세요."

SPEARHEAD_FACTION_PROMPT = """
이 문서는 워해머 에이지 오브 지그마의 특정 팩션 전용 '스피어헤드(Spearhead)' 규칙 문서입니다.
이 문서의 내용을 다음 JSON 구조로 완벽하게 추출하세요. 유닛의 워스크롤을 절대 생략해서는 안 됩니다.

{
  "spearhead": {
    "spearhead_name": "스피어헤드 팩트 이름",
    "spearhead_rules": [
      {"name": "규칙명/어빌리티명/강화명", "effect": "설명"}
    ],
    "warscrolls": [
      {
        "unit_name": "유닛 이름",
        "stats": {"M": "이동", "S": "세이브", "C": "컨트롤", "H": "체력"},
        "weapons": [
          {"name": "무기명", "type": "Melee/Ranged", "range": "거리", "attacks": "횟수", "hit": "명중", "wound": "운드", "rend": "관통", "damage": "피해", "ability": "특수 능력"}
        ],
        "abilities": [
          {"title": "능력 이름", "timing": "사용 단계", "effect": "능력 효과 설명"}
        ],
        "keywords": ["키워드1", "키워드2"]
      }
    ]
  }
}
"""

OTHER_PROMPT = """
이 문서는 워해머 에이지 오브 지그마의 특수 규칙(예: 기란의 재앙 등)입니다.
문서의 내용을 다음 표준 구조에 최대한 맞추어 JSON 배열로 추출하세요:

[
  {
    "name": "팩션 또는 규칙 세트 이름",
    "army_rules": [
      {"name": "규칙명/특성명", "effect": "설명"}
    ],
    "warscrolls": [
      {
        "unit_name": "유닛 이름",
        "stats": {"M": "이동", "S": "세이브", "C": "컨트롤", "H": "체력"},
        "weapons": [
          {"name": "무기명", "type": "Melee/Ranged", "range": "거리", "attacks": "횟수", "hit": "명중", "wound": "운드", "rend": "관통", "damage": "피해", "ability": "특수 능력"}
        ],
        "abilities": [
          {"title": "능력 이름", "timing": "사용 단계", "effect": "능력 효과 설명"}
        ],
        "keywords": ["키워드1", "키워드2"]
      }
    ]
  }
]
"""