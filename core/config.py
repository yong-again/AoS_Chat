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

# 병렬 처리: API Rate Limit을 고려한 동시 워커 수
PIPELINE_MAX_WORKERS = 5

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
본편 매치드 플레이(aos_matched_play)와 스피어헤드(spearhead) 규칙을 완벽하게 분리하여 추출하세요.

[매우 중요한 지시사항]
1. 데이터가 없는 항목은 null로 표시하세요.
2. 스피어헤드 로고가 있는 마지막 섹션의 데이터가 본편과 섞이지 않도록 각별히 주의하세요.
3. 본편(aos_matched_play)에 등장한 유닛과 스피어헤드(spearhead)에 등장하는 유닛의 이름이 같더라도,
   스피어헤드에서는 수치(Stats, Weapons)와 특수 능력(Abilities)이 완전히 다르게 재설정되어 있습니다.
4. 스피어헤드의 워스크롤을 추출할 때 "본편과 동일" 등의 이유로 절대 생략하지 말고,
   스피어헤드 페이지에 적힌 고유의 능력치와 무기를 빠짐없이 독립적으로 추출해야 합니다.
5. spearhead_name은 스피어헤드 고유 이름만 추출하고 팩션명은 절대 포함하지 마세요.
   (예: "Vanari Bladelords", "Grundstok Trailblazers" — "Lumineth Realm-lords Vanari Bladelords" 형태 금지)
"""

RULE_PROMPT = """
이 문서는 워해머 에이지 오브 지그마의 코어 룰 및 용어집입니다.
일반 코어 룰 메커니즘(core_rules)과 스피어헤드 전용 코어 룰(spearhead_rules)을 분리하여 추출하세요.
core_rules에는 mechanics, terrain, glossary 항목을 포함하세요.
"""

BALANCE_PROMPT = """
이 문서는 워해머 에이지 오브 지그마의 배틀 프로필(포인트 및 편성 제한)입니다.
문서에 등장하는 모든 유닛과 유명 연대(Regiments of Renown)의 정보를 units 배열에 빠짐없이 추출하세요.

[매우 중요한 지시사항]
1. unit_name은 영어 원문을 유지하세요.
2. points는 "120" 또는 "150 (+10)" 형태의 문자열로 추출하세요.
3. 해당 항목의 데이터가 없다면 반드시 null을 사용하세요.
"""

DEFAULT_PROMPT = "문서의 핵심 내용을 구조화된 JSON으로 추출하세요."

SPEARHEAD_FACTION_PROMPT = """
이 문서는 워해머 에이지 오브 지그마의 특정 팩션 전용 스피어헤드(Spearhead) 규칙 문서입니다.
spearhead_name, spearhead_rules, warscrolls를 빠짐없이 추출하세요.
유닛의 워스크롤(stats, weapons, abilities, keywords)을 절대 생략해서는 안 됩니다.
"""

OTHER_PROMPT = """
이 문서는 워해머 에이지 오브 지그마의 특수 규칙(예: 기란의 재앙 등)입니다.
팩션 또는 규칙 세트별로 name, army_rules, warscrolls를 entries 배열에 추출하세요.
"""