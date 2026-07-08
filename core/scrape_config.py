"""
Wahapedia 스크래핑 설정 모음.

pipeline/wahapedia.py(워스크롤), pipeline/wahapedia_rules.py(룰 페이지),
pipeline/wahapedia_factions.py(팩션 룰)에 흩어져 있던 HTML/CSS 셀렉터와
파싱용 정규식을 한곳에서 관리한다. Wahapedia 마크업이 바뀌면 이 파일만
수정하면 된다.
"""
import re

# ─── HTTP 요청 공통 설정 ─────────────────────────────────────────────────────
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
REQUEST_DELAY_S = 1.0   # 페이지 요청 사이 대기(예의상)
REQUEST_TIMEOUT_S = 60

# ─── 룰/팩션 페이지 공통 구조 ─────────────────────────────────────────────────
# 잘못 중첩된 HTML 때문에 #wrapper에는 본문 일부만 담긴다 → 상위 #wrapper2 우선
WRAPPER_SELECTORS = ("#wrapper2", "#wrapper")
TOC_HEADER_SELECTOR = ".contents_header"   # 목차 (부모 span까지 제거)
STRIP_TAGS = ("script", "style", "noscript")

HEADING_LEVELS = {"h1": 0, "h2": 1, "h3": 2, "h4": 3}
NOISE_SELECTORS = (
    ".tooltip_templates", ".noprint", ".NavBtn", ".NavWrapper_M",
    ".NavDropdown", ".tooltipContents",
    ".page_ads_wrapper", ".page_breaker_ads", ".page_ads_br1",
    ".page_ads_br2", ".page_ads_br3",
)
SKIP_SECTIONS = {"Books"}   # 책 구매/개정 이력 테이블 — 룰 내용 아님

# ─── 청킹 파라미터 ───────────────────────────────────────────────────────────
MIN_CHUNK_CHARS = 40        # 목차 조각/빈 섹션 버리는 기준
MAX_CHUNK_CHARS = 1800      # 이보다 길면 문장 경계에서 분할 (e5-base 512토큰 한계 내)
CHUNK_OVERLAP_CHARS = 150   # 슬라이딩 윈도우: 직전 청크 꼬리를 다음 청크 앞에 겹침

# ─── 팩션 페이지 구간(모드) 마커 ─────────────────────────────────────────────
SPEARHEAD_MARKER = "SPEARHEAD"          # h2 제목이 이 값이면 스피어헤드 구간 시작
SKIP_MODE_MARKERS = {"PATH TO GLORY"}   # 앱 DB 분류에 없는 모드 — 스킵

# ─── 워스크롤(datasheet) 셀렉터 ──────────────────────────────────────────────
WS_DATASHEET = "div.datasheet"
WS_HEADER = ".wsHeaderIn"
WS_ADD_NAME = ".wsAddName"              # 이름 부제 (예: "on Dracoth")
WS_PIC_SEARCH = ".picSearch"            # 이미지 검색 아이콘 (제거 대상)
# 워드 세이브가 있는 유닛은 .AoS_profile_Ward(.wsWard div 추가) 사용
WS_STATS_PROFILE = ".AoS_profile, .AoS_profile_Ward"
WS_STAT_MOVE = ".wsMove"
WS_STAT_HEALTH = ".wsWounds"
WS_STAT_SAVE = ".wsSave"
WS_STAT_CONTROL = ".wsBravery"
WS_STAT_WARD = ".wsWard"

# 무기 테이블
WS_WEAPON_TABLE = "table.wTable"
WS_ROW_HEADER_CLASS = "wsHeaderRow"     # 섹션 헤더 행 (RANGED/MELEE 구분)
WS_ROW_DATA_CLASS = "wsDataRow"
WS_ROW_DATA_SHORT_CLASS = "wsDataRow_short"
WS_CELL_CLASS = "wsCell"
WS_DATA_CELL_LONG = "td.wsDataCell_long"
WS_WEAPON_ABILITY = ".wsWeaponAbility"

# 배틀 프로필 / 어빌리티 / 키워드
WS_BATTLE_PROFILE = ".PitchedBattleProfile"
WS_ABILITY_BODY = ".abBody"
WS_ABILITY_HEADER_CLASS = "abHeader"    # 발동 타이밍 (바디 앞 테이블의 td)
WS_KEYWORD_LINES = ".wsKeywordLine1, .wsKeywordLine2"

# ─── 파싱용 정규식 ───────────────────────────────────────────────────────────
RE_WHITESPACE = re.compile(r"\s+")
RE_NON_WORD = re.compile(r"\W+")

# 배틀 프로필 텍스트 파싱
RE_UNIT_SIZE = re.compile(r"Unit Size\s*:?\s*(\d+)")
RE_POINTS = re.compile(r"Points\s*:?\s*(\d+)")
RE_BASE_SIZE = re.compile(r"Base size\s*:?\s*([\d×x.\s]+mm)")
RE_REINFORCED = re.compile(r"Can be reinforced\s*:?\s*(Yes|No)", re.I)
RE_REGIMENT_OPTIONS = re.compile(r"Regiment Options\s*:?\s*(.+)$")

# 어빌리티 본문 파싱 (Declare: ... Effect: ...)
RE_DECLARE = re.compile(r"Declare\s*:\s*(.*?)(?:Effect\s*:|$)", re.S)
RE_EFFECT = re.compile(r"Effect\s*:\s*(.*)$", re.S)
