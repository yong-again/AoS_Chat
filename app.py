import json
import os
import re
import threading
import uuid
from datetime import datetime
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
from dotenv import dotenv_values
from tools import calculate_expected_damage
import pprint
import time
from core.logging_config import setup_logging, get_logger, set_log_session
from core.hybrid_search import BM25Index, rrf_fuse

# 파이프라인 전 과정 로그: 콘솔 + runtime/logs/aos_chat.log (AOS_LOG_LEVEL로 레벨 조정)
setup_logging()
log = get_logger("aos.pipeline")

# 임베딩 모델은 st.cache_resource로 전 세션이 공유하는 단일 인스턴스인데,
# Streamlit은 세션마다 스레드를 띄우므로 동시 encode() 호출이 발생할 수 있다.
# MPS(Apple GPU) 백엔드는 동시 호출에 취약해 락으로 직렬화한다.
_EMBED_LOCK = threading.Lock()

# ─── 채팅 내역 저장 ───────────────────────────────────────────────────────────
HISTORY_DIR = "runtime/chat_history"
LOG_DIR = "runtime/chat_logs"
os.makedirs(HISTORY_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def append_qa_log(session_id: str, user_msg: str, bot_msg: str, db_name: str, search_query: str) -> None:
    """Q&A 한 쌍을 일별 로그 파일에 추가합니다."""
    today = datetime.now().strftime("%Y-%m-%d")
    log_path = os.path.join(LOG_DIR, f"{today}.log")
    now = datetime.now().strftime("%H:%M:%S")
    separator = "─" * 60
    # 동시 접속 시 사용자 간 줄 교차를 막기 위해 한 번의 write로 기록
    entry = (
        f"[{now}] SESSION: {session_id}\n"
        f"[{now}] USER: {user_msg}\n"
        f"[{now}] DB: {db_name}  |  키워드: {search_query}\n"
        f"[{now}] BOT: {bot_msg}\n"
        f"{separator}\n"
    )
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(entry)

def _session_file(session_id: str) -> str:
    return os.path.join(HISTORY_DIR, f"{session_id}.json")

def save_chat_history(session_id: str, messages: list) -> None:
    """현재 세션의 채팅 내역을 JSON 파일로 저장합니다."""
    data = {
        "session_id": session_id,
        "saved_at": datetime.now().isoformat(),
        "messages": messages,
    }
    with open(_session_file(session_id), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_chat_history(session_id: str) -> list:
    """저장된 채팅 내역을 불러옵니다. 없으면 빈 리스트 반환."""
    path = _session_file(session_id)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get("messages", [])
    return []

def list_saved_sessions() -> list[dict]:
    """저장된 세션 목록을 최신순으로 반환합니다."""
    sessions = []
    for fname in sorted(os.listdir(HISTORY_DIR), reverse=True):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(HISTORY_DIR, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            sessions.append({
                "session_id": data.get("session_id", fname[:-5]),
                "saved_at": data.get("saved_at", ""),
                "message_count": len(data.get("messages", [])),
                "path": path,
            })
        except Exception:
            continue
    return sessions

st.set_page_config(page_title="Warhammer AI 룰마스터", page_icon="⚔️", layout="wide")

# ─── 테마 CSS ─────────────────────────────────────────────────────────────────
def inject_theme():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;700&family=Cinzel+Decorative:wght@400;700&display=swap');

    /* 전체 배경 */
    .stApp {
        background-color: #0f0d0a;
        background-image:
            radial-gradient(ellipse at top, #1a1208 0%, #0f0d0a 60%);
    }

    /* 메인(채팅) 영역: wide 레이아웃에서 최대폭을 제한해 가독성 유지 */
    div.block-container,
    [data-testid="stMainBlockContainer"] {
        max-width: 1100px !important;
        margin: 0 auto;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    /* 제목 */
    h1 {
        font-family: 'Cinzel Decorative', serif !important;
        color: #c9a84c !important;
        text-shadow: 0 0 18px #7a5c1e88, 0 2px 4px #000;
        letter-spacing: 0.05em;
    }

    /* 부제목 / caption */
    .stCaption, .stCaption p {
        font-family: 'Cinzel', serif !important;
        color: #8a7040 !important;
        letter-spacing: 0.04em;
    }

    /* 사이드바: 폭을 고정해 위젯 겹침 방지 */
    section[data-testid="stSidebar"] {
        background-color: #130f08 !important;
        border-right: 1px solid #3a2c10;
        min-width: 330px !important;
        max-width: 330px !important;
    }
    section[data-testid="stSidebar"] * {
        color: #c9a84c !important;
    }
    /* 장식 폰트는 제목/라벨/본문 텍스트에만 적용
       (입력 위젯 내부까지 적용하면 letter-spacing 때문에 글자가 겹침) */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] summary {
        font-family: 'Cinzel', serif !important;
        letter-spacing: 0.02em;
    }
    /* 입력/선택/버튼 위젯 내부는 기본 폰트·자간 유지 + 줄바꿈 허용 */
    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] [data-baseweb="select"] *,
    section[data-testid="stSidebar"] button {
        font-family: sans-serif !important;
        letter-spacing: normal !important;
    }
    section[data-testid="stSidebar"] .stButton button,
    section[data-testid="stSidebar"] .stDownloadButton button {
        width: 100%;
        white-space: normal;
    }

    /* 채팅 입력창 */
    .stChatInput textarea, .stChatInput input {
        background-color: #1a1510 !important;
        color: #e8d9b0 !important;
        border: 1px solid #5a4420 !important;
        border-radius: 6px !important;
        font-family: 'Cinzel', serif !important;
    }
    .stChatInput textarea:focus, .stChatInput input:focus {
        border-color: #c9a84c !important;
        box-shadow: 0 0 8px #c9a84c44 !important;
    }

    /* 채팅 메시지 버블 */
    .stChatMessage {
        background-color: #1a1510 !important;
        border: 1px solid #2e2210 !important;
        border-radius: 8px !important;
    }
    .stChatMessage p, .stChatMessage li, .stChatMessage td {
        font-family: 'Cinzel', serif !important;
        color: #e8d9b0 !important;
        line-height: 1.8 !important;
    }
    .stChatMessage strong {
        color: #c9a84c !important;
    }
    .stChatMessage code {
        background-color: #2a1f0e !important;
        color: #f0c060 !important;
        border: 1px solid #4a3520 !important;
    }

    /* 구분선 (---) */
    hr {
        border-color: #3a2c10 !important;
    }

    /* expander (추론 과정) */
    .streamlit-expanderHeader {
        font-family: 'Cinzel', serif !important;
        color: #8a7040 !important;
        background-color: #1a1510 !important;
        border: 1px solid #3a2c10 !important;
    }
    .streamlit-expanderContent {
        background-color: #130f08 !important;
        border: 1px solid #2e2210 !important;
        color: #7a6a48 !important;
        font-size: 0.85em !important;
    }

    /* 스크롤바 */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0f0d0a; }
    ::-webkit-scrollbar-thumb { background: #3a2c10; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #c9a84c; }
    </style>
    """, unsafe_allow_html=True)

inject_theme()

# ─── 접근 코드 게이트 (선택) ──────────────────────────────────────────────────
# 외부 공유(Cloudflare Tunnel 등) 시 APP_ACCESS_CODE 환경변수를 설정하면
# 코드를 아는 사람만 사용 가능. 미설정이면 게이트 없음 (로컬 사용 영향 없음).
_ACCESS_CODE = os.environ.get("APP_ACCESS_CODE", "")
if _ACCESS_CODE and not st.session_state.get("access_granted"):
    st.title("Warhammer Age of Sigmar AI 룰마스터")
    _code = st.text_input("접근 코드를 입력하세요", type="password")
    if _code == _ACCESS_CODE:
        st.session_state.access_granted = True
        st.rerun()
    elif _code:
        st.error("코드가 올바르지 않습니다.")
    st.stop()

st.title("Warhammer Age of Sigmar AI 룰마스터")
st.caption("공식 규칙 문서와 FAQ를 기반으로 답변하는 AI 심판입니다.")

# ─── 모델 설정 ────────────────────────────────────────────────────────────────
ROUTER_MODEL = "gemini-2.5-flash-lite"
EMBED_MODEL  = "intfloat/multilingual-e5-base"

# 질문 유형별 답변 프로필: 모드 → (모델, DB별 thinking 토큰 예산)
# - lookup  : 검색 결과 발췌·정리 — 추론 예산 낮게 (빠르고 저렴)
# - analysis: 규칙 근거 해석·평가 — 추론 예산 크게
#   분석 품질이 부족하면 "analysis"의 model만 "gemini-2.5-pro"로 교체
ANSWER_PROFILES = {
    "lookup": {
        "model": "gemini-2.5-flash",
        "thinking": {
            "rule_db":      4000,   # 룰 조회도 FAQ 교차 참조가 있어 약간의 추론 필요
            "faction_db":   2000,
            "balance_db":   1000,
            "spearhead_db": 2000,
            "other_db":     2000,
        },
    },
    "analysis": {
        "model": "gemini-2.5-flash",
        "thinking": {
            "rule_db":      16000,
            "faction_db":   12000,
            "balance_db":   8000,
            "spearhead_db": 12000,
            "other_db":     12000,
        },
    },
}

# ─── 라우터 프롬프트 ──────────────────────────────────────────────────────────
ROUTER_PROMPT = """
사용자의 질문을 분석하여 아래 5개 카테고리 중 가장 적합한 것을 하나만 선택하세요.
아래 규칙을 1번부터 순서대로 검사하여, 가장 먼저 해당하는 규칙 하나만 적용하세요.
1. 질문에 포인트, 점수, 비용, points, 부대 편성 등 비용/편성 관련 단어가 있으면 balance_db.
2. 질문에 "스피어헤드", "뱅가드" 단어가 있더라도, "규칙", "진행 순서", "게임 방법" 등 게임을 플레이하는 방식에 대한 질문이라면 rule_db.
   단, 특정 배틀팩 이름(Fire and Jade, Sand and Bone, City of Ash, Spearhead Doubles)이나
   특정 스피어헤드/유닛 이름이 함께 언급되면 spearhead_db.
   (예: "sand and bone에서 grundstok trailblazers로 플레이하는 과정 알려줘" → spearhead_db)
3. 질문에 "스피어헤드"가 있고 유닛/구성/스탯/정보를 묻는다면 spearhead_db.
   특정 팩션의 스피어헤드 고유 명칭(예: Hurakan Vanguard, Grundstok Trailblazers)이나 스피어헤드 배틀팩
   (Fire and Jade, Sand and Bone, City of Ash, Spearhead Doubles)의 규칙을 묻는 경우도 spearhead_db.
   (예: "카라드론 오버로드 팩션의 스피어헤드 유닛 정보를 알려줘" → spearhead_db)
   'Regiment Ability(레지먼트 어빌리티)'는 스피어헤드 전용 룰 명칭이므로 spearhead_db.
   단, 'Enhancement(인핸스먼트)'는 정규 게임 공통 용어이므로 그것만으로 spearhead_db를 선택하지 마세요.
4. 룰 용어/키워드(예: Ward, Rend, 차지)의 "의미", "뜻", "정의"를 묻는다면 rule_db.
   단, 특정 유닛이나 팩션에 대한 설명 요청("~가 무엇인가요?" 형태라도 대상이 유닛/팩션이면)은 faction_db.
5. 위 규칙에 해당하지 않으면 아래 카테고리 설명 중 가장 적합한 것을 선택.

rule_db      : 코어 룰, 용어집, 키워드 정의, 일반/스피어헤드 게임 진행 순서 및 메커니즘
faction_db   : 특정 유닛의 스탯, 무기, 팩션 고유 능력, 워스크롤
balance_db   : 유닛의 포인트 가격, 부대 편성 제한, 레지먼트
spearhead_db : 스피어헤드 모드 전용 룰, 스피어헤드 세트 구성(warscrolls), 팩션 고유 규칙(spearhead_rules)
other_db     : 특수 캠페인 룰 (예: 기란의 재앙)

추가로 질문 유형을 판별하세요:
lookup   : 규칙 원문/용어 정의/스탯/목록 등 문서 내용을 그대로 조회하는 질문
analysis : 장단점, 평가, 전략, 활용법, 시너지, 비교, 상성 등 해석·의견을 요청하는 질문

반드시 "<카테고리>|<유형>" 형식으로만 출력하세요. (예: rule_db|lookup, faction_db|analysis)

사용자 질문: {query}"""

# 분석/의견형 질문 감지: 규칙 조회가 아니라 해석·평가·전략을 묻는 질문.
# 감지되면 답변 프롬프트에 분석 허용 지침(ANALYSIS_ADDENDUM)을 덧붙인다
ANALYSIS_RE = re.compile(
    r"장점|단점|유리|불리|강점|약점|평가|전략|활용|시너지|추천|비교|차이"
    r"|왜\s*좋|왜\s*나쁘|어떻게\s*쓰|쓸만|좋은\s*(?:이유|점)|나쁜\s*(?:이유|점)"
    r"|의견|생각|어때|카운터|상성|효율적|가치가?\s*있"
)

ANALYSIS_ADDENDUM = (
    " [분석 모드] 이 질문은 규칙 원문 조회가 아니라 해석·평가·전략 의견을 요청하는 질문입니다. "
    "이 경우 '문서에 없으면 찾을 수 없다고만 답하라'는 규칙 대신 다음을 따르세요: "
    "(1) 먼저 제공된 문서에서 관련 규칙의 원문 내용을 근거로 요약하세요. "
    "(2) 그 다음 '**분석 (의견)**' 소제목 아래에서, 요약한 규칙을 근거로 게임적 의미·장단점·활용법을 "
    "논리적으로 해석하세요. 각 주장에는 근거가 되는 규칙을 함께 언급하세요. "
    "(3) 분석 부분은 공식 규칙이 아닌 심판의 해석·의견임을 답변 끝에 한 줄로 명시하세요. "
    "(4) 단, 존재하지 않는 규칙·수치·유닛을 지어내는 것은 여전히 금지입니다. "
    "관련 규칙 자체가 문서에 전혀 없으면 분석도 하지 말고 찾을 수 없다고 답하세요."
)

# 스피어헤드 '게임 진행' 질문 감지: 진행 규칙(트위스트/배틀 택틱/배틀플랜)은
# 배틀팩마다 다르므로, 배틀팩 미지정 시 검색 대신 배틀팩을 되묻는다
SPEARHEAD_PROGRESSION_RE = re.compile(
    r"진행|게임\s*방법|플레이\s*방법|어떻게\s*(?:하|해|플레이)|순서|시작부터"
)

CLARIFY_BATTLEPACK_MSG = (
    "스피어헤드의 게임 진행 규칙(트위스트 카드, 배틀 택틱, 배틀플랜)은 "
    "**배틀팩(시즌)마다 다릅니다.** 어떤 배틀팩 기준으로 알려드릴까요?\n\n"
    "- **Fire and Jade** — 시즌 배틀팩\n"
    "- **Sand and Bone** — 시즌 배틀팩\n"
    "- **City of Ash** — 시즌 배틀팩\n"
    "- **Spearhead Doubles** — 2대2 협동전 배틀팩\n\n"
    "예: *\"sand and bone에서 게임 진행 순서 알려줘\"* 처럼 질문해 주시면 "
    "배틀 시작부터 라운드 종료까지 해당 배틀팩 규칙으로 정리해 드립니다."
)

# 무기 특수 능력 용어 감지: 질문이나 검색된 워스크롤에 이 용어가 보이면
# 코어 룰 '20.0 Weapon Abilities' 정의 청크를 컨텍스트에 주입
WEAPON_ABILITY_RE = re.compile(
    r"companion|shoot\s+in\s+combat|crit\s*\(|anti-\s*\w+|charge\s*\(\+1\s*damage\)"
    r"|blood-hungry|컴패니언|크릿|무기\s*(?:특수\s*)?능력",
    re.I,
)

# 지시 표현 감지: 재작성 안전망에서 사용 (직전 답변의 대상을 가리키는 질문)
DEMONSTRATIVE_RE = re.compile(
    r"(?:^|\s)(?:각|각각|해당|그|이|저|위)\s"
    r"|이것|그것|저것|이거|그거|저거|이들|그들|얘|걔"
    r"|전부|모두|나머지|둘\s*다"
)

# 스피어헤드 배틀팩 감지 패턴 → 소스 파일 매핑
# (질문에 특정 배틀팩이 명시되면 라우팅 교정 + 다른 배틀팩 청크 오염 차단에 사용)
BATTLEPACK_SOURCES = {
    r"sand\s*(?:and|&|앤)?\s*bone|샌드\s*앤\s*본": "wahapedia_sand-and-bone.json",
    r"city\s*of\s*ash|시티\s*오브\s*애[쉬시]": "wahapedia_city-of-ash.json",
    r"fire\s*(?:and|&|앤)?\s*jade|파이어\s*앤\s*제이드": "wahapedia_fire-and-jade.json",
    r"spearhead\s*doubles|스피어헤드\s*더블": "wahapedia_spearhead-doubles.json",
}

# 현행 GHB 시즌. 과거 시즌 소스는 기본 검색에서 제외 (사이드바 토글로 포함 가능)
CURRENT_GHB_SEASON = "2026-27"
OLD_GHB_SOURCES = [
    "wahapedia_general-s-handbook-2024-25.json",
    "wahapedia_general-s-handbook-2025-26.json",
]

# ─── DB별 설정 ────────────────────────────────────────────────────────────────
DB_LABELS = {
    "rule_db":      "📖 코어 룰",
    "faction_db":   "⚔️ 팩션 DB",
    "balance_db":   "⚖️ 포인트 DB",
    "spearhead_db": "🏹 스피어헤드 DB",
    "other_db":     "📜 특수 캠페인 DB",
}

# 각 DB에서 가져올 문서 수 (balance_db는 청크가 작아 더 많이 가져옴)
N_RESULTS = {
    "rule_db":      10,
    "faction_db":   15,
    "balance_db":   60,
    "spearhead_db": 20,
    "other_db":     5,
}

SYSTEM_PROMPTS = {
    "rule_db": (
        "당신은 워해머 에이지 오브 지그마의 공인 심판입니다. "
        "제공된 코어 룰 및 용어집 문서를 바탕으로 정확하게 한국어로 답변하세요. "
        "▶ 용어/키워드 질문: 특정 키워드의 의미를 물어보면 정의와 효과를 설명해주세요. "
        "▶ 스피어헤드 룰 질문: 검색된 문서에 '일반 스피어헤드'와 '스피어헤드 더블즈' 규칙이 혼재되어 있다면, 임의로 하나만 골라 설명하지 마세요. 반드시 '일반 스피어헤드(1대1)와 스피어헤드 더블즈(다대다) 중 어떤 모드의 규칙을 알고 싶으신가요?'라고 먼저 되물어보세요. "
        "▶ 스피어헤드 일반/진행 규칙 질문에서 특정 배틀팩이 지정되지 않았고 검색된 문서가 소개 수준(제품 설명)뿐이라면, 개요만 장황하게 늘어놓지 말고 다음을 안내하세요: "
        "'진행 규칙(트위스트, 배틀 택틱, 배틀플랜)은 배틀팩마다 다르니 Fire and Jade / Sand and Bone / City of Ash / Spearhead Doubles 중 어떤 배틀팩인지 알려달라'고 되물어보세요. "
        "[절대 금지]: 제공된 문서에 질문과 직접 일치하는 내용이 없다면 다음 행동을 하지 마세요: "
        "(1) 기존 지식이나 외부 설정으로 답변 생성, "
        "(2) 유사한 룰을 찾아 유추하거나 비유하는 설명. "
        "문서에 없으면 반드시 '제공된 문서에서 해당 정보를 찾을 수 없습니다'라고만 답하세요."
    ),
    "faction_db": (
        "당신은 워해머 에이지 오브 지그마의 팩션 전문가입니다. "
        "제공된 JSON 데이터는 팩션 팩, 스피어헤드, wahapedia 워스크롤 데이터일 수 있습니다. "
        "유닛 이름은 unit_name 또는 name 필드에 있으며 대문자/일반 표기가 혼재합니다. 대소문자를 무시하고 매칭하세요. "
        "▶ 특정 유닛 질문: 스탯(이동/세이브/컨트롤/체력/워드), 무기 프로필(ranged_weapons/melee_weapons 또는 weapons), abilities(특수 능력), keywords를 모두 정리해서 보여주세요. "
        "   데이터 출처가 스피어헤드인 경우 '이 정보는 스피어헤드 데이터 기준입니다'라고 명시하세요. "
        "▶ 스탯 표기 형식: 라벨을 정확히 '이동(Move)', '체력(Health)', '방호(Save)', '점령(Control)' 형태(한글+영문 병기)로, 이 순서대로 표기하세요. "
        "   ward 값이 있으면(예: 5+) 방호 다음에 '와드(Ward)'를 넣어 이동→체력→방호→와드→점령 순서로 5개를, "
        "   ward가 빈 값이면 와드 항목 자체를 생략하고 4개만 표기하세요 ('-'로 표기하지 마세요). "
        "▶ 무기 프로필 표기 형식: 반드시 다음 컬럼 순서의 마크다운 표로 정리하세요 — "
        "| 무기 | 사거리 | 공격횟수(Atk) | 명중(Hit) | 피해(Wnd) | 관통(Rnd) | 대미지(Dmg) | 특수 능력 |. "
        "   range 값이 비어 있으면 근접 무기이므로 사거리를 '근접'으로, 값이 있으면 인치 그대로(예: 8\") 표기하세요. "
        "▶ 팩션 유닛 목록/종류 질문: '전체 유닛(워스크롤) 목록' 문서가 제공되면 그 문서의 카테고리 구분(영웅/보병/기병/괴수 등)을 유지하여 카테고리별 소제목 아래 빠짐없이 나열하세요. "
        "   그런 문서가 없으면 스탯과 무기 프로필을 갖춘 워스크롤 형태의 JSON 항목만 유닛으로 취급하고, 능력/주문/룰만 서술된 항목은 유닛이 아닙니다. "
        "[절대 금지]: 제공된 JSON에 없는 유닛을 유추하거나, 다른 팩션 데이터로 대답하거나, 외부 지식으로 정보를 보충하지 마세요. "
        "제공된 JSON 데이터에 워스크롤 형태의 항목이 없을 때만 '찾지 못했습니다'라고 하세요."
    ),
    "balance_db": (
        "당신은 워해머 에이지 오브 지그마의 포인트 및 편성 전문가입니다. "
        "제공된 배틀 프로필을 바탕으로 포인트, 유닛 사이즈, 편성 제한을 한국어로 정확히 답변하세요. "
        "[경고]: 반드시 unit_name 필드가 일치하는 데이터를 찾으세요. 일치하는 데이터가 없다면, 다른 영웅의 편성 제한 등을 대신 말하지 마세요. "
        "대신 '해당 유닛의 포인트 정보를 찾을 수 없습니다. 찾으시는 유닛의 정확한 영문 이름을 알려주시겠어요?'라고 되물어보세요."
    ),
    "spearhead_db": (
        "당신은 워해머 에이지 오브 지그마 스피어헤드 모드 전문가입니다. "
        "제공된 JSON 데이터와 메타데이터를 바탕으로 한국어로 답변하세요. "
        "▶ 팩션의 스피어헤드 종류 질문: 각 문서 앞에 표시된 '(스피어헤드 이름: ...)' 값만을 사용해 중복 없이 목록을 만드세요. "
        "   이 값이 없는 항목은 목록에서 제외하세요. 파일명(출처)은 절대 답변에 노출하지 마세요. "
        "▶ 특정 스피어헤드 정보 질문: 해당 스피어헤드의 이름, 규칙(spearhead_rules), 포함 유닛 목록(unit_name)을 정리하세요. "
        "▶ 특정 스피어헤드 유닛 스탯 질문: 스탯과 무기 프로필을 설명하고 '스피어헤드 전용 데이터입니다'라고 명시하세요. "
        "▶ 스탯 표기 형식: 라벨을 정확히 '이동(Move)', '체력(Health)', '방호(Save)', '점령(Control)' 형태(한글+영문 병기)로, 이 순서대로 표기하세요. "
        "   와드 값이 있으면 방호 다음에 '와드(Ward)'를 넣어 이동→체력→방호→와드→점령 순서로, 없으면 와드를 생략하세요. "
        "▶ 무기 프로필 표기 형식: 반드시 다음 컬럼 순서의 마크다운 표로 정리하세요 — "
        "| 무기 | 사거리 | 공격횟수(Atk) | 명중(Hit) | 피해(Wnd) | 관통(Rnd) | 대미지(Dmg) | 특수 능력 |. "
        "   문서의 Rng/Atk/Hit/Wnd/Rnd/Dmg 값을 그대로 옮기되, 사거리(Rng)가 없는 무기는 근접 무기이므로 '근접'으로 표기하세요. "
        "▶ 스피어헤드 배틀팩(Fire and Jade, Sand and Bone, City of Ash, Spearhead Doubles) 규칙 질문: "
        "   해당 배틀팩 문서([배틀팩 이름 | 섹션] 형식 텍스트)의 배틀플랜, 트위스트, 진행 규칙을 정리해 설명하세요. "
        "▶ 특정 스피어헤드로 배틀을 진행하는 과정 질문: 배틀팩 문서의 시퀀스를 골격으로 하되, "
        "   능력 예시는 반드시 질문한 스피어헤드의 것(배틀 트레잇, 레지먼트 어빌리티, 인핸스먼트, 워스크롤 능력)을 "
        "   각 단계에 연결해 설명하세요. 다른 팩션이나 다른 배틀팩의 능력을 예시로 들지 마세요. "
        "   스피어헤드 모드에는 배틀플랜 선택이 없으므로 배틀플랜을 고르라는 안내를 하지 마세요. "
        "[절대 금지]: JSON 파일명을 답변에 포함하거나, 파일명에서 스피어헤드 이름을 추론하거나, 문서에 없는 내용을 지어내지 마세요. "
        "스피어헤드 이름 정보가 없으면 '해당 팩션의 스피어헤드 이름 정보를 DB에서 찾을 수 없습니다'라고 안내하세요."
    ),
    "other_db": (
        "당신은 워해머 에이지 오브 지그마 특수 캠페인 규칙 전문가입니다. "
        "제공된 문서를 바탕으로 캠페인 전용 규칙을 한국어로 정확히 설명하세요. "
        "[절대 금지]: 문서에 없는 내용을 유추하거나 외부 지식으로 보충하지 마세요. "
        "내용이 없다면 지어내지 말고, '제공된 문서에서 찾을 수 없습니다. 키워드를 다시 알려주시겠어요?'라고 답하세요."
    ),
}

# ─── 리소스 초기화 (캐시) ─────────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    config = dotenv_values(".env")
    gemini_client = genai.Client(api_key=config["GEMINI_API_KEY"])
    chroma_client = chromadb.PersistentClient(path="./my_warhammer_db")
    embed_model = SentenceTransformer(EMBED_MODEL)
    collections = {
        name: chroma_client.get_collection(name=name)
        for name in DB_LABELS
    }
    return gemini_client, embed_model, collections

@st.cache_resource
def load_bm25_index(db_name: str) -> BM25Index:
    """컬렉션 전체 문서로 BM25 인덱스를 구축 (DB별 1회, 캐시)."""
    _, _, cols = load_resources()
    return BM25Index.from_collection(cols[db_name])

@st.cache_resource
def load_weapon_ability_chunks() -> tuple[list[str], list[dict], list[str]]:
    """rule_db의 '20.0 Weapon Abilities' 정의 청크 (시작 시 1회, 캐시).

    Companion, Crit (Mortal) 등 무기 특수 능력의 정의는 코어 룰에만 있어
    유닛/스피어헤드 질문의 컨텍스트에서 빠진다. 용어 감지 시 이 청크들을
    컨텍스트에 함께 주입하는 데 사용."""
    _, _, cols = load_resources()
    got = cols["rule_db"].get(include=["documents", "metadatas"])
    docs, metas, ids = [], [], []
    for _id, doc, meta in zip(got["ids"], got["documents"], got["metadatas"]):
        if "weapon abilit" in ((meta or {}).get("section") or "").lower():
            ids.append(_id)
            docs.append(doc)
            metas.append(meta)
    log.info("무기 능력 정의 청크: %d개 로드", len(ids))
    return docs, metas, ids

@st.cache_resource
def load_spearhead_names() -> list[str]:
    """spearhead_db 메타데이터에서 스피어헤드 고유명 사전을 구축 (시작 시 1회).
    질문에 스피어헤드 이름이 있는데 '스피어헤드' 단어가 없어 라우터가
    faction_db 등으로 보내는 오분류를 교정하는 데 사용."""
    _, _, cols = load_resources()
    got = cols["spearhead_db"].get(include=["metadatas"])
    names = sorted({
        m.get("spearhead_name") for m in got["metadatas"]
        if m and m.get("spearhead_name")
    })
    log.info("스피어헤드 고유명 사전: %d종", len(names))
    return names

# ─── LLM 유틸리티 함수 ────────────────────────────────────────────────────────
# 2. 쿼리 확장 함수 수정 (원본 질문 유지 + 영어 번역 병기)
def generate_search_query(query: str, db_name: str, client) -> str:
    """벡터 DB 검색에 최적화되도록 팩션 이름과 한국어 서술어를 버리고 오직 '핵심 영어 키워드'만 추출합니다."""
    
    extraction_prompt = f'''
    사용자의 워해머 에이지 오브 지그마 질문에서 벡터 DB 검색에 쓸 영어 키워드를 추출하세요.
    ⚠️[주의]: 아래 규칙을 반드시 따르세요.
    1. 특정 스피어헤드 이름을 묻는 경우: 팩션 이름을 제외하고 스피어헤드의 정확한 영어 고유명만 출력하세요.
    2. 팩션의 스피어헤드 종류/목록을 묻는 경우: 팩션의 정확한 영어 키워드를 출력하세요.
    3. 특정 유닛을 묻는 경우: 유닛 전체 영어 이름만 출력하세요. 팩션 이름은 포함하지 마세요.
    4. 팩션 전체 유닛 목록/종류를 묻는 경우: 팩션의 정확한 영어 워스크롤 키워드를 출력하세요.
       - 루미네스 렐름 로드 → LUMINETH REALM-LORDS
       - 스톰캐스트 이터널스 → STORMCAST ETERNALS
       - 나이트헌트 → NIGHTHAUNT
       - 젠취의 제자들 → DISCIPLES OF TZEENTCH
       - 슬라네쉬의 쾌락주의자들 → HEDONITES OF SLAANESH
       - 넉글의 마고트킨 → MAGGOTKIN OF NURGLE
       - 오시아크 본리퍼스 → OSSIARCH BONEREAPERS
       - 피어슬레이어스 → FYRESLAYERS
       - 오고르 모트라이브스 → OGOR MAWTRIBES
       - 세라폰 → SERAPHON
       - 실바네스 → SYLVANETH
       - 카인의 딸들 → DAUGHTERS OF KHAINE
       - 카하드론 오버로드 → KHARADRON OVERLORDS
       - 아이언조즈 → IRONJAWZ
       - 오러크스 앤 고블린스 → ORRUK WARCLANS
       - 스톰캐스트 → STORMCAST ETERNALS
       - 시티즈 오브 지그마 → CITIES OF SIGMAR
       - 아이도네스 딥킨 → IDONETH DEEPKIN
       - 슬레이브즈 투 다크니스 → SLAVES TO DARKNESS
       - 블레이즈 오브 코른(코른의 칼날들) → BLADES OF KHORNE
       - 스케이븐 → SKAVEN
       - 비스츠 오브 카오스 → BEASTS OF CHAOS
       - 헬스미스 오브 하슈트 → HELSMITHS OF HASHUT
       - 소울블라이트 그레이브로드 → SOULBLIGHT GRAVELORDS
       - 플레쉬이터 코트 → FLESH-EATER COURTS
       - 크룰보이즈 → KRULEBOYZ
       - 글룸스파이트 깃츠 → GLOOMSPITE GITZ
       - 선즈 오브 베헤마트 → SONS OF BEHEMAT
       - 본스플리터즈 → BONESPLITTERZ
    5. 룰 용어/키워드의 뜻·정의·규칙을 묻는 경우: 그 용어의 영어 키워드만 출력하세요.
       '코어룰', 'core rules', '용어집', '규칙서' 같은 문서 이름은 검색어에 절대 포함하지 마세요.
    6. 질문에 유닛/스피어헤드 고유명이 있으면 절대 버리지 말고 반드시 키워드에 포함하세요.
       (예: 'Grundstok Trailblazers의 Regiment Abilities' 질문에서 'Regiment Abilities'만
       추출하면 안 됩니다 — 고유명이 없으면 다른 부대의 문서가 검색됩니다.)
    7. 키워드가 여러 개여도 줄바꿈 없이 한 줄에 공백으로 구분해 출력하세요.
    8. 한국어 서술어는 모두 제거하세요.

    예시: grundstok trailblazers에 대해 알고 싶어 -> Grundstok Trailblazers
    예시: 카하드론 그런드스탁 트레일블레이저스 정보 -> Grundstok Trailblazers
    예시: 인드라스타의 스피어헤드 규칙 -> Yndrasta's Spearhead
    예시: 카하드론 오버로드 스피어헤드 목록 -> KHARADRON OVERLORDS
    예시: 스톰캐스트 스피어헤드 종류 -> STORMCAST ETERNALS
    예시: 루미네스 보병 워든 스탯 -> Vanari Auralan Wardens
    예시: 루미네스 렐름의 유닛들 목록 -> LUMINETH REALM-LORDS
    예시: 스톰캐스트 이터널스 팩션 유닛 목록 -> STORMCAST ETERNALS
    예시: 코어룰에서 healing이라는 용어를 찾아서 알려줘 -> Healing
    예시: ward 세이브가 뭐야 -> Ward
    예시: 렌드가 무슨 뜻이야 -> Rend
    예시: Grundstok Trailblazers의 Regiment Abilities와 Enhancements를 알려줘 -> Grundstok Trailblazers Regiment Abilities Enhancements
    예시: 후라칸 뱅가드의 인핸스먼트 알려줘 -> Hurakan Vanguard Enhancements

    질문: {query}
    출력:
    '''
    try:
        response = client.models.generate_content(
            model=ROUTER_MODEL,
            contents=extraction_prompt,
            config=types.GenerateContentConfig(temperature=0.0),
        )
        return response.text.strip()
    except Exception:
        return query

def route_query(query: str, client) -> tuple[str, str]:
    """Gemini Flash-Lite로 (대상 DB, 질문 유형)을 분류합니다.
    반환: (db_name, query_type) — 분류/API 실패 시 ("rule_db", "lookup") 폴백."""
    prompt = ROUTER_PROMPT.format(query=query)
    try:
        response = client.models.generate_content(
            model=ROUTER_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.0),
        )
    except Exception:
        # 동시 접속으로 레이트 리밋(429) 등 API 오류 시 앱을 죽이지 않고
        # 기본 라우팅으로 폴백 (코드 레벨 안전장치들이 상당 부분 교정)
        log.warning("라우터 API 호출 실패 — rule_db/lookup 폴백", exc_info=True)
        return "rule_db", "lookup"
    raw = response.text.strip().lower()
    parts = [p.strip() for p in raw.split("|")]
    db_name = parts[0] if parts and parts[0] in DB_LABELS else "rule_db"
    query_type = (parts[1] if len(parts) > 1 and parts[1] in ANSWER_PROFILES
                  else "lookup")
    return db_name, query_type

def rewrite_query_with_context(current_query: str, history: list, client) -> str:
    """이전 대화 문맥을 파악하여 현재 질문을 독립적인 검색용 질문으로 재작성합니다."""
    # 대화 기록이 없거나 1개(인사말)뿐이면 재작성 불필요
    if len(history) <= 1:
        return current_query

    # 최근 대화 6개(3턴)를 문맥으로 사용 (마지막 항목은 현재 질문이므로 제외)
    history_text = ""
    recent_history = history[-7:-1] if len(history) >= 7 else history[:-1]

    for msg in recent_history:
        role = "사용자" if msg["role"] == "user" else "AI 심판"
        content = msg["content"]
        # 어시스턴트 답변의 출처 푸터(--- 이후)는 문맥 판단에 불필요 — 제거해
        # 절단 예산을 본문(유닛 목록 등)에 쓴다
        if msg["role"] == "assistant":
            content = content.split("\n---\n")[0].strip()
        content = content[:400] + "..." if len(content) > 400 else content
        history_text += f"{role}: {content}\n"

    prompt = f"""당신은 워해머 에이지 오브 지그마 챗봇의 문맥 분석기입니다.
    아래 대화 기록을 참고하여, 사용자의 최신 질문을 검색용 독립 질문으로 만드세요.

    [규칙]
    1. 최신 질문에 대명사·지시 표현(이거, 그 유닛, 저 팩션, 해당 스피어헤드,
       "각 유닛", "각각", "전부", "모두", "나머지", "이들" 등)이나 생략된 주어가 있을 때만,
       대화 기록에서 그 대상을 찾아 명확한 고유명사로 바꿔 재작성하세요.
    2. 최신 질문이 직전 AI 답변에 나열된 목록(유닛 목록, 스피어헤드 목록 등)을 가리키는 경우,
       반드시 그 목록의 주체(팩션/스피어헤드 이름)를 붙여 재작성하세요. 이것은 규칙 4의 금지에
       해당하지 않습니다 — 지시 표현이 가리키는 대상을 밝히는 것이기 때문입니다.
    3. 최신 질문이 그 자체로 완결된 질문이면(새 주제로 넘어간 경우 포함) 원문을 한 글자도 바꾸지 말고 그대로 출력하세요.
    4. 최신 질문에 지시 표현이 전혀 없는데 대화 기록의 팩션/유닛 이름을 추가하는 것은 금지합니다.
    5. 질문의 의도(규칙 질문인지, 유닛 질문인지)를 바꾸지 마세요.
    6. 다른 말은 절대 덧붙이지 마세요.

    [예시]
    - 기록에 "카라드론 오버로드 스피어헤드 유닛 알려줘"가 있고 최신 질문이 "그 중 리더는 누구야?"
      → "카라드론 오버로드 스피어헤드 유닛 중 리더는 누구인가요?" (대명사 '그 중'을 해소)
    - 기록에 "Grundstok Trailblazers 부대에는 다음 유닛이 포함됩니다: General Endrinmaster, ..."가 있고
      최신 질문이 "각 유닛의 워스크롤을 알려줘"
      → "Grundstok Trailblazers의 각 유닛의 워스크롤을 알려줘" (목록의 주체를 붙임)
    - 기록에 "카라드론 오버로드 스피어헤드 유닛 알려줘"가 있고 최신 질문이 "스피어헤드 sand and bone에 대한 규칙을 알려줘"
      → "스피어헤드 sand and bone에 대한 규칙을 알려줘" (새 주제의 완결된 질문이므로 그대로)

    [대화기록]
    {history_text}

    [최신질문]
    사용자: {current_query}

    [재작성된 질문]
"""
    try:
        response = client.models.generate_content(
            model=ROUTER_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.0),
        )
        return response.text.strip()
    except Exception:
        return current_query

# ─── UI ──────────────────────────────────────────────────────────────────────
gemini_client, embed_model, collections = load_resources()

# session_state 초기화 (사이드바보다 먼저 실행되어야 함)
# 동시 접속 대비: 초 단위 타임스탬프만으로는 같은 초에 접속한 두 사용자가
# 같은 ID를 받아 기록이 섞이므로 uuid 접미사로 유일성 보장
def _new_session_id() -> str:
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

if "session_id" not in st.session_state:
    st.session_state.session_id = _new_session_id()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "안녕하세요! 워해머 규칙에 대해 무엇이든 물어보세요."}
    ]

st.sidebar.title("검색 설정")
include_patch = st.sidebar.checkbox(
    "최신 패치 내역 포함 (FAQ/Errata)",
    value=True,
    help="체크하면 rule_db 검색 시 rules_updates 문서도 포함합니다.",
)
include_old_ghb = st.sidebar.checkbox(
    "과거 시즌 GHB 포함",
    value=False,
    help=f"체크하면 {CURRENT_GHB_SEASON} 이전 제너럴스 핸드북 규칙도 검색합니다.",
)

# ─── 채팅 내역 관리 사이드바 ─────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.subheader("💾 채팅 내역")

# 현재 세션 다운로드
current_history_json = json.dumps(
    {
        "session_id": st.session_state.session_id,
        "saved_at": datetime.now().isoformat(),
        "messages": st.session_state.messages,
    },
    ensure_ascii=False,
    indent=2,
)
st.sidebar.download_button(
    label="현재 세션 다운로드 (JSON)",
    data=current_history_json,
    file_name=f"chat_{st.session_state.session_id}.json",
    mime="application/json",
)

# 새 대화 시작
if st.sidebar.button("새 대화 시작"):
    save_chat_history(st.session_state.session_id, st.session_state.messages)
    st.session_state.session_id = _new_session_id()
    st.session_state.messages = [
        {"role": "assistant", "content": "안녕하세요! 워해머 규칙에 대해 무엇이든 물어보세요."}
    ]
    st.rerun()

# 저장된 이전 세션 목록
# 공유 모드(APP_ACCESS_CODE 설정)에서는 서버의 모든 세션이 노출되므로 숨김
# — 접속자가 서로의(호스트 포함) 대화 기록을 열람하는 것을 방지
_SHARED_MODE = bool(_ACCESS_CODE)
saved = [] if _SHARED_MODE else list_saved_sessions()
if saved:
    st.sidebar.caption(f"저장된 세션: {len(saved)}개")
    session_options = {
        f"{s['saved_at'][:16].replace('T', ' ')} ({s['message_count']}개)": s
        for s in saved
    }
    selected_label = st.sidebar.selectbox(
        "이전 세션 불러오기",
        options=["(선택)"] + list(session_options.keys()),
        key="session_select",
    )
    if selected_label != "(선택)":
        selected = session_options[selected_label]
        if st.sidebar.button("불러오기"):
            save_chat_history(st.session_state.session_id, st.session_state.messages)
            st.session_state.session_id = selected["session_id"]
            st.session_state.messages = load_chat_history(selected["session_id"])
            st.rerun()

# ─── 개발자 로그 뷰어 ────────────────────────────────────────────────────────
# 공유 모드에서는 전 사용자의 Q&A 로그가 노출되므로 숨김
if not _SHARED_MODE:
    st.sidebar.divider()
    with st.sidebar.expander("🛠️ 개발자 로그 보기"):
        today = datetime.now().strftime("%Y-%m-%d")
        log_path = os.path.join(LOG_DIR, f"{today}.log")
        n_entries = st.sidebar.number_input("최근 대화 수", min_value=1, max_value=50, value=5, step=1)
        if st.sidebar.button("🔄 새로고침", key="log_refresh"):
            st.rerun()
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8") as f:
                raw = f.read()
            entries = [e.strip() for e in raw.split("─" * 60) if e.strip()]
            recent = entries[-n_entries:]
            st.code("\n\n".join(recent), language=None)
        else:
            st.caption("오늘 기록된 로그가 없습니다.")

AVATAR_USER      = "⚔️"   # 질문하는 플레이어
AVATAR_ASSISTANT = "🏛️"   # AI 심판 (지그마의 서고)

for msg in st.session_state.messages:
    avatar = AVATAR_ASSISTANT if msg["role"] == "assistant" else AVATAR_USER
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

if user_query := st.chat_input("질문을 입력하세요..."):
    with st.chat_message("user", avatar=AVATAR_USER):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    # 메인 UI 검색 로직 순서 수정
    with st.chat_message("assistant", avatar=AVATAR_ASSISTANT):
        with st.spinner("지그마의 서고를 뒤적이는 중..."):

            pipeline_t0 = time.monotonic()
            # 이 스레드의 모든 로그 줄에 세션 식별자(뒤 6자리) 자동 표기
            set_log_session(st.session_state.session_id[-6:])
            log.info("═══ 새 질문 (session=%s): %s", st.session_state.session_id, user_query)

            # 0. 문백을 반영한 쿼리 재작성 (대화 누적)
            rewritten_query = rewrite_query_with_context(user_query, st.session_state.messages, gemini_client)
            # 디버깅용
            st.caption(f"문맥 변환: {rewritten_query}")
            if rewritten_query != user_query:
                log.info("[1.재작성] %s", rewritten_query)
            else:
                log.info("[1.재작성] 변경 없음")

            # 재작성 안전망: LLM이 지시 표현을 못 풀고 원문 그대로 반환한
            # 경우(경계 사례에서 비결정적으로 발생), 직전 턴의 검색 주제를
            # 결합해 결정적으로 보정한다. 지시 표현이 없는 새 주제 질문에는
            # 발동하지 않는다.
            last_topic = st.session_state.get("last_search_query", "")
            if (
                rewritten_query == user_query
                and last_topic
                and len(st.session_state.messages) > 1
                and DEMONSTRATIVE_RE.search(user_query)
                and last_topic.lower() not in user_query.lower()
            ):
                rewritten_query = f"{last_topic} — {user_query}"
                log.info("[1.재작성] 안전망 발동: 직전 주제 %r 결합 → %s",
                         last_topic, rewritten_query)

            # 1. 라우터: '사용자의 원본 질문'으로 (대상 DB, 질문 유형) 결정
            db_name, query_type = route_query(rewritten_query, gemini_client)
            # 안전망: 라우터가 lookup으로 판정해도 분석 표현이 명시돼 있으면 분석 모드
            query_mode = ("analysis"
                          if query_type == "analysis" or ANALYSIS_RE.search(rewritten_query)
                          else "lookup")
            log.info("[2.라우터] 판정: %s / %s (라우터=%s, regex=%s)",
                     db_name, query_mode, query_type,
                     bool(ANALYSIS_RE.search(rewritten_query)))

            # 라우터 안전장치: 스피어헤드 질문이 faction_db로 새는 오분류 교정
            # (rule_db 판정은 스피어헤드 '진행 규칙' 질문일 수 있으므로 유지)
            if db_name == "faction_db" and re.search(r"스피어헤드|spearhead", rewritten_query, re.I):
                db_name = "spearhead_db"
                log.info("[2.라우터] 안전장치: 스피어헤드 키워드 감지 → spearhead_db 교정")

            # 라우터 안전장치 2: 특정 배틀팩이 명시된 질문은 spearhead_db로 교정.
            # rule_db에는 팩션 스피어헤드 데이터가 없어 유닛/능력 예시를 찾지 못하고,
            # GHB 매치드 플레이 배틀플랜이 잘못 섞여 들어온다.
            matched_battlepack = next(
                (src for pat, src in BATTLEPACK_SOURCES.items()
                 if re.search(pat, rewritten_query, re.I)),
                None,
            )
            if matched_battlepack:
                log.info("[2.라우터] 배틀팩 감지: %s", matched_battlepack)
                if db_name == "rule_db":
                    db_name = "spearhead_db"
                    log.info("[2.라우터] 안전장치: 배틀팩 질문 → spearhead_db 교정")

            # 라우터 안전장치 3: 질문에 스피어헤드 고유명(DB 메타데이터 사전)이
            # 있으면 spearhead_db로 교정. '스피어헤드' 단어 없이 이름만 언급된
            # 질문(예: "Grundstok Trailblazers의 Regiment Abilities")이 대상.
            if db_name != "spearhead_db":
                q_norm = re.sub(r"[^a-z0-9]+", " ", rewritten_query.lower())
                matched_sp_name = next(
                    (nm for nm in load_spearhead_names()
                     if re.sub(r"[^a-z0-9]+", " ", nm.lower()).strip() in q_norm),
                    None,
                )
                if matched_sp_name:
                    db_name = "spearhead_db"
                    log.info("[2.라우터] 안전장치: 스피어헤드 고유명 %r 감지 → spearhead_db 교정",
                             matched_sp_name)

            # 라우터 안전장치 4: 'Regiment Ability'는 스피어헤드 전용 룰 명칭.
            # (Enhancement는 정규 게임 공통 용어이므로 단독으로는 신호로 쓰지 않음)
            if db_name != "spearhead_db" and re.search(
                    r"regiment\s*abilit|레지먼트\s*어빌리티", rewritten_query, re.I):
                db_name = "spearhead_db"
                log.info("[2.라우터] 안전장치: Regiment Ability(스피어헤드 전용 용어) → spearhead_db 교정")

            # 스피어헤드 진행 질문 + 배틀팩 미지정: 진행 규칙은 배틀팩마다
            # 달라 개요 청크만 검색되므로, 검색 대신 배틀팩을 되묻는다
            if (
                not matched_battlepack
                and re.search(r"스피어헤드|스피어\s*모드|spearhead", rewritten_query, re.I)
                and SPEARHEAD_PROGRESSION_RE.search(rewritten_query)
            ):
                log.info("[2.라우터] 스피어헤드 진행 질문 + 배틀팩 미지정 → 배틀팩 확인 반문")
                st.markdown(CLARIFY_BATTLEPACK_MSG)
                st.session_state.messages.append(
                    {"role": "assistant", "content": CLARIFY_BATTLEPACK_MSG})
                save_chat_history(st.session_state.session_id, st.session_state.messages)
                append_qa_log(st.session_state.session_id, user_query,
                              CLARIFY_BATTLEPACK_MSG, "clarify", "")
                st.stop()

            # 2. 검색 쿼리 추출: 벡터 DB에 던질 '순수 영어 키워드'만 생성
            search_query = generate_search_query(rewritten_query, db_name, gemini_client)
            # 빈 문자열("")이나 따옴표만 있는 경우 정리
            search_query = search_query.strip().strip('"').strip("'").strip()
            # LLM이 키워드를 여러 줄로 출력하는 경우가 있어 한 줄로 정규화
            search_query = re.sub(r"\s+", " ", search_query)
            faction_hint = search_query.lower().replace("-", " ").strip()
            log.info("[3.키워드] 추출: %r (faction_hint=%r)", search_query, faction_hint)
            # 다음 턴의 재작성 안전망에서 쓸 직전 검색 주제 저장
            st.session_state.last_search_query = search_query

            # 3. 쿼리 임베딩 & 벡터 검색 (순수 영어 키워드로 검색하여 정확도 100% 달성)
            _q = search_query if search_query else rewritten_query
            query_texts = ["query: " + _q]
            # rule_db/spearhead_db: 키워드 추출이 검색어를 훼손하거나(예: 'healing'
            # 탈락) 배틀팩+유닛처럼 대상이 둘인 질문에서 한쪽만 추출돼도 원문
            # 질문 임베딩 검색이 받쳐주도록 병행 검색 후 병합한다.
            if db_name in ("rule_db", "spearhead_db") and rewritten_query and rewritten_query != _q:
                query_texts.append("query: " + rewritten_query)
            with _EMBED_LOCK:
                query_embeddings = embed_model.encode(query_texts).tolist()
            collection = collections[db_name]
            if len(query_texts) > 1:
                log.info("[4.검색] 병행 검색: 키워드 임베딩 + 원문 질문 임베딩")

            # 참고용 진단: 키워드 임베딩으로 모든 DB의 최상위 유사도를 조사
            # (라우터 판정과 실제 유사도가 어긋나는지 확인하는 용도)
            # 빈 컬렉션은 HNSW 인덱스가 없어 쿼리가 실패하므로 건너뛰고,
            # 특정 DB의 실패가 나머지 조사를 막지 않도록 개별 처리한다.
            probe, probe_failed = {}, []
            best_db = None
            for _db, _col in collections.items():
                try:
                    if _col.count() == 0:
                        continue
                    r = _col.query(query_embeddings=[query_embeddings[0]],
                                   n_results=1, include=["distances"])
                    if r["distances"] and r["distances"][0]:
                        probe[_db] = round(r["distances"][0][0], 4)
                except Exception as e:
                    probe_failed.append(f"{_db}({type(e).__name__})")
            if probe:
                best_db = min(probe, key=probe.get)
                log.info("[4.검색] DB별 최상위 거리(낮을수록 유사): %s | 최근접=%s, 선택=%s%s",
                         probe, best_db, db_name,
                         "" if best_db == db_name else " ← 라우터와 불일치")
            if probe_failed:
                log.warning("[4.검색] 유사도 조사 실패한 DB: %s", ", ".join(probe_failed))

            query_kwargs = dict(
                query_embeddings=query_embeddings,
                n_results=N_RESULTS[db_name],
                include=["documents", "metadatas", "distances"],
            )

            # [수정된 부분] 무조건 faction으로 필터링하지 않고, 검색어가 팩션 이름일 때만 적용합니다.
            KNOWN_FACTIONS = [
                "stormcast eternals", "nighthaunt", "lumineth realm lords",
                "disciples of tzeentch", "hedonites of slaanesh", "maggotkin of nurgle",
                "ossiarch bonereapers", "fyreslayers", "ogor mawtribes", "seraphon",
                "sylvaneth", "daughters of khaine", "kharadron overlords",
                "ironjawz", "orruk warclans", "gloomspite gitz", "slaves to darkness",
                "soulblight gravelords", "flesh eater courts", "cities of sigmar",
                "kruleboyz", "skaven", "blades of khorne", "idoneth deepkin",
                "beasts of chaos", "helsmiths of hashut", "sons of behemat",
                "bonesplitterz",
            ]

            if faction_hint in KNOWN_FACTIONS:
                query_kwargs["where"] = {"faction": faction_hint}

            # spearhead_db + 배틀팩 특정 시: 다른 배틀팩 청크 오염 차단
            # (예: sand and bone 질문에 city-of-ash의 배틀 택틱이 섞이는 문제)
            if db_name == "spearhead_db" and matched_battlepack:
                other_packs = [
                    s for s in BATTLEPACK_SOURCES.values() if s != matched_battlepack
                ]
                pack_filter = {"source": {"$nin": other_packs}}
                if "where" in query_kwargs:
                    query_kwargs["where"] = {"$and": [query_kwargs["where"], pack_filter]}
                else:
                    query_kwargs["where"] = pack_filter

            # rule_db: 소스 제외 필터 (패치/에라타, 과거 시즌 GHB)
            if db_name == "rule_db":
                excluded_sources = []
                if not include_patch:
                    excluded_sources += ["rules_updates.json", "wahapedia_faqs.json"]
                if not include_old_ghb:
                    excluded_sources += OLD_GHB_SOURCES
                if excluded_sources:
                    src_filter = {"source": {"$nin": excluded_sources}}
                    if "where" in query_kwargs:
                        query_kwargs["where"] = {"$and": [query_kwargs["where"], src_filter]}
                    else:
                        query_kwargs["where"] = src_filter

            if "where" in query_kwargs:
                log.info("[4.검색] 메타데이터 필터: %s", query_kwargs["where"])

            try:
                results = collection.query(**query_kwargs)
            except chromadb.errors.ChromaError as e:
                # 앱 실행 중 외부에서 DB가 재빌드/컴팩션되면 캐시된 컬렉션
                # 핸들이 사라진 세그먼트 파일을 가리켜 'hnsw segment reader:
                # Nothing found on disk' 오류가 난다.
                # 동시 접속 대비: st.cache_resource.clear()는 다른 사용자의
                # 진행 중인 쿼리와 임베딩 모델·BM25 인덱스까지 날리므로 쓰지
                # 않는다. 대신 chroma 클라이언트 캐시만 비워 새 핸들을 받고,
                # 공유 collections dict를 제자리 갱신해 다른 세션도 다음
                # 쿼리부터 새 핸들을 쓰게 한다.
                log.warning("[4.검색] Chroma 핸들 무효화 감지(%s: %s) → 핸들 갱신 후 재시도",
                            type(e).__name__, str(e)[:120])
                from chromadb.api.client import SharedSystemClient
                SharedSystemClient.clear_system_cache()
                fresh_client = chromadb.PersistentClient(path="./my_warhammer_db")
                for _db in list(collections):
                    collections[_db] = fresh_client.get_collection(name=_db)
                collection = collections[db_name]
                results = collection.query(**query_kwargs)
                log.info("[4.검색] 재시도 성공")
            #pprint.pp(results)

            # 쿼리별 상위 5건 로그 (거리·출처·섹션)
            for qi in range(len(results["ids"])):
                q_label = "키워드" if qi == 0 else "원문"
                for rank, (meta, dist) in enumerate(zip(
                        results["metadatas"][qi][:5], results["distances"][qi][:5])):
                    meta = meta or {}
                    log.info("[5.결과] %s검색 %d위 d=%.4f | %s | %s",
                             q_label, rank + 1, dist,
                             meta.get("source", "?"),
                             meta.get("section") or meta.get("unit_name") or meta.get("category") or "")

            # 병행 검색 시: 쿼리 간 거리 스케일이 달라 거리순 병합은 한쪽이
            # 독식하므로, 쿼리별 순위를 번갈아 채우는 방식으로 병합(중복 제거)
            if len(query_embeddings) > 1 and results["ids"]:
                per_query = [
                    list(zip(ids, docs, metas, dists))
                    for ids, docs, metas, dists in zip(
                        results["ids"], results["documents"],
                        results["metadatas"], results["distances"],
                    )
                ]
                seen_ids, ranked = set(), []
                for rank in range(max(len(rows) for rows in per_query)):
                    for rows in per_query:
                        if rank < len(rows) and rows[rank][0] not in seen_ids:
                            seen_ids.add(rows[rank][0])
                            ranked.append(rows[rank])
                ranked = ranked[:N_RESULTS[db_name]]
                results["ids"] = [[r[0] for r in ranked]]
                results["documents"] = [[r[1] for r in ranked]]
                results["metadatas"] = [[r[2] for r in ranked]]
                results["distances"] = [[r[3] for r in ranked]]
                log.info("[5.결과] 병행 검색 병합: %d개 문서 (rank interleave)", len(ranked))

            # 하이브리드 검색: BM25 희소 검색 결과를 RRF로 병합.
            # 메타데이터 하드 필터(where)는 BM25 측에도 동일하게 적용된다.
            try:
                bm25_index = load_bm25_index(db_name)
                bm25_query = f"{search_query} {rewritten_query}".strip()
                bm25_ids = bm25_index.search(
                    bm25_query,
                    n_results=N_RESULTS[db_name],
                    where=query_kwargs.get("where"),
                )
                dense_ids = results["ids"][0] if results["ids"] else []
                if bm25_ids:
                    fused_ids, _ = rrf_fuse(
                        [dense_ids, bm25_ids], top_n=N_RESULTS[db_name])
                    dense_map = {
                        _id: (doc, meta, dist)
                        for _id, doc, meta, dist in zip(
                            dense_ids,
                            results["documents"][0],
                            results["metadatas"][0],
                            results["distances"][0],
                        )
                    }
                    fused_docs, fused_metas, fused_dists = [], [], []
                    for _id in fused_ids:
                        if _id in dense_map:
                            doc, meta, dist = dense_map[_id]
                        else:   # BM25 단독 진입 문서
                            doc = bm25_index.doc_by_id.get(_id, "")
                            meta = bm25_index.meta_by_id.get(_id, {})
                            dist = None
                        fused_docs.append(doc)
                        fused_metas.append(meta)
                        fused_dists.append(dist)
                    new_from_bm25 = sum(1 for _id in fused_ids if _id not in dense_map)
                    results["ids"] = [fused_ids]
                    results["documents"] = [fused_docs]
                    results["metadatas"] = [fused_metas]
                    results["distances"] = [fused_dists]
                    log.info("[5.하이브리드] 밀집 %d건 + BM25 %d건 → RRF 병합 %d건 (BM25 단독 진입 %d건)",
                             len(dense_ids), len(bm25_ids), len(fused_ids), new_from_bm25)
            except Exception:
                log.warning("[5.하이브리드] BM25 검색 실패 — 밀집 검색 결과만 사용", exc_info=True)

            # faction_db + 팩션 특정 시: 전체 유닛 목록을 합성 문서로 주입.
            # 상위 N 유사도 검색은 룰 청크에 밀려 유닛을 놓치고, 애초에 전체
            # 목록을 만들 수 없으므로 메타데이터(unit_name) 전수 조회로 보강.
            if db_name == "faction_db" and faction_hint in KNOWN_FACTIONS:
                try:
                    roster = collection.get(
                        where={"$and": [{"faction": faction_hint}, {"type": "warscroll"}]},
                        include=["metadatas"],
                    )
                    seen = {}
                    for meta in roster["metadatas"]:
                        name = (meta.get("unit_name") or "").strip()
                        role = (meta.get("role") or "").strip()
                        # 출처별 표기 차이(대소문자/구두점) 무시하고 중복 제거.
                        # role 정보가 있는 항목(wahapedia)을 우선 사용
                        key = re.sub(r"[^A-Z0-9]+", " ", name.upper()).strip()
                        if key and (key not in seen or (role and not seen[key][1])):
                            seen[key] = (name, role)
                    if seen:
                        ROLE_LABELS = [
                            ("Hero", "영웅 (Hero)"),
                            ("Infantry", "보병 (Infantry)"),
                            ("Cavalry", "기병 (Cavalry)"),
                            ("Monster", "괴수 (Monster)"),
                            ("Beast", "야수 (Beast)"),
                            ("War Machine", "워 머신 (War Machine)"),
                            ("Manifestation", "마법 현현 (Manifestation)"),
                            ("Faction Terrain", "팩션 지형 (Faction Terrain)"),
                            ("Regiment of Renown", "유명 연대 (Regiments of Renown)"),
                            ("Other", "기타"),
                        ]
                        groups = {}
                        for name, role in seen.values():
                            groups.setdefault(role or "Other", []).append(name)
                        lines = [
                            f"{faction_hint.upper()} 팩션의 전체 유닛(워스크롤) 목록"
                            f" (총 {len(seen)}개, 카테고리별):"
                        ]
                        for role, label in ROLE_LABELS:
                            if groups.get(role):
                                lines.append(f"- {label}: " + ", ".join(sorted(groups[role])))
                        roster_doc = "\n".join(lines)
                        results["documents"][0].insert(0, roster_doc)
                        results["metadatas"][0].insert(0, {
                            "source": "faction_unit_roster",
                            "faction": faction_hint,
                            "type": "unit_roster",
                        })
                        results["ids"][0].insert(0, f"unit_roster_{faction_hint}")
                        log.info("[5.보강] 팩션 로스터 주입: %s (유닛 %d개)", faction_hint, len(seen))
                except Exception:
                    log.warning("[5.보강] 팩션 로스터 주입 실패", exc_info=True)

            if db_name == "spearhead_db" and results["ids"] and len(results["ids"][0]) > 0:
                try:
                    def _norm_name(s: str) -> str:
                        return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()

                    # 검색된 결과에서 스피어헤드 이름을 순서 유지하며 중복 없이 수집
                    # (배틀팩 청크에는 spearhead_name이 없으므로 자동 제외됨)
                    found_names = []
                    for meta in results["metadatas"][0]:
                        nm = (meta or {}).get("spearhead_name")
                        if nm and nm not in found_names:
                            found_names.append(nm)

                    # 질문에 실제로 언급된 스피어헤드를 우선 확장.
                    # (search_query는 LLM이 오타를 교정한 정식 명칭이므로 함께 검사)
                    query_text = _norm_name(search_query + " " + rewritten_query)
                    named = [nm for nm in found_names if _norm_name(nm) in query_text]
                    expand_names = (named or found_names)[:2]
                    if found_names:
                        log.info("[5.보강] 스피어헤드 확장: 검색됨=%s / 질문 언급=%s / 확장 대상=%s",
                                 found_names, named, expand_names)

                    expanded_docs, expanded_metas, expanded_ids = [], [], []
                    # 스피어헤드 전체 구성(배틀 트레잇, 레지먼트 어빌리티,
                    # 인핸스먼트, 워스크롤)을 통째로 불러옴
                    for nm in expand_names:
                        box = collection.get(where={"spearhead_name": nm},
                                             include=["documents", "metadatas"])
                        if box["ids"]:
                            expanded_docs.extend(box["documents"])
                            expanded_metas.extend(box["metadatas"])
                            expanded_ids.extend(box["ids"])

                    # 배틀팩 특정 시: 배틀 진행/시퀀스 질문에 대비해 해당
                    # 배틀팩 문서 전체를 확장 (누락된 시퀀스 청크 방지)
                    if matched_battlepack:
                        box = collection.get(where={"source": matched_battlepack},
                                             include=["documents", "metadatas"])
                        if box["ids"]:
                            expanded_docs.extend(box["documents"])
                            expanded_metas.extend(box["metadatas"])
                            expanded_ids.extend(box["ids"])

                    # 확장분을 앞에 두고 기존 벡터 검색 결과는 유지하며 병합
                    # (id 기준 중복 제거). 질문이 특정 스피어헤드를 지목한
                    # 경우, 임베딩 노이즈로 걸린 다른 스피어헤드 청크는 제외
                    if expanded_ids:
                        seen_exp = set(expanded_ids)
                        for _id, doc, meta in zip(results["ids"][0],
                                                  results["documents"][0],
                                                  results["metadatas"][0]):
                            if _id in seen_exp:
                                continue
                            other_nm = (meta or {}).get("spearhead_name")
                            if named and other_nm and other_nm not in expand_names:
                                continue
                            expanded_ids.append(_id)
                            expanded_docs.append(doc)
                            expanded_metas.append(meta)
                        results["documents"][0] = expanded_docs
                        results["metadatas"][0] = expanded_metas
                        results["ids"][0] = expanded_ids
                        log.info("[5.보강] 확장 후 문서 %d개 (배틀팩 전체 포함: %s)",
                                 len(expanded_ids), bool(matched_battlepack))
                except Exception:
                    log.warning("[5.보강] 스피어헤드 확장 실패", exc_info=True)

            # 벡터 검색 결과에 search_query 키워드가 없으면 키워드 폴백 검색
            def _keyword_hit(docs: list, keyword: str) -> bool:
                kw = keyword.upper()
                return any(kw in d.upper() for d in docs)

            def _fallback_search(col, keyword: str, limit: int = 5, warscroll_only: bool = False):
                extra = {"where": {"type": "warscroll"}} if warscroll_only else {}

                # 1차: 대문자, 원문, 첫글자대문자(Title) 모두 찔러보기
                for kw in [keyword.upper(), keyword, keyword.title()]:
                    r = col.get(where_document={"$contains": kw},
                                include=["documents", "metadatas"], limit=limit, **extra)
                    if r["ids"]:
                        return r

                # 2차: 단어별 검색 (4글자 이상 단어 우선)
                words = [w for w in keyword.split() if len(w) >= 4]
                for word in words:
                    for w_case in [word.upper(), word, word.title()]:
                        r = col.get(where_document={"$contains": w_case},
                                    include=["documents", "metadatas"], limit=limit, **extra)
                        if r["ids"]:
                            return r
                return {"ids": [], "documents": [], "metadatas": []}

            flat_docs = results["documents"][0] if results["ids"] and results["ids"][0] else []
            # 특정 유닛 질의(faction_hint가 팩션명이 아닌 경우)인데 결과에
            # 워스크롤 청크가 하나도 없으면, 룰 청크가 유닛명을 언급해서
            # _keyword_hit이 통과하더라도 워스크롤 폴백을 강제한다.
            flat_metas = results["metadatas"][0] if results["ids"] and results["ids"][0] else []
            force_warscroll_fallback = (
                db_name == "faction_db"
                and faction_hint not in KNOWN_FACTIONS
                and not any((m or {}).get("type") == "warscroll" for m in flat_metas)
            )
            if search_query and (not _keyword_hit(flat_docs, search_query) or force_warscroll_fallback):
                log.info("[5.폴백] 벡터 결과에 키워드 %r 없음%s → 키워드 폴백 검색 시도",
                         search_query, " (워스크롤 강제)" if force_warscroll_fallback else "")
                try:
                    warscroll_only = db_name in ("faction_db", "spearhead_db")
                    fallback = _fallback_search(collection, search_query,
                                                limit=30 if warscroll_only else 5,
                                                warscroll_only=warscroll_only)
                    if fallback["ids"]:
                        results["documents"][0] = fallback["documents"] + flat_docs
                        results["metadatas"][0] = fallback["metadatas"] + results["metadatas"][0]
                        flat_docs = results["documents"][0]
                        log.info("[5.폴백] 키워드 폴백 성공: %d개 문서 추가", len(fallback["ids"]))
                    else:
                        log.info("[5.폴백] 키워드 폴백 결과 없음")
                except Exception:
                    log.warning("[5.폴백] 키워드 폴백 실패", exc_info=True)

            # faction_db에서도 못 찾으면 spearhead_db에서 크로스 검색
            if db_name == "faction_db" and search_query and not _keyword_hit(flat_docs, search_query):
                try:
                    spearhead_col = collections["spearhead_db"]
                    sp_fallback = {"ids": []}

                    # 1. 키워드로 아무 조각이나 하나 낚아채기 (_fallback_search는 1차원 리스트 반환)
                    temp_search = _fallback_search(spearhead_col, search_query, limit=5, warscroll_only=False)

                    if temp_search["ids"] and len(temp_search["ids"]) > 0:
                        # 2. 낚아챈 첫 번째 조각의 출처 파일명(source) 확인 (1차원 접근)
                        target_source = temp_search["metadatas"][0].get("source")

                        if target_source:
                            # 3. 해당 파일명을 가진 모든 유닛과 룰을 통째로 끌어오기!
                            sp_fallback = spearhead_col.get(
                                where={"source": target_source},
                                include=["documents", "metadatas"]
                            )

                    # 4. 기존 결과 상단에 싹쓸이한 데이터 추가
                    if sp_fallback["ids"]:
                        results["documents"][0] = sp_fallback["documents"] + flat_docs
                        results["metadatas"][0] = sp_fallback["metadatas"] + results["metadatas"][0]
                        # flat_docs 변수 갱신
                        flat_docs = results["documents"][0]
                        log.info("[5.폴백] spearhead_db 크로스 검색 성공: %d개 문서 추가",
                                 len(sp_fallback["ids"]))
                except Exception:
                    log.warning("[5.폴백] spearhead_db 크로스 검색 실패", exc_info=True)

            # 프로브 기반 크로스 DB 폴백: 선택 DB 결과에 키워드가 여전히 없고,
            # 임베딩 최근접 DB가 선택 DB와 다르면 그 DB를 추가 검색해 병합.
            # 라우팅이 어떤 이유로든 틀렸을 때의 범용 회복 장치.
            if (
                best_db and best_db != db_name and search_query
                and not _keyword_hit(flat_docs, search_query)
            ):
                try:
                    alt = collections[best_db].query(
                        query_embeddings=[query_embeddings[0]],
                        n_results=N_RESULTS.get(best_db, 10),
                        include=["documents", "metadatas", "distances"],
                    )
                    if alt["ids"] and alt["ids"][0]:
                        results["documents"][0] = alt["documents"][0] + flat_docs
                        results["metadatas"][0] = alt["metadatas"][0] + results["metadatas"][0]
                        flat_docs = results["documents"][0]
                        log.info("[5.폴백] 프로브 크로스 DB 검색(%s → %s): %d개 문서 추가%s",
                                 db_name, best_db, len(alt["ids"][0]),
                                 "" if _keyword_hit(alt["documents"][0], search_query)
                                 else " (키워드 미포함)")
                except Exception:
                    log.warning("[5.폴백] 프로브 크로스 DB 검색 실패", exc_info=True)

            # 무기 능력 정의 주입: 질문 또는 검색된 문서에 무기 특수 능력
            # 용어(Companion, Crit 등)가 보이면 코어 룰 정의 청크를 추가.
            # 워스크롤(어느 무기에 붙었는지) + 정의(효과)를 한 컨텍스트로 제공.
            try:
                has_results = bool(results["ids"] and results["ids"][0])
                scan_text = rewritten_query + " " + (
                    " ".join(results["documents"][0]) if has_results else "")
                matched_term = WEAPON_ABILITY_RE.search(scan_text)
                if matched_term:
                    wa_docs, wa_metas, wa_ids = load_weapon_ability_chunks()
                    if not has_results:
                        results["ids"] = [[]]
                        results["documents"] = [[]]
                        results["metadatas"] = [[]]
                    existing = set(results["ids"][0])
                    added = 0
                    for _id, doc, meta in zip(wa_ids, wa_docs, wa_metas):
                        if _id in existing:
                            continue
                        results["ids"][0].append(_id)
                        results["documents"][0].append(doc)
                        results["metadatas"][0].append(meta)
                        added += 1
                    if added:
                        log.info("[5.보강] 무기 능력 정의 주입: %r 감지 → 코어 룰 청크 %d개 추가",
                                 matched_term.group(0), added)
            except Exception:
                log.warning("[5.보강] 무기 능력 정의 주입 실패", exc_info=True)

            # 4. 컨텍스트 조합
            retrieved_context = ""
            sources_info = []
            if results["ids"] and len(results["ids"][0]) > 0:
                for i, (doc, meta) in enumerate(
                        zip(results["documents"][0], results["metadatas"][0])
                ):
                    source_file = meta.get("source", "unknown")
                    spearhead_name = meta.get("spearhead_name", "")

                    # 메타데이터에 spearhead_name이 없으면 파일명에서 추출 시도
                    # spearhead_팩션명_-_스피어헤드명.json → 스피어헤드명 부분 추출
                    if not spearhead_name and source_file.startswith("spearhead_"):
                        stem = source_file.replace(".json", "")
                        parts = stem.split("_-_", 1)
                        if len(parts) == 2:
                            name_candidate = parts[1].replace("_", " ").strip()
                            # "none", "unknown" 등 무의미한 값은 제외
                            if name_candidate.lower() not in ("none", "unknown", ""):
                                spearhead_name = name_candidate.title()

                    # 스피어헤드 이름이 확보된 경우 텍스트에 주입
                    if spearhead_name:
                        retrieved_context += f"[{i + 1}] (스피어헤드 이름: {spearhead_name}, 출처: {source_file}) {doc.replace(chr(10), ' ')}\n\n"
                    else:
                        retrieved_context += f"[{i + 1}] (출처: {source_file}) {doc.replace(chr(10), ' ')}\n\n"

                    source = spearhead_name or meta.get("unit_name") or source_file
                    #sources_info.append(f"- {source}")

                    display_name = (
                            spearhead_name or
                            meta.get("unit_name") or
                            meta.get("category") or
                            meta.get("section") or
                            "규칙/설명"
                    )
                    # 보기 좋게 첫 글자 대문자로 변환 (영문일 경우)
                    if isinstance(display_name, str):
                        display_name = display_name.replace("_", " ").title()

                    sources_info.append(f"- {display_name} ({source_file})")
            else:
                retrieved_context = "관련 문서를 찾을 수 없습니다."
                log.warning("[6.컨텍스트] 검색 결과 0건")

            n_ctx_docs = len(results["documents"][0]) if results["ids"] and results["ids"][0] else 0
            log.info("[6.컨텍스트] 최종: 문서 %d개, %d자", n_ctx_docs, len(retrieved_context))

            # 5. 에이전틱 루프: RAG 컨텍스트 + Tool Use
            # 답변 모델은 대화 이력을 받지 않으므로, 문맥이 반영된 재작성 질문을 전달
            user_prompt_text = (
                f"[검색 키워드 힌트: '{search_query}']\n\n"
                f"[참고 규칙]\n{retrieved_context}\n"
                f"사용자 질문: {rewritten_query}"
            )
            contents = [
                types.Content(role="user",
                              parts=[types.Part.from_text(text=user_prompt_text)])
            ]

            # 질문 유형별 프로필 적용: 모델·thinking 예산·프롬프트를 모드에 맞게 구성
            profile = ANSWER_PROFILES[query_mode]
            system_prompt = SYSTEM_PROMPTS[db_name]
            if query_mode == "analysis":
                system_prompt += ANALYSIS_ADDENDUM
            log.info("[7.답변] 프로필: %s (model=%s, thinking=%d)",
                     query_mode, profile["model"], profile["thinking"][db_name])

            gen_cfg = types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=1.0,
                tools=[calculate_expected_damage],
                thinking_config=types.ThinkingConfig(
                    thinking_budget=profile["thinking"][db_name],
                    include_thoughts=True,
                ),
            )

            all_thinking = []
            tool_calls_log = []      # UI 표시용
            MAX_TURNS = 5
            answer_t0 = time.monotonic()

            for _ in range(MAX_TURNS):
                response = gemini_client.models.generate_content(
                    model=profile["model"],
                    contents=contents,
                    config=gen_cfg,
                )

                # thinking 파트 수집
                for part in response.candidates[0].content.parts:
                    if getattr(part, "thought", False):
                        all_thinking.append(part.text)

                # 함수 호출 파트 확인
                fn_calls = [
                    p for p in response.candidates[0].content.parts
                    if p.function_call
                ]
                if not fn_calls:
                    break   # 최종 답변 완성

                # 함수 실행 및 결과 반환
                contents.append(response.candidates[0].content)
                fn_response_parts = []
                for p in fn_calls:
                    args   = dict(p.function_call.args)
                    result = calculate_expected_damage(**args)
                    tool_calls_log.append({
                        "name": p.function_call.name,
                        "args": args,
                        "result": result,
                    })
                    fn_response_parts.append(
                        types.Part.from_function_response(
                            name=p.function_call.name,
                            response={"result": result},
                        )
                    )
                contents.append(
                    types.Content(role="user", parts=fn_response_parts)
                )

        # ── UI 렌더링 ─────────────────────────────────────────────────────────
        # 최종 answer 텍스트 추출 (thinking·function_call 파트 제외)
        answer_text = "".join(
            p.text
            for p in response.candidates[0].content.parts
            if not getattr(p, "thought", False) and not p.function_call
        )
        log.info("[7.답변] 완료: model=%s, 생성 %.1fs / 전체 %.1fs, 도구호출 %d회, %d자",
                 profile["model"], time.monotonic() - answer_t0,
                 time.monotonic() - pipeline_t0, len(tool_calls_log), len(answer_text))

        # 도구 실행 결과 표시
        if tool_calls_log:
            with st.expander("🎲 전투 계산 엔진 실행 결과", expanded=True):
                for call in tool_calls_log:
                    st.caption(f"도구: {call['name']}")
                    st.json(call["args"])
                    st.code(call["result"], language=None)

        # 추론 과정 표시
        if all_thinking:
            with st.expander("💭 추론 과정 보기", expanded=False):
                st.markdown("\n\n---\n\n".join(all_thinking))

        db_label = DB_LABELS[db_name]
        unique_sources = list(dict.fromkeys(s.lstrip("- ") for s in sources_info))
        shown = unique_sources[:3]
        source_text = ", ".join(f"`{s}`" for s in shown)
        if len(unique_sources) > 3:
            source_text += f" 외 {len(unique_sources) - 3}건"
        footer = (
            f"---\n{db_label}  |  `{search_query}`  |  {source_text}"
        )
        response_text = f"{answer_text}\n\n{footer}"
        st.markdown(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        save_chat_history(st.session_state.session_id, st.session_state.messages)
        append_qa_log(st.session_state.session_id, user_query, answer_text, db_name, search_query)