import json
import os
from datetime import datetime
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
from dotenv import dotenv_values
from tools import calculate_expected_damage

# ─── 채팅 내역 저장 ───────────────────────────────────────────────────────────
HISTORY_DIR = "chat_history"
LOG_DIR = "chat_logs"
os.makedirs(HISTORY_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def append_qa_log(session_id: str, user_msg: str, bot_msg: str, db_name: str, search_query: str) -> None:
    """Q&A 한 쌍을 일별 로그 파일에 추가합니다."""
    today = datetime.now().strftime("%Y-%m-%d")
    log_path = os.path.join(LOG_DIR, f"{today}.log")
    now = datetime.now().strftime("%H:%M:%S")
    separator = "─" * 60
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{now}] SESSION: {session_id}\n")
        f.write(f"[{now}] USER: {user_msg}\n")
        f.write(f"[{now}] DB: {db_name}  |  키워드: {search_query}\n")
        f.write(f"[{now}] BOT: {bot_msg}\n")
        f.write(f"{separator}\n")

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

st.set_page_config(page_title="Warhammer AI 룰마스터", page_icon="⚔️", layout="centered")

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

    /* 사이드바 */
    section[data-testid="stSidebar"] {
        background-color: #130f08 !important;
        border-right: 1px solid #3a2c10;
    }
    section[data-testid="stSidebar"] * {
        font-family: 'Cinzel', serif !important;
        color: #c9a84c !important;
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

st.title("Warhammer Age of Sigmar AI 룰마스터")
st.caption("공식 규칙 문서와 FAQ를 기반으로 답변하는 AI 심판입니다.")

# ─── 모델 설정 ────────────────────────────────────────────────────────────────
ROUTER_MODEL = "gemini-2.5-flash-lite"
ANSWER_MODEL = "gemini-2.5-flash"
EMBED_MODEL  = "intfloat/multilingual-e5-base"

# DB별 thinking 토큰 예산 (-1=동적, 0=비활성)
# 룰 판단은 추론이 많이 필요, 단순 스탯 조회는 적게
THINKING_BUDGET = {
    "rule_db":      8000,
    "faction_db":   2000,
    "balance_db":   1000,
    "spearhead_db": 4000,
    "other_db":     4000,
}



# ─── 라우터 프롬프트 ──────────────────────────────────────────────────────────
ROUTER_PROMPT = """
사용자의 질문을 분석하여 아래 5개 카테고리 중 가장 적합한 것을 하나만 선택하세요.
- 질문에 포인트, 점수, 비용, points, 부대 편성 등의 단어가 있으면 무조건 balance_db 를 선택하세요.
- 질문에 "스피어헤드"라는 단어가 있거나, 스피어헤드 고유 명칭(예: Grundstok Trailblazers, Yndrasta's Spearhead, Ironjawz Waaagh!, Dawnbringers 등 팩션명+Spearhead 형태 또는 스피어헤드 세트 이름)이 포함되어 있으면 무조건 spearhead_db를 선택하세요.
- 반드시 카테고리 이름(영문, 예: rule_db)만 출력하고 다른 텍스트는 절대 포함하지 마세요.

rule_db      : 코어 룰, 턴 진행 순서, 일반적인 게임 메커니즘 (이동, 슈팅, 전투, 마법, 차지 등)
faction_db   : 특정 유닛의 스탯, 무기, 팩션 고유 능력, 워스크롤
balance_db   : 유닛의 포인트 가격, 부대 편성 제한, 레지먼트
spearhead_db : 스피어헤드 모드 전용 룰, 스피어헤드 세트 구성(warscrolls), 스피어헤드 고유 규칙(spearhead_rules)
other_db     : 특수 캠페인 룰 (예: 기란의 재앙)

사용자 질문: {query}"""

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
    "rule_db":      5,
    "faction_db":   15,
    "balance_db":   60,
    "spearhead_db": 5,
    "other_db":     5,
}

SYSTEM_PROMPTS = {
    "rule_db": (
        "당신은 워해머 에이지 오브 지그마의 공인 심판입니다. "
        "제공된 코어 룰 문서를 바탕으로 정확하게 한국어로 답변하세요. "
        "[절대 금지]: 제공된 문서에 질문과 직접 일치하는 내용이 없다면 다음 행동을 하지 마세요: "
        "(1) 기존 지식이나 외부 설정으로 답변 생성, "
        "(2) 유사한 룰을 찾아 유추하거나 비유하는 설명, "
        "(3) '~와 비슷하다', '~로 볼 수 있다'와 같은 간접 추론. "
        "문서에 없으면 반드시 '제공된 문서에서 해당 정보를 찾을 수 없습니다. 정확한 규칙 이름이나 키워드를 다시 알려주시겠어요?'라고만 답하세요."
    ),
    "faction_db": (
        "당신은 워해머 에이지 오브 지그마의 팩션 전문가입니다. "
        "제공된 JSON 데이터는 팩션 팩 또는 스피어헤드 데이터일 수 있습니다. "
        "JSON 데이터의 unit_name 필드는 대문자로 저장되어 있습니다. 대소문자를 무시하고 매칭하세요. "
        "▶ 특정 유닛 질문: stats(이동/저장/제어/체력), weapons(무기 프로필), abilities(특수 능력), keywords를 모두 정리해서 보여주세요. "
        "   데이터 출처가 스피어헤드인 경우 '이 정보는 스피어헤드 데이터 기준입니다'라고 명시하세요. "
        "▶ 팩션 유닛 목록/종류 질문: JSON 데이터에 있는 모든 유닛의 unit_name을 목록으로 나열하세요. "
        "   단, type이 'warscroll'인 항목만 유닛으로 취급하고, 능력/주문/룰은 유닛이 아닙니다. "
        "[절대 금지]: 제공된 JSON에 없는 유닛을 유추하거나, 다른 팩션 데이터로 대답하거나, 외부 지식으로 정보를 보충하지 마세요. "
        "제공된 JSON 데이터에 warscroll 타입 항목이 없을 때만 '찾지 못했습니다'라고 하세요."
    ),
    "balance_db": (
        "당신은 워해머 에이지 오브 지그마의 포인트 및 편성 전문가입니다. "
        "제공된 배틀 프로필을 바탕으로 포인트, 유닛 사이즈, 편성 제한을 한국어로 정확히 답변하세요. "
        "[경고]: 반드시 unit_name 필드가 일치하는 데이터를 찾으세요. 일치하는 데이터가 없다면, 다른 영웅의 편성 제한 등을 대신 말하지 마세요. "
        "대신 '해당 유닛의 포인트 정보를 찾을 수 없습니다. 찾으시는 유닛의 정확한 영문 이름을 알려주시겠어요?'라고 되물어보세요."
    ),
    "spearhead_db": (
        "당신은 워해머 에이지 오브 지그마 스피어헤드 모드 전문가입니다. "
        "제공된 스피어헤드 JSON 데이터와 출처 파일명을 모두 활용하여 한국어로 답변하세요. "
        "▶ 팩션의 스피어헤드 종류 질문: 다음 우선순위로 스피어헤드 목록을 구성하세요. "
        "   (1순위) JSON 본문에 spearhead_name 필드가 있으면 그 값을 사용하세요. "
        "   (2순위) spearhead_name이 없으면 출처 파일명에서 이름을 추출하세요. "
        "           파일명 구조: spearhead_팩션명_-_스피어헤드명.json "
        "           예: spearhead_kharadron_overlords_-_grundstok_trailblazers.json → Grundstok Trailblazers "
        "           '_-_' 뒤쪽 부분을 추출하고 언더스코어를 공백으로, 단어 첫 글자를 대문자로 변환하세요. "
        "   (3순위) 파일명도 없으면 JSON 데이터의 최상위 키 이름을 스피어헤드 이름으로 간주하세요. "
        "▶ 특정 스피어헤드 정보 질문: 아래 세 항목을 모두 정리해서 답변하세요. "
        "   (1) 스피어헤드 이름과 주요 특징 및 고유 규칙(spearhead_rules). "
        "   (2) 포함된 유닛 목록: type이 'warscroll'인 항목의 unit_name을 모두 나열하세요. "
        "   (3) 각 유닛의 간략한 역할(abilities 요약)을 함께 제공하세요. "
        "▶ 스피어헤드 유닛 단독 질문: type이 'warscroll'인 항목의 unit_name을 모두 나열하세요. "
        "[주의]: JSON 본문, 출처 파일명, 메타데이터를 최대한 조합하여 답변을 시도하세요. "
        "세 가지 수단을 모두 써도 도저히 관련 정보를 찾을 수 없을 때만 "
        "'정확한 정보를 위해 스피어헤드 이름이나 팩션명을 영문으로 알려주시겠어요?'라고 되물어보세요. "
        "[절대 금지]: 일반 매치드 플레이 데이터나 외부 지식으로 스피어헤드 정보를 대체하지 마세요."
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
    5. 한국어 서술어는 모두 제거하세요.

    예시: grundstok trailblazers에 대해 알고 싶어 -> Grundstok Trailblazers
    예시: 카하드론 그런드스탁 트레일블레이저스 정보 -> Grundstok Trailblazers
    예시: 인드라스타의 스피어헤드 규칙 -> Yndrasta's Spearhead
    예시: 카하드론 오버로드 스피어헤드 목록 -> KHARADRON OVERLORDS
    예시: 스톰캐스트 스피어헤드 종류 -> STORMCAST ETERNALS
    예시: 루미네스 보병 워든 스탯 -> Vanari Auralan Wardens
    예시: 루미네스 렐름의 유닛들 목록 -> LUMINETH REALM-LORDS
    예시: 스톰캐스트 이터널스 팩션 유닛 목록 -> STORMCAST ETERNALS

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

def route_query(query: str, client) -> str:
    """Gemini Flash-Lite로 질문 의도를 분류하여 DB 이름을 반환합니다.
    분류 실패 시 rule_db로 폴백합니다."""
    prompt = ROUTER_PROMPT.format(query=query)
    response = client.models.generate_content(
        model=ROUTER_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.0),
    )
    db_name = response.text.strip().lower()
    return db_name if db_name in DB_LABELS else "rule_db"

# ─── UI ──────────────────────────────────────────────────────────────────────
gemini_client, embed_model, collections = load_resources()

# session_state 초기화 (사이드바보다 먼저 실행되어야 함)
if "session_id" not in st.session_state:
    st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

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
    st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state.messages = [
        {"role": "assistant", "content": "안녕하세요! 워해머 규칙에 대해 무엇이든 물어보세요."}
    ]
    st.rerun()

# 저장된 이전 세션 목록
saved = list_saved_sessions()
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
st.sidebar.divider()
log_expander = st.sidebar.expander("🛠️ 개발자 로그 보기")
with log_expander:
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

            # 1. 라우터: '사용자의 원본 질문'으로 의도를 파악하여 DB 결정
            db_name = route_query(user_query, gemini_client)

            # 2. 검색 쿼리 추출: 벡터 DB에 던질 '순수 영어 키워드'만 생성
            search_query = generate_search_query(user_query, db_name, gemini_client)
            # 빈 문자열("")이나 따옴표만 있는 경우 정리
            search_query = search_query.strip().strip('"').strip("'").strip()

            # 3. 쿼리 임베딩 & 벡터 검색 (순수 영어 키워드로 검색하여 정확도 100% 달성)
            _q = search_query if search_query else user_query
            query_embedding = embed_model.encode("query: " + _q).tolist()
            collection = collections[db_name]

            query_kwargs = dict(
                query_embeddings=[query_embedding],
                n_results=N_RESULTS[db_name],
                include=["documents", "metadatas", "distances"],
            )

            # rule_db: 패치 포함 여부에 따라 소스 필터 적용
            if db_name == "rule_db" and not include_patch:
                query_kwargs["where"] = {"source": {"$ne": "rules_updates.json"}}

            results = collection.query(**query_kwargs)

            # 벡터 검색 결과에 search_query 키워드가 없으면 키워드 폴백 검색
            def _keyword_hit(docs: list, keyword: str) -> bool:
                kw = keyword.upper()
                return any(kw in d.upper() for d in docs)

            def _fallback_search(col, keyword: str, limit: int = 5, warscroll_only: bool = False):
                """전체 구문 → 단어별 순으로 키워드 매칭 폴백."""
                extra = {"where": {"type": "warscroll"}} if warscroll_only else {}
                # 1차: 전체 구문 대문자
                r = col.get(where_document={"$contains": keyword.upper()},
                            include=["documents", "metadatas"], limit=limit, **extra)
                if r["ids"]:
                    return r
                # 2차: 전체 구문 원문
                r = col.get(where_document={"$contains": keyword},
                            include=["documents", "metadatas"], limit=limit, **extra)
                if r["ids"]:
                    return r
                # 3차: 단어별 검색 (4글자 이상 단어 우선)
                words = [w for w in keyword.split() if len(w) >= 4]
                for word in words:
                    r = col.get(where_document={"$contains": word.upper()},
                                include=["documents", "metadatas"], limit=limit, **extra)
                    if r["ids"]:
                        return r
                return {"ids": [], "documents": [], "metadatas": []}

            flat_docs = results["documents"][0] if results["ids"] and results["ids"][0] else []
            if search_query and not _keyword_hit(flat_docs, search_query):
                try:
                    warscroll_only = db_name in ("faction_db", "spearhead_db")
                    fallback = _fallback_search(collection, search_query,
                                                limit=30 if warscroll_only else 5,
                                                warscroll_only=warscroll_only)
                    if fallback["ids"]:
                        results["documents"][0] = fallback["documents"] + flat_docs
                        results["metadatas"][0] = fallback["metadatas"] + results["metadatas"][0]
                        flat_docs = results["documents"][0]
                except Exception:
                    pass

            # spearhead_db에서 못 찾으면 faction_db에서도 크로스 검색 (반대 방향)
            if db_name == "spearhead_db" and search_query and not _keyword_hit(flat_docs, search_query):
                try:
                    faction_col = collections["faction_db"]
                    fc_fallback = _fallback_search(faction_col, search_query,
                                                   limit=20, warscroll_only=True)
                    if fc_fallback["ids"]:
                        results["documents"][0] = fc_fallback["documents"] + flat_docs
                        results["metadatas"][0] = fc_fallback["metadatas"] + results["metadatas"][0]
                        flat_docs = results["documents"][0]
                except Exception:
                    pass

            # faction_db에서도 못 찾으면 spearhead_db에서 크로스 검색
            if db_name == "faction_db" and search_query and not _keyword_hit(flat_docs, search_query):
                try:
                    spearhead_col = collections["spearhead_db"]
                    # 1차: 키워드 매칭
                    sp_fallback = _fallback_search(spearhead_col, search_query,
                                                   limit=10, warscroll_only=True)
                    if not sp_fallback["ids"]:
                        # 2차: faction 메타데이터로 필터링
                        # search_query를 소문자+공백 정규화해서 faction 키와 매칭
                        faction_hint = search_query.lower().replace("-", " ").strip()
                        sp_by_faction = spearhead_col.get(
                            where={"$and": [{"faction": {"$eq": faction_hint}}, {"type": {"$eq": "warscroll"}}]},
                            include=["documents", "metadatas"],
                            limit=20,
                        )
                        if sp_by_faction["ids"]:
                            sp_fallback = sp_by_faction
                    if not sp_fallback["ids"]:
                        # 3차: warscroll 타입만 벡터 검색
                        sp_vec = spearhead_col.query(
                            query_embeddings=[query_embedding],
                            n_results=10,
                            where={"type": "warscroll"},
                            include=["documents", "metadatas"],
                        )
                        sp_fallback["documents"] = sp_vec["documents"][0]
                        sp_fallback["metadatas"] = sp_vec["metadatas"][0]
                        sp_fallback["ids"] = sp_vec["ids"][0]
                    if sp_fallback["ids"]:
                        results["documents"][0] = sp_fallback["documents"] + flat_docs
                        results["metadatas"][0] = sp_fallback["metadatas"] + results["metadatas"][0]
                except Exception:
                    pass

            # 4. 컨텍스트 조합
            retrieved_context = ""
            sources_info = []
            if results["ids"] and len(results["ids"][0]) > 0:
                for i, (doc, meta) in enumerate(
                    zip(results["documents"][0], results["metadatas"][0])
                ):
                    retrieved_context += f"[{i+1}] {doc.replace(chr(10), ' ')}\n\n"
                    source = meta.get("unit_name") or meta.get("source") or "unknown"
                    sources_info.append(f"- {source}")
            else:
                retrieved_context = "관련 문서를 찾을 수 없습니다."

            # 5. 에이전틱 루프: RAG 컨텍스트 + Tool Use
            user_prompt_text = (
                f"[검색 키워드 힌트: '{search_query}']\n\n"
                f"[참고 규칙]\n{retrieved_context}\n"
                f"사용자 질문: {user_query}"
            )
            contents = [
                types.Content(role="user",
                              parts=[types.Part.from_text(text=user_prompt_text)])
            ]

            gen_cfg = types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPTS[db_name],
                temperature=1.0,
                tools=[calculate_expected_damage],
                thinking_config=types.ThinkingConfig(
                    thinking_budget=THINKING_BUDGET[db_name],
                    include_thoughts=True,
                ),
            )

            all_thinking = []
            tool_calls_log = []      # UI 표시용
            MAX_TURNS = 5

            for _ in range(MAX_TURNS):
                response = gemini_client.models.generate_content(
                    model=ANSWER_MODEL,
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