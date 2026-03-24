"""
Warhammer AoS AI 룰마스터 - Qwen3-8B 로컬 버전
Gemini API 대신 로컬 Qwen3-8B 모델을 사용합니다.
"""

import json
import os
from datetime import datetime

import streamlit as st
import chromadb
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─── 채팅 내역 저장 ───────────────────────────────────────────────────────────
HISTORY_DIR = "chat_history"
LOG_DIR = "chat_logs"
os.makedirs(HISTORY_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

QWEN_MODEL_NAME = "Qwen/Qwen3-8B"
EMBED_MODEL     = "intfloat/multilingual-e5-base"

# ─── 채팅 내역 유틸 ──────────────────────────────────────────────────────────
def append_qa_log(session_id, user_msg, bot_msg, db_name, search_query):
    today = datetime.now().strftime("%Y-%m-%d")
    log_path = os.path.join(LOG_DIR, f"{today}.log")
    now = datetime.now().strftime("%H:%M:%S")
    sep = "─" * 60
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{now}] SESSION: {session_id}\n")
        f.write(f"[{now}] USER: {user_msg}\n")
        f.write(f"[{now}] DB: {db_name}  |  키워드: {search_query}\n")
        f.write(f"[{now}] BOT: {bot_msg}\n")
        f.write(f"{sep}\n")

def _session_file(session_id):
    return os.path.join(HISTORY_DIR, f"{session_id}.json")

def save_chat_history(session_id, messages):
    data = {
        "session_id": session_id,
        "saved_at": datetime.now().isoformat(),
        "messages": messages,
    }
    with open(_session_file(session_id), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_chat_history(session_id):
    path = _session_file(session_id)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get("messages", [])
    return []

def list_saved_sessions():
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

# ─── Streamlit 설정 ───────────────────────────────────────────────────────────
st.set_page_config(page_title="Warhammer AI 룰마스터 (Qwen3)", page_icon="⚔️", layout="centered")

def inject_theme():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;700&family=Cinzel+Decorative:wght@400;700&display=swap');
    .stApp { background-color: #0f0d0a; background-image: radial-gradient(ellipse at top, #1a1208 0%, #0f0d0a 60%); }
    h1 { font-family: 'Cinzel Decorative', serif !important; color: #c9a84c !important; text-shadow: 0 0 18px #7a5c1e88, 0 2px 4px #000; letter-spacing: 0.05em; }
    .stCaption, .stCaption p { font-family: 'Cinzel', serif !important; color: #8a7040 !important; }
    section[data-testid="stSidebar"] { background-color: #130f08 !important; border-right: 1px solid #3a2c10; }
    section[data-testid="stSidebar"] * { font-family: 'Cinzel', serif !important; color: #c9a84c !important; }
    .stChatInput textarea, .stChatInput input { background-color: #1a1510 !important; color: #e8d9b0 !important; border: 1px solid #5a4420 !important; border-radius: 6px !important; font-family: 'Cinzel', serif !important; }
    .stChatMessage { background-color: #1a1510 !important; border: 1px solid #2e2210 !important; border-radius: 8px !important; }
    .stChatMessage p, .stChatMessage li, .stChatMessage td { font-family: 'Cinzel', serif !important; color: #e8d9b0 !important; line-height: 1.8 !important; }
    .stChatMessage strong { color: #c9a84c !important; }
    hr { border-color: #3a2c10 !important; }
    .streamlit-expanderHeader { font-family: 'Cinzel', serif !important; color: #8a7040 !important; background-color: #1a1510 !important; border: 1px solid #3a2c10 !important; }
    .streamlit-expanderContent { background-color: #130f08 !important; border: 1px solid #2e2210 !important; color: #7a6a48 !important; font-size: 0.85em !important; }
    ::-webkit-scrollbar { width: 6px; } ::-webkit-scrollbar-track { background: #0f0d0a; }
    ::-webkit-scrollbar-thumb { background: #3a2c10; border-radius: 3px; } ::-webkit-scrollbar-thumb:hover { background: #c9a84c; }
    </style>
    """, unsafe_allow_html=True)

inject_theme()
st.title("Warhammer Age of Sigmar AI 룰마스터")
st.caption("Qwen3-8B 로컬 모델 기반 AI 심판입니다.")

# ─── DB별 설정 ────────────────────────────────────────────────────────────────
DB_LABELS = {
    "rule_db":      "📖 코어 룰",
    "faction_db":   "⚔️ 팩션 DB",
    "balance_db":   "⚖️ 포인트 DB",
    "spearhead_db": "🏹 스피어헤드 DB",
    "other_db":     "📜 특수 캠페인 DB",
}

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
        "(1) 기존 지식이나 외부 설정으로 답변 생성, (2) 유사한 룰을 찾아 유추하거나 비유하는 설명, "
        "(3) '~와 비슷하다', '~로 볼 수 있다'와 같은 간접 추론. "
        "문서에 없으면 반드시 '제공된 문서에서 해당 정보를 찾을 수 없습니다. 정확한 규칙 이름이나 키워드를 다시 알려주시겠어요?'라고만 답하세요."
    ),
    "faction_db": (
        "당신은 워해머 에이지 오브 지그마의 팩션 전문가입니다. "
        "제공된 JSON 데이터를 바탕으로 unit_name 필드를 대소문자 무시하고 매칭하세요. "
        "특정 유닛 질문: stats, weapons, abilities, keywords를 모두 정리해서 보여주세요. "
        "팩션 유닛 목록 질문: type이 'warscroll'인 항목의 unit_name만 목록으로 나열하세요. "
        "[절대 금지]: 제공된 JSON에 없는 유닛을 유추하거나 외부 지식으로 보충하지 마세요."
    ),
    "balance_db": (
        "당신은 워해머 에이지 오브 지그마의 포인트 및 편성 전문가입니다. "
        "제공된 배틀 프로필을 바탕으로 포인트, 유닛 사이즈, 편성 제한을 한국어로 정확히 답변하세요. "
        "[경고]: 반드시 unit_name 필드가 일치하는 데이터를 찾으세요. "
        "일치하는 데이터가 없다면 '해당 유닛의 포인트 정보를 찾을 수 없습니다.'라고 답하세요."
    ),
    "spearhead_db": (
        "당신은 워해머 에이지 오브 지그마 스피어헤드 모드 전문가입니다. "
        "제공된 JSON 데이터와 메타데이터를 바탕으로 한국어로 답변하세요. "
        "팩션의 스피어헤드 종류 질문: '(스피어헤드 이름: ...)' 메타데이터를 최우선 확인하세요. "
        "특정 스피어헤드 정보: 이름, 규칙(spearhead_rules), 포함 유닛 목록을 정리하세요."
    ),
    "other_db": (
        "당신은 워해머 에이지 오브 지그마 특수 캠페인 규칙 전문가입니다. "
        "제공된 문서를 바탕으로 캠페인 전용 규칙을 한국어로 정확히 설명하세요. "
        "[절대 금지]: 문서에 없는 내용을 유추하거나 외부 지식으로 보충하지 마세요."
    ),
}

ROUTER_PROMPT = """사용자의 질문을 분석하여 아래 5개 카테고리 중 가장 적합한 것을 하나만 선택하세요.
- 질문에 포인트, 점수, 비용, points, 부대 편성 등의 단어가 있으면 무조건 balance_db를 선택하세요.
- 질문에 "스피어헤드"라는 단어가 있거나 스피어헤드 고유 명칭이 포함되면 spearhead_db를 선택하세요.
- 반드시 카테고리 이름(영문)만 출력하고 다른 텍스트는 절대 포함하지 마세요.

rule_db      : 코어 룰, 턴 진행 순서, 일반 게임 메커니즘
faction_db   : 특정 유닛의 스탯, 무기, 팩션 고유 능력, 워스크롤
balance_db   : 유닛의 포인트 가격, 부대 편성 제한, 레지먼트
spearhead_db : 스피어헤드 모드 전용 룰, 세트 구성, 고유 규칙
other_db     : 특수 캠페인 룰

사용자 질문: {query}
출력:"""

SEARCH_QUERY_PROMPT = """사용자의 워해머 에이지 오브 지그마 질문에서 벡터 DB 검색에 쓸 영어 키워드를 추출하세요.
규칙:
1. 특정 유닛: 유닛 전체 영어 이름만 출력 (팩션 이름 제외)
2. 팩션 유닛 목록: 팩션의 영어 키워드 (예: STORMCAST ETERNALS)
3. 스피어헤드 이름: 정확한 영어 고유명 (예: Yndrasta's Spearhead)
4. 한국어 서술어 제거

예시: 카하드론 그런드스탁 트레일블레이저스 정보 → Grundstok Trailblazers
예시: 스톰캐스트 이터널스 팩션 유닛 목록 → STORMCAST ETERNALS
예시: 인드라스타의 스피어헤드 규칙 → Yndrasta's Spearhead

질문: {query}
출력:"""

# ─── Qwen3 추론 헬퍼 ─────────────────────────────────────────────────────────
def _qwen_generate(model, tokenizer, messages: list[dict], enable_thinking: bool = True,
                   max_new_tokens: int = 4096) -> tuple[str, str]:
    """
    Qwen3 모델로 텍스트를 생성합니다.
    Returns: (thinking_content, answer_content)
    """
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,        # greedy — 라우터·키워드 추출에 적합
            temperature=None,
            top_p=None,
        )

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    # </think> 토큰(151668) 위치 파싱
    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip()
    answer   = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()
    return thinking, answer


def _qwen_generate_answer(model, tokenizer, messages: list[dict],
                           max_new_tokens: int = 8192) -> tuple[str, str]:
    """답변 생성용: temperature sampling 활성화."""
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip()
    answer   = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()
    return thinking, answer


# ─── RAG 유틸 ────────────────────────────────────────────────────────────────
def route_query(query: str, model, tokenizer) -> str:
    """Qwen3로 질문 의도를 분류하여 DB 이름을 반환합니다."""
    messages = [{"role": "user", "content": ROUTER_PROMPT.format(query=query)}]
    _, db_name = _qwen_generate(model, tokenizer, messages,
                                enable_thinking=False, max_new_tokens=16)
    db_name = db_name.strip().lower().split()[0] if db_name.strip() else "rule_db"
    return db_name if db_name in DB_LABELS else "rule_db"


def generate_search_query(query: str, model, tokenizer) -> str:
    """벡터 DB 검색에 최적화된 영어 키워드를 추출합니다."""
    messages = [{"role": "user", "content": SEARCH_QUERY_PROMPT.format(query=query)}]
    _, keyword = _qwen_generate(model, tokenizer, messages,
                                enable_thinking=False, max_new_tokens=64)
    return keyword.strip().strip('"').strip("'").strip()


def _keyword_hit(docs: list, keyword: str) -> bool:
    kw = keyword.upper()
    return any(kw in d.upper() for d in docs)


def _fallback_search(col, keyword: str, limit: int = 5, warscroll_only: bool = False):
    extra = {"where": {"type": "warscroll"}} if warscroll_only else {}
    for kw in [keyword.upper(), keyword]:
        r = col.get(where_document={"$contains": kw},
                    include=["documents", "metadatas"], limit=limit, **extra)
        if r["ids"]:
            return r
    words = [w for w in keyword.split() if len(w) >= 4]
    for word in words:
        r = col.get(where_document={"$contains": word.upper()},
                    include=["documents", "metadatas"], limit=limit, **extra)
        if r["ids"]:
            return r
    return {"ids": [], "documents": [], "metadatas": []}


# ─── 리소스 초기화 (캐시) ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Qwen3-8B 모델 로딩 중... (최초 1회)")
def load_resources():
    qwen_model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_NAME,
        torch_dtype="auto",
        device_map="auto",
    )
    qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME)

    chroma_client = chromadb.PersistentClient(path="./my_warhammer_db")
    embed_model = SentenceTransformer(EMBED_MODEL)
    collections = {
        name: chroma_client.get_collection(name=name)
        for name in DB_LABELS
    }
    return qwen_model, qwen_tokenizer, embed_model, collections

qwen_model, qwen_tokenizer, embed_model, collections = load_resources()

# ─── session_state 초기화 ────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "안녕하세요! 워해머 규칙에 대해 무엇이든 물어보세요."}
    ]

# ─── 사이드바 ─────────────────────────────────────────────────────────────────
st.sidebar.title("검색 설정")
include_patch = st.sidebar.checkbox("최신 패치 내역 포함 (FAQ/Errata)", value=True)

st.sidebar.divider()
st.sidebar.subheader("💾 채팅 내역")

current_history_json = json.dumps(
    {
        "session_id": st.session_state.session_id,
        "saved_at": datetime.now().isoformat(),
        "messages": st.session_state.messages,
    },
    ensure_ascii=False, indent=2,
)
st.sidebar.download_button(
    label="현재 세션 다운로드 (JSON)",
    data=current_history_json,
    file_name=f"chat_{st.session_state.session_id}.json",
    mime="application/json",
)

if st.sidebar.button("새 대화 시작"):
    save_chat_history(st.session_state.session_id, st.session_state.messages)
    st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state.messages = [
        {"role": "assistant", "content": "안녕하세요! 워해머 규칙에 대해 무엇이든 물어보세요."}
    ]
    st.rerun()

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
        st.code("\n\n".join(entries[-n_entries:]), language=None)
    else:
        st.caption("오늘 기록된 로그가 없습니다.")

# ─── 채팅 UI ─────────────────────────────────────────────────────────────────
AVATAR_USER      = "⚔️"
AVATAR_ASSISTANT = "🏛️"

for msg in st.session_state.messages:
    avatar = AVATAR_ASSISTANT if msg["role"] == "assistant" else AVATAR_USER
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

if user_query := st.chat_input("질문을 입력하세요..."):
    with st.chat_message("user", avatar=AVATAR_USER):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("assistant", avatar=AVATAR_ASSISTANT):
        with st.spinner("지그마의 서고를 뒤적이는 중..."):

            # 1. 라우터: DB 결정
            db_name = route_query(user_query, qwen_model, qwen_tokenizer)

            # 2. 검색 쿼리 추출
            search_query = generate_search_query(user_query, qwen_model, qwen_tokenizer)
            search_query = search_query.strip().strip('"').strip("'").strip()

            # 3. 벡터 검색
            _q = search_query if search_query else user_query
            query_embedding = embed_model.encode("query: " + _q).tolist()
            collection = collections[db_name]

            query_kwargs = dict(
                query_embeddings=[query_embedding],
                n_results=N_RESULTS[db_name],
                include=["documents", "metadatas", "distances"],
            )
            if db_name == "rule_db" and not include_patch:
                query_kwargs["where"] = {"source": {"$ne": "rules_updates.json"}}

            results = collection.query(**query_kwargs)

            flat_docs = results["documents"][0] if results["ids"] and results["ids"][0] else []

            # 키워드 폴백 검색
            if search_query and not _keyword_hit(flat_docs, search_query):
                try:
                    warscroll_only = db_name in ("faction_db", "spearhead_db")
                    fallback = _fallback_search(collection, search_query,
                                                limit=30 if warscroll_only else 5,
                                                warscroll_only=warscroll_only)
                    if fallback["ids"]:
                        results["documents"][0] = fallback["documents"] + flat_docs
                        results["metadatas"][0]  = fallback["metadatas"] + results["metadatas"][0]
                        flat_docs = results["documents"][0]
                except Exception:
                    pass

            # spearhead ↔ faction 크로스 검색
            if db_name == "spearhead_db" and search_query and not _keyword_hit(flat_docs, search_query):
                try:
                    fc_fallback = _fallback_search(collections["faction_db"], search_query,
                                                   limit=20, warscroll_only=True)
                    if fc_fallback["ids"]:
                        results["documents"][0] = fc_fallback["documents"] + flat_docs
                        results["metadatas"][0]  = fc_fallback["metadatas"] + results["metadatas"][0]
                        flat_docs = results["documents"][0]
                except Exception:
                    pass

            if db_name == "faction_db" and search_query and not _keyword_hit(flat_docs, search_query):
                try:
                    sp_fallback = _fallback_search(collections["spearhead_db"], search_query,
                                                   limit=10, warscroll_only=True)
                    if sp_fallback["ids"]:
                        results["documents"][0] = sp_fallback["documents"] + flat_docs
                        results["metadatas"][0]  = sp_fallback["metadatas"] + results["metadatas"][0]
                        flat_docs = results["documents"][0]
                except Exception:
                    pass

            # 4. 컨텍스트 조합
            retrieved_context = ""
            sources_info = []
            if results["ids"] and len(results["ids"][0]) > 0:
                for i, (doc, meta) in enumerate(
                        zip(results["documents"][0], results["metadatas"][0])):
                    source_file   = meta.get("source", "unknown")
                    spearhead_name = meta.get("spearhead_name", "")
                    if spearhead_name:
                        retrieved_context += f"[{i+1}] (스피어헤드 이름: {spearhead_name}, 출처: {source_file}) {doc.replace(chr(10), ' ')}\n\n"
                    else:
                        retrieved_context += f"[{i+1}] (출처: {source_file}) {doc.replace(chr(10), ' ')}\n\n"
                    source = meta.get("unit_name") or spearhead_name or source_file
                    sources_info.append(f"- {source}")
            else:
                retrieved_context = "관련 문서를 찾을 수 없습니다."

            # 5. Qwen3 답변 생성
            user_prompt_text = (
                f"[검색 키워드 힌트: '{search_query}']\n\n"
                f"[참고 규칙]\n{retrieved_context}\n"
                f"사용자 질문: {user_query}"
            )
            messages_for_llm = [
                {"role": "system", "content": SYSTEM_PROMPTS[db_name]},
                {"role": "user",   "content": user_prompt_text},
            ]
            thinking_content, answer_text = _qwen_generate_answer(
                qwen_model, qwen_tokenizer, messages_for_llm
            )

        # ── UI 렌더링 ─────────────────────────────────────────────────────────
        if thinking_content:
            with st.expander("💭 추론 과정 보기", expanded=False):
                st.markdown(thinking_content)

        db_label = DB_LABELS[db_name]
        unique_sources = list(dict.fromkeys(s.lstrip("- ") for s in sources_info))
        shown = unique_sources[:3]
        source_text = ", ".join(f"`{s}`" for s in shown)
        if len(unique_sources) > 3:
            source_text += f" 외 {len(unique_sources) - 3}건"
        footer = f"---\n{db_label}  |  `{search_query}`  |  {source_text}"
        response_text = f"{answer_text}\n\n{footer}"

        st.markdown(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        save_chat_history(st.session_state.session_id, st.session_state.messages)
        append_qa_log(st.session_state.session_id, user_query, answer_text, db_name, search_query)
