import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
from dotenv import dotenv_values

st.set_page_config(page_title="Warhammer AI 룰마스터", page_icon="🎲")
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
- 반드시 카테고리 이름(영문, 예: rule_db)만 출력하고 다른 텍스트는 절대 포함하지 마세요.

rule_db      : 코어 룰, 턴 진행 순서, 일반적인 게임 메커니즘 (이동, 슈팅, 전투, 마법, 차지 등)
faction_db   : 특정 유닛의 스탯, 무기, 팩션 고유 능력, 워스크롤
balance_db   : 유닛의 포인트 가격, 부대 편성 제한, 레지먼트
spearhead_db : 스피어헤드 모드 전용 룰 및 뱅가드 유닛 정보
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
        "[경고]: 제공된 문서에 질문과 직접적으로 관련된 내용이 없다면 절대 기존 지식으로 지어내거나 엉뚱한 룰을 설명하지 마세요. "
        "대신 '제공된 문서에서 해당 정보를 찾을 수 없습니다. 원하시는 정확한 규칙 이름이나 키워드를 다시 말씀해 주시겠어요?'와 같이 정중하게 되물어보세요."
    ),
    "faction_db": (
        "당신은 워해머 에이지 오브 지그마의 팩션 전문가입니다. "
        "제공된 JSON 데이터는 팩션 팩 또는 스피어헤드 데이터일 수 있습니다. "
        "JSON 데이터의 unit_name 필드는 대문자로 저장되어 있습니다. 대소문자를 무시하고 매칭하세요. "
        "▶ 특정 유닛 질문: stats(이동/저장/제어/체력), weapons(무기 프로필), abilities(특수 능력), keywords를 모두 정리해서 보여주세요. "
        "   데이터 출처가 스피어헤드인 경우 '이 정보는 스피어헤드 데이터 기준입니다'라고 명시하세요. "
        "▶ 팩션 유닛 목록/종류 질문: JSON 데이터에 있는 모든 유닛의 unit_name을 목록으로 나열하세요. "
        "   단, type이 'warscroll'인 항목만 유닛으로 취급하고, 능력/주문/룰은 유닛이 아닙니다. "
        "[경고]: 제공된 JSON 데이터에 warscroll 타입 항목이 없을 때만 '찾지 못했습니다'라고 하세요."
    ),
    "balance_db": (
        "당신은 워해머 에이지 오브 지그마의 포인트 및 편성 전문가입니다. "
        "제공된 배틀 프로필을 바탕으로 포인트, 유닛 사이즈, 편성 제한을 한국어로 정확히 답변하세요. "
        "[경고]: 반드시 unit_name 필드가 일치하는 데이터를 찾으세요. 일치하는 데이터가 없다면, 다른 영웅의 편성 제한 등을 대신 말하지 마세요. "
        "대신 '해당 유닛의 포인트 정보를 찾을 수 없습니다. 찾으시는 유닛의 정확한 영문 이름을 알려주시겠어요?'라고 되물어보세요."
    ),
    "spearhead_db": (
        "당신은 워해머 에이지 오브 지그마 스피어헤드 모드 전문가입니다. "
        "제공된 스피어헤드 규칙을 바탕으로 한국어로 답변하세요. "
        "[경고]: 제공된 문서에 없다면 절대 지어내지 말고, '정확한 정보를 위해 다른 키워드나 영문 명칭을 알려주시겠어요?'라고 되물어보세요."
    ),
    "other_db": (
        "당신은 워해머 에이지 오브 지그마 특수 캠페인 규칙 전문가입니다. "
        "제공된 문서를 바탕으로 캠페인 전용 규칙을 한국어로 정확히 설명하세요. "
        "[경고]: 내용이 없다면 지어내지 말고, 확인을 위해 사용자에게 키워드를 다시 물어보세요."
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
    1. 특정 유닛을 묻는 경우: 유닛 전체 영어 이름만 출력하세요. 팩션 이름은 포함하지 마세요.
    2. 팩션 전체 유닛 목록/종류를 묻는 경우: 팩션의 정확한 영어 워스크롤 키워드를 출력하세요.
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
    3. 한국어 서술어는 모두 제거하세요.

    예시: 루미네스 보병 워든 스탯 -> Vanari Auralan Wardens
    예시: 루미네스 렐름 로드의 워든 정보 -> Vanari Auralan Wardens
    예시: 젠취의 카이로스 페이트위버 마법 -> Kairos Fateweaver
    예시: 루미네스 렐름의 유닛들 목록 -> LUMINETH REALM-LORDS
    예시: 루미네스 렐름 로드 유닛 종류 -> LUMINETH REALM-LORDS
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

st.sidebar.title("검색 설정")
include_patch = st.sidebar.checkbox(
    "최신 패치 내역 포함 (FAQ/Errata)",
    value=True,
    help="체크하면 rule_db 검색 시 rules_updates 문서도 포함합니다.",
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "안녕하세요! 워해머 규칙에 대해 무엇이든 물어보세요."}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_query := st.chat_input("질문을 입력하세요..."):
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    # 메인 UI 검색 로직 순서 수정
    with st.chat_message("assistant"):
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
                    warscroll_only = db_name == "faction_db"
                    fallback = _fallback_search(collection, search_query,
                                                limit=30 if warscroll_only else 5,
                                                warscroll_only=warscroll_only)
                    if fallback["ids"]:
                        results["documents"][0] = fallback["documents"] + flat_docs
                        results["metadatas"][0] = fallback["metadatas"] + results["metadatas"][0]
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

            # 5. 최종 답변 생성 (사용자의 원본 질문과 검색된 정확한 컨텍스트 사용)
            # LLM이 한국어 질문과 영어 원문을 매칭할 수 있도록 search_query를 힌트로 줍니다.
            user_prompt = (
                f"[AI 검색 내비게이터 힌트: 사용자가 찾는 유닛의 영문 이름은 '{search_query}'와 일치할 확률이 높습니다.]\n\n"
                f"[참고 규칙]\n{retrieved_context}\n"
                f"사용자 질문: {user_query}"
            )
            
            response = gemini_client.models.generate_content(
                model=ANSWER_MODEL,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPTS[db_name],
                    temperature=1.0,
                    thinking_config=types.ThinkingConfig(
                        thinking_budget=THINKING_BUDGET[db_name],
                        include_thoughts=True,
                    ),
                ),
            )

        # thinking 파트와 answer 파트 분리
        thinking_text = ""
        answer_text = ""
        for part in response.candidates[0].content.parts:
            if hasattr(part, "thought") and part.thought:
                thinking_text = part.text
            else:
                answer_text += part.text

        db_label = DB_LABELS[db_name]
        source_text = "\n".join(list(set(sources_info))) if sources_info else "참고할 문서가 없습니다."
        footer = f"---\n검색한 DB: {db_label}  |  검색 키워드: `{search_query}`  |  참고 문서:\n{source_text}"

        if thinking_text:
            with st.expander("💭 추론 과정 보기", expanded=False):
                st.markdown(thinking_text)

        response_text = f"{answer_text}\n\n{footer}"
        st.markdown(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})