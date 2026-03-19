import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
from dotenv import dotenv_values

st.set_page_config(page_title="Warhammer AI 룰마스터", page_icon="🎲")
st.title("Warhammer Age of Sigmar AI 룰마스터")
st.caption("공식 규칙 문서와 FAQ를 기반으로 답변하는 AI 심판입니다.")

@st.cache_resource
def load_resources():
    config = dotenv_values(".env")
    gemini_client = genai.Client(api_key=config["GEMINI_API_KEY"])
    chroma_client = chromadb.PersistentClient(path="./my_warhammer_db")
    embed_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    rule_coll = chroma_client.get_collection(name="rule_db")
    return gemini_client, embed_model, rule_coll

client, embed_model, rule_collection = load_resources()

# 사이드바 사용자 제어 UI 추가
st.sidebar.title("검색 설정")
include_patch = st.sidebar.checkbox(
    "최신 패치 내역 포함 (FAQ/Errata)", 
    value=True, 
    help="체크하면 최신 업데이트 내역을 우선적으로 검색하여 답변에 반영합니다."
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

    with st.chat_message("assistant"):
        with st.spinner("지그마의 서고를 뒤적이는 중..."):
            query_embedding = embed_model.encode(user_query).tolist()
            
            # 1. 코어 룰 검색 (항상 수행)
            core_results = rule_collection.query(
                query_embeddings=[query_embedding],
                n_results=2,
                where={"doc_type": "core"},
                include=["documents", "metadatas", "distances"]
            )
            
            retrieved_context = "[코어 룰 원본]\n"
            sources_info = []
            
            if core_results['ids'][0]:
                for i in range(len(core_results['documents'][0])):
                    doc_text = core_results['documents'][0][i].replace('\n', ' ')
                    title = core_results['metadatas'][0][i].get('title')
                    retrieved_context += f"- 출처: {title}\n내용: {doc_text}\n\n"
                    sources_info.append(f"- [코어] {title}")
            else:
                retrieved_context += "코어 룰에서 관련 내용을 찾을 수 없습니다.\n\n"

            # 2. 패치 룰 검색 및 프롬프트 분기 (사용자 설정에 따라)
            if include_patch:
                patch_results = rule_collection.query(
                    query_embeddings=[query_embedding],
                    n_results=1,
                    where={"doc_type": "patch"},
                    include=["documents", "metadatas", "distances"]
                )
                
                retrieved_context += "[최신 패치 및 FAQ 내역]\n"
                if patch_results['ids'][0]:
                    for i in range(len(patch_results['documents'][0])):
                        doc_text = patch_results['documents'][0][i].replace('\n', ' ')
                        title = patch_results['metadatas'][0][i].get('title')
                        retrieved_context += f"- 출처: {title}\n내용: {doc_text}\n\n"
                        sources_info.append(f"- [패치] {title}")
                else:
                    retrieved_context += "해당 규칙과 관련된 패치 내역이 없습니다.\n\n"
                    
                system_instruction = (
                    "당신은 미니어처 게임 워해머 에이지 오브 지그마의 심판입니다. "
                    "사용자의 질문에 답할 때, 반드시 [코어 룰 원본]을 기준으로 설명하되, "
                    "[최신 패치 및 FAQ 내역]에 변경 사항이 있다면 이를 가장 우선시하여 답변에 반영해 주세요. "
                    "답변 시 마크다운 볼드체 기호를 절대 사용하지 마세요."
                )
            else:
                system_instruction = (
                    "당신은 미니어처 게임 워해머 에이지 오브 지그마의 심판입니다. "
                    "제공된 [코어 룰 원본]만을 바탕으로 사용자의 질문에 한국어로 답변해 주세요. "
                    "답변 시 마크다운 볼드체 기호를 절대 사용하지 마세요."
                )

            user_prompt = f"[참고 규칙]\n{retrieved_context}\n\n사용자 질문: {user_query}"

            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2,
            )

            # Gemini 답변 생성
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=user_prompt,
                config=config
            )
            
            source_text = "\n".join(sources_info) if sources_info else "참고할 문서가 없습니다."
            response_text = f"{response.text}\n\n---\n참고한 문서 출처:\n{source_text}"

        st.markdown(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})