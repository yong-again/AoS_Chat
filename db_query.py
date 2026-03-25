import chromadb
from app import generate_search_query, route_query
from dotenv import dotenv_values
from sentence_transformers import SentenceTransformer
import google.genai as genai
import pprint


ROUTER_MODEL = "gemini-2.5-flash-lite"
ANSWER_MODEL = "gemini-2.5-flash"
EMBED_MODEL  = "intfloat/multilingual-e5-base"

DB_LABELS = {
    "rule_db":      "📖 코어 룰",
    "faction_db":   "⚔️ 팩션 DB",
    "balance_db":   "⚖️ 포인트 DB",
    "spearhead_db": "🏹 스피어헤드 DB",
    "other_db":     "📜 특수 캠페인 DB",
}
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


DB_PATH = "./my_warhammer_db"
config = dotenv_values(".env")

gemini_client = genai.Client(api_key=config["GEMINI_API_KEY"])
chroma_client = chromadb.PersistentClient(path="./my_warhammer_db")
embed_model = SentenceTransformer(EMBED_MODEL)
collections = {
    name: chroma_client.get_collection(name=name)
    for name in DB_LABELS
}

user_query = "루미네스 렐름로드의 스피어헤드 종류를 알려줘."

# 1. 라우터: '사용자의 원본 질문'으로 의도를 파악하여 DB 결정
db_name = route_query(user_query, gemini_client)

#print(db_name)

# 2. 검색 쿼리 추출: 벡터 DB에 던질 '순수 영어 키워드'만 생성
search_query = generate_search_query(user_query, db_name, gemini_client)
faction_hint = search_query.lower().replace("-", " ").strip()
#print(search_query)

_q = search_query if search_query else user_query
query_embedding = embed_model.encode("query: " + _q).tolist()
collection = collections[db_name]
#print(collection)

#pprint.pp(collection.get())
# query_texts uses Chroma's default embedder (384-dim), not EMBED_MODEL (768).
pprint.pp(collection.query(
    query_embeddings=[query_embedding],
    n_results=10,
    where={"faction":faction_hint},
    include=["documents", "metadatas", "distances"],
))

exit()

N_RESULTS = {
    "rule_db":      5,
    "faction_db":   15,
    "balance_db":   60,
    "spearhead_db": 10,
    "other_db":     5,
}

query_kwargs = dict(
    query_embeddings=[query_embedding],
    n_results=N_RESULTS[db_name],
    include=["documents", "metadatas", "distances"],
)

#print(query_kwargs)

if db_name == "rule_db":
    query_kwargs["where"] = {"source": {"$ne": "rules_updates.json"}}

results = collection.query(**query_kwargs)

print(results)


for id in collections:
    print(id)




