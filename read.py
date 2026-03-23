import chromadb
from sentence_transformers import SentenceTransformer

# 설정 경로 및 모델 (build_db.py와 동일하게 맞춤)
DB_PATH = "./my_warhammer_db"
EMBEDDING_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
DB_NAMES = ["rule_db", "faction_db", "balance_db", "spearhead_db", "other_db"]

def test_database():
    print("=== ChromaDB 데이터베이스 검증 시작 ===")
    
    print("\n모델 및 DB 클라이언트 로드 중...")
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME, 
                                      device="cuda",
                                      )

    # 1. 컬렉션별 데이터 개수 확인
    print("\n[1] 컬렉션별 데이터 적재량 확인")
    total_chunks = 0
    for db_name in DB_NAMES:
        try:
            collection = chroma_client.get_collection(name=db_name)
            count = collection.count()
            total_chunks += count
            print(f" - {db_name}: {count}개의 청크(Chunk) 적재 완료")
        except Exception as e:
            print(f" - {db_name}: 컬렉션을 찾을 수 없습니다. ({e})")
            
    print(f"\n총 {total_chunks}개의 데이터 청크가 DB에 존재합니다.")

    # 2. 샘플 검색 (Semantic Search) 테스트
    print("\n[2] 의미 기반 검색(Semantic Search) 테스트")
    try:
        test_collection = chroma_client.get_collection(name="faction_db")
        
        # 테스트용 질문 (이 부분을 자유롭게 바꿔보세요)
        test_query = "돌격(charge)했을 때 데미지가 증가하거나 보너스를 받는 기병 유닛"
        print(f"테스트 질문: '{test_query}'")
        
        query_embedding = embed_model.encode(test_query).tolist()
        
        # 가장 유사도가 높은 3개의 결과 검색
        results = test_collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
            include=["documents", "metadatas", "distances"]
        )

        for i in range(len(results['documents'][0])):
            doc = results['documents'][0][i]
            meta = results['metadatas'][0][i]
            dist = results['distances'][0][i] # 거리가 가까울수록(숫자가 작을수록) 유사도가 높음
            
            print(f"\n--- 검색 결과 {i+1} (거리: {dist:.4f}) ---")
            print(f"출처 문서: {meta.get('source', '알 수 없음')}")
            print(f"데이터 타입: {meta.get('type', '알 수 없음')}")
            if 'unit_name' in meta:
                print(f"유닛 이름: {meta['unit_name']}")
            print(f"내용 요약: {doc[:200]}...") # 텍스트가 너무 길면 200자까지만 출력
            
    except Exception as e:
        print(f"검색 테스트 중 에러 발생: {e}")

if __name__ == "__main__":
    test_database()