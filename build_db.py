import json
import os
from pathlib import Path
import chromadb
from dotenv import dotenv_values
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import core.config as cfg
from core.logging_config import get_logger

log = get_logger(__name__)

_env = dotenv_values(".env")
if _env.get("HF_TOKEN"):
    os.environ["HF_TOKEN"] = _env["HF_TOKEN"]

# 설정 경로
OUTPUT_DIR = Path("./outputs")
DB_PATH = "./my_warhammer_db"
EMBEDDING_MODEL_NAME = 'intfloat/multilingual-e5-base'

def build_database():
    log.info("임베딩 모델 로드 중: %s", EMBEDDING_MODEL_NAME)
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    log.info("ChromaDB 클라이언트 초기화 중 (경로: %s)", DB_PATH)
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    
    # DB 타겟별 컬렉션 생성 (기존 데이터가 있다면 초기화)
    collections = {}
    for db_name in cfg.DB_NAMES:
        try:
            chroma_client.delete_collection(name=db_name)
            log.info("기존 컬렉션 삭제됨: %s", db_name)
        except Exception:
            pass  # 컬렉션이 없는 경우 패스
        collections[db_name] = chroma_client.create_collection(name=db_name)
        
    log.info("데이터 청킹 및 적재 시작...")
    
    # outputs 하위 폴더 순회
    for db_name in cfg.DB_NAMES:
        db_dir = OUTPUT_DIR / db_name
        if not db_dir.exists():
            log.warning("폴더가 없습니다 스킵: %s", db_dir)
            continue
            
        collection = collections[db_name]
        files = list(db_dir.glob("*.json"))
        
        for filepath in tqdm(files, desc=f"적재 중 [{db_name}]"):
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            docs = []
            metadatas = []
            ids = []
            
            source_file = filepath.name

            # 파일명에서 정규화된 팩션 키 추출
            # 예: spearhead_stormcast_eternals_-_yndrastas_spearhead.json → "stormcast eternals"
            # 예: faction_pack_lumineth_realm-lords.json → "lumineth realm lords"
            stem = filepath.stem
            faction_key = stem.replace("faction_pack_", "").replace("spearhead_", "")
            faction_key = faction_key.split("_-_")[0]          # "xxx - yyy" 앞부분만
            faction_key = faction_key.replace("_", " ").replace("-", " ").strip()

            # 1. 팩션 팩 & 스피어헤드 데이터 청킹
            if db_name in ("faction_db", "spearhead_db"):
                # spearhead_*.json은 {"spearhead": {"warscrolls": [], "spearhead_rules": []}} 구조
                # faction_pack_*.json은 {"army_rules": {}, "warscrolls": []} 구조
                spearhead_block = data.get("spearhead") if "spearhead" in data else None
                root = spearhead_block if spearhead_block else data

                spearhead_name = root.get("spearhead_name", "")

                # 아미 룰 / 스피어헤드 룰 청킹
                for rule_key in ("army_rules", "spearhead_rules"):
                    rule_data = root.get(rule_key)
                    if isinstance(rule_data, list):
                        for idx, rule in enumerate(rule_data):
                            doc_text = json.dumps(rule, ensure_ascii=False)
                            docs.append(doc_text)
                            meta_dict = {"source": source_file, "faction": faction_key, "type": "rule", "category": rule_key}
                            if spearhead_name:
                                meta_dict["spearhead_name"] = spearhead_name
                            metadatas.append(meta_dict)
                            ids.append(f"{source_file}_rule_{rule_key}_{idx}")
                    elif isinstance(rule_data, dict):
                        for rule_category, rules in rule_data.items():
                            if isinstance(rules, list):
                                for idx, rule in enumerate(rules):
                                    doc_text = json.dumps(rule, ensure_ascii=False)
                                    docs.append(doc_text)
                                    meta_dict = {"source": source_file, "faction": faction_key, "type": "rule", "category": rule_key}
                                    if spearhead_name:
                                        meta_dict["spearhead_name"] = spearhead_name
                                    metadatas.append(meta_dict)
                                    ids.append(f"{source_file}_rule_{rule_category}_{idx}")

                # 워스크롤(유닛) 청킹
                for idx, unit in enumerate(root.get("warscrolls") or []):
                    unit_name = unit.get("unit_name", f"unit_{idx}")
                    doc_text = json.dumps(unit, ensure_ascii=False)
                    docs.append(doc_text)
                    meta_dict = {"source": source_file, "faction": faction_key, "type": "warscroll", "unit_name": unit_name}
                    if spearhead_name:
                        meta_dict["spearhead_name"] = spearhead_name
                    metadatas.append(meta_dict)
                    ids.append(f"{source_file}_warscroll_{idx}")
            
            # 2. 배틀 프로필(포인트) 데이터 청킹
            elif db_name == "balance_db":
                if isinstance(data, list):
                    for idx, unit in enumerate(data):
                        unit_name = unit.get("unit_name", f"unit_{idx}")
                        doc_text = json.dumps(unit, ensure_ascii=False)
                        docs.append(doc_text)
                        metadatas.append({"source": source_file, "type": "point", "unit_name": unit_name})
                        ids.append(f"{source_file}_point_{idx}")
                    
            # 3. 코어 룰 & 기타 특수 문서 청킹
            else:
                if isinstance(data, dict):
                    for key, val in data.items():
                        doc_text = json.dumps(val, ensure_ascii=False)
                        docs.append(doc_text)
                        metadatas.append({"source": source_file, "type": "core_or_other", "section": key})
                        ids.append(f"{source_file}_{key}")
                elif isinstance(data, list):
                    for idx, val in enumerate(data):
                        doc_text = json.dumps(val, ensure_ascii=False)
                        docs.append(doc_text)
                        metadatas.append({"source": source_file, "type": "core_or_other"})
                        ids.append(f"{source_file}_{idx}")

            # 추출된 청크들을 임베딩하여 컬렉션에 추가
            if docs:
                embeddings = embed_model.encode(["passage: " + d for d in docs]).tolist()
                collection.add(
                    documents=docs,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids
                )

    log.info("=== 모든 데이터베이스 구축이 성공적으로 완료되었습니다 ===")

if __name__ == "__main__":
    build_database()