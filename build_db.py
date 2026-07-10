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
OUTPUT_DIR = Path("./data/outputs")
WARSCROLLS_DIR = Path("./data/warscolls")  # pipeline.wahapedia 스크래핑 결과
DB_PATH = "./my_warhammer_db"
EMBEDDING_MODEL_NAME = 'intfloat/multilingual-e5-base'


def load_wahapedia_warscrolls(collection, embed_model):
    """warscolls/<slug>.json(pipeline.wahapedia 캐시)을 faction_db에 적재.

    기존 faction_db 청크와 동일한 스키마(type=warscroll, faction, unit_name)에
    유닛 카테고리(role) 메타데이터를 더해 사용하므로 app.py의 팩션 필터/
    키워드 폴백 검색/유닛 목록 주입에 그대로 걸린다.
    source는 "wahapedia_<slug>.json"으로 구분한다.
    """
    from pipeline.wahapedia import chunk_payload as warscroll_chunk_payload

    if not WARSCROLLS_DIR.exists():
        log.warning("warscolls 폴더가 없습니다 스킵: %s (python -m pipeline.wahapedia 로 생성)", WARSCROLLS_DIR)
        return

    files = sorted(WARSCROLLS_DIR.glob("*.json"))
    if not files:
        log.warning("warscolls 폴더에 JSON이 없습니다 스킵: %s", WARSCROLLS_DIR)
        return

    for filepath in tqdm(files, desc="적재 중 [faction_db/wahapedia]"):
        docs, embed_texts, metadatas, ids = warscroll_chunk_payload(filepath)
        if docs:
            # 임베딩은 자연어 요약으로, 문서는 전체 JSON으로 저장
            embeddings = embed_model.encode(["passage: " + t for t in embed_texts]).tolist()
            collection.add(
                documents=docs,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids,
            )


def load_wahapedia_rules(collections, embed_model, targets=None):
    """wahapedia_rules/<slug>.json을 페이지별 대상 DB(rule_db/spearhead_db)에 적재.

    스피어헤드 배틀팩 페이지는 룰 질문/스피어헤드 질문 어느 쪽 라우팅에도
    걸리도록 두 컬렉션에 모두 들어간다.
    targets가 주어지면 그 컬렉션에만 적재한다 (--only 부분 재구성용).
    """
    from pipeline.wahapedia_rules import DATA_DIR as RULES_DIR, chunk_payload, page_db_targets

    if not RULES_DIR.exists():
        log.warning("wahapedia_rules 폴더가 없습니다 스킵: %s (python -m pipeline.wahapedia_rules 로 생성)", RULES_DIR)
        return

    files = sorted(RULES_DIR.glob("*.json"))
    for filepath in tqdm(files, desc="적재 중 [rule_db/wahapedia]"):
        docs, metadatas, ids = chunk_payload(filepath)
        if docs:
            embeddings = embed_model.encode(["passage: " + d for d in docs]).tolist()
            for name in page_db_targets(filepath.stem):
                if targets and name not in targets:
                    continue
                collections[name].add(
                    documents=docs,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids,
                )


def load_wahapedia_balance(collection, embed_model):
    """warscolls/<slug>.json의 포인트/편성 정보를 balance_db에 적재."""
    from pipeline.wahapedia import balance_chunk_payload

    if not WARSCROLLS_DIR.exists():
        log.warning("warscolls 폴더가 없습니다 스킵: %s", WARSCROLLS_DIR)
        return

    files = sorted(WARSCROLLS_DIR.glob("*.json"))
    for filepath in tqdm(files, desc="적재 중 [balance_db/wahapedia]"):
        docs, metadatas, ids = balance_chunk_payload(filepath)
        if docs:
            embeddings = embed_model.encode(["passage: " + d for d in docs]).tolist()
            collection.add(
                documents=docs,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids,
            )


def load_wahapedia_faction_rules(collections, embed_model, targets=None):
    """wahapedia_factions/<slug>.json을 faction_db/spearhead_db에 나눠 적재.
    targets가 주어지면 그 컬렉션에만 적재한다."""
    from pipeline.wahapedia_factions import DATA_DIR as FACTIONS_DIR
    from pipeline.wahapedia_factions import chunk_payload as faction_chunk_payload

    if not FACTIONS_DIR.exists():
        log.warning("wahapedia_factions 폴더가 없습니다 스킵: %s (python -m pipeline.wahapedia_factions 로 생성)", FACTIONS_DIR)
        return

    files = sorted(FACTIONS_DIR.glob("*.json"))
    for filepath in tqdm(files, desc="적재 중 [faction/spearhead_db/wahapedia]"):
        payload = faction_chunk_payload(filepath)
        for db_name, (docs, metadatas, ids) in payload.items():
            if targets and db_name not in targets:
                continue
            if docs:
                embeddings = embed_model.encode(["passage: " + d for d in docs]).tolist()
                collections[db_name].add(
                    documents=docs,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids,
                )

def _iter_rule_chunks(node, path: str = "", max_chars: int = 2000):
    """rule_db/other_db JSON을 검색 가능한 크기의 리프 항목 단위로 재귀 분할.

    직렬화 크기가 max_chars 이하면 그대로 청크로 내보내고,
    크면 dict는 키별로, list는 항목별로 내려가며 섹션 경로를 누적한다.
    """
    text = json.dumps(node, ensure_ascii=False)
    if len(text) <= max_chars or not isinstance(node, (dict, list)):
        yield path, text
        return
    if isinstance(node, dict):
        for key, val in node.items():
            yield from _iter_rule_chunks(val, f"{path} > {key}" if path else str(key), max_chars)
    else:
        for item in node:
            yield from _iter_rule_chunks(item, path, max_chars)


def build_database(with_pdf: bool = False, only: str | None = None):
    """ChromaDB 재구성.

    기본은 wahapedia 스크래핑 데이터만 사용한다 (PDF 추출본은 패러프레이징된
    텍스트라 원문 정확도가 낮음). PDF outputs/ 데이터를 함께 적재하려면
    with_pdf=True (CLI: --with-pdf).
    단, other_db(특수 캠페인 룰)는 wahapedia 소스가 없어 PDF 추출본이 유일한
    소스이므로 --with-pdf 없이도 항상 outputs/other_db에서 적재한다.
    only가 주어지면 해당 컬렉션만 삭제·재적재하고 나머지는 건드리지 않는다.
    """
    targets = {only} if only else set(cfg.DB_NAMES)
    log.info("임베딩 모델 로드 중: %s", EMBEDDING_MODEL_NAME)
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    log.info("ChromaDB 클라이언트 초기화 중 (경로: %s)", DB_PATH)
    chroma_client = chromadb.PersistentClient(path=DB_PATH)

    # DB 타겟별 컬렉션 생성 (재구성 대상만 초기화, 나머지는 기존 것 유지)
    collections = {}
    for db_name in cfg.DB_NAMES:
        if db_name in targets:
            try:
                chroma_client.delete_collection(name=db_name)
                log.info("기존 컬렉션 삭제됨: %s", db_name)
            except Exception:
                pass  # 컬렉션이 없는 경우 패스
            collections[db_name] = chroma_client.create_collection(name=db_name)
        else:
            collections[db_name] = chroma_client.get_or_create_collection(name=db_name)

    log.info("데이터 청킹 및 적재 시작... (PDF 포함: %s, 대상: %s)",
             with_pdf, ", ".join(sorted(targets)))

    # outputs 하위 폴더 순회 (기본은 other_db만, --with-pdf면 전체)
    pdf_dbs = cfg.DB_NAMES if with_pdf else ("other_db",)
    for db_name in pdf_dbs:
        if db_name not in targets:
            continue
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
            # 최상위 키 단위 청킹은 코어 룰 전체(10만 자+)가 청크 하나가 되어
            # 임베딩(512토큰)이 앞부분만 반영하므로, 리프 항목 단위로 재귀 분할
            else:
                for idx, (sec_path, doc_text) in enumerate(_iter_rule_chunks(data)):
                    docs.append(doc_text)
                    meta = {"source": source_file, "type": "core_or_other"}
                    if sec_path:
                        meta["section"] = sec_path
                    metadatas.append(meta)
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

    # wahapedia warscroll 데이터를 faction_db에 추가 적재
    if "faction_db" in targets:
        load_wahapedia_warscrolls(collections["faction_db"], embed_model)

    # wahapedia The Rules 데이터를 rule_db(+배틀팩은 spearhead_db)에 추가 적재
    if targets & {"rule_db", "spearhead_db"}:
        load_wahapedia_rules(collections, embed_model, targets)

    # wahapedia 팩션 룰/스피어헤드 룰을 faction_db/spearhead_db에 추가 적재
    if targets & {"faction_db", "spearhead_db"}:
        load_wahapedia_faction_rules(collections, embed_model, targets)

    # wahapedia 워스크롤의 포인트 정보를 balance_db에 적재
    # (PDF 미포함 모드에서 balance_db가 비지 않도록; PDF 포함 시에는
    # 중복 포인트가 충돌할 수 있어 PDF 배틀 프로필만 사용)
    if not with_pdf and "balance_db" in targets:
        load_wahapedia_balance(collections["balance_db"], embed_model)

    log.info("=== 모든 데이터베이스 구축이 성공적으로 완료되었습니다 ===")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ChromaDB 재구성 (기본: wahapedia 데이터만)")
    parser.add_argument("--with-pdf", action="store_true",
                        help="outputs/의 PDF 추출 데이터도 함께 적재")
    parser.add_argument("--only", choices=cfg.DB_NAMES,
                        help="지정한 컬렉션만 재구성 (나머지는 유지)")
    args = parser.parse_args()
    build_database(with_pdf=args.with_pdf, only=args.only)