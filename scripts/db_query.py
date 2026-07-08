"""
벡터(밀집) / BM25(희소) / 하이브리드(RRF) 검색 비교 CLI.

LLM 라우팅·키워드 추출 없이 검색 계층만 직접 테스트한다.

사용법 (프로젝트 루트에서):
    uv run python -m scripts.db_query "healing" --db rule_db
    uv run python -m scripts.db_query "Grundstok Trailblazers" --db spearhead_db
    uv run python -m scripts.db_query "Vanari Auralan Wardens" --db faction_db --faction "lumineth realm lords"
"""
import argparse

import chromadb
from sentence_transformers import SentenceTransformer

from core.hybrid_search import BM25Index, rrf_fuse
from core.logging_config import get_logger

log = get_logger(__name__)

EMBED_MODEL = "intfloat/multilingual-e5-base"
DB_PATH = "./my_warhammer_db"
DBS = ("rule_db", "faction_db", "balance_db", "spearhead_db", "other_db")


def _label(meta: dict) -> str:
    meta = meta or {}
    src = (meta.get("source") or "?").replace("wahapedia_", "").replace(".json", "")
    sec = meta.get("section") or meta.get("unit_name") or meta.get("category") or ""
    return f"{src} | {sec}"


def main() -> None:
    ap = argparse.ArgumentParser(description="밀집/BM25/하이브리드 검색 비교")
    ap.add_argument("query", help="검색 질의 (영어 키워드 권장)")
    ap.add_argument("--db", default="rule_db", choices=DBS)
    ap.add_argument("--n", type=int, default=10, help="결과 수 (기본 10)")
    ap.add_argument("--faction", default=None,
                    help='메타데이터 하드 필터, 예: "kharadron overlords"')
    args = ap.parse_args()

    where = {"faction": args.faction} if args.faction else None

    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(args.db)
    print(f"DB: {args.db} ({collection.count()}개 문서)  |  질의: {args.query!r}"
          f"  |  필터: {where}")

    # 1) 밀집 벡터 검색
    model = SentenceTransformer(EMBED_MODEL)
    emb = model.encode("query: " + args.query).tolist()
    kwargs = dict(query_embeddings=[emb], n_results=args.n,
                  include=["documents", "metadatas", "distances"])
    if where:
        kwargs["where"] = where
    dense = collection.query(**kwargs)
    dense_ids = dense["ids"][0]
    dense_meta = dict(zip(dense_ids, dense["metadatas"][0]))
    dense_dist = dict(zip(dense_ids, dense["distances"][0]))

    # 2) BM25 희소 검색 (동일 where 필터 적용)
    index = BM25Index.from_collection(collection)
    bm25_ids = index.search(args.query, n_results=args.n, where=where)

    # 3) RRF 병합
    fused_ids, rrf_scores = rrf_fuse([dense_ids, bm25_ids], top_n=args.n)

    def meta_of(_id: str) -> dict:
        return dense_meta.get(_id) or index.meta_by_id.get(_id) or {}

    print(f"\n─── 밀집 벡터 (top {len(dense_ids)}) " + "─" * 40)
    for i, _id in enumerate(dense_ids):
        print(f"{i+1:2d}. d={dense_dist[_id]:.4f}  {_label(meta_of(_id))}")

    print(f"\n─── BM25 (top {len(bm25_ids)}) " + "─" * 40)
    for i, _id in enumerate(bm25_ids):
        print(f"{i+1:2d}. {_label(meta_of(_id))}")

    print(f"\n─── 하이브리드 RRF (top {len(fused_ids)}) " + "─" * 40)
    for i, _id in enumerate(fused_ids):
        origin = ("D" if _id in dense_dist else "-") + ("B" if _id in bm25_ids else "-")
        print(f"{i+1:2d}. rrf={rrf_scores[_id]:.4f} [{origin}]  {_label(meta_of(_id))}")
    print("\n[D-]=밀집 단독, [-B]=BM25 단독, [DB]=양쪽 모두")

    # 하드 필터 검증
    if where:
        bad = [(_id, meta_of(_id)) for _id in fused_ids
               if meta_of(_id).get("faction") != args.faction]
        print(f"\n필터 검증: 병합 결과 중 faction != {args.faction!r} 인 문서 {len(bad)}건"
              + (" ⚠️" if bad else " ✅"))


if __name__ == "__main__":
    main()
