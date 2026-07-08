"""
하이브리드 검색 유틸: BM25 희소 검색 + RRF(Reciprocal Rank Fusion) 병합.

ChromaDB 밀집 벡터 검색(의미 유사도)과 BM25 키워드 검색(정확한 용어 매칭)을
병행한 뒤 RRF로 순위를 병합한다. Chroma의 메타데이터 하드 필터(where)와
동일한 조건을 BM25 결과에도 적용할 수 있도록 간단한 where 매처를 제공한다.

Usage:
    from core.hybrid_search import BM25Index, rrf_fuse

    index = BM25Index.from_collection(collection)
    bm25_ids = index.search("Grundstok Trailblazers", n_results=20, where=...)
    fused_ids, scores = rrf_fuse([dense_ids, bm25_ids], top_n=20)
"""
from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from core.logging_config import get_logger

log = get_logger(__name__)

# 영문/숫자 토큰 + 한글 토큰 (문서는 대부분 영어, 질문은 한국어 혼용)
_TOKEN_RE = re.compile(r"[a-z0-9']+|[가-힣]+")

RRF_K = 60  # RRF 상수 (관행값 60 — 상위권 순위 차이를 완만하게 반영)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def matches_where(meta: dict, where: dict | None) -> bool:
    """Chroma where 필터의 부분집합을 메타데이터 dict에 적용.

    지원 형태 (이 프로젝트에서 사용하는 필터 전부):
      {"field": value}                      — 동등 비교
      {"field": {"$eq"/"$ne"/"$in"/"$nin"}} — 연산자
      {"$and": [...]}, {"$or": [...]}       — 논리 결합
    """
    if not where:
        return True
    meta = meta or {}
    for key, cond in where.items():
        if key == "$and":
            if not all(matches_where(meta, c) for c in cond):
                return False
        elif key == "$or":
            if not any(matches_where(meta, c) for c in cond):
                return False
        elif isinstance(cond, dict):
            val = meta.get(key)
            for op, ref in cond.items():
                if op == "$eq" and val != ref:
                    return False
                if op == "$ne" and val == ref:
                    return False
                if op == "$in" and val not in ref:
                    return False
                if op == "$nin" and val in ref:
                    return False
        else:
            if meta.get(key) != cond:
                return False
    return True


class BM25Index:
    """Chroma 컬렉션 전체 문서에 대한 BM25 희소 검색 인덱스."""

    def __init__(self, ids: list[str], documents: list[str], metadatas: list[dict]):
        self.ids = ids
        self.documents = documents
        self.metadatas = metadatas
        self.doc_by_id = dict(zip(ids, documents))
        self.meta_by_id = dict(zip(ids, metadatas))
        # BM25Okapi는 빈 코퍼스에서 ZeroDivisionError → 빈 인덱스는 None으로 표시
        self._bm25 = BM25Okapi([tokenize(d) for d in documents]) if documents else None

    @classmethod
    def from_collection(cls, collection) -> "BM25Index":
        got = collection.get(include=["documents", "metadatas"])
        index = cls(got["ids"], got["documents"], got["metadatas"])
        log.info("BM25 인덱스 구축: %s (%d개 문서)", collection.name, len(got["ids"]))
        return index

    def search(self, query: str, n_results: int = 20, where: dict | None = None) -> list[str]:
        """BM25 점수 내림차순 문서 id 리스트 (점수 0 이하 제외, where 필터 적용)."""
        if self._bm25 is None:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        order = scores.argsort()[::-1]
        out: list[str] = []
        for i in order:
            if scores[i] <= 0:
                break
            if not matches_where(self.metadatas[i], where):
                continue
            out.append(self.ids[i])
            if len(out) >= n_results:
                break
        return out


def rrf_fuse(
    ranked_lists: list[list[str]],
    k: int = RRF_K,
    top_n: int | None = None,
) -> tuple[list[str], dict[str, float]]:
    """RRF: score(d) = Σ 1/(k + rank_i(d)). 순위 리스트들을 병합.

    Returns: (RRF 점수 내림차순 id 리스트, id → 점수 dict)
    """
    scores: dict[str, float] = {}
    for lst in ranked_lists:
        for rank, _id in enumerate(lst):
            scores[_id] = scores.get(_id, 0.0) + 1.0 / (k + rank + 1)
    ordered = sorted(scores, key=lambda x: scores[x], reverse=True)
    if top_n:
        ordered = ordered[:top_n]
    return ordered, scores
