"""
임베딩 모델 성능 벤치마크 (코사인 유사도 기준)

평가 방법:
  - Corpus : outputs/rule_db/ 의 청크 (실제 ChromaDB에 들어가는 단위)
  - Query  : 한국어 게임 규칙 질문 15개 (실제 사용 패턴)
  - 관련성 : 각 쿼리에 미리 정의한 키워드가 검색된 문서에 포함되면 relevant
  - 지표   : Recall@1/3/5, MRR@10, 인코딩 시간, 임베딩 차원

사용법:
  python benchmark_embeddings.py
  python benchmark_embeddings.py --top-k 5   # 각 쿼리의 상위 K개 결과 표시
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

# ─── 비교할 모델 목록 ────────────────────────────────────────────────────────
# query_prefix / passage_prefix : E5 계열 모델은 프리픽스가 필요
MODELS = [
    {
        "name": "paraphrase-multilingual-MiniLM-L12-v2",
        "label": "MiniLM-L12 (현재)",
        "query_prefix": "",
        "passage_prefix": "",
    },
    {
        "name": "paraphrase-multilingual-mpnet-base-v2",
        "label": "mpnet-base",
        "query_prefix": "",
        "passage_prefix": "",
    },
    {
        "name": "intfloat/multilingual-e5-small",
        "label": "e5-small",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
    {
        "name": "intfloat/multilingual-e5-base",
        "label": "e5-base",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
    {
        "name": "jhgan/ko-sroberta-multitask",
        "label": "ko-sroberta",
        "query_prefix": "",
        "passage_prefix": "",
    },
    {
        "name": "BAAI/bge-m3",
        "label": "bge-m3",
        "query_prefix": "",
        "passage_prefix": "",
    },
]

# ─── 평가 쿼리 (한국어 질문 + 관련 영문 키워드) ──────────────────────────────
# keywords 중 하나라도 문서 텍스트에 포함되면 relevant로 판정
TEST_QUERIES = [
    {
        "query": "유닛이 이동할 수 있는 거리와 이동 규칙은 무엇인가요?",
        "keywords": ["move characteristic", "normal move", "movement"],
    },
    {
        "query": "슈팅(Shooting) 단계에서 원거리 무기로 공격하는 방법은?",
        "keywords": ["shoot", "missile weapon", "shooting phase"],
    },
    {
        "query": "차지(Charge) 이동의 최소 거리와 규칙은?",
        "keywords": ["charge", "charging move", "charge roll"],
    },
    {
        "query": "전투(Fight) 단계에서 근접 공격 순서는 어떻게 되나요?",
        "keywords": ["fight", "combat", "pile in", "attack sequence"],
    },
    {
        "query": "히어로 단계(Hero Phase)에서 할 수 있는 행동은?",
        "keywords": ["hero phase", "hero"],
    },
    {
        "query": "모델이 상처(Wound)를 입으면 어떻게 처리하나요?",
        "keywords": ["wound", "slain", "damage", "model is removed"],
    },
    {
        "query": "세이브(Save) 굴림과 갑옷 세이브의 메커니즘은?",
        "keywords": ["save", "armour save", "save roll"],
    },
    {
        "query": "지형지물(Terrain)이 전투에 미치는 효과는?",
        "keywords": ["terrain", "cover", "obstacle", "terrain feature"],
    },
    {
        "query": "주문(Spell) 시전 방법과 마법사의 역할은?",
        "keywords": ["spell", "casting", "wizard", "unbind"],
    },
    {
        "query": "달리기(Run) 이동과 일반 이동의 차이는?",
        "keywords": ["run", "run roll", "normal move"],
    },
    {
        "query": "워드 세이브(Ward Save)란 무엇인가요?",
        "keywords": ["ward", "ward save"],
    },
    {
        "query": "목표 지점(Objective)을 점령하는 조건은?",
        "keywords": ["objective", "control", "dominate"],
    },
    {
        "query": "적중 굴림과 상처 굴림은 어떻게 계산하나요?",
        "keywords": ["hit roll", "wound roll", "to hit", "to wound"],
    },
    {
        "query": "배틀쇼크(Battleshock) 테스트는 언제 하나요?",
        "keywords": ["battleshock", "bravery", "flee"],
    },
    {
        "query": "영웅과의 거리 조건이 필요한 능력 사용 규칙은?",
        "keywords": ["wholly within", "within range", "hero", "ability"],
    },
]

OUTPUT_DIR = Path("./outputs/rule_db")


# ─── 코퍼스 로드 ─────────────────────────────────────────────────────────────

def load_corpus() -> list[str]:
    """rule_db의 JSON 파일을 build_db.py와 동일한 방식으로 청킹하여 반환."""
    corpus = []
    for filepath in sorted(OUTPUT_DIR.glob("*.json")):
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            for val in data.values():
                if isinstance(val, dict):
                    for inner_val in val.values():
                        if isinstance(inner_val, list):
                            for item in inner_val:
                                corpus.append(json.dumps(item, ensure_ascii=False))
                        else:
                            corpus.append(json.dumps(inner_val, ensure_ascii=False))
                elif isinstance(val, list):
                    for item in val:
                        corpus.append(json.dumps(item, ensure_ascii=False))
                else:
                    corpus.append(json.dumps(val, ensure_ascii=False))
        elif isinstance(data, list):
            for item in data:
                corpus.append(json.dumps(item, ensure_ascii=False))
    return corpus


# ─── 코사인 유사도 ────────────────────────────────────────────────────────────

def cosine_similarity(query_embs: np.ndarray, doc_embs: np.ndarray) -> np.ndarray:
    """(n_queries, dim) × (n_docs, dim) → (n_queries, n_docs) 코사인 유사도 행렬."""
    q = query_embs / (np.linalg.norm(query_embs, axis=1, keepdims=True) + 1e-10)
    d = doc_embs / (np.linalg.norm(doc_embs, axis=1, keepdims=True) + 1e-10)
    return q @ d.T


def is_relevant(doc_text: str, keywords: list[str]) -> bool:
    doc_lower = doc_text.lower()
    return any(kw.lower() in doc_lower for kw in keywords)


# ─── 단일 모델 평가 ───────────────────────────────────────────────────────────

def evaluate(model_cfg: dict, corpus: list[str], top_k_display: int = 0) -> dict:
    name = model_cfg["name"]
    label = model_cfg["label"]
    qp = model_cfg["query_prefix"]
    pp = model_cfg["passage_prefix"]

    print(f"\n{'─'*60}")
    print(f"  모델: {label}  ({name})")
    print(f"{'─'*60}")

    model = SentenceTransformer(name)

    # 코퍼스 인코딩 시간 측정
    passages = [pp + doc for doc in corpus]
    t0 = time.perf_counter()
    doc_embs = model.encode(passages, batch_size=64, show_progress_bar=True,
                            convert_to_numpy=True)
    encode_time = time.perf_counter() - t0
    dim = doc_embs.shape[1]
    print(f"  코퍼스 {len(corpus)}개 인코딩: {encode_time:.1f}s  |  dim={dim}")

    # 쿼리 인코딩 및 평가
    queries = [qp + tq["query"] for tq in TEST_QUERIES]
    query_embs = model.encode(queries, convert_to_numpy=True)
    sims = cosine_similarity(query_embs, doc_embs)  # (n_queries, n_docs)

    recall = {1: [], 3: [], 5: []}
    mrr_scores = []

    for qi, tq in enumerate(TEST_QUERIES):
        ranked_idx = np.argsort(-sims[qi])  # 내림차순 정렬
        relevant_ranks = [
            rank + 1
            for rank, doc_idx in enumerate(ranked_idx[:10])
            if is_relevant(corpus[doc_idx], tq["keywords"])
        ]

        for k in recall:
            recall[k].append(1.0 if any(r <= k for r in relevant_ranks) else 0.0)

        mrr_scores.append(1.0 / min(relevant_ranks) if relevant_ranks else 0.0)

        # 상위 K개 결과 출력 (옵션)
        if top_k_display > 0:
            print(f"\n  Q{qi+1}: {tq['query']}")
            print(f"  기대 키워드: {tq['keywords']}")
            for rank, doc_idx in enumerate(ranked_idx[:top_k_display]):
                rel = "✓" if is_relevant(corpus[doc_idx], tq["keywords"]) else " "
                snippet = corpus[doc_idx][:100].replace("\n", " ")
                print(f"    [{rel}] #{rank+1} (sim={sims[qi][doc_idx]:.3f}) {snippet}...")

    result = {
        "label": label,
        "name": name,
        "dim": dim,
        "encode_time_s": round(encode_time, 1),
        "Recall@1": round(np.mean(recall[1]), 3),
        "Recall@3": round(np.mean(recall[3]), 3),
        "Recall@5": round(np.mean(recall[5]), 3),
        "MRR@10":   round(np.mean(mrr_scores), 3),
    }

    print(
        f"  Recall@1={result['Recall@1']:.3f}  "
        f"Recall@3={result['Recall@3']:.3f}  "
        f"Recall@5={result['Recall@5']:.3f}  "
        f"MRR@10={result['MRR@10']:.3f}"
    )
    return result


# ─── 결과 테이블 출력 ─────────────────────────────────────────────────────────

def print_table(results: list[dict]) -> None:
    header = f"{'모델':<22} {'dim':>4}  {'시간(s)':>7}  {'R@1':>5}  {'R@3':>5}  {'R@5':>5}  {'MRR@10':>7}"
    sep = "─" * len(header)
    print(f"\n{'='*60}")
    print("  최종 비교 결과 (코사인 유사도 기반)")
    print(f"{'='*60}")
    print(header)
    print(sep)

    best = max(results, key=lambda r: r["MRR@10"])
    for r in sorted(results, key=lambda r: -r["MRR@10"]):
        marker = " ◀ best" if r["label"] == best["label"] else ""
        print(
            f"  {r['label']:<20} {r['dim']:>4}  {r['encode_time_s']:>7.1f}  "
            f"{r['Recall@1']:>5.3f}  {r['Recall@3']:>5.3f}  "
            f"{r['Recall@5']:>5.3f}  {r['MRR@10']:>7.3f}{marker}"
        )
    print(sep)
    print()
    print("  * Recall@K : 상위 K개 안에 관련 문서가 1개 이상 있는 쿼리 비율")
    print("  * MRR@10   : 첫 번째 관련 문서 순위의 역수 평균 (높을수록 좋음)")
    print("  * 시간(s)  : 전체 코퍼스 인코딩 시간")
    print()


# ─── 메인 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="임베딩 모델 벤치마크")
    parser.add_argument(
        "--top-k", type=int, default=0, metavar="K",
        help="각 쿼리의 상위 K개 검색 결과 출력 (0=출력 안 함)",
    )
    parser.add_argument(
        "--models", nargs="+", metavar="LABEL",
        help="비교할 모델 label 지정 (기본: 전체). 예: --models 'MiniLM-L12 (현재)' e5-small",
    )
    args = parser.parse_args()

    if not OUTPUT_DIR.exists():
        raise FileNotFoundError(f"rule_db 폴더가 없습니다: {OUTPUT_DIR}")

    print("코퍼스 로딩 중...")
    corpus = load_corpus()
    print(f"총 {len(corpus)}개 청크 로드 완료\n")

    target_models = MODELS
    if args.models:
        target_models = [m for m in MODELS if m["label"] in args.models]
        if not target_models:
            print(f"[오류] 지정한 모델을 찾을 수 없습니다. 가능한 label: {[m['label'] for m in MODELS]}")
            return

    results = []
    for model_cfg in target_models:
        results.append(evaluate(model_cfg, corpus, top_k_display=args.top_k))

    print_table(results)


if __name__ == "__main__":
    main()
