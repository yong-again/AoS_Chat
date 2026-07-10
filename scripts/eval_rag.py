"""
RAG 파이프라인 오프라인 평가 하네스.

data/eval/rag_eval_qa.json의 QA 세트로 app.py의 실제 파이프라인(라우팅 →
안전장치 → 키워드 추출 → 하이브리드 검색 → 보강/폴백 → 답변 생성)을
단일 턴으로 재현 실행하고 다음을 측정한다:

  1. 라우팅 정확도       : 안전장치 적용 후 db_name == expected_db_route
  2. 검색 적중률         : mandatory_keywords가 검색 컨텍스트에 포함된 비율
  3. 답변 키워드 커버리지 : mandatory_keywords가 최종 답변에 포함된 비율
  4. LLM 심사            : 기대 답변 대비 사실 일치 (correct/partial/incorrect)

사용:
    python -m scripts.eval_rag                 # 전체 문항
    python -m scripts.eval_rag --only 1,5,18   # 특정 문항만 (1부터)
    python -m scripts.eval_rag --no-judge      # LLM 심사 생략

주의: app.py를 bare 모드로 import하므로 Streamlit 위젯은 기본값으로
동작한다 (include_patch=True, include_old_ghb=False와 동일 조건).
"""
import argparse
import functools
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path

# Streamlit bare 모드 경고(missing ScriptRunContext 등) 억제 — app import 전에 설정
for _name in ("streamlit", "streamlit.runtime", "streamlit.runtime.state",
              "streamlit.runtime.caching", "streamlit.runtime.scriptrunner_utils"):
    logging.getLogger(_name).setLevel(logging.ERROR)

import app  # noqa: E402  (bare import: chat_input이 None이라 UI 파이프라인은 실행 안 됨)
import chromadb  # noqa: E402
from google.genai import types  # noqa: E402
from core.hybrid_search import rrf_fuse  # noqa: E402
from core.logging_config import get_logger  # noqa: E402

log = get_logger("aos.eval")

EVAL_SET = Path("data/eval/rag_eval_qa.json")
RESULT_DIR = Path("data/eval/results")

JUDGE_MODEL = "gemini-2.5-flash-lite"

# ─── Gemini 호출 재시도 래퍼 ──────────────────────────────────────────────────
# route_query/generate_search_query는 내부에서 예외를 삼키고 폴백하므로,
# 레이트 리밋(429)/과부하(503)가 오분류로 기록되지 않게 클라이언트 레벨에서
# 재시도를 심는다 (모든 파이프라인 호출 + 심사 호출에 일괄 적용).
_orig_generate = app.gemini_client.models.generate_content
_RETRYABLE = ("429", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "overloaded")


def _generate_with_retry(**kwargs):
    delays = [5, 15, 30, 60]
    for attempt in range(len(delays) + 1):
        try:
            return _orig_generate(**kwargs)
        except Exception as e:
            if attempt < len(delays) and any(t in str(e) for t in _RETRYABLE):
                log.warning("Gemini 일시 오류(%s) — %ds 후 재시도 (%d/%d)",
                            str(e)[:80], delays[attempt], attempt + 1, len(delays))
                time.sleep(delays[attempt])
                continue
            raise


app.gemini_client.models.generate_content = _generate_with_retry


# bare 모드에서 st.cache_resource가 캐시하지 않을 수 있어 로컬 캐시로 보강
@functools.lru_cache(maxsize=None)
def _bm25(db_name: str):
    return app.load_bm25_index(db_name)


@functools.lru_cache(maxsize=None)
def _spearhead_names():
    return tuple(app.load_spearhead_names())


@functools.lru_cache(maxsize=None)
def _weapon_ability_chunks():
    return app.load_weapon_ability_chunks()


# ─── 키워드 매칭 ──────────────────────────────────────────────────────────────
def _norm(s: str) -> str:
    """대소문자·공백·괄호 등을 무시한 완화 매칭용 정규화. +, - 는 유지
    (5+, D3+1, 1-4 같은 게임 수치 표기를 보존)."""
    return re.sub(r"[^0-9a-z가-힣+\-]+", "", s.lower())


def keyword_in(keyword: str, text: str) -> bool:
    if keyword.lower() in text.lower():
        return True
    return _norm(keyword) in _norm(text)


# ─── 파이프라인 재현 (app.py 채팅 흐름의 단일 턴 버전) ────────────────────────
def run_pipeline(user_query: str) -> dict:
    """app.py의 검색·답변 파이프라인을 단일 턴으로 실행하고 진단 정보를 반환."""
    out = {
        "rewritten_query": user_query,  # 단일 턴 → 재작성 생략(히스토리 없음과 동일)
        "db_name": None, "query_mode": None, "router_raw": None,
        "corrections": [], "search_query": None, "probe": None, "best_db": None,
        "n_docs": 0, "context": "", "answer": None, "error": None,
        "route_s": 0.0, "search_s": 0.0, "answer_s": 0.0,
    }
    client = app.gemini_client
    rewritten_query = user_query

    # 1. 라우팅 + 안전장치 (app.py 859~922행과 동일 로직)
    t0 = time.monotonic()
    db_name, query_type = app.route_query(rewritten_query, client)
    out["router_raw"] = f"{db_name}|{query_type}"
    if query_type == "chat":
        query_mode = "chat"
    else:
        query_mode = ("analysis"
                      if query_type == "analysis" or app.ANALYSIS_RE.search(rewritten_query)
                      else "lookup")

    if db_name == "faction_db" and re.search(r"스피어헤드|spearhead", rewritten_query, re.I):
        db_name = "spearhead_db"
        out["corrections"].append("스피어헤드 키워드 → spearhead_db")

    matched_battlepack = next(
        (src for pat, src in app.BATTLEPACK_SOURCES.items()
         if re.search(pat, rewritten_query, re.I)),
        None,
    )
    if matched_battlepack and db_name == "rule_db":
        db_name = "spearhead_db"
        out["corrections"].append("배틀팩 질문 → spearhead_db")

    if db_name != "spearhead_db":
        q_norm = re.sub(r"[^a-z0-9]+", " ", rewritten_query.lower())
        matched_sp_name = next(
            (nm for nm in _spearhead_names()
             if re.sub(r"[^a-z0-9]+", " ", nm.lower()).strip() in q_norm),
            None,
        )
        if matched_sp_name:
            db_name = "spearhead_db"
            out["corrections"].append(f"스피어헤드 고유명 {matched_sp_name!r} → spearhead_db")

    if db_name != "spearhead_db" and re.search(
            r"regiment\s*abilit|레지먼트\s*어빌리티", rewritten_query, re.I):
        db_name = "spearhead_db"
        out["corrections"].append("Regiment Ability 용어 → spearhead_db")

    if db_name != "other_db" and app.SCOURGE_RE.search(rewritten_query):
        db_name = "other_db"
        out["corrections"].append("Scourge of Ghyran 캠페인 → other_db")

    needs_battlepack_clarify = (
        not matched_battlepack
        and re.search(r"스피어헤드|스피어\s*모드|spearhead", rewritten_query, re.I)
        and app.SPEARHEAD_PROGRESSION_RE.search(rewritten_query)
    )
    out["route_s"] = round(time.monotonic() - t0, 2)
    out["db_name"] = db_name
    out["query_mode"] = query_mode

    if needs_battlepack_clarify:
        out["db_name"] = "clarify"
        out["answer"] = app.CLARIFY_BATTLEPACK_MSG
        return out

    if db_name == "chat":
        t0 = time.monotonic()
        try:
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=rewritten_query,
                config=types.GenerateContentConfig(
                    system_instruction=app.CHAT_SYSTEM_PROMPT,
                    temperature=1.0,
                    thinking_config=types.ThinkingConfig(thinking_budget=1000),
                ),
            )
            out["answer"] = resp.text
        except Exception as e:
            out["error"] = f"chat 응답 실패: {e}"
        out["answer_s"] = round(time.monotonic() - t0, 2)
        return out
    elif query_mode == "chat":
        query_mode = "analysis" if app.ANALYSIS_RE.search(rewritten_query) else "lookup"
        out["query_mode"] = query_mode
        out["corrections"].append("chat 판정이 안전장치로 교정됨")

    # 2. 검색 (app.py 971~1465행과 동일 로직, 사이드바 기본값 기준)
    t0 = time.monotonic()
    try:
        search_query = app.generate_search_query(rewritten_query, db_name, client)
        search_query = search_query.strip().strip('"').strip("'").strip()
        search_query = re.sub(r"\s+", " ", search_query)
        faction_hint = search_query.lower().replace("-", " ").strip()
        out["search_query"] = search_query

        # 안전장치 6(추출 후): 추출 검색어의 스피어헤드 고유명 → spearhead_db
        if db_name != "spearhead_db" and search_query:
            sq_norm = re.sub(r"[^a-z0-9]+", " ", search_query.lower())
            matched_sq_name = next(
                (nm for nm in _spearhead_names()
                 if re.sub(r"[^a-z0-9]+", " ", nm.lower()).strip() in sq_norm),
                None,
            )
            if matched_sq_name:
                db_name = "spearhead_db"
                out["db_name"] = db_name
                out["corrections"].append(
                    f"추출 검색어의 스피어헤드 고유명 {matched_sq_name!r} → spearhead_db")

        _q = search_query if search_query else rewritten_query
        query_texts = ["query: " + _q]
        if db_name in ("rule_db", "spearhead_db", "other_db") and rewritten_query and rewritten_query != _q:
            query_texts.append("query: " + rewritten_query)
        query_embeddings = app.embed_model.encode(query_texts).tolist()
        collections = app.collections
        collection = collections[db_name]

        probe, best_db = {}, None
        for _db, _col in collections.items():
            try:
                if _col.count() == 0:
                    continue
                r = _col.query(query_embeddings=[query_embeddings[0]],
                               n_results=1, include=["distances"])
                if r["distances"] and r["distances"][0]:
                    probe[_db] = round(r["distances"][0][0], 4)
            except Exception:
                pass
        if probe:
            best_db = min(probe, key=probe.get)
        out["probe"] = probe
        out["best_db"] = best_db

        query_kwargs = dict(
            query_embeddings=query_embeddings,
            n_results=app.N_RESULTS[db_name],
            include=["documents", "metadatas", "distances"],
        )
        if faction_hint in app.KNOWN_FACTIONS:
            query_kwargs["where"] = {"faction": faction_hint}
        if db_name == "spearhead_db" and matched_battlepack:
            other_packs = [s for s in app.BATTLEPACK_SOURCES.values() if s != matched_battlepack]
            pack_filter = {"source": {"$nin": other_packs}}
            query_kwargs["where"] = ({"$and": [query_kwargs["where"], pack_filter]}
                                     if "where" in query_kwargs else pack_filter)
        if db_name == "rule_db":
            # 사이드바 기본값: 패치 포함(True), 과거 GHB 제외(False)
            excluded_sources = list(app.OLD_GHB_SOURCES)
            if excluded_sources:
                src_filter = {"source": {"$nin": excluded_sources}}
                query_kwargs["where"] = ({"$and": [query_kwargs["where"], src_filter]}
                                         if "where" in query_kwargs else src_filter)

        try:
            results = collection.query(**query_kwargs)
        except chromadb.errors.ChromaError:
            from chromadb.api.client import SharedSystemClient
            SharedSystemClient.clear_system_cache()
            fresh_client = chromadb.PersistentClient(path="./my_warhammer_db")
            for _db in list(collections):
                collections[_db] = fresh_client.get_collection(name=_db)
            collection = collections[db_name]
            results = collection.query(**query_kwargs)

        # 병행 검색 병합 (rank interleave)
        if len(query_embeddings) > 1 and results["ids"]:
            per_query = [
                list(zip(ids, docs, metas, dists))
                for ids, docs, metas, dists in zip(
                    results["ids"], results["documents"],
                    results["metadatas"], results["distances"],
                )
            ]
            seen_ids, ranked = set(), []
            for rank in range(max(len(rows) for rows in per_query)):
                for rows in per_query:
                    if rank < len(rows) and rows[rank][0] not in seen_ids:
                        seen_ids.add(rows[rank][0])
                        ranked.append(rows[rank])
            ranked = ranked[:app.N_RESULTS[db_name]]
            results["ids"] = [[r[0] for r in ranked]]
            results["documents"] = [[r[1] for r in ranked]]
            results["metadatas"] = [[r[2] for r in ranked]]
            results["distances"] = [[r[3] for r in ranked]]

        # 하이브리드 (BM25 + RRF)
        try:
            bm25_index = _bm25(db_name)
            bm25_query = f"{search_query} {rewritten_query}".strip()
            bm25_ids = bm25_index.search(
                bm25_query, n_results=app.N_RESULTS[db_name],
                where=query_kwargs.get("where"))
            dense_ids = results["ids"][0] if results["ids"] else []
            if bm25_ids:
                fused_ids, _ = rrf_fuse([dense_ids, bm25_ids], top_n=app.N_RESULTS[db_name])
                dense_map = {
                    _id: (doc, meta, dist)
                    for _id, doc, meta, dist in zip(
                        dense_ids, results["documents"][0],
                        results["metadatas"][0], results["distances"][0])
                }
                fused_docs, fused_metas, fused_dists = [], [], []
                for _id in fused_ids:
                    if _id in dense_map:
                        doc, meta, dist = dense_map[_id]
                    else:
                        doc = bm25_index.doc_by_id.get(_id, "")
                        meta = bm25_index.meta_by_id.get(_id, {})
                        dist = None
                    fused_docs.append(doc)
                    fused_metas.append(meta)
                    fused_dists.append(dist)
                results["ids"] = [fused_ids]
                results["documents"] = [fused_docs]
                results["metadatas"] = [fused_metas]
                results["distances"] = [fused_dists]
        except Exception:
            log.warning("BM25 하이브리드 실패 — 밀집 결과만 사용", exc_info=True)

        # 팩션 로스터 주입
        if db_name == "faction_db" and faction_hint in app.KNOWN_FACTIONS:
            try:
                roster = collection.get(
                    where={"$and": [{"faction": faction_hint}, {"type": "warscroll"}]},
                    include=["metadatas"])
                seen = {}
                for meta in roster["metadatas"]:
                    name = (meta.get("unit_name") or "").strip()
                    role = (meta.get("role") or "").strip()
                    key = re.sub(r"[^A-Z0-9]+", " ", name.upper()).strip()
                    if key and (key not in seen or (role and not seen[key][1])):
                        seen[key] = (name, role)
                if seen:
                    groups = {}
                    for name, role in seen.values():
                        groups.setdefault(role or "Other", []).append(name)
                    lines = [f"{faction_hint.upper()} 팩션의 전체 유닛(워스크롤) 목록 (총 {len(seen)}개):"]
                    for role, names in groups.items():
                        lines.append(f"- {role}: " + ", ".join(sorted(names)))
                    results["documents"][0].insert(0, "\n".join(lines))
                    results["metadatas"][0].insert(0, {
                        "source": "faction_unit_roster", "faction": faction_hint,
                        "type": "unit_roster"})
                    results["ids"][0].insert(0, f"unit_roster_{faction_hint}")
            except Exception:
                pass

        # 스피어헤드 확장
        if db_name == "spearhead_db" and results["ids"] and len(results["ids"][0]) > 0:
            try:
                def _norm_name(s: str) -> str:
                    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()

                found_names = []
                for meta in results["metadatas"][0]:
                    nm = (meta or {}).get("spearhead_name")
                    if nm and nm not in found_names:
                        found_names.append(nm)
                query_text = _norm_name(search_query + " " + rewritten_query)
                named = [nm for nm in found_names if _norm_name(nm) in query_text]
                expand_names = (named or found_names)[:2]

                expanded_docs, expanded_metas, expanded_ids = [], [], []
                for nm in expand_names:
                    box = collection.get(where={"spearhead_name": nm},
                                         include=["documents", "metadatas"])
                    if box["ids"]:
                        expanded_docs.extend(box["documents"])
                        expanded_metas.extend(box["metadatas"])
                        expanded_ids.extend(box["ids"])
                if matched_battlepack:
                    box = collection.get(where={"source": matched_battlepack},
                                         include=["documents", "metadatas"])
                    if box["ids"]:
                        expanded_docs.extend(box["documents"])
                        expanded_metas.extend(box["metadatas"])
                        expanded_ids.extend(box["ids"])
                if expanded_ids:
                    seen_exp = set(expanded_ids)
                    for _id, doc, meta in zip(results["ids"][0],
                                              results["documents"][0],
                                              results["metadatas"][0]):
                        if _id in seen_exp:
                            continue
                        other_nm = (meta or {}).get("spearhead_name")
                        if named and other_nm and other_nm not in expand_names:
                            continue
                        expanded_ids.append(_id)
                        expanded_docs.append(doc)
                        expanded_metas.append(meta)
                    results["documents"][0] = expanded_docs
                    results["metadatas"][0] = expanded_metas
                    results["ids"][0] = expanded_ids
            except Exception:
                pass

        # 키워드 폴백 + 크로스 DB 폴백
        def _keyword_hit(docs: list, keyword: str) -> bool:
            kw = keyword.upper()
            return any(kw in d.upper() for d in docs)

        def _fallback_search(col, keyword: str, limit: int = 5, warscroll_only: bool = False):
            extra = {"where": {"type": "warscroll"}} if warscroll_only else {}
            for kw in [keyword.upper(), keyword, keyword.title()]:
                r = col.get(where_document={"$contains": kw},
                            include=["documents", "metadatas"], limit=limit, **extra)
                if r["ids"]:
                    return r
            words = [w for w in keyword.split() if len(w) >= 4]
            for word in words:
                for w_case in [word.upper(), word, word.title()]:
                    r = col.get(where_document={"$contains": w_case},
                                include=["documents", "metadatas"], limit=limit, **extra)
                    if r["ids"]:
                        return r
            return {"ids": [], "documents": [], "metadatas": []}

        flat_docs = results["documents"][0] if results["ids"] and results["ids"][0] else []
        flat_metas = results["metadatas"][0] if results["ids"] and results["ids"][0] else []
        force_warscroll_fallback = (
            db_name == "faction_db"
            and faction_hint not in app.KNOWN_FACTIONS
            and not any((m or {}).get("type") == "warscroll" for m in flat_metas)
        )
        if search_query and (not _keyword_hit(flat_docs, search_query) or force_warscroll_fallback):
            try:
                warscroll_only = db_name in ("faction_db", "spearhead_db")
                fallback = _fallback_search(collection, search_query,
                                            limit=30 if warscroll_only else 5,
                                            warscroll_only=warscroll_only)
                if fallback["ids"]:
                    results["ids"][0] = fallback["ids"] + results["ids"][0]
                    results["documents"][0] = fallback["documents"] + flat_docs
                    results["metadatas"][0] = fallback["metadatas"] + results["metadatas"][0]
                    flat_docs = results["documents"][0]
            except Exception:
                pass

        if db_name == "faction_db" and search_query and not _keyword_hit(flat_docs, search_query):
            try:
                spearhead_col = collections["spearhead_db"]
                sp_fallback = {"ids": []}
                temp_search = _fallback_search(spearhead_col, search_query, limit=5,
                                               warscroll_only=False)
                if temp_search["ids"] and len(temp_search["ids"]) > 0:
                    target_source = temp_search["metadatas"][0].get("source")
                    if target_source:
                        sp_fallback = spearhead_col.get(
                            where={"source": target_source},
                            include=["documents", "metadatas"])
                if sp_fallback["ids"]:
                    results["ids"][0] = sp_fallback["ids"] + results["ids"][0]
                    results["documents"][0] = sp_fallback["documents"] + flat_docs
                    results["metadatas"][0] = sp_fallback["metadatas"] + results["metadatas"][0]
                    flat_docs = results["documents"][0]
            except Exception:
                pass

        if (best_db and best_db != db_name and search_query
                and not _keyword_hit(flat_docs, search_query)):
            try:
                alt = collections[best_db].query(
                    query_embeddings=[query_embeddings[0]],
                    n_results=app.N_RESULTS.get(best_db, 10),
                    include=["documents", "metadatas", "distances"])
                if alt["ids"] and alt["ids"][0]:
                    results["ids"][0] = alt["ids"][0] + results["ids"][0]
                    results["documents"][0] = alt["documents"][0] + flat_docs
                    results["metadatas"][0] = alt["metadatas"][0] + results["metadatas"][0]
                    flat_docs = results["documents"][0]
                    out["corrections"].append(f"프로브 크로스 DB 폴백: {db_name} → {best_db}")
            except Exception:
                pass

        # 무기 능력 정의 주입
        try:
            has_results = bool(results["ids"] and results["ids"][0])
            scan_text = rewritten_query + " " + (
                " ".join(results["documents"][0]) if has_results else "")
            if app.WEAPON_ABILITY_RE.search(scan_text):
                wa_docs, wa_metas, wa_ids = _weapon_ability_chunks()
                if not has_results:
                    results["ids"] = [[]]
                    results["documents"] = [[]]
                    results["metadatas"] = [[]]
                existing = set(results["ids"][0])
                for _id, doc, meta in zip(wa_ids, wa_docs, wa_metas):
                    if _id in existing:
                        continue
                    results["ids"][0].append(_id)
                    results["documents"][0].append(doc)
                    results["metadatas"][0].append(meta)
        except Exception:
            pass

        # 컨텍스트 조합
        retrieved_context = ""
        if results["ids"] and len(results["ids"][0]) > 0:
            for i, (doc, meta) in enumerate(
                    zip(results["documents"][0], results["metadatas"][0])):
                meta = meta or {}
                source_file = meta.get("source", "unknown")
                spearhead_name = meta.get("spearhead_name", "")
                if not spearhead_name and source_file.startswith("spearhead_"):
                    stem = source_file.replace(".json", "")
                    parts = stem.split("_-_", 1)
                    if len(parts) == 2:
                        cand = parts[1].replace("_", " ").strip()
                        if cand.lower() not in ("none", "unknown", ""):
                            spearhead_name = cand.title()
                if spearhead_name:
                    retrieved_context += (f"[{i + 1}] (스피어헤드 이름: {spearhead_name}, "
                                          f"출처: {source_file}) {doc.replace(chr(10), ' ')}\n\n")
                else:
                    retrieved_context += f"[{i + 1}] (출처: {source_file}) {doc.replace(chr(10), ' ')}\n\n"
        else:
            retrieved_context = "관련 문서를 찾을 수 없습니다."
        out["n_docs"] = len(results["documents"][0]) if results["ids"] and results["ids"][0] else 0
        out["context"] = retrieved_context
    except Exception as e:
        out["error"] = f"검색 단계 실패: {type(e).__name__}: {e}"
        out["search_s"] = round(time.monotonic() - t0, 2)
        return out
    out["search_s"] = round(time.monotonic() - t0, 2)

    # 3. 답변 생성 (tool-use 루프 포함)
    t0 = time.monotonic()
    try:
        user_prompt_text = (
            f"[검색 키워드 힌트: '{search_query}']\n\n"
            f"[참고 규칙]\n{retrieved_context}\n"
            f"사용자 질문: {rewritten_query}"
        )
        contents = [types.Content(role="user",
                                  parts=[types.Part.from_text(text=user_prompt_text)])]
        profile = app.ANSWER_PROFILES[query_mode]
        system_prompt = app.SYSTEM_PROMPTS[db_name]
        if query_mode == "analysis":
            system_prompt += app.ANALYSIS_ADDENDUM
        gen_cfg = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=1.0,
            tools=[app.calculate_expected_damage],
            thinking_config=types.ThinkingConfig(
                thinking_budget=profile["thinking"][db_name],
                include_thoughts=True,
            ),
        )
        response = None
        for _ in range(5):
            response = client.models.generate_content(
                model=profile["model"], contents=contents, config=gen_cfg)
            fn_calls = [p for p in response.candidates[0].content.parts if p.function_call]
            if not fn_calls:
                break
            contents.append(response.candidates[0].content)
            fn_response_parts = []
            for p in fn_calls:
                args = dict(p.function_call.args)
                result = app.calculate_expected_damage(**args)
                fn_response_parts.append(types.Part.from_function_response(
                    name=p.function_call.name, response={"result": result}))
            contents.append(types.Content(role="user", parts=fn_response_parts))
        final_text = "".join(
            p.text for p in response.candidates[0].content.parts
            if p.text and not getattr(p, "thought", False))
        out["answer"] = final_text
    except Exception as e:
        out["error"] = f"답변 단계 실패: {type(e).__name__}: {e}"
    out["answer_s"] = round(time.monotonic() - t0, 2)
    return out


# ─── LLM 심사 ─────────────────────────────────────────────────────────────────
JUDGE_PROMPT = """당신은 워해머 에이지 오브 지그마 챗봇의 답변 채점관입니다.
[질문]에 대한 [실제 답변]이 [기대 답변]의 핵심 사실(능력 이름, 수치, 조건, 타이밍)과
일치하는지 판정하세요. 표현 차이·상세도 차이는 무시하고 사실 일치만 봅니다.

판정 기준:
- correct   : 기대 답변의 핵심 사실을 모두 담고 있고 틀린 수치/조건이 없음
- partial   : 핵심 사실 일부만 맞거나, 맞는 내용에 사소한 누락이 있음
- incorrect : 핵심 사실이 틀렸거나, 답을 찾지 못했다고 했거나, 다른 대상을 설명함

JSON 한 개만 출력하세요: {{"verdict": "correct|partial|incorrect", "reason": "한 문장 근거"}}

[질문]
{question}

[기대 답변]
{expected}

[실제 답변]
{answer}
"""


def judge_answer(question: str, expected: str, answer: str) -> dict:
    try:
        resp = app.gemini_client.models.generate_content(
            model=JUDGE_MODEL,
            contents=JUDGE_PROMPT.format(question=question, expected=expected,
                                         answer=answer or "(답변 생성 실패)"),
            config=types.GenerateContentConfig(
                temperature=0.0, response_mime_type="application/json"),
        )
        # 모델이 JSON 뒤에 부연을 붙이는 경우가 있어 첫 JSON 객체만 파싱
        m = re.search(r"\{.*?\}", resp.text, re.S)
        data = json.loads(m.group(0) if m else resp.text)
        verdict = data.get("verdict", "").strip().lower()
        if verdict not in ("correct", "partial", "incorrect"):
            verdict = "parse_error"
        return {"verdict": verdict, "reason": data.get("reason", "")}
    except Exception as e:
        return {"verdict": "judge_error", "reason": f"{type(e).__name__}: {e}"}


# ─── 실행 ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="RAG 파이프라인 평가")
    parser.add_argument("--only", help="실행할 문항 번호(1부터), 쉼표 구분. 예: 1,5,18")
    parser.add_argument("--no-judge", action="store_true", help="LLM 심사 생략")
    parser.add_argument("--sleep", type=float, default=3.0,
                        help="문항 간 대기 초 (레이트 리밋 완화, 기본 3)")
    args = parser.parse_args()

    qa_set = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    indices = list(range(len(qa_set)))
    if args.only:
        indices = [int(x) - 1 for x in args.only.split(",")]

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    records = []

    for n, idx in enumerate(indices):
        item = qa_set[idx]
        q = item["question"]
        log.info("═══ [평가 %d/%d] Q%d: %s", n + 1, len(indices), idx + 1, q)
        result = run_pipeline(q)

        route_ok = result["db_name"] == item["expected_db_route"]
        answer_text = result["answer"] or ""
        ctx = result["context"] or ""
        kw_ans = {kw: keyword_in(kw, answer_text) for kw in item["mandatory_keywords"]}
        # 컨텍스트는 영어 원문이므로 영문/숫자 키워드만 검색 적중률에 반영
        ascii_kws = [kw for kw in item["mandatory_keywords"]
                     if re.search(r"[A-Za-z0-9]", kw) and not re.search(r"[가-힣]", kw)]
        kw_ctx = {kw: keyword_in(kw, ctx) for kw in ascii_kws}

        rec = {
            "index": idx + 1,
            "question": q,
            "expected_db": item["expected_db_route"],
            "routed_db": result["db_name"],
            "router_raw": result["router_raw"],
            "corrections": result["corrections"],
            "route_ok": route_ok,
            "query_mode": result["query_mode"],
            "search_query": result["search_query"],
            "probe_best_db": result["best_db"],
            "n_docs": result["n_docs"],
            "context_kw_hits": kw_ctx,
            "context_recall": (round(sum(kw_ctx.values()) / len(kw_ctx), 3)
                               if kw_ctx else None),
            "answer_kw_hits": kw_ans,
            "answer_recall": round(sum(kw_ans.values()) / len(kw_ans), 3),
            "missing_keywords": [k for k, v in kw_ans.items() if not v],
            "answer": answer_text,
            "error": result["error"],
            "latency": {"route": result["route_s"], "search": result["search_s"],
                        "answer": result["answer_s"]},
        }
        if not args.no_judge and answer_text:
            rec["judge"] = judge_answer(q, item["expected_answer"], answer_text)
        records.append(rec)
        log.info("[평가 %d] 라우팅=%s(%s) 검색적중=%s 답변커버리지=%.0f%% 심사=%s%s",
                 idx + 1, result["db_name"],
                 "OK" if route_ok else f"기대 {item['expected_db_route']}",
                 rec["context_recall"], rec["answer_recall"] * 100,
                 rec.get("judge", {}).get("verdict", "-"),
                 f" 오류={result['error']}" if result["error"] else "")

        # 중간 저장 (중단돼도 결과 보존)
        out_path = RESULT_DIR / f"eval_{ts}.json"
        out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        if n + 1 < len(indices):
            time.sleep(args.sleep)

    # ── 요약 ──
    total = len(records)
    route_acc = sum(r["route_ok"] for r in records) / total
    ans_recall = sum(r["answer_recall"] for r in records) / total
    ctx_vals = [r["context_recall"] for r in records if r["context_recall"] is not None]
    ctx_recall = sum(ctx_vals) / len(ctx_vals) if ctx_vals else 0
    verdicts = {}
    for r in records:
        v = r.get("judge", {}).get("verdict", "미심사")
        verdicts[v] = verdicts.get(v, 0) + 1
    errors = [r for r in records if r["error"]]

    print("\n" + "═" * 70)
    print(f"평가 완료: {total}문항 → {RESULT_DIR / f'eval_{ts}.json'}")
    print(f"  라우팅 정확도       : {route_acc:.1%}")
    print(f"  검색 적중률(영문 kw) : {ctx_recall:.1%}")
    print(f"  답변 키워드 커버리지 : {ans_recall:.1%}")
    print(f"  LLM 심사            : {verdicts}")
    if errors:
        print(f"  파이프라인 오류      : {len(errors)}건 — " +
              ", ".join(f"Q{r['index']}" for r in errors))
    print("═" * 70)
    for r in records:
        judge_v = r.get("judge", {}).get("verdict", "-")
        flag = "✅" if (r["route_ok"] and judge_v == "correct") else "❌"
        print(f"  {flag} Q{r['index']:>2} [{r['expected_db']:>12}→{r['routed_db']:>12}] "
              f"kw {r['answer_recall']:.0%} judge={judge_v}"
              + (f" ERROR: {r['error'][:60]}" if r["error"] else ""))


if __name__ == "__main__":
    main()
