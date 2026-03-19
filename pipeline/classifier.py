"""
문서 분류/태스크 구성 모듈.
"""

from __future__ import annotations

from typing import Optional

from core.logging_config import get_logger
from core import config as cfg

log = get_logger(__name__)


def classify_document(doc_name: str) -> Optional[tuple[str, str]]:
    if any(ex in doc_name for ex in cfg.EXCLUDE_KEYWORDS):
        return None

    if "Core Rules" in doc_name or "Rules Updates" in doc_name or "Glossary" in doc_name:
        return ("rule_db", cfg.RULE_PROMPT)
    if "Battle Profiles" in doc_name:
        return ("balance_db", cfg.BALANCE_PROMPT)

    # 스피어헤드 라우팅 세분화
    if "Spearhead Reference" in doc_name or "Spearhead Doubles" in doc_name:
        return ("spearhead_db", cfg.RULE_PROMPT)  # 코어 룰 성격의 스피어헤드 문서
    elif "Spearhead" in doc_name:
        return ("spearhead_db", cfg.SPEARHEAD_FACTION_PROMPT)  # 특정 팩션의 스피어헤드 뱅가드 문서

    if "Faction Pack:" in doc_name:
        return ("faction_db", cfg.FACTION_PROMPT)
    if "Scourge of Ghyran" in doc_name:
        return ("other_db", cfg.OTHER_PROMPT)

    return None


def build_db_tasks(data: dict[str, dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    """섹션/문서 딕셔너리를 DB별 태스크 리스트로 변환. 각 태스크에 name, url, prompt 포함."""
    tasks: dict[str, list[dict[str, str]]] = {name: [] for name in cfg.DB_NAMES}
    for section, items in data.items():
        for doc_name, url in items.items():
            result = classify_document(doc_name)
            if result is None:
                continue
            db_target, prompt = result
            tasks[db_target].append({"name": doc_name, "url": url, "prompt": prompt})
    return tasks


def print_db_tasks_summary(db_tasks: dict[str, list], top_n: int = 3) -> None:
    """분류 결과 요약 출력 (디버깅/확인용)."""
    for db_name, task_list in db_tasks.items():
        log.info("--- %s (총 %s개) ---", db_name, len(task_list))
        for task in task_list[:top_n]:
            log.info("  [%s]", task["name"])
