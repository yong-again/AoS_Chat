"""
체크포인트 모듈.

- 이미 파싱 완료된 문서를 output 디렉터리 파일 존재 여부로 판단
- 미완료 태스크만 추려 재시작 지점 결정
- 사용자에게 이어서 진행할지 여부를 물어봄
"""

from __future__ import annotations

from core.logging_config import get_logger
from core.utils import build_output_path

log = get_logger(__name__)


def find_completed(
    all_tasks: list[tuple[str, dict]],
    output_dir: str,
) -> set[tuple[str, str]]:
    """
    output_dir 내 파일 존재 여부로 완료된 (db_target, doc_name) 집합을 반환.
    faction_db는 분리 저장되므로 faction_db 경로 기준으로만 판단.
    """
    completed: set[tuple[str, str]] = set()
    for db_target, task in all_tasks:
        path = build_output_path(
            db_target=db_target,
            doc_name=task["name"],
            outputs_dir=output_dir,
        )
        if path.exists():
            completed.add((db_target, task["name"]))
    return completed


def print_checkpoint_status(
    all_tasks: list[tuple[str, dict]],
    completed: set[tuple[str, str]],
) -> None:
    """완료/미완료 상태를 DB별로 출력."""
    db_stats: dict[str, dict[str, int]] = {}
    for db_target, task in all_tasks:
        stat = db_stats.setdefault(db_target, {"done": 0, "pending": 0})
        if (db_target, task["name"]) in completed:
            stat["done"] += 1
        else:
            stat["pending"] += 1

    log.info("체크포인트 상태:")
    for db_target, stat in db_stats.items():
        log.info("  %-14s 완료 %d개 / 미완료 %d개", db_target, stat["done"], stat["pending"])


def ask_resume(completed_count: int, total_count: int) -> bool:
    """사용자에게 이어서 진행할지 물어봄. True=이어서, False=처음부터."""
    print()
    print(f"  체크포인트 감지: {completed_count}/{total_count}개 문서가 이미 파싱 완료")
    print("  [y] 이어서 진행  (미완료 문서만 처리)")
    print("  [n] 처음부터 시작 (전체 문서 재처리)")
    while True:
        ans = input("  선택 > ").strip().lower()
        if ans in ("y", "yes", ""):
            return True
        if ans in ("n", "no"):
            return False
        print("  y 또는 n을 입력해주세요.")


def filter_pending(
    all_tasks: list[tuple[str, dict]],
    completed: set[tuple[str, str]],
) -> list[tuple[str, dict]]:
    """완료된 태스크를 제외한 미완료 태스크 리스트 반환."""
    return [(db, task) for db, task in all_tasks if (db, task["name"]) not in completed]
