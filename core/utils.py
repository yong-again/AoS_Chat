"""
프로젝트 공용 유틸리티.
- 경로 생성(디렉터리 보장)
- JSON load/save (원자적 저장)
- 파일명 정규화(가장 보편적인 규칙: 소문자/언더스코어/특수문자 제거)
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any


def project_dir() -> Path:
    """`AoS_Chat` 디렉터리 반환."""
    return Path(__file__).resolve().parent.parent  # core/ → AoS_Chat/


def ensure_dir(path: str | Path) -> Path:
    """디렉터리가 없으면 생성하고 Path 반환."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_filename(name: str, *, max_len: int = 120) -> str:
    """
    파일명에 안전한 형태로 정규화.
    - 소문자
    - 공백/구분자 -> underscore
    - 허용: a-z 0-9 _ . -
    """
    s = name.strip().lower()
    s = re.sub(r"[\s/\\:|]+", "_", s)
    s = re.sub(r"[^a-z0-9_.-]+", "", s)
    s = re.sub(r"_+", "_", s).strip("._-")
    if not s:
        s = "output"
    return s[:max_len]


def atomic_write_text(path: str | Path, text: str, *, encoding: str = "utf-8") -> None:
    """임시 파일에 쓴 뒤 rename 하는 방식으로 원자적 저장."""
    target = Path(path)
    ensure_dir(target.parent)

    fd, tmp_path = tempfile.mkstemp(
        prefix=target.stem + ".",
        suffix=target.suffix + ".tmp",
        dir=str(target.parent),
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(text)
        os.replace(tmp_path, target)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass


def save_json(path: str | Path, data: Any, *, indent: int = 2) -> None:
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=indent))


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def default_cache_path() -> Path:
    """스크래핑 결과(pdf index) 캐시 경로."""
    return project_dir() / "data.json"


def default_outputs_dir() -> Path:
    """추출 결과(JSON) 저장 루트 디렉터리."""
    return ensure_dir(project_dir() / "outputs")


def build_output_path(
    *,
    db_target: str,
    doc_name: str,
    outputs_dir: str | Path | None = None,
    ext: str = ".json",
) -> Path:
    """
    결과물 저장 경로 생성.
    기본 규칙: outputs/<db_target>/<doc_name>.json
    """
    root = Path(outputs_dir) if outputs_dir is not None else default_outputs_dir()
    target_dir = ensure_dir(root / safe_filename(db_target))
    filename = safe_filename(doc_name) + ext
    return target_dir / filename
