"""
AoS_Chat 공통 로깅 설정.
콘솔 + 파일 출력, 로그 레벨/경로 설정, get_logger()로 모듈별 로거 사용.

Usage:
    from logging_config import setup_logging, get_logger

    setup_logging(level="DEBUG")  # 앱 시작 시 한 번 (선택)
    log = get_logger(__name__)
    log.info("시작")
    log.error("실패: %s", err)
"""
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# 기본값
DEFAULT_LOG_LEVEL = os.environ.get("AOS_LOG_LEVEL", "INFO").upper()
DEFAULT_LOG_DIR = Path(__file__).resolve().parent / "logs"
DEFAULT_LOG_FILE = "aos_chat.log"
MAX_BYTES = 2 * 1024 * 1024  # 2MB
BACKUP_COUNT = 5

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_root_configured = False


def setup_logging(
    level: str = DEFAULT_LOG_LEVEL,
    log_dir: str | Path | None = None,
    log_file: str = DEFAULT_LOG_FILE,
    console: bool = True,
    max_bytes: int = MAX_BYTES,
    backup_count: int = BACKUP_COUNT,
) -> None:
    """
    전역 로깅 설정. 프로젝트 진입점(예: main, app 실행)에서 한 번 호출.
    - level: DEBUG, INFO, WARNING, ERROR, CRITICAL
    - log_dir: 로그 파일 디렉터리 (None이면 DEFAULT_LOG_DIR)
    - log_file: 파일 이름
    - console: True면 stderr로도 출력
    """
    global _root_configured
    if _root_configured:
        return

    log_level = getattr(logging, level, logging.INFO)
    root = logging.getLogger()
    root.setLevel(log_level)
    root.handlers.clear()

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    if console:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(log_level)
        handler.setFormatter(formatter)
        root.addHandler(handler)

    dir_path = Path(log_dir) if log_dir is not None else DEFAULT_LOG_DIR
    dir_path.mkdir(parents=True, exist_ok=True)
    filepath = dir_path / log_file

    try:
        fh = RotatingFileHandler(
            filepath,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        root.addHandler(fh)
    except OSError:
        root.warning("로그 파일을 열 수 없어 파일 로깅을 건너뜁니다: %s", filepath)

    _root_configured = True


def get_logger(name: str) -> logging.Logger:
    """
    모듈/컴포넌트별 로거 반환. setup_logging()을 먼저 호출하지 않았으면
    기본적으로 루트 로거만 사용(콘솔만 출력)되도록 루트 로거를 사용.
    """
    if not _root_configured:
        setup_logging()
    return logging.getLogger(name)
