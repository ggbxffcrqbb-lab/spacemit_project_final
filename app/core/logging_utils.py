from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(log_dir: Path, level: str, runtime_file: str) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / runtime_file

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    return log_path
