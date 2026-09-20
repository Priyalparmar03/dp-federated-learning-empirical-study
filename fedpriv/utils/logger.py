"""Structured experiment logging.

Provides a single logger that writes human-readable logs to stdout + a
run-specific log file, and a JSONL metrics stream that downstream
visualization / table-generation code consumes.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict


def get_logger(name: str, log_dir: str | Path, run_id: str) -> logging.Logger:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"{name}.{run_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s: %(message)s", "%H:%M:%S"
    )

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    file_handler = logging.FileHandler(log_dir / f"{run_id}.log")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger


class MetricsWriter:
    """Appends one JSON object per line -> easy to load with pandas.read_json(lines=True)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a")

    def log(self, **kwargs: Any) -> None:
        record: Dict[str, Any] = {"timestamp": time.time()}
        for k, v in kwargs.items():
            if is_dataclass(v):
                v = asdict(v)
            record[k] = v
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()
