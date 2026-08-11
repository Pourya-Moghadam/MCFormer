"""Consistent console and JSON-lines logging for experiments."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class JsonFormatter(logging.Formatter):
    """Format standard log records as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for key in ("event", "run_id", "epoch", "step", "metric", "value"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def configure_logging(
    *,
    level: str = "INFO",
    log_file: Path | None = None,
    json_lines: bool = True,
) -> logging.Logger:
    """Configure and return the package logger without duplicating handlers."""

    logger = logging.getLogger("mcformer")
    for existing_handler in logger.handlers:
        existing_handler.close()
    logger.handlers.clear()
    logger.setLevel(level.upper())
    logger.propagate = False

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        formatter: logging.Formatter
        formatter = (
            JsonFormatter()
            if json_lines
            else logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger
