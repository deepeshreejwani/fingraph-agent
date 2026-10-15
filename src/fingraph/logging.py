"""Structured JSON logging utilities."""

import json
import logging
from datetime import UTC, datetime
from typing import Any


class JsonFormatter(logging.Formatter):
    """Serialize log records as one JSON object per line."""

    _reserved_fields = frozenset(logging.makeLogRecord({}).__dict__)

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        extra_fields = {
            key: value
            for key, value in record.__dict__.items()
            if key not in self._reserved_fields and key not in {"message", "asctime"}
        }

        if extra_fields:
            payload["context"] = extra_fields

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(
            payload,
            default=str,
            ensure_ascii=False,
        )


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging for the application."""

    normalized_level = level.upper()

    if normalized_level not in logging.getLevelNamesMapping():
        raise ValueError(f"Invalid logging level: {level}")

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(normalized_level)
    root_logger.addHandler(handler)
