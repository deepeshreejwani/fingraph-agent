"""Tests for structured logging."""

import json
import logging

from fingraph.logging import JsonFormatter


def test_json_formatter_emits_expected_fields() -> None:
    formatter = JsonFormatter()

    record = logging.LogRecord(
        name="fingraph.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )

    payload = json.loads(formatter.format(record))

    assert payload["level"] == "INFO"
    assert payload["logger"] == "fingraph.test"
    assert payload["message"] == "hello"
    assert "timestamp" in payload


def test_json_formatter_preserves_context_fields() -> None:
    formatter = JsonFormatter()

    record = logging.LogRecord(
        name="fingraph.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="retrieval complete",
        args=(),
        exc_info=None,
    )
    record.request_id = "request-123"
    record.route = "VECTOR"

    payload = json.loads(formatter.format(record))

    assert payload["context"]["request_id"] == "request-123"
    assert payload["context"]["route"] == "VECTOR"
