"""Tests for settings validation."""

import pytest
from pydantic import ValidationError

from fingraph.config.settings import Settings


def test_log_level_is_normalized() -> None:
    settings = Settings(
        environment="test",
        langfuse_enabled=False,
        _env_file=None,
        log_level="debug",
    )

    assert settings.log_level == "DEBUG"


def test_invalid_log_level_is_rejected() -> None:
    with pytest.raises(
        ValidationError,
        match="Invalid logging level",
    ):
        Settings(
            environment="test",
            langfuse_enabled=False,
            _env_file=None,
            log_level="definitely-not-a-level",
        )
