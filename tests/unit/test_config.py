"""Tests for application configuration."""

from pytest import MonkeyPatch

from fingraph.config.settings import Settings


def test_settings_defaults(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv("FINGRAPH_ENVIRONMENT", raising=False)
    monkeypatch.delenv("FINGRAPH_LOG_LEVEL", raising=False)
    monkeypatch.delenv("FINGRAPH_SEC_USER_AGENT", raising=False)

    settings = Settings(_env_file=None)

    assert settings.environment == "development"
    assert settings.log_level == "INFO"
    assert settings.sec_user_agent is None


def test_settings_reads_prefixed_environment(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv(
        "FINGRAPH_ENVIRONMENT",
        "test",
    )
    monkeypatch.setenv(
        "FINGRAPH_LOG_LEVEL",
        "DEBUG",
    )

    settings = Settings(_env_file=None)

    assert settings.environment == "test"
    assert settings.log_level == "DEBUG"
