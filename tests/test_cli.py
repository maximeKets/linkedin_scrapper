from typer.testing import CliRunner

from linkedin_scrapper.cli import app
from linkedin_scrapper.config import Settings


def test_run_pipeline_dry_run_starts_without_secrets() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["run-pipeline", "--dry-run"])

    assert result.exit_code == 0
    assert "parse_cv" in result.output
    assert "send_digest" in result.output


def test_config_command_does_not_print_secret_values(monkeypatch) -> None:
    monkeypatch.setenv("APIFY_API_TOKEN", "apify-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    monkeypatch.setenv("RESEND_API_KEY", "resend-secret")

    runner = CliRunner()
    result = runner.invoke(app, ["config"])

    assert result.exit_code == 0
    assert "apify-secret" not in result.output
    assert "openai-secret" not in result.output
    assert "resend-secret" not in result.output
    assert "'apify_configured': True" in result.output


def test_empty_environment_values_are_missing(monkeypatch) -> None:
    monkeypatch.setenv("APIFY_API_TOKEN", "")

    settings = Settings()

    assert "APIFY_API_TOKEN" in settings.missing_runtime_values()
    assert settings.safe_dump()["apify_configured"] is False


def test_openai_model_is_configurable(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4.1")

    settings = Settings()

    assert settings.openai_model == "gpt-4.1"
    assert settings.safe_dump()["openai_model"] == "gpt-4.1"


def test_cv_parser_model_is_configurable_without_changing_default_model(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4.1-mini")
    monkeypatch.setenv("OPENAI_CV_PARSER_MODEL", "gpt-4.1")

    settings = Settings()

    assert settings.openai_model == "gpt-4.1-mini"
    assert settings.openai_cv_parser_model == "gpt-4.1"
    assert settings.safe_dump()["openai_cv_parser_model"] == "gpt-4.1"
