from pathlib import Path
from typing import Any

import pytest
from pypdf import PdfWriter
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from linkedin_scrapper.cli import app
from linkedin_scrapper.cv_parser import (
    CandidateProfileExtraction,
    extract_cv_text,
    parse_cv,
    parse_cv_text,
)
from linkedin_scrapper.models import Base, CandidateProfile
from linkedin_scrapper.services.profiles import save_candidate_profile


class StubCVParserAgent:
    def __init__(self, extraction: CandidateProfileExtraction | None = None) -> None:
        self.extraction = extraction or CandidateProfileExtraction(
            target_roles=["Backend Engineer", "AI Engineer", "Data Engineer"],
            skills=["Python", "PostgreSQL", "FastAPI", "Docker", "LangGraph"],
            locations=["Paris", "France", "Remote"],
            remote_preference="remote",
            seniority="senior",
            exclusions=["Not interested in PHP roles."],
            extraction_notes=["stubbed evidence"],
        )
        self.inputs: list[Any] = []

    def invoke(self, input: Any) -> CandidateProfileExtraction:
        self.inputs.append(input)
        return self.extraction


class StubChatModel:
    def __init__(self, agent: StubCVParserAgent) -> None:
        self.agent = agent

    def with_structured_output(self, schema: type[CandidateProfileExtraction]) -> StubCVParserAgent:
        return self.agent


def test_parse_cv_text_uses_agent_to_extract_structured_profile() -> None:
    agent = StubCVParserAgent()

    profile = parse_cv_text(
        """
        Senior Backend Engineer based in Paris, France.
        6 years building Python, PostgreSQL, FastAPI, Docker and LangGraph systems.
        Looking for remote AI Engineer or Data Engineer roles.
        Not interested in PHP roles.
        """,
        agent,
    )

    assert "Backend Engineer" in profile.target_roles
    assert "AI Engineer" in profile.target_roles
    assert "Data Engineer" in profile.target_roles
    assert {"Python", "PostgreSQL", "FastAPI", "Docker", "LangGraph"}.issubset(
        profile.skills
    )
    assert {"Paris", "France", "Remote"}.issubset(profile.locations)
    assert profile.remote_preference == "remote"
    assert profile.seniority == "senior"
    assert profile.exclusions == ["Not interested in PHP roles."]
    assert profile.profile_payload["parser"] == "llm-agent-v1"
    assert profile.profile_payload["extraction_notes"] == ["stubbed evidence"]
    assert agent.inputs


def test_parse_cv_reads_text_file(tmp_path: Path) -> None:
    cv_path = tmp_path / "cv.txt"
    cv_path.write_text("Python Software Engineer in Berlin", encoding="utf-8")
    agent = StubCVParserAgent(
        CandidateProfileExtraction(
            target_roles=["Software Engineer"],
            skills=["Python"],
            locations=["Berlin"],
        )
    )

    profile = parse_cv(cv_path, agent)

    assert profile.cv_text == "Python Software Engineer in Berlin"
    assert "Software Engineer" in profile.target_roles
    assert "Python" in profile.skills
    assert "Berlin" in profile.locations


def test_extract_cv_text_accepts_pdf_and_rejects_empty_pdf(tmp_path: Path) -> None:
    cv_path = tmp_path / "cv.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with cv_path.open("wb") as file:
        writer.write(file)

    with pytest.raises(ValueError, match="does not contain extractable text"):
        extract_cv_text(cv_path)


def test_save_candidate_profile_persists_parsed_profile() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    parsed_profile = parse_cv_text(
        "Senior Python Backend Engineer in Paris",
        StubCVParserAgent(),
    )

    with Session(engine) as session:
        saved = save_candidate_profile(session, parsed_profile)

        profile = session.scalars(select(CandidateProfile)).one()
        assert profile.id == saved.id
        assert profile.cv_text == parsed_profile.cv_text
        assert profile.target_roles == parsed_profile.target_roles
        assert profile.skills == parsed_profile.skills
        assert profile.locations == parsed_profile.locations


def test_parse_cv_cli_outputs_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cv_path = tmp_path / "cv.md"
    cv_path.write_text("Senior Python Data Engineer - Remote Europe", encoding="utf-8")
    _patch_cli_agent(monkeypatch)
    runner = CliRunner()

    result = runner.invoke(app, ["parse-cv", str(cv_path)])

    assert result.exit_code == 0
    assert "Data Engineer" in result.output
    assert "Python" in result.output
    assert "Remote" in result.output


def test_parse_cv_cli_save_requires_database_url(tmp_path: Path, monkeypatch) -> None:
    cv_path = tmp_path / "cv.txt"
    cv_path.write_text("Python Backend Engineer", encoding="utf-8")
    _patch_cli_agent(monkeypatch)
    monkeypatch.setenv("DATABASE_URL", "")
    runner = CliRunner()

    result = runner.invoke(app, ["parse-cv", str(cv_path), "--save"])

    assert result.exit_code == 1
    assert "DATABASE_URL" in result.output


def test_parse_cv_cli_save_persists_profile(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "profiles.db"
    cv_path = tmp_path / "cv.txt"
    cv_path.write_text("Senior Python Backend Engineer in Paris", encoding="utf-8")
    monkeypatch.setenv("DATABASE_URL", f"sqlite+pysqlite:///{db_path}")
    _patch_cli_agent(monkeypatch)
    runner = CliRunner()

    init_result = runner.invoke(app, ["init-db"])
    parse_result = runner.invoke(app, ["parse-cv", str(cv_path), "--save"])

    assert init_result.exit_code == 0
    assert parse_result.exit_code == 0

    engine = create_engine(f"sqlite+pysqlite:///{db_path}")
    with Session(engine) as session:
        profile = session.scalars(select(CandidateProfile)).one()
        assert profile.cv_text == "Senior Python Backend Engineer in Paris"
        assert "Backend Engineer" in profile.target_roles
        assert "Python" in profile.skills


def _patch_cli_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = StubCVParserAgent()
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "linkedin_scrapper.cli.build_cv_parser_chat_model",
        lambda settings: StubChatModel(agent),
    )
