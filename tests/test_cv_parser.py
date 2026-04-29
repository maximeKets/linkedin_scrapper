from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from pypdf import PdfWriter
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from linkedin_scrapper.cli import app
from linkedin_scrapper.cv_parser import (
    CV_PARSER_SYSTEM_PROMPT,
    CV_PROFILE_EXTRACTION_SKILL_NAME,
    CandidateProfileExtraction,
    CandidateMarkdownProfile,
    CandidateSkill,
    ConsolidatedStack,
    EducationItem,
    KeyExperience,
    Language,
    ProfileSnapshot,
    RemotePreference,
    SkillContext,
    SkillName,
    build_cv_parser_agent,
    extract_cv_text,
    load_skill,
    parse_cv,
    parse_cv_text,
)
from linkedin_scrapper.models import Base, CandidateProfile
from linkedin_scrapper.services.profiles import save_candidate_profile


class StubCVParserAgent:
    def __init__(self, extraction: CandidateProfileExtraction | None = None) -> None:
        self.extraction = extraction or CandidateProfileExtraction(
            full_name="Maxime Kets",
            target_roles=["Backend Engineer", "AI Engineer", "Data Engineer"],
            locations=["Montpellier", "Occitanie", "France"],
            total_years_of_experience=6,
            remote_preference=[RemotePreference.FULL_REMOTE, RemotePreference.HYBRID],
            languages_spoken=[Language.FR, Language.EN],
            industries_experienced=["HR tech", "E-commerce"],
            skills=[
                CandidateSkill(
                    name=SkillName.PYTHON,
                    years_of_experience=6,
                    context=SkillContext.PRODUCTION,
                    last_used_year=2026,
                ),
                CandidateSkill(
                    name=SkillName.FASTAPI,
                    years_of_experience=3,
                    context=SkillContext.PRODUCTION,
                    last_used_year=2026,
                ),
                CandidateSkill(
                    name=SkillName.REACT,
                    years_of_experience=4,
                    context=SkillContext.PRODUCTION,
                    last_used_year=2025,
                ),
            ],
            markdown_profile=_markdown_profile(),
            extraction_notes=["stubbed evidence"],
        )
        self.inputs: list[Any] = []

    def invoke(self, input: Any) -> CandidateProfileExtraction:
        self.inputs.append(input)
        return self.extraction


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

    assert profile.full_name == "Maxime Kets"
    assert "Backend Engineer" in profile.target_roles
    assert "AI Engineer" in profile.target_roles
    assert {"Montpellier", "Occitanie", "France"}.issubset(profile.locations)
    assert profile.total_years_of_experience == 6
    assert profile.remote_preference == [RemotePreference.FULL_REMOTE, RemotePreference.HYBRID]
    assert profile.languages_spoken == [Language.FR, Language.EN]
    assert profile.industries_experienced == ["HR tech", "E-commerce"]
    assert profile.skills[0].name == SkillName.PYTHON
    assert profile.skills[0].context == SkillContext.PRODUCTION
    assert profile.skills[0].years_of_experience == 6
    assert profile.skills[0].last_used_year == 2026
    assert profile.cv_text == agent.extraction.markdown_profile.markdown
    assert profile.cv_text.startswith("# Maxime Kets")
    assert "**Localisation** : Montpellier, Occitanie, France" in profile.cv_text
    assert "(Ouvert Remote : Oui)" in profile.cv_text
    assert "## 🎯 Snapshot Profil (Persona)" in profile.cv_text
    assert "## 🛠 Stack Technique Consolidée" in profile.cv_text
    assert "### Backend Engineer | 2021-2026" in profile.cv_text
    assert "## 🎓 Formation" in profile.cv_text
    assert "years_of_experience" not in profile.cv_text
    assert "last_used_year" not in profile.cv_text
    assert profile.profile_payload["parser"] == "llm-cv-profile-v2"
    assert profile.profile_payload["extraction_notes"] == ["stubbed evidence"]
    assert profile.profile_payload["raw_text_sha256"]
    assert profile.profile_payload["narrative"]["full_name"] == "Maxime Kets"
    assert profile.profile_payload["narrative"]["profile_snapshot"]["dna"].startswith(
        "Développeur backend"
    )
    assert agent.inputs
    assert agent.inputs[0]["messages"][0]["role"] == "user"
    assert CV_PROFILE_EXTRACTION_SKILL_NAME in agent.inputs[0]["messages"][0]["content"]


def test_cv_parser_contract_lives_in_load_skill_tool_not_system_prompt() -> None:
    skill_prompt = load_skill.invoke({"skill_name": CV_PROFILE_EXTRACTION_SKILL_NAME})

    assert "Use load_skill when detailed domain instructions are needed" in (
        CV_PARSER_SYSTEM_PROMPT
    )
    assert "Allowed skill names" not in CV_PARSER_SYSTEM_PROMPT
    assert "Do not include a redundant skills table" not in CV_PARSER_SYSTEM_PROMPT
    assert "Allowed skill names" in skill_prompt
    assert "Python, JavaScript, TypeScript" in skill_prompt
    assert "markdown_profile" in skill_prompt
    assert "Do not include a redundant skills table" in skill_prompt


def test_build_cv_parser_agent_uses_load_skill_tool_and_response_format(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_create_agent(**kwargs):
        captured.update(kwargs)
        return StubCVParserAgent()

    monkeypatch.setattr("linkedin_scrapper.cv_parser.create_agent", fake_create_agent)

    agent = build_cv_parser_agent(object())

    assert isinstance(agent, StubCVParserAgent)
    assert captured["tools"] == [load_skill]
    assert captured["system_prompt"] == CV_PARSER_SYSTEM_PROMPT
    assert captured["response_format"] is CandidateProfileExtraction


def test_candidate_skill_rejects_values_outside_controlled_enum() -> None:
    with pytest.raises(ValidationError):
        CandidateProfileExtraction.model_validate(
            {
                "skills": [
                    {
                        "name": "PHP",
                        "years_of_experience": 5,
                        "context": "PRODUCTION",
                        "last_used_year": 2024,
                    }
                ]
            }
        )


def test_parse_cv_reads_text_file(tmp_path: Path) -> None:
    cv_path = tmp_path / "cv.txt"
    cv_path.write_text("Python Software Engineer in Berlin", encoding="utf-8")
    agent = StubCVParserAgent(
        CandidateProfileExtraction(
            full_name="Ada Lovelace",
            target_roles=["Software Engineer"],
            locations=["Berlin"],
            total_years_of_experience=2,
            remote_preference=[RemotePreference.HYBRID],
            languages_spoken=[Language.EN],
            skills=[
                CandidateSkill(
                    name=SkillName.PYTHON,
                    years_of_experience=2,
                    context=SkillContext.ACADEMIC,
                    last_used_year=2026,
                )
            ],
            markdown_profile=CandidateMarkdownProfile(markdown="# Ada Lovelace"),
        )
    )

    profile = parse_cv(cv_path, agent)

    assert profile.cv_text.startswith("# Ada Lovelace")
    assert "Software Engineer" in profile.target_roles
    assert profile.skills[0].name == SkillName.PYTHON
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
        assert profile.locations == parsed_profile.locations
        assert profile.total_years_of_experience == parsed_profile.total_years_of_experience
        assert profile.remote_preference == ["FULL_REMOTE", "HYBRID"]
        assert profile.languages_spoken == ["FR", "EN"]
        assert profile.industries_experienced == ["HR tech", "E-commerce"]
        assert profile.skills[0]["name"] == "Python"
        assert profile.skills[0]["context"] == "PRODUCTION"


def test_parse_cv_cli_outputs_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cv_path = tmp_path / "cv.md"
    cv_path.write_text("Senior Python Data Engineer - Remote Europe", encoding="utf-8")
    _patch_cli_agent(monkeypatch)
    runner = CliRunner()

    result = runner.invoke(app, ["parse-cv", str(cv_path)])

    assert result.exit_code == 0
    assert "Data Engineer" in result.output
    assert "Python" in result.output
    assert "FULL_REMOTE" in result.output
    assert "# Maxime Kets" in result.output


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
        assert profile.cv_text.startswith("# Maxime Kets")
        assert "Backend Engineer" in profile.target_roles
        assert profile.skills[0]["name"] == "Python"
        assert profile.total_years_of_experience == 6


def _patch_cli_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = StubCVParserAgent()
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "linkedin_scrapper.cli.build_cv_parser_chat_model",
        lambda settings: object(),
    )
    monkeypatch.setattr(
        "linkedin_scrapper.cli.build_cv_parser_agent",
        lambda chat_model: agent,
    )


def _markdown_profile() -> CandidateMarkdownProfile:
    return CandidateMarkdownProfile(
        profile_snapshot=ProfileSnapshot(
            dna="Développeur backend orienté produit et automatisation IA.",
            current_focus="Agents IA et matching d'offres.",
            strengths=["Python", "Architecture API", "Automatisation"],
        ),
        consolidated_stack=ConsolidatedStack(
            core_backend_ai=["Python", "FastAPI", "LangGraph"],
            core_frontend=["React", "TypeScript"],
            infra_tools=["Docker", "PostgreSQL"],
            business_domains=["HR tech", "E-commerce"],
        ),
        key_experiences=[
            KeyExperience(
                title="Backend Engineer",
                dates="2021-2026",
                mission="Construire des APIs et agents IA pour automatiser des workflows.",
                achievements=["Déploiement d'agents LLM", "Industrialisation FastAPI"],
                stack=["Python", "FastAPI", "PostgreSQL"],
            )
        ],
        education=[
            EducationItem(
                degree="Master Informatique",
                school="Université de Montpellier",
                years="2018-2020",
            )
        ],
        markdown="\n".join(
            [
                "# Maxime Kets",
                "",
                "**Titre cible** : Backend Engineer, AI Engineer, Data Engineer",
                "**Localisation** : Montpellier, Occitanie, France (Ouvert Remote : Oui)",
                "**Langues** : FR, EN",
                "**Expérience globale** : 6 ans",
                "",
                "## 🎯 Snapshot Profil (Persona)",
                "",
                "* **ADN** : Développeur backend orienté produit et automatisation IA.",
                "* **Focus actuel** : Agents IA et matching d'offres.",
                "* **Points forts** : Python, Architecture API, Automatisation",
                "",
                "## 🛠 Stack Technique Consolidée",
                "",
                "* **Core Backend & IA** : Python, FastAPI, LangGraph",
                "* **Core Frontend** : React, TypeScript",
                "* **Infra & Outils** : Docker, PostgreSQL",
                "* **Domaines métiers** : HR tech, E-commerce",
                "",
                "## 💼 Expériences Clés Synthétisées",
                "",
                "### Backend Engineer | 2021-2026",
                "",
                "* **Mission** : Construire des APIs et agents IA pour automatiser des workflows.",
                "* **Réalisations** :",
                "    * Déploiement d'agents LLM",
                "    * Industrialisation FastAPI",
                "* **Stack** : Python, FastAPI, PostgreSQL",
                "",
                "## 🎓 Formation",
                "",
                "* Master Informatique - Université de Montpellier (2018-2020)",
            ]
        ),
    )
