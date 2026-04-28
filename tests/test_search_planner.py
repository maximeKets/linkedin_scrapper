from uuid import UUID

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from linkedin_scrapper.cli import app
from linkedin_scrapper.cv_parser import ParsedCandidateProfile
from linkedin_scrapper.models import Base, CandidateProfile, SearchRun, SearchRunStatus
from linkedin_scrapper.search_planner import (
    LinkedInSearchPlan,
    LinkedInSearchSuggestion,
    build_linkedin_jobs_url,
    generate_linkedin_searches,
)
from linkedin_scrapper.services.searches import save_search_runs


class StubSearchPlannerAgent:
    def __init__(self, plan: LinkedInSearchPlan | None = None) -> None:
        self.plan = plan or LinkedInSearchPlan(
            searches=[
                LinkedInSearchSuggestion(
                    title="Backend Python",
                    keywords="Python Backend Engineer",
                    location="Paris",
                    remote=True,
                    rationale="Matches backend Python profile.",
                ),
                LinkedInSearchSuggestion(
                    title="Data Engineer",
                    keywords="Data Engineer Python",
                    location="France",
                    remote=True,
                    rationale="Adjacent data role.",
                ),
                LinkedInSearchSuggestion(
                    title="AI Engineer",
                    keywords="AI Engineer LangChain",
                    location="Europe",
                    remote=True,
                    rationale="Matches AI tooling.",
                ),
                LinkedInSearchSuggestion(
                    title="ML Engineer",
                    keywords="Machine Learning Engineer Python",
                    location="Paris",
                    remote=False,
                    rationale="ML-adjacent search.",
                ),
                LinkedInSearchSuggestion(
                    title="Software Engineer",
                    keywords="Software Engineer Python",
                    location="France",
                    remote=None,
                    rationale="Broad fallback search.",
                ),
            ]
        )
        self.inputs = []

    def invoke(self, input):
        self.inputs.append(input)
        return self.plan


class StubChatModel:
    def __init__(self, agent: StubSearchPlannerAgent) -> None:
        self.agent = agent

    def with_structured_output(self, schema):
        return self.agent


def test_build_linkedin_jobs_url_is_public_actor_compatible() -> None:
    url = build_linkedin_jobs_url(
        LinkedInSearchSuggestion(
            title="Backend",
            keywords="Python Backend Engineer",
            location="Paris",
            remote=True,
            rationale="Relevant.",
        )
    )

    assert url.startswith("https://www.linkedin.com/jobs/search/?")
    assert "keywords=Python+Backend+Engineer" in url
    assert "location=Paris" in url
    assert "f_WT=2" in url
    assert "pageNum=0" in url


def test_generate_linkedin_searches_returns_limited_deduped_urls() -> None:
    profile = _candidate_profile()
    agent = StubSearchPlannerAgent(
        LinkedInSearchPlan(
            searches=[
                *_default_suggestions(),
                _default_suggestions()[0],
            ]
        )
    )

    searches = generate_linkedin_searches(profile, agent, _settings(max_searches=5))

    assert len(searches) == 5
    assert all(search.linkedin_url.startswith("https://www.linkedin.com/jobs/search/?") for search in searches)
    assert agent.inputs


def test_generate_linkedin_searches_accepts_parsed_profile() -> None:
    profile = ParsedCandidateProfile(
        cv_text="Senior Python backend engineer in Paris",
        target_roles=["Backend Engineer", "Data Engineer"],
        skills=["Python", "PostgreSQL"],
        locations=["Paris", "Remote"],
        remote_preference="remote",
        seniority="senior",
    )

    searches = generate_linkedin_searches(profile, StubSearchPlannerAgent(), _settings())

    assert len(searches) == 5


def test_generate_linkedin_searches_rejects_too_few_searches() -> None:
    profile = _candidate_profile()
    agent = StubSearchPlannerAgent(
        LinkedInSearchPlan(
            searches=[
                LinkedInSearchSuggestion(
                    title="Only one",
                    keywords="Python",
                    location="Paris",
                    remote=True,
                    rationale="Too narrow.",
                )
            ]
        )
    )

    with pytest.raises(ValueError, match="fewer than 5"):
        generate_linkedin_searches(profile, agent, _settings())


def test_save_search_runs_persists_traceable_searches() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        profile = _candidate_profile()
        session.add(profile)
        session.commit()
        searches = generate_linkedin_searches(profile, StubSearchPlannerAgent(), _settings())

        search_runs = save_search_runs(
            session,
            profile,
            searches,
            "curious_coder/linkedin-jobs-scraper",
        )

        saved_runs = list(session.scalars(select(SearchRun)).all())
        assert len(saved_runs) == 5
        assert len(search_runs) == 5
        assert saved_runs[0].profile_id == profile.id
        assert saved_runs[0].status == SearchRunStatus.PENDING.value
        assert saved_runs[0].linkedin_url.startswith("https://www.linkedin.com/jobs/search/?")


def test_generate_searches_cli_saves_search_runs(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "searches.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+pysqlite:///{db_path}")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "linkedin_scrapper.cli.build_chat_model",
        lambda settings: StubChatModel(StubSearchPlannerAgent()),
    )
    runner = CliRunner()

    init_result = runner.invoke(app, ["init-db"])
    profile_id = _insert_profile(db_path)
    result = runner.invoke(app, ["generate-searches", str(profile_id)])

    assert init_result.exit_code == 0
    assert result.exit_code == 0
    assert "search_run_id" in result.output

    engine = create_engine(f"sqlite+pysqlite:///{db_path}")
    with Session(engine) as session:
        assert len(session.scalars(select(SearchRun)).all()) == 5


def _candidate_profile() -> CandidateProfile:
    return CandidateProfile(
        cv_text="Senior Python backend engineer in Paris",
        target_roles=["Backend Engineer", "Data Engineer", "AI Engineer"],
        skills=["Python", "PostgreSQL", "LangChain", "Docker"],
        locations=["Paris", "France", "Remote"],
        remote_preference="remote",
        seniority="senior",
        exclusions=["PHP roles"],
    )


def _default_suggestions() -> list[LinkedInSearchSuggestion]:
    return StubSearchPlannerAgent().plan.searches


def _settings(max_searches: int = 10):
    from linkedin_scrapper.config import Settings

    return Settings(
        MIN_SEARCH_QUERIES=5,
        MAX_SEARCH_QUERIES=max_searches,
        LINKEDIN_JOBS_ACTOR_ID="curious_coder/linkedin-jobs-scraper",
    )


def _insert_profile(db_path) -> UUID:
    engine = create_engine(f"sqlite+pysqlite:///{db_path}")
    with Session(engine) as session:
        profile = _candidate_profile()
        session.add(profile)
        session.commit()
        return profile.id
