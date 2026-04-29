from uuid import UUID

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from linkedin_scrapper.cli import app
from linkedin_scrapper.cv_parser import (
    CandidateMarkdownProfile,
    CandidateSkill,
    Language,
    ParsedCandidateProfile,
    RemotePreference,
    SkillContext,
    SkillName,
)
from linkedin_scrapper.models import Base, CandidateProfile, SearchRun, SearchRunStatus
from linkedin_scrapper.search_planner import (
    LinkedInSearchPlan,
    LinkedInSearchSuggestion,
    SEARCH_PLANNER_SYSTEM_PROMPT,
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
    assert "f_TPR=" in url
    assert "f_WT=2" in url
    assert "pageNum=0" in url


def test_build_linkedin_jobs_url_adds_known_geo_ids() -> None:
    montpellier_url = build_linkedin_jobs_url(
        LinkedInSearchSuggestion(
            title="Dev IA Montpellier",
            keywords="Développeur IA",
            location="Montpellier, Occitanie, France",
            remote=False,
            rationale="Local search.",
        )
    )
    france_remote_url = build_linkedin_jobs_url(
        LinkedInSearchSuggestion(
            title="Dev IA France remote",
            keywords="Développeur IA",
            location="France",
            remote=True,
            rationale="Remote search.",
        )
    )

    assert "location=Montpellier%2C+Occitanie%2C+France" in montpellier_url
    assert "geoId=106719766" in montpellier_url
    assert "f_WT=2" not in montpellier_url
    assert "location=France" in france_remote_url
    assert "geoId=105015875" in france_remote_url
    assert "f_WT=2" in france_remote_url
    assert "f_TPR=" in france_remote_url


def test_generate_linkedin_searches_returns_two_urls_per_limited_deduped_title() -> None:
    profile = _candidate_profile()
    agent = StubSearchPlannerAgent(
        LinkedInSearchPlan(
            searches=[
                *_default_suggestions(),
                _default_suggestions()[0],
            ]
        )
    )

    searches = generate_linkedin_searches(profile, agent, _settings(max_search_titles=5))

    assert len(searches) == 10
    assert all(search.linkedin_url.startswith("https://www.linkedin.com/jobs/search/?") for search in searches)
    assert searches[0].keywords == "Python Backend Engineer"
    assert searches[0].location == "Montpellier, Occitanie, France"
    assert searches[0].remote is False
    assert "geoId=106719766" in searches[0].linkedin_url
    assert searches[1].keywords == "Python Backend Engineer"
    assert searches[1].location == "France"
    assert searches[1].remote is True
    assert "geoId=105015875" in searches[1].linkedin_url
    assert "f_WT=2" in searches[1].linkedin_url
    assert agent.inputs
    assert "Maximum job titles: 5" in agent.inputs[0][1][1]
    assert "Candidate markdown context:" in agent.inputs[0][1][1]
    assert "# Maxime Kets" in agent.inputs[0][1][1]
    assert "Python (6y, PRODUCTION, last used 2026)" in agent.inputs[0][1][1]


def test_generate_linkedin_searches_accepts_parsed_profile() -> None:
    profile = ParsedCandidateProfile(
        cv_text="# Maxime Kets\n\n**Titre cible** : Backend Engineer, Data Engineer",
        full_name="Maxime Kets",
        target_roles=["Backend Engineer", "Data Engineer"],
        total_years_of_experience=6,
        skills=[
            CandidateSkill(
                name=SkillName.PYTHON,
                years_of_experience=6,
                context=SkillContext.PRODUCTION,
                last_used_year=2026,
            ),
            CandidateSkill(
                name=SkillName.POSTGRESQL,
                years_of_experience=4,
                context=SkillContext.PRODUCTION,
                last_used_year=2026,
            ),
        ],
        locations=["Paris", "Remote"],
        remote_preference=[RemotePreference.FULL_REMOTE],
        languages_spoken=[Language.FR, Language.EN],
        industries_experienced=["HR tech"],
        markdown_profile=CandidateMarkdownProfile(
            markdown="# Maxime Kets\n\n**Titre cible** : Backend Engineer, Data Engineer"
        ),
        seniority="senior",
    )

    searches = generate_linkedin_searches(profile, StubSearchPlannerAgent(), _settings())

    assert len(searches) == 10


def test_generate_linkedin_searches_accepts_fewer_than_default_max() -> None:
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

    searches = generate_linkedin_searches(profile, agent, _settings())

    assert len(searches) == 2
    assert searches[0].keywords == "Python"
    assert searches[0].location == "Montpellier, Occitanie, France"
    assert searches[1].keywords == "Python"
    assert searches[1].location == "France"


def test_search_planner_prompt_favors_concise_distinct_queries() -> None:
    assert "fewer, stronger searches" in SEARCH_PLANNER_SYSTEM_PROMPT
    assert "distinct hiring angle" in SEARCH_PLANNER_SYSTEM_PROMPT
    assert "job titles only" in SEARCH_PLANNER_SYSTEM_PROMPT
    assert "2 to 4 words total" in SEARCH_PLANNER_SYSTEM_PROMPT
    assert "Avoid long keyword lists" in SEARCH_PLANNER_SYSTEM_PROMPT
    assert "Do not stuff every matching skill" in SEARCH_PLANNER_SYSTEM_PROMPT
    assert "AI Engineer Python LLMs" in SEARCH_PLANNER_SYSTEM_PROMPT
    assert "Pinecone, Wagtail" in SEARCH_PLANNER_SYSTEM_PROMPT
    assert "Django React Developer" in SEARCH_PLANNER_SYSTEM_PROMPT


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
        assert len(saved_runs) == 10
        assert len(search_runs) == 10
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
        assert len(session.scalars(select(SearchRun)).all()) == 10


def _candidate_profile() -> CandidateProfile:
    return CandidateProfile(
        cv_text="# Maxime Kets\n\n**Titre cible** : Backend Engineer, Data Engineer",
        target_roles=["Backend Engineer", "Data Engineer", "AI Engineer"],
        total_years_of_experience=6,
        skills=[
            {
                "name": "Python",
                "years_of_experience": 6,
                "context": "PRODUCTION",
                "last_used_year": 2026,
            },
            {
                "name": "PostgreSQL",
                "years_of_experience": 4,
                "context": "PRODUCTION",
                "last_used_year": 2026,
            },
            {
                "name": "LangChain",
                "years_of_experience": 2,
                "context": "PERSONAL",
                "last_used_year": 2025,
            },
            {
                "name": "Docker",
                "years_of_experience": 4,
                "context": "PRODUCTION",
                "last_used_year": 2026,
            },
        ],
        locations=["Paris", "France", "Remote"],
        remote_preference=["FULL_REMOTE"],
        languages_spoken=["FR", "EN"],
        industries_experienced=["HR tech"],
        seniority="senior",
        exclusions=["PHP roles"],
    )


def _default_suggestions() -> list[LinkedInSearchSuggestion]:
    return StubSearchPlannerAgent().plan.searches


def _settings(max_search_titles: int = 10):
    from linkedin_scrapper.config import Settings

    return Settings(
        MAX_SEARCH_TITLES=max_search_titles,
        LINKEDIN_JOBS_ACTOR_ID="curious_coder/linkedin-jobs-scraper",
    )


def _insert_profile(db_path) -> UUID:
    engine = create_engine(f"sqlite+pysqlite:///{db_path}")
    with Session(engine) as session:
        profile = _candidate_profile()
        session.add(profile)
        session.commit()
        return profile.id
