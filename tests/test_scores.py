from uuid import uuid4

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from linkedin_scrapper.cli import app
from linkedin_scrapper.job_scorer import JobScoreExtraction
from linkedin_scrapper.models import (
    Base,
    CandidateProfile,
    Job,
    JobScore,
    SearchRun,
    SearchRunJob,
)
from linkedin_scrapper.services.scores import (
    list_jobs_for_scoring,
    score_jobs_for_profile,
)


class StubJobScorerAgent:
    def __init__(self, score: int = 82) -> None:
        self.score = score
        self.calls = []

    def invoke(self, input):
        self.calls.append(input)
        return JobScoreExtraction(
            score=self.score,
            summary=f"Score {self.score}",
            match_reasons=["Role aligns"],
            missing_skills=["Kubernetes"],
            risk_flags=["Seniority unclear"],
        )


def test_list_jobs_for_scoring_returns_profile_jobs_without_scores() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        profile = _create_profile_with_jobs(session, job_count=2)
        unrelated_profile = _create_profile_with_jobs(session, job_count=1)
        scored_job = profile.search_runs[0].job_links[0].job
        session.add(JobScore(job=scored_job, score=70, summary="Existing"))
        session.commit()

        jobs = list_jobs_for_scoring(session, profile)
        unrelated_jobs = list_jobs_for_scoring(session, unrelated_profile)

        assert len(jobs) == 1
        assert jobs[0].title == "Job 1"
        assert len(unrelated_jobs) == 1
        assert unrelated_jobs[0].title == "Job 0"


def test_score_jobs_for_profile_persists_scores() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        profile = _create_profile_with_jobs(session, job_count=1)
        job = list_jobs_for_scoring(session, profile)[0]

        results = score_jobs_for_profile(
            session=session,
            profile=profile,
            jobs=[job],
            agent=StubJobScorerAgent(score=88),
            model_name="gpt-test",
        )

        saved_score = session.scalars(select(JobScore)).one()
        assert results[0].score == 88
        assert saved_score.score == 88
        assert saved_score.summary == "Score 88"
        assert saved_score.match_reasons == ["Role aligns"]
        assert saved_score.missing_skills == ["Kubernetes"]
        assert saved_score.risk_flags == ["Seniority unclear"]
        assert saved_score.scoring_payload["model"] == "gpt-test"
        assert saved_score.scoring_payload["profile_id"] == str(profile.id)


def test_score_jobs_skips_existing_scores_unless_rescore() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        profile = _create_profile_with_jobs(session, job_count=1)
        job = profile.search_runs[0].job_links[0].job
        session.add(JobScore(job=job, score=60, summary="Existing"))
        session.commit()
        agent = StubJobScorerAgent(score=90)

        skipped = score_jobs_for_profile(
            session=session,
            profile=profile,
            jobs=[job],
            agent=agent,
            model_name="gpt-test",
        )
        rescored = score_jobs_for_profile(
            session=session,
            profile=profile,
            jobs=[job],
            agent=agent,
            model_name="gpt-test",
            rescore=True,
        )

        saved_score = session.scalars(select(JobScore)).one()
        assert skipped[0].skipped is True
        assert skipped[0].score == 60
        assert len(agent.calls) == 1
        assert rescored[0].skipped is False
        assert saved_score.score == 90


def test_list_jobs_for_scoring_honors_limit_and_include_scored() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        profile = _create_profile_with_jobs(session, job_count=3)
        session.add(JobScore(job=profile.search_runs[0].job_links[0].job, score=75))
        session.commit()

        unscored = list_jobs_for_scoring(session, profile, limit=1)
        all_jobs = list_jobs_for_scoring(session, profile, limit=3, include_scored=True)

        assert len(unscored) == 1
        assert len(all_jobs) == 3


def test_score_jobs_cli_requires_openai_key(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_URL", f"sqlite+pysqlite:///{tmp_path / 'scores.db'}")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    runner = CliRunner()

    result = runner.invoke(app, ["score-jobs", "00000000-0000-0000-0000-000000000000"])

    assert result.exit_code == 1
    assert "OPENAI_API_KEY" in result.output


def test_score_jobs_cli_scores_limited_jobs(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "scores.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+pysqlite:///{db_path}")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "linkedin_scrapper.cli.build_chat_model",
        lambda settings: StubChatModel(StubJobScorerAgent(score=91)),
    )
    runner = CliRunner()

    init_result = runner.invoke(app, ["init-db"])
    profile_id = _insert_profile_with_jobs(db_path, job_count=2)
    result = runner.invoke(app, ["score-jobs", str(profile_id), "--limit-jobs", "1"])

    assert init_result.exit_code == 0
    assert result.exit_code == 0
    assert "jobs_selected" in result.output
    assert "scores_created" in result.output

    engine = create_engine(f"sqlite+pysqlite:///{db_path}")
    with Session(engine) as session:
        scores = session.scalars(select(JobScore)).all()
        assert len(scores) == 1
        assert scores[0].score == 91


class StubChatModel:
    def __init__(self, agent: StubJobScorerAgent) -> None:
        self.agent = agent

    def with_structured_output(self, schema):
        return self.agent


def _create_profile_with_jobs(session: Session, job_count: int) -> CandidateProfile:
    profile_token = str(uuid4())
    profile = CandidateProfile(
        cv_text="Python AI engineer",
        target_roles=["AI Engineer"],
        skills=["Python", "FastAPI"],
        locations=["France"],
        remote_preference="remote",
        seniority="mid",
    )
    search_run = SearchRun(
        candidate_profile=profile,
        query="AI Engineer",
        linkedin_url="https://www.linkedin.com/jobs/search/?keywords=AI+Engineer",
        actor_name="curious_coder/linkedin-jobs-scraper",
    )
    for index in range(job_count):
        search_run.job_links.append(
            SearchRunJob(
                job=Job(
                    title=f"Job {index}",
                    company="Example",
                    url=f"https://www.linkedin.com/jobs/view/{profile_token}-{index}",
                    description="Build AI tools with Python.",
                )
            )
        )
    session.add(search_run)
    session.commit()
    session.refresh(profile)
    return profile


def _insert_profile_with_jobs(db_path, job_count: int):
    engine = create_engine(f"sqlite+pysqlite:///{db_path}")
    with Session(engine) as session:
        profile = _create_profile_with_jobs(session, job_count)
        return profile.id
