from decimal import Decimal
from types import SimpleNamespace

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from linkedin_scrapper.cli import app
from linkedin_scrapper.config import Settings
from linkedin_scrapper.models import (
    Base,
    CandidateProfile,
    Job,
    SearchRun,
    SearchRunJob,
    SearchRunStatus,
)
from linkedin_scrapper.services.apify import (
    ApifySearchRunResult,
    scrape_linkedin_jobs_for_search_run,
)


class FakeApifyClient:
    def __init__(
        self,
        items: list[dict] | None = None,
        run: dict | None = None,
        error: Exception | None = None,
    ) -> None:
        self.items = items or []
        self.run = run if run is not None else {"defaultDatasetId": "dataset-123"}
        self.error = error
        self.actor_name = None
        self.dataset_id = None
        self.actor_call_kwargs = None
        self.dataset_list_kwargs = None

    def actor(self, actor_name: str):
        self.actor_name = actor_name
        return self

    def call(self, **kwargs):
        self.actor_call_kwargs = kwargs
        if self.error is not None:
            raise self.error
        return self.run

    def dataset(self, dataset_id: str):
        self.dataset_id = dataset_id
        return self

    def list_items(self, **kwargs):
        self.dataset_list_kwargs = kwargs
        return SimpleNamespace(items=self.items)


def test_scrape_search_run_calls_actor_and_persists_jobs() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    settings = _settings()
    client = FakeApifyClient(
        items=[
            {
                "id": "123",
                "title": "AI Engineer",
                "companyName": "Example",
                "location": "Montpellier, France",
                "url": "https://www.linkedin.com/jobs/view/123",
                "applyUrl": "https://example.com/apply",
                "description": "Build RAG systems.",
                "salary": "50k",
                "remote": True,
                "postedAt": "2026-04-28T10:00:00Z",
            }
        ]
    )

    with Session(engine) as session:
        search_run = _create_search_run(session)

        result = scrape_linkedin_jobs_for_search_run(session, client, search_run, settings, count=20)

        assert client.actor_name == "curious_coder/linkedin-jobs-scraper"
        assert client.actor_call_kwargs == {
            "run_input": {
                "urls": [search_run.linkedin_url],
                "count": 20,
                "scrapeCompany": True,
            },
            "max_items": 20,
            "max_total_charge_usd": Decimal("1.00"),
        }
        assert client.dataset_id == "dataset-123"
        assert client.dataset_list_kwargs == {"limit": 20, "clean": True}
        assert result.status == SearchRunStatus.SUCCEEDED.value
        assert result.jobs_received == 1
        assert result.jobs_saved == 1

        job = session.scalars(select(Job)).one()
        assert job.external_id == "123"
        assert job.title == "AI Engineer"
        assert job.company == "Example"
        assert job.remote is True
        assert job.raw_payload["description"] == "Build RAG systems."
        assert session.scalars(select(SearchRunJob)).one().job_id == job.id


def test_scrape_search_run_reuses_existing_job_by_url() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    client = FakeApifyClient(
        items=[
            {
                "title": "Backend Engineer",
                "url": "https://www.linkedin.com/jobs/view/456",
                "companyName": "Updated",
            }
        ]
    )

    with Session(engine) as session:
        search_run = _create_search_run(session)
        session.add(
            Job(
                title="Old title",
                company="Old company",
                url="https://www.linkedin.com/jobs/view/456",
            )
        )
        session.commit()

        result = scrape_linkedin_jobs_for_search_run(session, client, search_run, _settings(), count=10)

        jobs = list(session.scalars(select(Job)))
        assert len(jobs) == 1
        assert jobs[0].title == "Backend Engineer"
        assert jobs[0].company == "Updated"
        assert len(session.scalars(select(SearchRunJob)).all()) == 1
        assert result.jobs_saved == 1


def test_scrape_search_run_empty_dataset_succeeds() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        search_run = _create_search_run(session)

        result = scrape_linkedin_jobs_for_search_run(
            session,
            FakeApifyClient(items=[]),
            search_run,
            _settings(),
            count=10,
        )

        assert result.status == SearchRunStatus.SUCCEEDED.value
        assert result.jobs_received == 0
        assert result.jobs_saved == 0
        assert session.get(SearchRun, search_run.id).status == SearchRunStatus.SUCCEEDED.value


def test_scrape_search_run_actor_error_marks_run_failed() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        search_run = _create_search_run(session)

        result = scrape_linkedin_jobs_for_search_run(
            session,
            FakeApifyClient(error=RuntimeError("Actor failed")),
            search_run,
            _settings(),
            count=10,
        )

        saved_run = session.get(SearchRun, search_run.id)
        assert result.status == SearchRunStatus.FAILED.value
        assert result.error_message == "Actor failed"
        assert saved_run.status == SearchRunStatus.FAILED.value
        assert saved_run.error_message == "Actor failed"


def test_scrape_jobs_cli_requires_apify_token(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_URL", f"sqlite+pysqlite:///{tmp_path / 'jobs.db'}")
    monkeypatch.setenv("APIFY_API_TOKEN", "")
    runner = CliRunner()

    result = runner.invoke(app, ["scrape-jobs", "00000000-0000-0000-0000-000000000000"])

    assert result.exit_code == 1
    assert "APIFY_API_TOKEN" in result.output


def test_scrape_jobs_cli_limits_pending_runs(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "jobs.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+pysqlite:///{db_path}")
    monkeypatch.setenv("APIFY_API_TOKEN", "test-token")
    calls = []

    def fake_scrape(session, client, search_run, settings, count):
        calls.append((search_run.id, count))
        search_run.status = SearchRunStatus.SUCCEEDED.value
        session.commit()
        return ApifySearchRunResult(
            search_run_id=str(search_run.id),
            status=SearchRunStatus.SUCCEEDED.value,
            jobs_received=0,
            jobs_saved=0,
        )

    monkeypatch.setattr("linkedin_scrapper.cli.build_apify_client", lambda settings: object())
    monkeypatch.setattr("linkedin_scrapper.cli.scrape_linkedin_jobs_for_search_run", fake_scrape)
    runner = CliRunner()

    init_result = runner.invoke(app, ["init-db"])
    profile_id = _insert_profile_with_search_runs(db_path, count=2)
    result = runner.invoke(
        app,
        ["scrape-jobs", str(profile_id), "--count", "20", "--limit-runs", "1"],
    )

    assert init_result.exit_code == 0
    assert result.exit_code == 0
    assert len(calls) == 1
    assert calls[0][1] == 20
    assert "pending_runs_selected" in result.output
    assert "jobs_saved" in result.output


def _settings() -> Settings:
    return Settings(
        APIFY_API_TOKEN="test-token",
        APIFY_MAX_TOTAL_CHARGE_USD="1.00",
        LINKEDIN_JOBS_ACTOR_ID="curious_coder/linkedin-jobs-scraper",
    )


def _create_search_run(session: Session) -> SearchRun:
    profile = CandidateProfile(cv_text="Python AI engineer")
    search_run = SearchRun(
        candidate_profile=profile,
        query="AI Engineer Python",
        linkedin_url="https://www.linkedin.com/jobs/search/?keywords=AI+Engineer+Python",
        actor_name="curious_coder/linkedin-jobs-scraper",
        status=SearchRunStatus.PENDING.value,
    )
    session.add(search_run)
    session.commit()
    session.refresh(search_run)
    return search_run


def _insert_profile_with_search_runs(db_path, count: int):
    engine = create_engine(f"sqlite+pysqlite:///{db_path}")
    with Session(engine) as session:
        profile = CandidateProfile(cv_text="Python AI engineer")
        session.add(profile)
        session.flush()
        for index in range(count):
            session.add(
                SearchRun(
                    candidate_profile=profile,
                    query=f"AI Engineer Python {index}",
                    linkedin_url=(
                        "https://www.linkedin.com/jobs/search/"
                        f"?keywords=AI+Engineer+Python+{index}"
                    ),
                    actor_name="curious_coder/linkedin-jobs-scraper",
                    status=SearchRunStatus.PENDING.value,
                )
            )
        session.commit()
        return profile.id
