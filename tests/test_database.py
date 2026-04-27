from sqlalchemy import create_engine, inspect, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from linkedin_scrapper.models import (
    Application,
    ApplicationStatus,
    Base,
    CandidateProfile,
    Job,
    JobScore,
    SearchRun,
    SearchRunJob,
)
from linkedin_scrapper.services.database import init_db


def test_init_db_creates_required_tables() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")

    init_db(engine)

    tables = set(inspect(engine).get_table_names())
    assert {
        "candidate_profiles",
        "search_runs",
        "jobs",
        "search_run_jobs",
        "job_scores",
        "applications",
    }.issubset(tables)


def test_job_url_unique_constraint_deduplicates_jobs() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        session.add(
            Job(
                title="Backend Engineer",
                company="Example",
                url="https://www.linkedin.com/jobs/view/123",
                raw_payload={"id": "123"},
            )
        )
        session.commit()

        session.add(
            Job(
                title="Backend Engineer",
                company="Example",
                url="https://www.linkedin.com/jobs/view/123",
                raw_payload={"id": "123"},
            )
        )

        try:
            session.commit()
        except IntegrityError:
            session.rollback()
        else:
            raise AssertionError("Expected duplicate job URL to violate unique constraint")


def test_application_status_can_be_updated_independently() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        job = Job(
            title="ML Engineer",
            company="Example",
            url="https://www.linkedin.com/jobs/view/456",
        )
        application = Application(job=job)
        session.add(application)
        session.commit()

        application.status = ApplicationStatus.APPLIED.value
        application.notes = "Applied from dashboard"
        session.commit()

        saved = session.scalars(select(Application)).one()
        assert saved.status == ApplicationStatus.APPLIED.value
        assert saved.notes == "Applied from dashboard"


def test_search_run_can_link_to_existing_deduplicated_job() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        profile = CandidateProfile(cv_text="Python backend engineer")
        search_run = SearchRun(
            candidate_profile=profile,
            query="backend engineer",
            linkedin_url="https://www.linkedin.com/jobs/search/?keywords=backend",
            actor_name="curious_coder/linkedin-jobs-scraper",
        )
        job = Job(
            title="Backend Engineer",
            company="Example",
            url="https://www.linkedin.com/jobs/view/789",
        )
        search_run.job_links.append(SearchRunJob(job=job))
        session.add(search_run)
        session.commit()

        saved_job = session.scalars(select(Job)).one()
        assert saved_job.search_run_links[0].search_run.query == "backend engineer"


def test_job_score_is_one_per_job_with_score_range() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        job = Job(
            title="Data Engineer",
            company="Example",
            url="https://www.linkedin.com/jobs/view/999",
        )
        session.add(JobScore(job=job, score=85, summary="Strong match"))
        session.commit()

        saved_score = session.scalars(select(JobScore)).one()
        assert saved_score.score == 85
        assert saved_score.job.title == "Data Engineer"
