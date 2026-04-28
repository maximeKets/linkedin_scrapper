from datetime import UTC, datetime

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from linkedin_scrapper.models import Base, CandidateProfile, Job, SearchRun, SearchRunJob
from linkedin_scrapper.services.jobs import normalize_apify_job, persist_apify_jobs


def test_normalize_apify_job_maps_real_linkedin_payload_fields() -> None:
    item = {
        "id": "4368302805",
        "link": "https://fr.linkedin.com/jobs/view/fullstack-software-engineer-at-alan-4368302805",
        "title": "Fullstack Software Engineer (x/f/m) - Ops AI Platform",
        "companyName": "Alan",
        "location": "Montpellier, Occitanie, France",
        "descriptionHtml": "<strong>HTML description</strong>",
        "descriptionText": "Plain text description",
        "applyUrl": "https://jobs.ashbyhq.com/alan/7094b0ed",
        "salary": "",
        "postedAt": "2026-04-28T14:21:25.000Z",
        "postedAtTimestamp": 1777386085000,
        "workplaceTypes": ["Hybrid"],
        "workRemoteAllowed": False,
        "standardizedTitle": "Artificial Intelligence Engineer",
        "expireAt": 1783003175000,
    }

    normalized = normalize_apify_job(item)

    assert normalized == {
        "external_id": "4368302805",
        "title": "Fullstack Software Engineer (x/f/m) - Ops AI Platform",
        "company": "Alan",
        "location": "Montpellier, Occitanie, France",
        "url": "https://fr.linkedin.com/jobs/view/fullstack-software-engineer-at-alan-4368302805",
        "apply_url": "https://jobs.ashbyhq.com/alan/7094b0ed",
        "description": "Plain text description",
        "salary": None,
        "remote": False,
        "posted_at": datetime(2026, 4, 28, 14, 21, 25, tzinfo=UTC),
        "raw_payload": item,
    }
    assert normalized["raw_payload"]["expireAt"] == 1783003175000


def test_normalize_apify_job_rebuilds_url_from_id_when_missing() -> None:
    normalized = normalize_apify_job(
        {
            "id": "123",
            "title": "Backend Engineer",
            "companyName": "Example",
        }
    )

    assert normalized["url"] == "https://www.linkedin.com/jobs/view/123"
    assert normalized["external_id"] == "123"


def test_normalize_apify_job_ignores_unidentifiable_items() -> None:
    assert normalize_apify_job({"title": "No identifiers"}) is None


def test_persist_apify_jobs_deduplicates_by_external_id() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        first_run = _create_search_run(session, "first")
        second_run = _create_search_run(session, "second")

        assert persist_apify_jobs(session, first_run, [_item("123", title="Old")]) == 1
        assert persist_apify_jobs(session, second_run, [_item("123", title="Updated")]) == 1
        session.commit()

        jobs = session.scalars(select(Job)).all()
        assert len(jobs) == 1
        assert jobs[0].title == "Updated"
        assert len(session.scalars(select(SearchRunJob)).all()) == 2


def test_persist_apify_jobs_deduplicates_by_url() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        search_run = _create_search_run(session, "url")
        session.add(
            Job(
                title="Existing",
                url="https://www.linkedin.com/jobs/view/456",
            )
        )
        session.commit()

        persist_apify_jobs(
            session,
            search_run,
            [
                {
                    "title": "Updated",
                    "link": "https://www.linkedin.com/jobs/view/456",
                    "companyName": "Example",
                }
            ],
        )
        session.commit()

        jobs = session.scalars(select(Job)).all()
        assert len(jobs) == 1
        assert jobs[0].title == "Updated"
        assert jobs[0].company == "Example"


def test_persist_apify_jobs_deduplicates_by_apply_url_and_enriches_linkedin_url() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        search_run = _create_search_run(session, "apply")
        session.add(
            Job(
                title="Existing",
                url="https://jobs.ashbyhq.com/alan/abc",
                apply_url="https://jobs.ashbyhq.com/alan/abc",
            )
        )
        session.commit()

        persist_apify_jobs(
            session,
            search_run,
            [
                {
                    "title": "Updated",
                    "link": "https://www.linkedin.com/jobs/view/789",
                    "applyUrl": "https://jobs.ashbyhq.com/alan/abc",
                }
            ],
        )
        session.commit()

        job = session.scalars(select(Job)).one()
        assert job.title == "Updated"
        assert job.url == "https://www.linkedin.com/jobs/view/789"
        assert job.apply_url == "https://jobs.ashbyhq.com/alan/abc"
        assert len(session.scalars(select(SearchRunJob)).all()) == 1


def _create_search_run(session: Session, suffix: str) -> SearchRun:
    profile = CandidateProfile(cv_text="Python AI engineer")
    search_run = SearchRun(
        candidate_profile=profile,
        query=f"AI Engineer {suffix}",
        linkedin_url=f"https://www.linkedin.com/jobs/search/?keywords=AI+Engineer+{suffix}",
        actor_name="curious_coder/linkedin-jobs-scraper",
    )
    session.add(search_run)
    session.commit()
    session.refresh(search_run)
    return search_run


def _item(external_id: str, title: str) -> dict:
    return {
        "id": external_id,
        "title": title,
        "link": f"https://www.linkedin.com/jobs/view/{external_id}",
        "companyName": "Example",
    }
