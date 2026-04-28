from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from apify_client import ApifyClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from linkedin_scrapper.config import Settings
from linkedin_scrapper.models import Job, SearchRun, SearchRunJob, SearchRunStatus


def build_apify_client(settings: Settings) -> ApifyClient:
    if settings.apify_api_token is None:
        raise RuntimeError("APIFY_API_TOKEN is required to build the Apify client.")

    return ApifyClient(settings.apify_api_token.get_secret_value())


@dataclass(frozen=True)
class ApifySearchRunResult:
    search_run_id: str
    status: str
    jobs_received: int
    jobs_saved: int
    error_message: str | None = None


def scrape_linkedin_jobs_for_search_run(
    session: Session,
    client: Any,
    search_run: SearchRun,
    settings: Settings,
    count: int,
) -> ApifySearchRunResult:
    search_run.status = SearchRunStatus.RUNNING.value
    search_run.started_at = datetime.now(UTC)
    search_run.error_message = None
    session.commit()

    try:
        run_input = {
            "urls": [search_run.linkedin_url],
            "count": count,
            "scrapeCompany": True,
        }
        run = client.actor(search_run.actor_name).call(
            run_input=run_input,
            max_items=count,
            max_total_charge_usd=settings.apify_max_total_charge_usd,
        )
        if not run:
            raise RuntimeError("Apify Actor returned no run metadata.")

        dataset_id = run.get("defaultDatasetId")
        if not dataset_id:
            raise RuntimeError("Apify Actor run did not provide defaultDatasetId.")

        dataset_items = client.dataset(dataset_id).list_items(
            limit=count,
            clean=True,
        )
        raw_items = list(getattr(dataset_items, "items", []))
        jobs_saved = _persist_apify_jobs(session, search_run, raw_items)

        search_run.status = SearchRunStatus.SUCCEEDED.value
        search_run.completed_at = datetime.now(UTC)
        search_run.error_message = None
        session.commit()

        return ApifySearchRunResult(
            search_run_id=str(search_run.id),
            status=search_run.status,
            jobs_received=len(raw_items),
            jobs_saved=jobs_saved,
        )
    except Exception as exc:
        session.rollback()
        search_run.status = SearchRunStatus.FAILED.value
        search_run.completed_at = datetime.now(UTC)
        search_run.error_message = str(exc)
        session.commit()
        return ApifySearchRunResult(
            search_run_id=str(search_run.id),
            status=search_run.status,
            jobs_received=0,
            jobs_saved=0,
            error_message=search_run.error_message,
        )


def _persist_apify_jobs(
    session: Session,
    search_run: SearchRun,
    raw_items: list[dict[str, Any]],
) -> int:
    jobs_saved = 0
    for item in raw_items:
        job = _upsert_job_from_apify_item(session, item)
        if job is None:
            continue
        session.flush()
        link = session.get(SearchRunJob, (search_run.id, job.id))
        if link is None:
            session.add(SearchRunJob(search_run=search_run, job=job))
        jobs_saved += 1
    session.flush()
    return jobs_saved


def _upsert_job_from_apify_item(session: Session, item: dict[str, Any]) -> Job | None:
    url = _extract_job_url(item)
    external_id = _extract_external_id(item, url)
    if not url:
        return None

    job_by_url = session.scalars(select(Job).where(Job.url == url)).first()
    job_by_external_id = (
        session.scalars(select(Job).where(Job.external_id == external_id)).first()
        if external_id
        else None
    )
    job = job_by_url or job_by_external_id

    if job is None:
        job = Job(title=_extract_title(item), url=url)
        session.add(job)

    if job.external_id is None and external_id and (
        job_by_external_id is None or job_by_external_id.id == job.id
    ):
        job.external_id = external_id

    job.title = _extract_title(item)
    job.company = _extract_company(item)
    job.location = _first_text(item, "location", "jobLocation", "formattedLocation")
    job.apply_url = _first_text(item, "applyUrl", "apply_url", "externalApplyUrl")
    job.description = _first_text(item, "description", "jobDescription", "text")
    job.salary = _first_text(item, "salary", "salaryInfo", "compensation")
    job.remote = _extract_remote(item)
    job.posted_at = _extract_posted_at(item)
    job.raw_payload = item
    return job


def _extract_job_url(item: dict[str, Any]) -> str | None:
    url = _first_text(
        item,
        "url",
        "jobUrl",
        "job_url",
        "link",
        "jobLink",
        "linkedinUrl",
    )
    if url:
        return url

    external_id = _first_text(item, "id", "jobId", "job_id", "linkedinJobId")
    if external_id:
        return f"https://www.linkedin.com/jobs/view/{external_id}"
    return None


def _extract_external_id(item: dict[str, Any], url: str | None) -> str | None:
    external_id = _first_text(item, "id", "jobId", "job_id", "linkedinJobId")
    if external_id:
        return external_id
    if not url:
        return None
    match = re.search(r"/jobs/view/(\d+)", url)
    return match.group(1) if match else None


def _extract_title(item: dict[str, Any]) -> str:
    return _first_text(item, "title", "jobTitle", "positionName") or "Untitled LinkedIn job"


def _extract_company(item: dict[str, Any]) -> str | None:
    company = item.get("company")
    if isinstance(company, dict):
        nested = _first_text(company, "name", "companyName")
        if nested:
            return nested
    return _first_text(item, "companyName", "company", "company_name")


def _extract_remote(item: dict[str, Any]) -> bool | None:
    for key in ("remote", "isRemote", "remoteAllowed"):
        value = item.get(key)
        if isinstance(value, bool):
            return value

    workplace = _first_text(item, "workplaceType", "workType", "workplace")
    if workplace:
        normalized = workplace.lower()
        if "remote" in normalized or "à distance" in normalized:
            return True
        if "on-site" in normalized or "onsite" in normalized or "hybrid" in normalized:
            return False
    return None


def _extract_posted_at(item: dict[str, Any]) -> datetime | None:
    value = _first_text(item, "postedAt", "posted_at", "listedAt", "createdAt")
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _first_text(item: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, int | float):
            return str(value)
        if isinstance(value, dict | list) and value:
            return str(value)
    return None
