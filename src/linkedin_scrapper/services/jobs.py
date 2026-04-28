from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from linkedin_scrapper.models import Job, SearchRun, SearchRunJob


def persist_apify_jobs(
    session: Session,
    search_run: SearchRun,
    raw_items: list[dict[str, Any]],
) -> int:
    jobs_saved = 0
    for item in raw_items:
        normalized = normalize_apify_job(item)
        if normalized is None:
            continue

        job = upsert_job(session, normalized)
        session.flush()

        link = session.get(SearchRunJob, (search_run.id, job.id))
        if link is None:
            session.add(SearchRunJob(search_run=search_run, job=job))
        jobs_saved += 1

    session.flush()
    return jobs_saved


def normalize_apify_job(item: dict[str, Any]) -> dict[str, Any] | None:
    url = _extract_job_url(item)
    external_id = _extract_external_id(item, url)
    apply_url = _first_text(item, "applyUrl", "apply_url", "externalApplyUrl")

    if not any([external_id, url, apply_url]):
        return None
    if not url and external_id:
        url = f"https://www.linkedin.com/jobs/view/{external_id}"

    return {
        "external_id": external_id,
        "title": _extract_title(item),
        "company": _extract_company(item),
        "location": _first_text(item, "location", "jobLocation", "formattedLocation"),
        "url": url,
        "apply_url": apply_url,
        "description": _first_text(
            item,
            "descriptionText",
            "description",
            "jobDescription",
            "text",
            "descriptionHtml",
        ),
        "salary": _extract_salary(item),
        "remote": _extract_remote(item),
        "posted_at": _extract_posted_at(item),
        "raw_payload": item,
    }


def upsert_job(session: Session, normalized: dict[str, Any]) -> Job:
    job = _find_existing_job(session, normalized)
    if job is None:
        job = Job(
            title=normalized["title"],
            url=normalized["url"] or normalized["apply_url"],
        )
        session.add(job)

    if job.external_id is None:
        job.external_id = normalized["external_id"]
    _update_url_if_safe(session, job, normalized["url"])

    job.title = normalized["title"]
    job.company = normalized["company"]
    job.location = normalized["location"]
    job.apply_url = normalized["apply_url"]
    job.description = normalized["description"]
    job.salary = normalized["salary"]
    job.remote = normalized["remote"]
    job.posted_at = normalized["posted_at"]
    job.raw_payload = normalized["raw_payload"]
    return job


def _update_url_if_safe(session: Session, job: Job, url: str | None) -> None:
    if not url or job.url == url:
        return

    existing = session.scalars(select(Job).where(Job.url == url)).first()
    if existing is None or existing.id == job.id:
        job.url = url


def _find_existing_job(session: Session, normalized: dict[str, Any]) -> Job | None:
    external_id = normalized["external_id"]
    if external_id:
        job = session.scalars(select(Job).where(Job.external_id == external_id)).first()
        if job is not None:
            return job

    url = normalized["url"]
    if url:
        job = session.scalars(select(Job).where(Job.url == url)).first()
        if job is not None:
            return job

    apply_url = normalized["apply_url"]
    if apply_url:
        return session.scalars(select(Job).where(Job.apply_url == apply_url)).first()

    return None


def _extract_job_url(item: dict[str, Any]) -> str | None:
    return _first_text(
        item,
        "link",
        "url",
        "jobUrl",
        "job_url",
        "jobLink",
        "linkedinUrl",
    )


def _extract_external_id(item: dict[str, Any], url: str | None) -> str | None:
    external_id = _first_text(item, "id", "jobId", "job_id", "linkedinJobId")
    if external_id:
        return external_id
    if not url:
        return None
    match = re.search(r"/jobs/view/([^/?]+)", url)
    return match.group(1) if match else None


def _extract_title(item: dict[str, Any]) -> str:
    return _first_text(item, "title", "standardizedTitle", "jobTitle", "positionName") or (
        "Untitled LinkedIn job"
    )


def _extract_company(item: dict[str, Any]) -> str | None:
    company = item.get("company")
    if isinstance(company, dict):
        nested = _first_text(company, "name", "companyName")
        if nested:
            return nested
    return _first_text(item, "companyName", "company", "company_name")


def _extract_salary(item: dict[str, Any]) -> str | None:
    salary = _first_text(item, "salary", "salaryInfo", "compensation")
    if salary:
        return salary
    salary_insights = item.get("salaryInsights")
    if isinstance(salary_insights, dict) and salary_insights:
        return str(salary_insights)
    return None


def _extract_remote(item: dict[str, Any]) -> bool | None:
    for key in ("workRemoteAllowed", "remote", "isRemote", "remoteAllowed"):
        value = item.get(key)
        if isinstance(value, bool):
            return value

    workplace_types = item.get("workplaceTypes")
    if isinstance(workplace_types, list):
        normalized_types = {str(value).lower() for value in workplace_types}
        if "remote" in normalized_types:
            return True
        if normalized_types.intersection({"hybrid", "on-site", "onsite"}):
            return False

    workplace = _first_text(item, "workplaceType", "workType", "workplace")
    if workplace:
        normalized = workplace.lower()
        if "remote" in normalized or "à distance" in normalized:
            return True
        if "on-site" in normalized or "onsite" in normalized or "hybrid" in normalized:
            return False
    return None


def _extract_posted_at(item: dict[str, Any]) -> datetime | None:
    timestamp = item.get("postedAtTimestamp")
    if isinstance(timestamp, int | float):
        return datetime.fromtimestamp(timestamp / 1000, tz=UTC)

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
    return None
