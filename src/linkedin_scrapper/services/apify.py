from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from apify_client import ApifyClient
from sqlalchemy.orm import Session

from linkedin_scrapper.config import Settings
from linkedin_scrapper.models import SearchRun, SearchRunStatus
from linkedin_scrapper.services.jobs import persist_apify_jobs


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
        jobs_saved = persist_apify_jobs(session, search_run, raw_items)

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
