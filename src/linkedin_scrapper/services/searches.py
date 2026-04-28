from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from linkedin_scrapper.models import CandidateProfile, SearchRun, SearchRunStatus
from linkedin_scrapper.search_planner import LinkedInJobSearch


def load_candidate_profile(session: Session, profile_id: UUID) -> CandidateProfile:
    profile = session.get(CandidateProfile, profile_id)
    if profile is None:
        raise ValueError(f"Candidate profile not found: {profile_id}")
    return profile


def save_search_runs(
    session: Session,
    profile: CandidateProfile,
    searches: list[LinkedInJobSearch],
    actor_name: str,
) -> list[SearchRun]:
    search_runs = [
        SearchRun(
            candidate_profile=profile,
            query=search.keywords,
            linkedin_url=search.linkedin_url,
            actor_name=actor_name,
            status=SearchRunStatus.PENDING.value,
        )
        for search in searches
    ]
    session.add_all(search_runs)
    session.commit()
    for search_run in search_runs:
        session.refresh(search_run)
    return search_runs


def list_search_runs_for_profile(
    session: Session,
    profile: CandidateProfile,
) -> list[SearchRun]:
    return list(
        session.scalars(
            select(SearchRun)
            .where(SearchRun.profile_id == profile.id)
            .order_by(SearchRun.created_at)
        )
    )
