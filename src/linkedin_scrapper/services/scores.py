from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from linkedin_scrapper.job_scorer import JobScorerAgent, JobScoreExtraction, score_job
from linkedin_scrapper.models import CandidateProfile, Job, JobScore, SearchRun, SearchRunJob


@dataclass(frozen=True)
class JobScoringResult:
    job_id: str
    title: str
    skipped: bool
    score: int | None = None
    summary: str | None = None


def list_jobs_for_scoring(
    session: Session,
    profile: CandidateProfile,
    limit: int | None = None,
    include_scored: bool = False,
) -> list[Job]:
    statement = (
        select(Job)
        .join(SearchRunJob, SearchRunJob.job_id == Job.id)
        .join(SearchRun, SearchRun.id == SearchRunJob.search_run_id)
        .where(SearchRun.profile_id == profile.id)
        .order_by(Job.created_at)
        .distinct()
    )
    if not include_scored:
        statement = statement.outerjoin(JobScore, JobScore.job_id == Job.id).where(
            JobScore.id.is_(None)
        )
    if limit is not None:
        statement = statement.limit(limit)
    return list(session.scalars(statement))


def score_jobs_for_profile(
    session: Session,
    profile: CandidateProfile,
    jobs: list[Job],
    agent: JobScorerAgent,
    model_name: str,
    rescore: bool = False,
) -> list[JobScoringResult]:
    results = []
    for job in jobs:
        existing_score = job.score
        if existing_score is not None and not rescore:
            results.append(
                JobScoringResult(
                    job_id=str(job.id),
                    title=job.title,
                    skipped=True,
                    score=existing_score.score,
                    summary=existing_score.summary,
                )
            )
            continue

        extraction = score_job(profile, job, agent)
        saved_score = save_job_score(
            session=session,
            profile_id=profile.id,
            job=job,
            extraction=extraction,
            model_name=model_name,
            existing_score=existing_score,
        )
        results.append(
            JobScoringResult(
                job_id=str(job.id),
                title=job.title,
                skipped=False,
                score=saved_score.score,
                summary=saved_score.summary,
            )
        )

    session.commit()
    return results


def save_job_score(
    session: Session,
    profile_id: UUID,
    job: Job,
    extraction: JobScoreExtraction,
    model_name: str,
    existing_score: JobScore | None = None,
) -> JobScore:
    job_score = existing_score or JobScore(job=job)
    job_score.score = extraction.score
    job_score.summary = extraction.summary
    job_score.match_reasons = extraction.match_reasons
    job_score.missing_skills = extraction.missing_skills
    job_score.risk_flags = extraction.risk_flags
    job_score.scoring_payload = {
        "scorer": "llm-job-scorer-v1",
        "profile_id": str(profile_id),
        "job_id": str(job.id),
        "model": model_name,
    }
    session.add(job_score)
    session.flush()
    return job_score
