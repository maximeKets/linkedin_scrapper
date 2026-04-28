from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field

from linkedin_scrapper.models import CandidateProfile, Job


class JobScoreExtraction(BaseModel):
    score: int = Field(ge=0, le=100)
    summary: str
    match_reasons: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)


class JobScorerAgent(Protocol):
    def invoke(self, input: Any) -> JobScoreExtraction | dict[str, Any]:
        pass


JOB_SCORER_SYSTEM_PROMPT = """
You score how well a job matches a candidate profile for job-search automation.

Return only evidence-backed scoring fields. Compare the candidate's target roles,
skills, seniority, location, remote preference, and exclusions against the job
title, description, company, location, and remote signal.

Scoring guidance:
- 80-100: strong match; role and core requirements align well.
- 60-79: plausible match with meaningful reservations.
- 40-59: partial or weak match.
- 0-39: poor match or clear mismatch.

Field guidance:
- summary: one concise sentence explaining the fit.
- match_reasons: concrete reasons the job fits the profile.
- missing_skills: important concrete skills or experience requested by the job but
  absent or weak in the candidate profile.
- risk_flags: non-skill or contextual reservations, such as seniority mismatch,
  mostly off-target role, incompatible location, onsite requirement, or language
  requirements.

Do not penalize minor wording differences or adjacent frameworks too heavily.
Do penalize explicit exclusions and clear seniority or location mismatches.
""".strip()


def build_job_scorer_agent(chat_model: Any) -> JobScorerAgent:
    return chat_model.with_structured_output(JobScoreExtraction)


def score_job(
    profile: CandidateProfile,
    job: Job,
    agent: JobScorerAgent,
) -> JobScoreExtraction:
    extraction = agent.invoke(
        [
            ("system", JOB_SCORER_SYSTEM_PROMPT),
            ("human", _score_prompt(profile, job)),
        ]
    )
    return _coerce_score(extraction)


def _score_prompt(profile: CandidateProfile, job: Job) -> str:
    return "\n".join(
        [
            "Candidate profile:",
            f"- Target roles: {', '.join(profile.target_roles) or 'none'}",
            f"- Skills: {', '.join(profile.skills) or 'none'}",
            f"- Locations: {', '.join(profile.locations) or 'none'}",
            f"- Remote preference: {profile.remote_preference or 'none'}",
            f"- Seniority: {profile.seniority or 'none'}",
            f"- Exclusions: {', '.join(profile.exclusions) or 'none'}",
            "",
            "Job:",
            f"- Title: {job.title}",
            f"- Company: {job.company or 'none'}",
            f"- Location: {job.location or 'none'}",
            f"- Remote: {job.remote if job.remote is not None else 'unknown'}",
            f"- Salary: {job.salary or 'none'}",
            f"- Description: {_truncate(job.description or 'none')}",
        ]
    )


def _truncate(text: str, max_chars: int = 6000) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()}..."


def _coerce_score(extraction: JobScoreExtraction | dict[str, Any]) -> JobScoreExtraction:
    if isinstance(extraction, JobScoreExtraction):
        return extraction
    return JobScoreExtraction.model_validate(extraction)
