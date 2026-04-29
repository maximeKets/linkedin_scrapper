from typing import Any

from linkedin_scrapper.job_scorer import (
    JOB_SCORER_SYSTEM_PROMPT,
    JobScoreExtraction,
    score_job,
)
from linkedin_scrapper.models import CandidateProfile, Job


class StubJobScorerAgent:
    def __init__(self, extraction: JobScoreExtraction | None = None) -> None:
        self.extraction = extraction or JobScoreExtraction(
            score=84,
            summary="Strong AI/full-stack match with some seniority risk.",
            match_reasons=["AI platform focus", "Full-stack engineering alignment"],
            missing_skills=["Large-scale ML platform experience"],
            risk_flags=["Seniority may be high"],
        )
        self.inputs: list[Any] = []

    def invoke(self, input: Any) -> JobScoreExtraction:
        self.inputs.append(input)
        return self.extraction


def test_score_job_uses_structured_agent_and_profile_context() -> None:
    profile = CandidateProfile(
        cv_text="# Maxime Kets\n\n**Titre cible** : AI Engineer",
        target_roles=["AI Engineer"],
        total_years_of_experience=6,
        skills=[
            {
                "name": "Python",
                "years_of_experience": 6,
                "context": "PRODUCTION",
                "last_used_year": 2026,
            },
            {
                "name": "FastAPI",
                "years_of_experience": 3,
                "context": "PRODUCTION",
                "last_used_year": 2026,
            },
            {
                "name": "RAG",
                "years_of_experience": 2,
                "context": "PERSONAL",
                "last_used_year": 2026,
            },
        ],
        locations=["Montpellier"],
        remote_preference=["FULL_REMOTE"],
        languages_spoken=["FR", "EN"],
        industries_experienced=["HR tech"],
        seniority="mid",
        exclusions=["PHP roles"],
    )
    job = Job(
        title="AI Engineer",
        company="Example",
        location="France",
        remote=True,
        description="Build AI agents with Python and LLMs.",
        url="https://www.linkedin.com/jobs/view/123",
    )
    agent = StubJobScorerAgent()

    score = score_job(profile, job, agent)

    assert score.score == 84
    assert score.missing_skills == ["Large-scale ML platform experience"]
    assert score.risk_flags == ["Seniority may be high"]
    assert "Scoring guidance" in agent.inputs[0][0][1]
    assert "# Maxime Kets" in agent.inputs[0][1][1]
    assert "Target roles: AI Engineer" in agent.inputs[0][1][1]
    assert "Python (6y, PRODUCTION, last used 2026)" in agent.inputs[0][1][1]
    assert "Title: AI Engineer" in agent.inputs[0][1][1]


def test_job_scorer_prompt_defines_missing_skills_and_risk_flags() -> None:
    assert "missing_skills" in JOB_SCORER_SYSTEM_PROMPT
    assert "important concrete skills" in JOB_SCORER_SYSTEM_PROMPT
    assert "risk_flags" in JOB_SCORER_SYSTEM_PROMPT
    assert "non-skill or contextual reservations" in JOB_SCORER_SYSTEM_PROMPT
