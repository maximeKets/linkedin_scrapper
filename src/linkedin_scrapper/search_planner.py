from __future__ import annotations

from typing import Any, Protocol
from urllib.parse import urlencode

from pydantic import BaseModel, Field

from linkedin_scrapper.config import Settings


class LinkedInSearchSuggestion(BaseModel):
    title: str = Field(description="Short internal title for the search.")
    keywords: str = Field(description="LinkedIn Jobs keywords query.")
    location: str | None = Field(default=None, description="LinkedIn location query.")
    remote: bool | None = Field(default=None, description="Whether to force remote jobs.")
    rationale: str = Field(description="Why this search is relevant to the profile.")


class LinkedInSearchPlan(BaseModel):
    searches: list[LinkedInSearchSuggestion] = Field(default_factory=list)


class LinkedInJobSearch(LinkedInSearchSuggestion):
    linkedin_url: str


class SearchPlannerAgent(Protocol):
    def invoke(self, input: Any) -> LinkedInSearchPlan | dict[str, Any]:
        pass


class CandidateSearchProfile(Protocol):
    target_roles: list[str]
    skills: list[str]
    locations: list[str]
    remote_preference: str | None
    seniority: str | None
    exclusions: list[str]


SEARCH_PLANNER_SYSTEM_PROMPT = """
You generate LinkedIn Jobs search queries from a structured candidate profile.

Return focused searches that can be passed to the public LinkedIn Jobs search page.
Create enough variety to cover adjacent roles and keyword variants while keeping the
set small enough to control scraping cost.

Rules:
- Return up to the requested maximum number of searches.
- Prefer concise keyword strings that work well in LinkedIn Jobs.
- Use explicit locations from the profile when available.
- If the profile indicates remote preference, include remote-focused searches.
- Avoid searches that match explicit exclusions.
""".strip()


def build_search_planner_agent(chat_model: Any) -> SearchPlannerAgent:
    return chat_model.with_structured_output(LinkedInSearchPlan)


def generate_linkedin_searches(
    profile: CandidateSearchProfile,
    agent: SearchPlannerAgent,
    settings: Settings,
) -> list[LinkedInJobSearch]:
    max_searches = settings.max_search_queries
    plan = _coerce_plan(
        agent.invoke(
            [
                ("system", SEARCH_PLANNER_SYSTEM_PROMPT),
                ("human", _profile_prompt(profile, max_searches)),
            ]
        )
    )

    searches = _dedupe_searches(plan.searches)[:max_searches]
    return [
        LinkedInJobSearch(
            title=search.title,
            keywords=search.keywords,
            location=search.location,
            remote=search.remote,
            rationale=search.rationale,
            linkedin_url=build_linkedin_jobs_url(search),
        )
        for search in searches
    ]


def build_linkedin_jobs_url(search: LinkedInSearchSuggestion) -> str:
    params = {
        "keywords": search.keywords,
        "position": "1",
        "pageNum": "0",
    }
    if search.location:
        params["location"] = search.location
    if search.remote:
        params["f_WT"] = "2"

    return f"https://www.linkedin.com/jobs/search/?{urlencode(params)}"


def _profile_prompt(profile: CandidateSearchProfile, max_searches: int) -> str:
    return "\n".join(
        [
            f"Maximum searches: {max_searches}",
            f"Target roles: {', '.join(profile.target_roles) or 'none'}",
            f"Skills: {', '.join(profile.skills) or 'none'}",
            f"Locations: {', '.join(profile.locations) or 'none'}",
            f"Remote preference: {profile.remote_preference or 'none'}",
            f"Seniority: {profile.seniority or 'none'}",
            f"Exclusions: {', '.join(profile.exclusions) or 'none'}",
        ]
    )


def _coerce_plan(plan: LinkedInSearchPlan | dict[str, Any]) -> LinkedInSearchPlan:
    if isinstance(plan, LinkedInSearchPlan):
        return plan
    return LinkedInSearchPlan.model_validate(plan)


def _dedupe_searches(
    searches: list[LinkedInSearchSuggestion],
) -> list[LinkedInSearchSuggestion]:
    seen: set[tuple[str, str, bool | None]] = set()
    deduped = []
    for search in searches:
        key = (
            search.keywords.strip().lower(),
            (search.location or "").strip().lower(),
            search.remote,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(search)
    return deduped
