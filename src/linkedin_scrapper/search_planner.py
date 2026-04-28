from __future__ import annotations

from typing import Any, Protocol
from urllib.parse import urlencode

from pydantic import BaseModel, Field

from linkedin_scrapper.config import Settings


MONTPELLIER_LOCATION = "Montpellier, Occitanie, France"
MONTPELLIER_GEO_ID = "106719766"
FRANCE_LOCATION = "France"
FRANCE_GEO_ID = "105015875"


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

Return simple job-title searches that can be expanded into public LinkedIn Jobs
URLs. The goal is broad public LinkedIn recall with controlled scraping cost, not
an exhaustive list of the candidate's skills.

Rules:
- Return job titles only. The application will add locations and remote filters.
- Return up to the requested maximum number of job titles.
- Prefer fewer, stronger searches over many narrow variants.
- Each search must target a distinct hiring angle, for example AI/LLM, Python
  backend, full-stack, Django/React, or automation. Do not return near-duplicates
  such as "AI Engineer" and "AI Developer" unless the keywords cover clearly
  different job markets.
- Keywords must be concise LinkedIn job titles, usually 2 to 4 words total.
  Avoid long keyword lists.
- Do not stuff every matching skill into keywords. Put supporting technologies in
  the rationale instead.
- Do not combine a title with multiple secondary skills such as
  "AI Engineer Python LLMs". Prefer "AI Engineer" or "Développeur IA".
- Avoid niche tools as primary keywords unless they are central to a target role.
  Examples of niche terms to avoid in broad searches: Pinecone, Wagtail,
  Speech-to-Text, specific vector database vendors.
- Do not mix unrelated stacks in one query unless the target role genuinely
  requires them. Prefer "Django React Developer" over a long Django/Laravel/Wagtail
  combined query.
- Use market terms candidates and recruiters actually use on LinkedIn. For French
  profiles, include French role labels only when they are likely to improve recall.
- Avoid searches that match explicit exclusions.
- The rationale should explain the search angle and what it covers compared with
  the other returned searches.
""".strip()


def build_search_planner_agent(chat_model: Any) -> SearchPlannerAgent:
    return chat_model.with_structured_output(LinkedInSearchPlan)


def generate_linkedin_searches(
    profile: CandidateSearchProfile,
    agent: SearchPlannerAgent,
    settings: Settings,
) -> list[LinkedInJobSearch]:
    max_titles = settings.max_search_titles
    plan = _coerce_plan(
        agent.invoke(
            [
                ("system", SEARCH_PLANNER_SYSTEM_PROMPT),
                ("human", _profile_prompt(profile, max_titles)),
            ]
        )
    )

    titles = _dedupe_search_titles(plan.searches)[:max_titles]
    return _build_location_matrix_searches(titles)


def build_linkedin_jobs_url(search: LinkedInSearchSuggestion) -> str:
    params = {
        "keywords": search.keywords,
        "position": "1",
        "pageNum": "0",
    }
    if search.location:
        params["location"] = search.location
        if search.location == MONTPELLIER_LOCATION:
            params["geoId"] = MONTPELLIER_GEO_ID
        elif search.location == FRANCE_LOCATION:
            params["geoId"] = FRANCE_GEO_ID
    if search.remote:
        params["f_TPR"] = ""
        params["f_WT"] = "2"

    return f"https://www.linkedin.com/jobs/search/?{urlencode(params)}"


def _profile_prompt(profile: CandidateSearchProfile, max_titles: int) -> str:
    return "\n".join(
        [
            f"Maximum job titles: {max_titles}",
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


def _build_location_matrix_searches(
    searches: list[LinkedInSearchSuggestion],
) -> list[LinkedInJobSearch]:
    output: list[LinkedInJobSearch] = []
    for search in searches:
        local_search = LinkedInSearchSuggestion(
            title=f"{search.keywords} - Montpellier",
            keywords=search.keywords,
            location=MONTPELLIER_LOCATION,
            remote=False,
            rationale=f"{search.rationale} Local Montpellier search.",
        )
        france_remote_search = LinkedInSearchSuggestion(
            title=f"{search.keywords} - France remote",
            keywords=search.keywords,
            location=FRANCE_LOCATION,
            remote=True,
            rationale=f"{search.rationale} France remote search.",
        )
        for expanded in (local_search, france_remote_search):
            output.append(
                LinkedInJobSearch(
                    title=expanded.title,
                    keywords=expanded.keywords,
                    location=expanded.location,
                    remote=expanded.remote,
                    rationale=expanded.rationale,
                    linkedin_url=build_linkedin_jobs_url(expanded),
                )
            )
    return output


def _dedupe_search_titles(
    searches: list[LinkedInSearchSuggestion],
) -> list[LinkedInSearchSuggestion]:
    seen: set[str] = set()
    deduped = []
    for search in searches:
        key = search.keywords.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(
            LinkedInSearchSuggestion(
                title=search.title,
                keywords=search.keywords.strip(),
                location=None,
                remote=None,
                rationale=search.rationale,
            )
        )
    return deduped
