from __future__ import annotations

from typing import Any, Protocol
from urllib.parse import urlencode

from pydantic import BaseModel, Field

from linkedin_scrapper.config import Settings
from linkedin_scrapper.profile_formatting import format_candidate_skills, format_string_list
from linkedin_scrapper.search_locations import LOCAL_SEARCH_LOCATION, REMOTE_SEARCH_LOCATION


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
    cv_text: str
    target_roles: list[str]
    skills: list[Any]
    locations: list[str]
    remote_preference: list[str]
    languages_spoken: list[str]
    industries_experienced: list[str]
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
        if search.location == LOCAL_SEARCH_LOCATION.label:
            params["geoId"] = LOCAL_SEARCH_LOCATION.geo_id
        elif search.location == REMOTE_SEARCH_LOCATION.label:
            params["geoId"] = REMOTE_SEARCH_LOCATION.geo_id
    if search.remote:
        params["f_TPR"] = ""
        params["f_WT"] = "2"

    return f"https://www.linkedin.com/jobs/search/?{urlencode(params)}"


def _profile_prompt(profile: CandidateSearchProfile, max_titles: int) -> str:
    return "\n".join(
        [
            f"Maximum job titles: {max_titles}",
            "",
            "Candidate markdown context:",
            _truncate(getattr(profile, "cv_text", "") or "none", max_chars=4000),
            "",
            "Structured candidate fields:",
            f"Target roles: {format_string_list(profile.target_roles)}",
            f"Skills: {format_candidate_skills(profile.skills)}",
            f"Locations: {format_string_list(profile.locations)}",
            f"Remote preference: {format_string_list(profile.remote_preference)}",
            f"Languages: {format_string_list(getattr(profile, 'languages_spoken', []))}",
            f"Industries: {format_string_list(getattr(profile, 'industries_experienced', []))}",
            f"Seniority: {getattr(profile, 'seniority', None) or 'none'}",
            f"Exclusions: {format_string_list(getattr(profile, 'exclusions', []))}",
        ]
    )


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()}..."


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
            location=LOCAL_SEARCH_LOCATION.label,
            remote=False,
            rationale=f"{search.rationale} Local Montpellier search.",
        )
        france_remote_search = LinkedInSearchSuggestion(
            title=f"{search.keywords} - France remote",
            keywords=search.keywords,
            location=REMOTE_SEARCH_LOCATION.label,
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
