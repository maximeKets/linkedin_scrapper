from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, Field
from pypdf import PdfReader


class ParsedCandidateProfile(BaseModel):
    cv_text: str
    target_roles: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    remote_preference: str | None = None
    seniority: str | None = None
    exclusions: list[str] = Field(default_factory=list)
    profile_payload: dict = Field(default_factory=dict)


ROLE_PATTERNS = {
    "AI Engineer": ("ai engineer", "artificial intelligence engineer"),
    "Machine Learning Engineer": ("machine learning engineer", "ml engineer"),
    "Data Engineer": ("data engineer", "data engineering"),
    "Backend Engineer": ("backend engineer", "back-end engineer", "backend developer"),
    "Python Developer": ("python developer", "python engineer"),
    "Full Stack Developer": ("full stack", "full-stack"),
    "Software Engineer": ("software engineer", "software developer"),
}

SKILL_KEYWORDS = [
    "Python",
    "TypeScript",
    "JavaScript",
    "SQL",
    "PostgreSQL",
    "FastAPI",
    "Django",
    "React",
    "Astro",
    "Docker",
    "Kubernetes",
    "AWS",
    "GCP",
    "Azure",
    "LangChain",
    "LangGraph",
    "OpenAI",
    "Machine Learning",
    "Deep Learning",
    "NLP",
    "ETL",
    "Airflow",
    "dbt",
]

LOCATION_PATTERNS = [
    "Paris",
    "France",
    "Lyon",
    "Bordeaux",
    "Remote",
    "Europe",
    "London",
    "Berlin",
]


def parse_cv(path: Path) -> ParsedCandidateProfile:
    cv_text = extract_cv_text(path)
    return parse_cv_text(cv_text)


def extract_cv_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"CV file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8")
    elif suffix == ".pdf":
        text = _extract_pdf_text(path)
    else:
        raise ValueError("Unsupported CV format. Use .txt, .md, or .pdf.")

    normalized = _normalize_text(text)
    if not normalized:
        raise ValueError(f"CV file does not contain extractable text: {path}")
    return normalized


def parse_cv_text(cv_text: str) -> ParsedCandidateProfile:
    normalized = _normalize_text(cv_text)
    lower = normalized.lower()

    target_roles = _extract_target_roles(lower)
    skills = _extract_known_values(normalized, SKILL_KEYWORDS)
    locations = _extract_known_values(normalized, LOCATION_PATTERNS)
    remote_preference = _extract_remote_preference(lower)
    seniority = _extract_seniority(lower)
    exclusions = _extract_exclusions(normalized)

    return ParsedCandidateProfile(
        cv_text=normalized,
        target_roles=target_roles,
        skills=skills,
        locations=locations,
        remote_preference=remote_preference,
        seniority=seniority,
        exclusions=exclusions,
        profile_payload={
            "parser": "deterministic-v1",
            "text_length": len(normalized),
        },
    )


def _extract_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _normalize_text(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text.replace("\r\n", "\n")).strip()


def _extract_target_roles(lower_text: str) -> list[str]:
    roles = [
        role
        for role, patterns in ROLE_PATTERNS.items()
        if any(pattern in lower_text for pattern in patterns)
    ]
    return roles or ["Software Engineer"]


def _extract_known_values(text: str, values: list[str]) -> list[str]:
    found = []
    for value in values:
        if re.search(rf"(?<!\w){re.escape(value)}(?!\w)", text, flags=re.IGNORECASE):
            found.append(value)
    return found


def _extract_remote_preference(lower_text: str) -> str | None:
    if "remote" in lower_text or "télétravail" in lower_text or "teletravail" in lower_text:
        return "remote"
    if "hybrid" in lower_text or "hybride" in lower_text:
        return "hybrid"
    if "on-site" in lower_text or "onsite" in lower_text or "présentiel" in lower_text:
        return "onsite"
    return None


def _extract_seniority(lower_text: str) -> str | None:
    if "principal" in lower_text or "staff" in lower_text:
        return "staff"
    if "senior" in lower_text or "lead" in lower_text:
        return "senior"
    if "junior" in lower_text or "entry level" in lower_text:
        return "junior"
    if re.search(r"\b[4-9]\+?\s+(years|ans)\b", lower_text):
        return "senior"
    if re.search(r"\b[1-3]\+?\s+(years|ans)\b", lower_text):
        return "mid"
    return None


def _extract_exclusions(text: str) -> list[str]:
    exclusions = []
    for line in text.splitlines():
        lower = line.lower()
        if any(marker in lower for marker in ("not interested", "exclude", "avoid", "pas intéressé")):
            exclusions.append(line.strip())
    return exclusions
