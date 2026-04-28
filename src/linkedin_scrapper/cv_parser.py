from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field
from pypdf import PdfReader


class CandidateProfileExtraction(BaseModel):
    target_roles: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    remote_preference: str | None = None
    seniority: str | None = None
    exclusions: list[str] = Field(default_factory=list)
    extraction_notes: list[str] = Field(default_factory=list)


class ParsedCandidateProfile(CandidateProfileExtraction):
    cv_text: str
    profile_payload: dict[str, Any] = Field(default_factory=dict)


class CVParserAgent(Protocol):
    def invoke(self, input: Any) -> CandidateProfileExtraction | dict[str, Any]:
        pass


CV_PARSER_SYSTEM_PROMPT = """
You extract a structured candidate profile from a CV for job-search automation.

Return only fields that are supported by evidence in the CV. Use empty lists or null
when the CV does not provide enough information.

Field guidance:
- target_roles: normalized job titles the candidate is suited for.
- skills: concrete tools, languages, frameworks, platforms, and methods.
- locations: only current residence/base location and explicit target job-search
  locations. Do not include historical work, education, client, employer, travel,
  or project locations unless the CV explicitly states the candidate wants jobs
  there. For LinkedIn exports, prefer the profile header/contact location over
  experience locations.
- remote_preference: one of remote, hybrid, onsite, or null.
- seniority: junior, mid, senior, staff, principal, lead, or null.
- exclusions: explicit constraints, avoidances, or non-target roles.
- extraction_notes: concise evidence notes useful for debugging extraction decisions.

When location evidence conflicts, explain the decision in extraction_notes and keep
locations focused on job-search geography, not the candidate's full work history.
""".strip()


def build_cv_parser_agent(chat_model: Any) -> CVParserAgent:
    return chat_model.with_structured_output(CandidateProfileExtraction)


def parse_cv(path: Path, agent: CVParserAgent) -> ParsedCandidateProfile:
    cv_text = extract_cv_text(path)
    return parse_cv_text(cv_text, agent)


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


def parse_cv_text(cv_text: str, agent: CVParserAgent) -> ParsedCandidateProfile:
    normalized = _normalize_text(cv_text)
    extraction = _coerce_extraction(
        agent.invoke(
            [
                ("system", CV_PARSER_SYSTEM_PROMPT),
                ("human", f"CV text:\n\n{normalized}"),
            ]
        )
    )

    payload = {
        "parser": "llm-agent-v1",
        "text_length": len(normalized),
        "extraction_notes": extraction.extraction_notes,
    }

    return ParsedCandidateProfile(
        cv_text=normalized,
        target_roles=extraction.target_roles,
        skills=extraction.skills,
        locations=extraction.locations,
        remote_preference=extraction.remote_preference,
        seniority=extraction.seniority,
        exclusions=extraction.exclusions,
        profile_payload=payload,
    )


def _extract_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _normalize_text(text: str) -> str:
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    normalized_lines: list[str] = []
    previous_blank = False

    for line in lines:
        is_blank = not line.strip()
        if is_blank and previous_blank:
            continue
        normalized_lines.append("" if is_blank else line)
        previous_blank = is_blank

    return "\n".join(normalized_lines).strip()


def _coerce_extraction(
    extraction: CandidateProfileExtraction | dict[str, Any],
) -> CandidateProfileExtraction:
    if isinstance(extraction, CandidateProfileExtraction):
        return extraction
    return CandidateProfileExtraction.model_validate(extraction)
