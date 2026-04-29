from __future__ import annotations

from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field
from pypdf import PdfReader


class CVProfileModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RemotePreference(StrEnum):
    FULL_REMOTE = "FULL_REMOTE"
    HYBRID = "HYBRID"
    ONSITE = "ONSITE"


class Language(StrEnum):
    FR = "FR"
    EN = "EN"


class SkillContext(StrEnum):
    PRODUCTION = "PRODUCTION"
    ACADEMIC = "ACADEMIC"
    PERSONAL = "PERSONAL"


class SkillName(StrEnum):
    PYTHON = "Python"
    JAVASCRIPT = "JavaScript"
    TYPESCRIPT = "TypeScript"
    REACT = "React"
    FASTAPI = "FastAPI"
    DJANGO = "Django"
    POSTGRESQL = "PostgreSQL"
    DOCKER = "Docker"
    KUBERNETES = "Kubernetes"
    LANGCHAIN = "LangChain"
    LANGGRAPH = "LangGraph"
    OPENAI = "OpenAI"
    LLM = "LLM"
    RAG = "RAG"
    SQLALCHEMY = "SQLAlchemy"
    PYDANTIC = "Pydantic"
    TYPER = "Typer"
    APIFY = "Apify"


class CandidateSkill(CVProfileModel):
    name: SkillName
    years_of_experience: int = Field(ge=0)
    context: SkillContext
    last_used_year: int = Field(ge=1900)


class ProfileSnapshot(CVProfileModel):
    dna: str | None = None
    current_focus: str | None = None
    strengths: list[str] = Field(default_factory=list)


class ConsolidatedStack(CVProfileModel):
    core_backend_ai: list[str] = Field(default_factory=list)
    core_frontend: list[str] = Field(default_factory=list)
    infra_tools: list[str] = Field(default_factory=list)
    business_domains: list[str] = Field(default_factory=list)


class KeyExperience(CVProfileModel):
    title: str
    dates: str | None = None
    mission: str | None = None
    achievements: list[str] = Field(default_factory=list)
    stack: list[str] = Field(default_factory=list)


class EducationItem(CVProfileModel):
    degree: str
    school: str | None = None
    years: str | None = None


class CandidateMarkdownProfile(CVProfileModel):
    profile_snapshot: ProfileSnapshot = Field(default_factory=ProfileSnapshot)
    consolidated_stack: ConsolidatedStack = Field(default_factory=ConsolidatedStack)
    key_experiences: list[KeyExperience] = Field(default_factory=list)
    education: list[EducationItem] = Field(default_factory=list)
    markdown: str = Field(
        min_length=1,
        description="Synthetic markdown profile generated from the CV and markdown subobjects.",
    )


class CandidateProfileExtraction(CVProfileModel):
    full_name: str | None = None
    target_roles: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    total_years_of_experience: int = Field(default=0, ge=0)
    remote_preference: list[RemotePreference] = Field(default_factory=list)
    languages_spoken: list[Language] = Field(default_factory=list)
    industries_experienced: list[str] = Field(default_factory=list)
    skills: list[CandidateSkill] = Field(default_factory=list)
    markdown_profile: CandidateMarkdownProfile
    extraction_notes: list[str] = Field(default_factory=list)


class ParsedCandidateProfile(CandidateProfileExtraction):
    cv_text: str
    seniority: str | None = None
    exclusions: list[str] = Field(default_factory=list)
    profile_payload: dict[str, Any] = Field(default_factory=dict)


class CVParserAgent(Protocol):
    def invoke(self, input: Any) -> CandidateProfileExtraction | dict[str, Any]:
        pass


CV_PARSER_SYSTEM_PROMPT = """
You extract a structured candidate profile from a CV for job-search automation.

Return only fields that are supported by evidence in the CV. Do not infer visa status.
Use empty lists, empty nested objects, zero for total_years_of_experience, or
null only for nullable string fields when the CV does not provide enough
information.

Field guidance:
- full_name: candidate first and last name when present.
- target_roles: normalized target job titles for the candidate.
- locations: current residence/base location and explicit target job-search
  locations. This field is also rendered in the markdown context.
- total_years_of_experience: global professional development experience.
- remote_preference: use only FULL_REMOTE, HYBRID, or ONSITE.
- languages_spoken: use only FR and EN.
- industries_experienced: business sectors where the CV shows real experience.
- skills: controlled technical skills only. Allowed skill names are:
  Python, JavaScript, TypeScript, React, FastAPI, Django, PostgreSQL, Docker,
  Kubernetes, LangChain, LangGraph, OpenAI, LLM, RAG, SQLAlchemy, Pydantic,
  Typer, Apify.
  For each skill, include years_of_experience, context, and last_used_year.
  context must be PRODUCTION, ACADEMIC, or PERSONAL. Skip skills that are not
  in the allowed list; do not invent new labels.
- markdown_profile: generate the markdown context in the markdown field and fill
  its subobjects at the same time. The markdown must follow this exact structure:
  # prénom Nom
  **Titre cible** :
  **Localisation** : ville, region, pays (Ouvert Remote : [Oui/Non])
  **Langues** :
  **Expérience globale** :
  ## 🎯 Snapshot Profil (Persona)
  * **ADN** :
  * **Focus actuel** :
  * **Points forts** :
  ## 🛠 Stack Technique Consolidée
  * **Core Backend & IA** :
  * **Core Frontend** :
  * **Infra & Outils** :
  * **Domaines métiers** :
  ## 💼 Expériences Clés Synthétisées
  ### titre | dates
  * **Mission** :
  * **Réalisations** :
      *
  * **Stack** :
  ## 🎓 Formation
  * diplome - ecole (années)
  Do not include a redundant skills table in markdown.
- extraction_notes: concise evidence notes useful for debugging extraction decisions.

When evidence conflicts, explain the decision in extraction_notes and keep the
structured fields conservative.
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
    markdown_context = _normalize_text(extraction.markdown_profile.markdown)
    if not markdown_context:
        raise ValueError("CV parser did not return markdown context.")

    payload = {
        "parser": "llm-cv-profile-v2",
        "text_length": len(normalized),
        "raw_text_length": len(normalized),
        "raw_text_sha256": sha256(normalized.encode("utf-8")).hexdigest(),
        "extraction_notes": extraction.extraction_notes,
        "narrative": {
            "full_name": extraction.full_name,
            "profile_snapshot": extraction.markdown_profile.profile_snapshot.model_dump(
                mode="json"
            ),
            "consolidated_stack": (
                extraction.markdown_profile.consolidated_stack.model_dump(mode="json")
            ),
            "key_experiences": [
                experience.model_dump(mode="json")
                for experience in extraction.markdown_profile.key_experiences
            ],
            "education": [
                item.model_dump(mode="json")
                for item in extraction.markdown_profile.education
            ],
        },
    }

    return ParsedCandidateProfile(
        cv_text=markdown_context,
        full_name=extraction.full_name,
        target_roles=extraction.target_roles,
        locations=extraction.locations,
        total_years_of_experience=extraction.total_years_of_experience,
        remote_preference=extraction.remote_preference,
        languages_spoken=extraction.languages_spoken,
        industries_experienced=extraction.industries_experienced,
        skills=extraction.skills,
        markdown_profile=extraction.markdown_profile,
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
