from dataclasses import dataclass
from pathlib import Path

from linkedin_scrapper.config import Settings
from linkedin_scrapper.cv_parser import (
    ParsedCandidateProfile,
    build_cv_parser_agent,
    parse_cv,
)
from linkedin_scrapper.services.llm import build_chat_model


@dataclass(frozen=True)
class PipelineRequest:
    cv_path: Path


@dataclass(frozen=True)
class PipelineResult:
    status: str
    steps: list[str]
    candidate_profile: ParsedCandidateProfile | None = None


class JobMatchingPipeline:
    steps = [
        "parse_cv",
        "build_searches",
        "scrape_jobs",
        "normalize_and_persist_jobs",
        "score_jobs",
        "send_digest",
    ]

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def run(self, request: PipelineRequest) -> PipelineResult:
        if not request.cv_path.exists():
            raise FileNotFoundError(f"CV file not found: {request.cv_path}")

        chat_model = build_chat_model(self.settings)
        parser_agent = build_cv_parser_agent(chat_model)
        candidate_profile = parse_cv(request.cv_path, parser_agent)
        return PipelineResult(
            status="not_implemented",
            steps=self.steps,
            candidate_profile=candidate_profile,
        )


def build_pipeline(settings: Settings) -> JobMatchingPipeline:
    return JobMatchingPipeline(settings=settings)
