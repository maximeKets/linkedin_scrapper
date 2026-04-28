from dataclasses import asdict
from pathlib import Path
from uuid import UUID

import typer
from rich.console import Console
from sqlalchemy.orm import Session

from linkedin_scrapper.config import Settings
from linkedin_scrapper.cv_parser import build_cv_parser_agent, parse_cv
from linkedin_scrapper.pipeline import PipelineRequest, build_pipeline
from linkedin_scrapper.search_planner import build_search_planner_agent, generate_linkedin_searches
from linkedin_scrapper.services.apify import build_apify_client, scrape_linkedin_jobs_for_search_run
from linkedin_scrapper.services.database import build_engine, drop_db, init_db
from linkedin_scrapper.services.llm import build_chat_model, build_cv_parser_chat_model
from linkedin_scrapper.services.profiles import save_candidate_profile
from linkedin_scrapper.services.searches import (
    list_pending_search_runs_for_profile,
    load_candidate_profile,
    save_search_runs,
)

app = typer.Typer(help="LinkedIn job matching POC backend.")
console = Console()


@app.command("config")
def show_config() -> None:
    """Show resolved non-secret configuration."""
    settings = Settings()
    console.print(settings.safe_dump())


@app.command("init-db")
def initialize_database(
    drop_existing: bool = typer.Option(
        False,
        "--drop-existing",
        help="Drop existing POC tables before creating the schema.",
    ),
) -> None:
    """Create the database schema from SQLAlchemy metadata."""
    settings = Settings()
    if not settings.database_url:
        console.print({"missing_required_environment": ["DATABASE_URL"]})
        raise typer.Exit(code=1)

    engine = build_engine(settings)
    if drop_existing:
        drop_db(engine)
    init_db(engine)
    console.print({"database_initialized": True, "drop_existing": drop_existing})


@app.command("parse-cv")
def parse_candidate_cv(
    cv_path: Path = typer.Argument(..., help="Path to a candidate CV in .txt, .md, or .pdf format."),
    save: bool = typer.Option(
        False,
        "--save",
        help="Persist the parsed profile to the configured database.",
    ),
) -> None:
    """Parse a CV into a structured candidate profile."""
    settings = Settings()
    if not settings.openai_api_key:
        console.print({"missing_required_environment": ["OPENAI_API_KEY"]})
        raise typer.Exit(code=1)
    if save and not settings.database_url:
        console.print({"missing_required_environment": ["DATABASE_URL"]})
        raise typer.Exit(code=1)

    chat_model = build_cv_parser_chat_model(settings)
    parser_agent = build_cv_parser_agent(chat_model)
    parsed_profile = parse_cv(cv_path, parser_agent)

    output = parsed_profile.model_dump()
    if save:
        engine = build_engine(settings)
        with Session(engine) as session:
            profile = save_candidate_profile(session, parsed_profile)
            output["id"] = str(profile.id)

    console.print(output)


@app.command("generate-searches")
def generate_searches(
    profile_id: UUID = typer.Argument(..., help="Candidate profile ID saved in the database."),
    save: bool = typer.Option(
        True,
        "--save/--no-save",
        help="Persist generated searches to search_runs.",
    ),
) -> None:
    """Generate LinkedIn Jobs searches from a saved candidate profile."""
    settings = Settings()
    missing = []
    if not settings.openai_api_key:
        missing.append("OPENAI_API_KEY")
    if not settings.database_url:
        missing.append("DATABASE_URL")
    if missing:
        console.print({"missing_required_environment": missing})
        raise typer.Exit(code=1)

    engine = build_engine(settings)
    with Session(engine) as session:
        profile = load_candidate_profile(session, profile_id)
        chat_model = build_chat_model(settings)
        planner_agent = build_search_planner_agent(chat_model)
        searches = generate_linkedin_searches(profile, planner_agent, settings)
        output = [search.model_dump() for search in searches]
        if save:
            search_runs = save_search_runs(
                session,
                profile,
                searches,
                settings.linkedin_jobs_actor_id,
            )
            for item, search_run in zip(output, search_runs, strict=True):
                item["search_run_id"] = str(search_run.id)

    console.print(output)


@app.command("scrape-jobs")
def scrape_jobs(
    profile_id: UUID = typer.Argument(..., help="Candidate profile ID saved in the database."),
    count: int = typer.Option(
        20,
        "--count",
        help="Maximum jobs to fetch per LinkedIn search run.",
    ),
    limit_runs: int | None = typer.Option(
        None,
        "--limit-runs",
        help="Maximum number of pending search runs to execute.",
    ),
) -> None:
    """Run pending LinkedIn Jobs searches through Apify and persist results."""
    if count < 1:
        raise typer.BadParameter("--count must be greater than 0.")
    if limit_runs is not None and limit_runs < 1:
        raise typer.BadParameter("--limit-runs must be greater than 0.")

    settings = Settings()
    missing = []
    if not settings.apify_api_token:
        missing.append("APIFY_API_TOKEN")
    if not settings.database_url:
        missing.append("DATABASE_URL")
    if missing:
        console.print({"missing_required_environment": missing})
        raise typer.Exit(code=1)

    client = build_apify_client(settings)
    engine = build_engine(settings)
    with Session(engine) as session:
        profile = load_candidate_profile(session, profile_id)
        search_runs = list_pending_search_runs_for_profile(
            session,
            profile,
            limit=limit_runs,
        )
        results = [
            asdict(scrape_linkedin_jobs_for_search_run(session, client, search_run, settings, count))
            for search_run in search_runs
        ]

    console.print(
        {
            "profile_id": str(profile_id),
            "pending_runs_selected": len(results),
            "results": results,
        }
    )


@app.command("run-pipeline")
def run_pipeline(
    cv_path: Path | None = typer.Option(
        None,
        "--cv-path",
        help="Path to the candidate CV. Required outside dry-run mode.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate startup and print the planned pipeline steps without external calls.",
    ),
) -> None:
    """Run the job matching pipeline."""
    settings = Settings()
    missing = settings.missing_runtime_values()

    if dry_run:
        pipeline = build_pipeline(settings)
        console.print(
            {
                "mode": "dry-run",
                "settings": settings.safe_dump(),
                "steps": pipeline.steps,
            }
        )
        raise typer.Exit(code=0)

    if cv_path is None:
        raise typer.BadParameter("--cv-path is required unless --dry-run is set.")

    if missing:
        console.print({"missing_required_environment": missing})
        raise typer.Exit(code=1)

    pipeline = build_pipeline(settings)
    result = pipeline.run(PipelineRequest(cv_path=cv_path))
    console.print(result)
