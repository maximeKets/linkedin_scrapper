from pathlib import Path

import typer
from rich.console import Console
from sqlalchemy.orm import Session

from linkedin_scrapper.config import Settings
from linkedin_scrapper.cv_parser import build_cv_parser_agent, parse_cv
from linkedin_scrapper.pipeline import PipelineRequest, build_pipeline
from linkedin_scrapper.services.database import build_engine, drop_db, init_db
from linkedin_scrapper.services.llm import build_chat_model
from linkedin_scrapper.services.profiles import save_candidate_profile

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

    chat_model = build_chat_model(settings)
    parser_agent = build_cv_parser_agent(chat_model)
    parsed_profile = parse_cv(cv_path, parser_agent)

    output = parsed_profile.model_dump()
    if save:
        engine = build_engine(settings)
        with Session(engine) as session:
            profile = save_candidate_profile(session, parsed_profile)
            output["id"] = str(profile.id)

    console.print(output)


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
