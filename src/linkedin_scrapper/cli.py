from pathlib import Path

import typer
from rich.console import Console

from linkedin_scrapper.config import Settings
from linkedin_scrapper.pipeline import PipelineRequest, build_pipeline

app = typer.Typer(help="LinkedIn job matching POC backend.")
console = Console()


@app.command("config")
def show_config() -> None:
    """Show resolved non-secret configuration."""
    settings = Settings()
    console.print(settings.safe_dump())


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
