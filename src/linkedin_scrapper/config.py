from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    apify_api_token: SecretStr | None = Field(default=None, alias="APIFY_API_TOKEN")
    openai_api_key: SecretStr | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4.1-mini", alias="OPENAI_MODEL")
    openai_cv_parser_model: str = Field(
        default="gpt-4.1",
        alias="OPENAI_CV_PARSER_MODEL",
    )
    database_url: str | None = Field(default=None, alias="DATABASE_URL")
    resend_api_key: SecretStr | None = Field(default=None, alias="RESEND_API_KEY")
    resend_from_email: str | None = Field(default=None, alias="RESEND_FROM_EMAIL")
    digest_to_email: str | None = Field(default=None, alias="DIGEST_TO_EMAIL")

    linkedin_jobs_actor_id: str = Field(
        default="curious_coder/linkedin-jobs-scraper",
        alias="LINKEDIN_JOBS_ACTOR_ID",
    )
    min_search_queries: int = Field(default=5, alias="MIN_SEARCH_QUERIES", ge=1, le=15)
    max_search_queries: int = Field(default=10, alias="MAX_SEARCH_QUERIES", ge=5, le=15)
    default_job_count: int = Field(default=25, alias="DEFAULT_JOB_COUNT", ge=1)
    score_threshold: int = Field(default=75, alias="SCORE_THRESHOLD", ge=0, le=100)

    def missing_runtime_values(self) -> list[str]:
        required = {
            "APIFY_API_TOKEN": self.apify_api_token,
            "OPENAI_API_KEY": self.openai_api_key,
            "DATABASE_URL": self.database_url,
            "RESEND_API_KEY": self.resend_api_key,
            "RESEND_FROM_EMAIL": self.resend_from_email,
            "DIGEST_TO_EMAIL": self.digest_to_email,
        }
        return [name for name, value in required.items() if not _has_value(value)]

    def safe_dump(self) -> dict[str, str | int | bool | None]:
        return {
            "database_configured": _has_value(self.database_url),
            "apify_configured": _has_value(self.apify_api_token),
            "openai_configured": _has_value(self.openai_api_key),
            "resend_configured": _has_value(self.resend_api_key),
            "resend_from_email": self.resend_from_email,
            "digest_to_email": self.digest_to_email,
            "openai_model": self.openai_model,
            "openai_cv_parser_model": self.openai_cv_parser_model,
            "linkedin_jobs_actor_id": self.linkedin_jobs_actor_id,
            "min_search_queries": self.min_search_queries,
            "max_search_queries": self.max_search_queries,
            "default_job_count": self.default_job_count,
            "score_threshold": self.score_threshold,
        }


def _has_value(value: SecretStr | str | None) -> bool:
    if value is None:
        return False
    if isinstance(value, SecretStr):
        return bool(value.get_secret_value().strip())
    return bool(value.strip())
