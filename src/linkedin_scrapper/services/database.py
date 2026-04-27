from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from linkedin_scrapper.config import Settings


def build_engine(settings: Settings) -> Engine:
    if settings.database_url is None:
        raise RuntimeError("DATABASE_URL is required to build the database engine.")

    return create_engine(settings.database_url)
