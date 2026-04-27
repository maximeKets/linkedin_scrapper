from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from linkedin_scrapper.config import Settings
from linkedin_scrapper.models import Base


def build_engine(settings: Settings) -> Engine:
    if settings.database_url is None:
        raise RuntimeError("DATABASE_URL is required to build the database engine.")

    return create_engine(settings.database_url)


def init_db(engine: Engine) -> None:
    Base.metadata.create_all(bind=engine)


def drop_db(engine: Engine) -> None:
    Base.metadata.drop_all(bind=engine)
