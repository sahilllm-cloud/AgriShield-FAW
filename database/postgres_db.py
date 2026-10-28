
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from sqlalchemy.orm import sessionmaker

# Load environment variables from the project-root .env file.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))


def _build_database_url() -> URL:
    username = os.getenv("PGUSER")
    password = os.getenv("PGPASSWORD")

    if not username or not password:
        raise RuntimeError(
            "PGUSER and PGPASSWORD must be set in the project-root .env file."
        )

    return URL.create(
        drivername="postgresql+psycopg",
        username=username,
        password=password,
        host=os.getenv("PGHOST", "localhost"),
        port=int(os.getenv("PGPORT", "5432")),
        database=os.getenv("PGDATABASE", "agrishield"),
    )


def get_postgres_engine():
    return create_engine(
        _build_database_url(),
        pool_pre_ping=True,
    )


def test_postgres_connection() -> bool:
    engine = get_postgres_engine()
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    finally:
        engine.dispose()


def get_postgres_session_factory():
    engine = get_postgres_engine()
    return sessionmaker(
        bind=engine,
        autoflush=False,
        expire_on_commit=False,
    )
