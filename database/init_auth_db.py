from database.history_models import ScanHistory
from database.auth_models import AuthBase
from database.postgres_db import get_postgres_engine


def initialize_auth_database():
    engine = get_postgres_engine()
    try:
        AuthBase.metadata.create_all(bind=engine)
        print("PostgreSQL users table created successfully.")
    finally:
        engine.dispose()


if __name__ == "__main__":
    initialize_auth_database()
