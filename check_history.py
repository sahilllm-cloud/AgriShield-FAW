
from sqlalchemy import text
from database.postgres_db import get_postgres_engine

engine = get_postgres_engine()

try:
    with engine.connect() as connection:
        result = connection.execute(
            text("""
                SELECT id, user_id, guest_id, image_filename,
                       predicted_class, created_at
                FROM scan_history
                ORDER BY id DESC
                LIMIT 10
            """)
        )

        rows = result.mappings().all()

        print("\n--- LATEST HISTORY RECORDS ---")

        if not rows:
            print("No records found in scan_history.")

        for row in rows:
            print(dict(row))

finally:
    engine.dispose()
