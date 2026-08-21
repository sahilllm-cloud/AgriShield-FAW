import sqlite3
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_PATH = BASE_DIR / "database" / "agrishield.db"


def get_connection():
    return sqlite3.connect(DATABASE_PATH)


def initialize_database():
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS faw_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            district TEXT,
            taluk TEXT,
            village TEXT,
            latitude REAL,
            longitude REAL,
            month INTEGER,
            year INTEGER,
            temperature_c REAL,
            humidity_pct REAL,
            rainfall_mm REAL,
            soil_moisture_pct REAL,
            wind_speed_kmph REAL,
            crop_stage TEXT,
            maize_variety TEXT,
            crop_age_days INTEGER,
            previous_pest_count INTEGER,
            days_since_last_attack INTEGER,
            risk_level TEXT
        )
    """)

    connection.commit()
    connection.close()


if __name__ == "__main__":
    initialize_database()
    print("AgriShield database initialized successfully.")

def find_nearest_faw_record(latitude: float, longitude: float):
    connection = get_connection()

    query = """
        SELECT *
        FROM faw_records
        ORDER BY
            ((latitude - ?) * (latitude - ?)) +
            ((longitude - ?) * (longitude - ?))
        LIMIT 1
    """

    cursor = connection.cursor()
    cursor.execute(
        query,
        (latitude, latitude, longitude, longitude),
    )

    row = cursor.fetchone()
    columns = [column[0] for column in cursor.description]

    connection.close()

    if row is None:
        return None

    return dict(zip(columns, row))


if __name__ == "__main__":
    initialize_database()
    print("AgriShield database initialized successfully.")