import sqlite3
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

CSV_PATH = BASE_DIR / "data" / "faw" / "FAW_Maize_Karnataka_Dataset_rebalanced.csv"
DATABASE_PATH = BASE_DIR / "database" / "agrishield.db"


def import_faw_data():
    print("Loading FAW dataset...")

    df = pd.read_csv(CSV_PATH)

    print(f"Rows loaded: {len(df)}")

    connection = sqlite3.connect(DATABASE_PATH)

    df.to_sql(
        "faw_records",
        connection,
        if_exists="replace",
        index=False,
    )

    connection.close()

    print("FAW dataset imported successfully!")
    print(f"Records stored: {len(df)}")


if __name__ == "__main__":
    import_faw_data()