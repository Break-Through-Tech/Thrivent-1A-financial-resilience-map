import sqlite3
from pathlib import Path
from contextlib import closing

DATABASE_PATH = Path(__file__).resolve().parent / "financial_predictions.db"


def get_connection():
    return sqlite3.connect(DATABASE_PATH)

def initialize_database():
    with closing(get_connection()) as connection:
        with connection:
            # SQLite stores timestamps in UTC (Coordinated Universal Time)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY,
                    prediction INTEGER NOT NULL CHECK (prediction IN (0, 1)),
                    timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)

def save_prediction(prediction):
    if prediction not in (0, 1):
        raise ValueError("The prediction must be represented by the value 0 or 1.")

    with closing(get_connection()) as connection:
        with connection:
            cursor = connection.execute(
                "INSERT INTO predictions (prediction) VALUES (?)",
                (int(prediction),)
            )
            record_id = cursor.lastrowid

    return record_id

def get_predictions():
    with closing(get_connection()) as connection:
        cursor = connection.execute("""
            SELECT id, prediction, timestamp
            FROM predictions
            ORDER BY id DESC
        """)
        records = cursor.fetchall()

    return records
