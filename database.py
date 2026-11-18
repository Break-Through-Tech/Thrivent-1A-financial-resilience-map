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
                    probability REAL CHECK (probability BETWEEN 0 AND 1),
                    timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Databases created before the probability column was added
            columns = {row[1] for row in connection.execute("PRAGMA table_info(predictions)")}
            if "probability" not in columns:
                connection.execute(
                    "ALTER TABLE predictions ADD COLUMN probability REAL "
                    "CHECK (probability BETWEEN 0 AND 1)"
                )

def save_prediction(prediction, probability=None):
    if prediction not in (0, 1):
        raise ValueError("The prediction must be represented by the value 0 or 1.")
    if probability is not None and not 0 <= probability <= 1:
        raise ValueError("The probability must be between 0 and 1.")

    with closing(get_connection()) as connection:
        with connection:
            cursor = connection.execute(
                "INSERT INTO predictions (prediction, probability) VALUES (?, ?)",
                (int(prediction), None if probability is None else float(probability))
            )
            record_id = cursor.lastrowid

    return record_id

def get_predictions(limit=None):
    # SQLite treats LIMIT -1 as "no limit"
    with closing(get_connection()) as connection:
        cursor = connection.execute("""
            SELECT id, prediction, probability, timestamp
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
        """, (-1 if limit is None else limit,))
        records = cursor.fetchall()

    return records
