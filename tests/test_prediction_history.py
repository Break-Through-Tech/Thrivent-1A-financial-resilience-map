"""Prediction history: database storage and the app's history view."""

import sqlite3
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import database

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture
def temp_database(tmp_path, monkeypatch):
    # Every test writes to its own throwaway database, never the real one
    monkeypatch.setattr(database, "DATABASE_PATH", tmp_path / "predictions.db")
    database.initialize_database()
    return database.DATABASE_PATH


def test_saved_prediction_is_returned(temp_database):
    record_id = database.save_prediction(1, 0.82)
    records = database.get_predictions()
    assert len(records) == 1
    assert records[0][:3] == (record_id, 1, 0.82)
    assert records[0][3]  # timestamp filled in by SQLite


def test_predictions_are_newest_first_and_limited(temp_database):
    ids = [database.save_prediction(i % 2, i / 10) for i in range(5)]
    records = database.get_predictions(limit=3)
    assert [r[0] for r in records] == ids[::-1][:3]


def test_probability_is_optional(temp_database):
    database.save_prediction(0)
    assert database.get_predictions()[0][2] is None


@pytest.mark.parametrize("prediction, probability", [(2, 0.5), (1, 1.5), (0, -0.1)])
def test_invalid_values_are_rejected(temp_database, prediction, probability):
    with pytest.raises(ValueError):
        database.save_prediction(prediction, probability)
    assert database.get_predictions() == []


def test_old_database_gains_probability_column(tmp_path, monkeypatch):
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as connection:
        connection.execute("""
            CREATE TABLE predictions (
                id INTEGER PRIMARY KEY,
                prediction INTEGER NOT NULL CHECK (prediction IN (0, 1)),
                timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        connection.execute("INSERT INTO predictions (prediction) VALUES (1)")
    connection.close()
    monkeypatch.setattr(database, "DATABASE_PATH", path)

    database.initialize_database()
    database.save_prediction(0, 0.3)
    records = database.get_predictions()
    assert [(r[1], r[2]) for r in records] == [(0, 0.3), (1, None)]


def predict_with_profile(profile):
    app = AppTest.from_file(str(APP_PATH), default_timeout=300)
    app.run()
    app.selectbox(key="profile").set_value(profile).run()
    next(b for b in app.button if b.label == "Predict").click().run()
    return app


def test_saved_prediction_appears_in_history(temp_database):
    app = predict_with_profile("Financially stretched")
    assert not app.error, [e.value for e in app.error]

    records = database.get_predictions()
    assert len(records) == 1
    _, prediction, probability, _ = records[0]
    assert prediction == 1 and 0.8 <= probability <= 1

    history = app.dataframe[0].value
    assert len(history) == 1
    assert history.iloc[0]["Classification"] == "Financially fragile"
    assert history.iloc[0]["Probability"] == f"{probability:.1%}"

    # Other widgets rerun the script; that must not save the prediction again
    app.selectbox(key="profile").set_value("Getting by").run()
    assert len(database.get_predictions()) == 1


def test_failed_save_shows_error_but_keeps_prediction(temp_database, monkeypatch):
    def broken_save(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(database, "save_prediction", broken_save)
    app = predict_with_profile("Getting by")

    assert any("could not be saved" in e.value for e in app.error)
    assert any(m.label == "Estimated probability of financial fragility" for m in app.metric)
