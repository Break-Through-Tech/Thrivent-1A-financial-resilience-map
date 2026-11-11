
import json
from pathlib import Path

import joblib
import pandas as pd


# Locate the models folder relative to this file
ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "models"


def load_model(model_name="logistic_regression_with_ef"):
    model_path = MODEL_DIR / f"{model_name}.joblib"
    metadata_path = MODEL_DIR / f"{model_name}.json"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Model metadata not found: {metadata_path}"
        )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}. "
            "Train the model before running predictions."
        )

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    model = joblib.load(model_path)

    return model, metadata


def predict(model, metadata, data):
    """
    Generate financial fragility predictions.

    Returns a DataFrame containing predicted probabilities
    and binary classifications.
    """
    features = metadata["features"]
    threshold = metadata.get("threshold", 0.5)

    missing_features = [
        col for col in features if col not in data.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    X = data[features].copy()

    probabilities = model.predict_proba(X)[:, 1]

    results = pd.DataFrame({
        "fragility_probability": probabilities,
        "predicted_fragile": (
            probabilities >= threshold
        ).astype(int),
    }, index=data.index)

    return results
