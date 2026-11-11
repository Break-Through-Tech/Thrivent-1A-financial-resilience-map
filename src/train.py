
"""Train and save the financial fragility logistic regression model."""

import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"

MODEL_NAME = "logistic_regression_with_ef"
C_GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]


def build_model(c_value, spec):
    """Build the same preprocessing and model pipeline as notebook 04."""

    features = spec["features"]
    numeric = features["numeric"]
    categorical = features["categorical"]
    ordinal = features["ordinal"]

    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(
            strategy="median", add_indicator=True
        )),
        ("scale", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(
            strategy="constant", fill_value="__MISSING__"
        )),
        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
    ])

    ordinal_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OrdinalEncoder(
            categories=[
                features["ordinal_order"][col]
                for col in ordinal
            ]
        )),
        ("scale", StandardScaler()),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric),
        ("categorical", categorical_pipeline, categorical),
        ("ordinal", ordinal_pipeline, ordinal),
    ])

    return Pipeline([
        ("preprocess", preprocessor),
        ("model", LogisticRegression(
            C=c_value, max_iter=1000
        )),
    ])


def train_model():
    """Select regularization strength, fit, and save the model."""

    data_path = DATA_DIR / "model_dataset_2025.csv"
    spec_path = DATA_DIR / "model_dataset_2025_spec.json"

    with open(spec_path, encoding="utf-8") as f:
        spec = json.load(f)

    data = pd.read_csv(data_path, low_memory=False)

    features = (
        spec["features"]["numeric"]
        + spec["features"]["categorical"]
        + spec["features"]["ordinal"]
    )

    target = spec["target"]["column"]
    weight = spec["weight"]

    train = data.loc[data["split"] == "train"].copy()

    X = train[features]
    y = train[target]
    weights = train[weight]
    normalized_weights = weights / weights.mean()
    cv_folds = train["cv_fold"]

    # Use the same cross-validation procedure as notebook 04.
    cv_results = {}

    for c_value in C_GRID:
        fold_losses = []

        for fold in range(spec["cv"]["n_folds"]):
            training_rows = cv_folds != fold
            validation_rows = cv_folds == fold

            model = build_model(c_value, spec)

            model.fit(
                X.loc[training_rows],
                y.loc[training_rows],
                model__sample_weight=normalized_weights.loc[
                    training_rows
                ],
            )

            probabilities = model.predict_proba(
                X.loc[validation_rows]
            )[:, 1]

            loss = log_loss(
                y.loc[validation_rows],
                probabilities,
                sample_weight=weights.loc[validation_rows],
            )

            fold_losses.append(loss)

        cv_results[c_value] = sum(fold_losses) / len(fold_losses)
        print(f"C={c_value}: CV log loss={cv_results[c_value]:.4f}")

    best_c = min(cv_results, key=cv_results.get)
    print(f"Selected C: {best_c}")

    # Fit the final pipeline on the full training partition.
    final_model = build_model(best_c, spec)

    final_model.fit(
        X,
        y,
        model__sample_weight=normalized_weights,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / f"{MODEL_NAME}.joblib"
    metadata_path = MODEL_DIR / f"{MODEL_NAME}.json"

    joblib.dump(final_model, model_path)

    metadata = {
        "model": MODEL_NAME,
        "target": target,
        "features": features,
        "selected_params": {"C": best_c},
        "cv_mean_log_loss": cv_results[best_c],
        "threshold": 0.5,
        "sklearn_version": sklearn.__version__,
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Model saved to: {model_path}")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    train_model()
