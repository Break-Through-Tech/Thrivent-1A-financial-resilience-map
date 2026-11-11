"""Turn a short set of user answers into a financial fragility prediction.

The trained pipeline expects all 425 survey features, but the app only asks a
user about a dozen of the most influential ones. This module:

1. loads the saved model artifact through ``src.model.load_model``;
2. finds the training respondents whose answers best match the user's, and
   fills every feature the app does not ask about with the most common
   answer (or the median, for numeric features) among those respondents;
3. overwrites that profile with whatever the user answered, keeping linked
   features consistent (for example ``atleast_okay`` follows ``B2``);
4. runs the unchanged preprocessing + model pipeline via ``src.model.predict``.

Filling from similar respondents, rather than from the whole sample, keeps the
unasked answers consistent with the asked ones. A profile of the overall most
common answers describes a financially secure household and would pull every
prediction towards "not fragile". With no answers at all, the whole training
split is used, giving a typical respondent.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.model import MODEL_DIR, load_model, predict


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "processed" / "model_dataset_2025.csv"
SPEC_PATH = ROOT_DIR / "data" / "processed" / "model_dataset_2025_spec.json"

DEFAULT_MODEL = "logistic_regression_with_ef"

# Survey skip-pattern code: the question was not shown to this respondent.
NOT_ASKED = "__NOT_ASKED__"

# Number of most similar training respondents used to fill unasked features.
N_SIMILAR = 100

# Numeric answers within this distance count as matching when finding
# similar respondents.
NUMBER_TOLERANCE = {"ppage": 5, "pphhsize": 0}


@dataclass(frozen=True)
class InputField:
    """One question shown to the user.

    ``options`` maps each answer label to the feature values it sets. When it
    is empty, the options are the values of ``feature`` seen in training.
    ``derived`` lists features computed from ``feature`` (e.g. age bands from
    age); each takes the most common value seen alongside the user's answer.
    """

    key: str
    label: str
    feature: str
    section: str
    kind: str = "select"  # "select" or "number"
    options: dict = field(default_factory=dict)
    derived: tuple = ()
    help: str = ""


INPUT_FIELDS = [
    InputField(
        key="emergency_savings",
        label="Do you have emergency savings that would cover 3 months "
              "of expenses?",
        feature="EF1",
        section="finances",
        options={
            "Yes": {"EF1": "Yes", "EF2": NOT_ASKED},
            "No, but I could cover 3 months by borrowing, using other "
            "savings, or selling assets": {"EF1": "No", "EF2": "Yes"},
            "No, and I could not cover 3 months of expenses":
                {"EF1": "No", "EF2": "No"},
        },
    ),
    InputField(
        key="largest_emergency_expense",
        label="What is the largest emergency expense you could cover "
              "using only your savings?",
        feature="EF7",
        section="finances",
        options={
            value: {"EF7": value, "EF7A": NOT_ASKED}
            for value in [
                "Under $100",
                "$100 to $499",
                "$500 to $999",
                "$1,000 to $1,999",
                "$2,000 or more",
            ]
        },
    ),
    InputField(
        key="credit_card_balance",
        label="In the past 12 months, how often did you carry an unpaid "
              "credit card balance?",
        feature="C4A",
        section="finances",
        options={
            "Never (always pay in full)": {
                "C2A": "Yes",
                "C4A": "Never carried an unpaid balance "
                       "(always pay in full)",
            },
            "Once": {"C2A": "Yes", "C4A": "Once"},
            "Some of the time": {"C2A": "Yes", "C4A": "Some of the time"},
            "Most or all of the time":
                {"C2A": "Yes", "C4A": "Most or all of the time"},
            "I don't have a credit card": {"C2A": "No", "C4A": NOT_ASKED},
        },
    ),
    InputField(
        key="savings_account",
        label="Do you have a savings account, money market account, or CD?",
        feature="K21_d",
        section="finances",
    ),
    InputField(
        key="spending_vs_income",
        label="Over the past 12 months, was your spending less than, the "
              "same as, or more than your income?",
        feature="I20",
        section="finances",
    ),
    InputField(
        key="managing_financially",
        label="How well are you managing financially these days?",
        feature="B2",
        section="finances",
        options={
            "Finding it difficult to get by":
                {"B2": "Finding it difficult to get by",
                 "atleast_okay": "No"},
            "Just getting by":
                {"B2": "Just getting by", "atleast_okay": "No"},
            "Doing okay": {"B2": "Doing okay", "atleast_okay": "Yes"},
            "Living comfortably":
                {"B2": "Living comfortably", "atleast_okay": "Yes"},
        },
    ),
    InputField(
        key="food_sufficiency",
        label="In the past month, which best describes the food eaten in "
              "your household?",
        feature="FD3",
        section="finances",
    ),
    InputField(
        key="household_income",
        label="Household income",
        feature="ppinc7",
        section="household",
    ),
    InputField(
        key="employment",
        label="Employment status",
        feature="ppemploy",
        section="household",
    ),
    InputField(
        key="education",
        label="Highest level of education",
        feature="ppeduc5",
        section="household",
        derived=("ppeducat",),
    ),
    InputField(
        key="housing",
        label="Housing situation",
        feature="GH1",
        section="household",
    ),
    InputField(
        key="age",
        label="Age",
        feature="ppage",
        section="household",
        kind="number",
        derived=("ppagecat", "ppagect4"),
    ),
    InputField(
        key="household_size",
        label="Number of people in your household",
        feature="pphhsize",
        section="household",
        kind="number",
        derived=("pphhsize5",),
    ),
]

FIELDS_BY_KEY = {f.key: f for f in INPUT_FIELDS}


# Representative profiles used for the app's "load an example" menu and
# for the end-to-end tests.
EXAMPLE_PROFILES = {
    "Financially secure": {
        "emergency_savings": "Yes",
        "largest_emergency_expense": "$2,000 or more",
        "credit_card_balance": "Never (always pay in full)",
        "savings_account": "Yes",
        "spending_vs_income": "Less than your income",
        "managing_financially": "Living comfortably",
        "food_sufficiency": "Enough of the kinds of food we wanted to eat",
        "household_income": "$150,000 or more",
        "employment": "Working full-time",
        "education": "Bachelor's degree",
        "housing": "Own your home with a mortgage or loan",
        "age": 45,
        "household_size": 3,
    },
    "Getting by": {
        "emergency_savings": "No, but I could cover 3 months by borrowing, "
                             "using other savings, or selling assets",
        "largest_emergency_expense": "$500 to $999",
        "credit_card_balance": "Some of the time",
        "savings_account": "Yes",
        "spending_vs_income": "The same as your income",
        "managing_financially": "Just getting by",
        "food_sufficiency": "Enough, but not always the kinds of food "
                            "we wanted to eat",
        "household_income": "$50,000 to $74,999",
        "employment": "Working full-time",
        "education": "Some college or Associate's degree",
        "housing": "Pay rent",
        "age": 38,
        "household_size": 2,
    },
    "Financially stretched": {
        "emergency_savings": "No, and I could not cover 3 months of expenses",
        "largest_emergency_expense": "Under $100",
        "credit_card_balance": "Most or all of the time",
        "savings_account": "No",
        "spending_vs_income": "More than your income",
        "managing_financially": "Finding it difficult to get by",
        "food_sufficiency": "Sometimes not enough to eat",
        "household_income": "$10,000 to $24,999",
        "employment": "Working part-time",
        "education": "High school graduate (high school diploma or the "
                     "equivalent GED)",
        "housing": "Pay rent",
        "age": 29,
        "household_size": 4,
    },
}


@dataclass
class PredictionResult:
    probability: float
    is_fragile: bool
    threshold: float
    answered: dict
    n_answered: int
    n_features: int

    @property
    def label(self):
        if self.is_fragile:
            return "Financially fragile"
        return "Not financially fragile"


def _display(value):
    # Some survey labels were saved with a broken apostrophe/dash character.
    return str(value).replace("�", "'")


def _ensure_model(model_name, spec, training_data, save=True):
    """Load the saved pipeline, or refit it from the committed metadata.

    The ``.joblib`` file is git-ignored, so a fresh clone (or a hosted
    deployment) may not have it. For the logistic regression model the
    metadata records the selected regularisation strength, so the final fit
    from ``src/train.py`` can be reproduced without re-running the search.
    """
    try:
        return load_model(model_name)
    except FileNotFoundError:
        metadata_path = MODEL_DIR / f"{model_name}.json"
        if not metadata_path.exists() or not model_name.startswith(
            "logistic_regression"
        ):
            raise

    # Imported here so the app does not pay for it when the artifact exists.
    from src.train import build_model

    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)

    weights = training_data[spec["weight"]]
    model = build_model(metadata["selected_params"]["C"], spec)
    model.fit(
        training_data[metadata["features"]],
        training_data[spec["target"]["column"]],
        model__sample_weight=weights / weights.mean(),
    )

    if save:
        try:
            joblib.dump(model, MODEL_DIR / f"{model_name}.joblib")
        except OSError:
            pass  # read-only filesystem: keep the in-memory model

    return model, metadata


class FragilityPredictor:
    """Predict financial fragility from a few user answers."""

    def __init__(self, model, metadata, training_data):
        self.model = model
        self.metadata = metadata
        self.features = metadata["features"]
        self.threshold = metadata.get("threshold", 0.5)

        train = training_data[self.features].reset_index(drop=True)
        self._train = train
        self._numeric = [
            col for col in self.features
            if pd.api.types.is_numeric_dtype(train[col])
        ]
        self.reference = self._typical(train)

        self._options = {}
        self._derived = {}
        self._ranges = {}
        for f in INPUT_FIELDS:
            values = train[f.feature]
            if f.kind == "number":
                self._ranges[f.key] = (int(values.min()), int(values.max()))
            elif f.options:
                self._options[f.key] = dict(f.options)
            else:
                ordered = values[values != NOT_ASKED].value_counts().index
                self._options[f.key] = {
                    _display(v): {f.feature: v} for v in ordered
                }
            for target in f.derived:
                self._derived[(f.key, target)] = (
                    train.groupby(f.feature)[target]
                    .agg(lambda s: s.mode().iloc[0])
                )

    @classmethod
    def from_artifacts(cls, model_name=DEFAULT_MODEL, save_if_fitted=True):
        with open(SPEC_PATH, encoding="utf-8") as f:
            spec = json.load(f)
        data = pd.read_csv(DATA_PATH, low_memory=False)
        training_data = data.loc[data["split"] == "train"]
        model, metadata = _ensure_model(
            model_name, spec, training_data, save=save_if_fitted
        )
        return cls(model, metadata, training_data)

    def options(self, key):
        """Answer labels for a select question, in display order."""
        return list(self._options[key])

    def number_range(self, key):
        """(min, max) seen in training for a numeric question."""
        return self._ranges[key]

    def _typical(self, rows):
        """Median of numeric features and most common value of the rest."""
        typical = rows[self._numeric].median().to_dict()
        for col in self.features:
            if col not in typical:
                typical[col] = rows[col].value_counts().index[0]
        return typical

    def _similar_profile(self, answered):
        """Typical values among the respondents most similar to ``answered``.

        ``answered`` maps the features the user set to their values. Each
        matching answer adds ``-log(share of respondents giving it)`` to a
        respondent's similarity, so rare answers (such as often not having
        enough to eat) count for more than common ones (such as an age band).
        """
        if not answered:
            return dict(self.reference)

        score = pd.Series(0.0, index=self._train.index)
        for feat, value in answered.items():
            column = self._train[feat]
            if feat in NUMBER_TOLERANCE:
                match = (column - value).abs() <= NUMBER_TOLERANCE[feat]
            else:
                match = column == value
            share = match.mean()
            if share > 0:
                score += match * -np.log(share)
        nearest = score.sort_values(ascending=False, kind="stable")
        return self._typical(self._train.loc[nearest.index[:N_SIMILAR]])

    def _validated_values(self, answers):
        """Check answers and return the feature values each one sets."""
        unknown = set(answers) - set(FIELDS_BY_KEY)
        if unknown:
            raise KeyError(f"Unknown input field(s): {sorted(unknown)}")

        values = {}
        for key, answer in answers.items():
            if answer is None:
                continue
            f = FIELDS_BY_KEY[key]
            if f.kind == "number":
                low, high = self._ranges[key]
                number = int(answer)
                if not low <= number <= high:
                    raise ValueError(
                        f"{f.label} must be between {low} and {high}."
                    )
                values[f.feature] = number
            else:
                if answer not in self._options[key]:
                    raise ValueError(
                        f"{answer!r} is not a valid answer for {key!r}."
                    )
                values.update(self._options[key][answer])
        return values

    def build_row(self, answers):
        """Return a one-row DataFrame with every model feature filled in.

        ``answers`` maps field keys to an option label (select questions) or
        a number. Missing keys and ``None`` values are filled in from the
        respondents most similar to the answers given.
        """
        answered = self._validated_values(answers)
        row = self._similar_profile(answered)
        row.update(answered)

        # Keep derived features (age bands, etc.) in line with the answers.
        for f in INPUT_FIELDS:
            if f.feature not in answered:
                continue
            value = answered[f.feature]
            for target in f.derived:
                lookup = self._derived[(f.key, target)]
                if value not in lookup.index:
                    # Use the closest value seen in training.
                    value = min(lookup.index, key=lambda v: abs(v - value))
                row[target] = lookup[value]

        return pd.DataFrame([row], columns=self.features)

    def predict(self, answers):
        """Predict from user answers and return a ``PredictionResult``."""
        row = self.build_row(answers)
        result = predict(self.model, self.metadata, row).iloc[0]
        answered = {k: v for k, v in answers.items() if v is not None}
        return PredictionResult(
            probability=float(result["fragility_probability"]),
            is_fragile=bool(result["predicted_fragile"]),
            threshold=self.threshold,
            answered=answered,
            n_answered=len(answered),
            n_features=len(self.features),
        )
