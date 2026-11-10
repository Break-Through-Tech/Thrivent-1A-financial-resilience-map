"""End-to-end checks: user answers -> saved pipeline -> prediction."""

import pandas as pd
import pytest

from src.inference import (
    DATA_PATH,
    EXAMPLE_PROFILES,
    INPUT_FIELDS,
    FragilityPredictor,
)


@pytest.fixture(scope="module")
def predictor():
    # Loads models/<name>.joblib, or refits it from the committed metadata
    # when the artifact is not present (as in CI). Nothing is written.
    return FragilityPredictor.from_artifacts(save_if_fitted=False)


def test_every_answer_maps_to_a_value_seen_in_training(predictor):
    columns = {feat for f in INPUT_FIELDS
               for option in predictor._options.get(f.key, {}).values()
               for feat in option}
    data = pd.read_csv(DATA_PATH, usecols=sorted(columns), low_memory=False)
    for f in INPUT_FIELDS:
        if f.kind != "select":
            continue
        for label in predictor.options(f.key):
            for feat, value in predictor._options[f.key][label].items():
                assert value in set(data[feat]), (f.key, label, feat, value)


def test_example_profiles_answer_every_question():
    keys = {f.key for f in INPUT_FIELDS}
    for name, profile in EXAMPLE_PROFILES.items():
        assert set(profile) == keys, name


def test_built_row_is_complete(predictor):
    row = predictor.build_row(EXAMPLE_PROFILES["Getting by"])
    assert list(row.columns) == predictor.features
    assert len(row) == 1
    assert not row.isna().any().any()


@pytest.mark.parametrize(
    "profile, expected_fragile, low, high",
    [
        ("Financially secure", False, 0.0, 0.2),
        ("Getting by", None, 0.2, 0.8),
        ("Financially stretched", True, 0.8, 1.0),
    ],
)
def test_representative_profiles(predictor, profile, expected_fragile,
                                 low, high):
    result = predictor.predict(EXAMPLE_PROFILES[profile])
    assert low <= result.probability <= high
    if expected_fragile is not None:
        assert result.is_fragile is expected_fragile
    assert result.is_fragile == (result.probability >= result.threshold)
    assert result.n_answered == len(INPUT_FIELDS)


def test_profiles_are_ranked_by_risk(predictor):
    probs = [predictor.predict(EXAMPLE_PROFILES[name]).probability
             for name in ["Financially secure", "Getting by",
                          "Financially stretched"]]
    assert probs == sorted(probs)


def test_skipped_questions_use_typical_profile(predictor):
    blank = predictor.predict({})
    all_none = predictor.predict({f.key: None for f in INPUT_FIELDS})
    assert 0.0 <= blank.probability <= 1.0
    assert blank.n_answered == 0
    assert blank.probability == pytest.approx(all_none.probability)


def test_rare_answer_outweighs_common_one(predictor):
    # Few respondents often lack food; many are around 70. The unasked
    # features should come from people like the former, not the latter.
    result = predictor.predict({
        "food_sufficiency": "Often not enough to eat",
        "age": 70,
    })
    assert result.is_fragile


def test_linked_features_follow_the_answer(predictor):
    row = predictor.build_row({
        "age": 45,
        "managing_financially": "Just getting by",
        "credit_card_balance": "I don't have a credit card",
    }).iloc[0]
    assert row["ppage"] == 45
    assert row["ppagecat"] == "45-54"
    assert row["atleast_okay"] == "No"
    assert row["C2A"] == "No"
    assert row["C4A"] == "__NOT_ASKED__"


@pytest.mark.parametrize(
    "answers, error",
    [
        ({"not_a_question": "Yes"}, KeyError),
        ({"savings_account": "Maybe"}, ValueError),
        ({"age": 5}, ValueError),
    ],
)
def test_invalid_input_is_rejected(predictor, answers, error):
    with pytest.raises(error):
        predictor.predict(answers)
