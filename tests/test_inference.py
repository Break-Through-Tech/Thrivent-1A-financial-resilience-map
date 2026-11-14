"""End-to-end checks: user answers -> saved pipeline -> prediction."""

import pandas as pd
import pytest

from src.inference import (
    DATA_PATH,
    EXAMPLE_PROFILES,
    INPUT_FIELDS,
    N_SIMILAR,
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


def test_built_rows_are_complete_and_carry_the_answers(predictor):
    answers = EXAMPLE_PROFILES["Getting by"]
    rows = predictor.build_rows(answers)
    assert list(rows.columns) == predictor.features
    assert len(rows) == N_SIMILAR
    # Unasked answers come from real respondents, so skipped numeric
    # questions stay blank, as in training; the pipeline imputes them.
    for feat, value in predictor._validated_values(answers).items():
        assert (rows[feat] == value).all(), feat


def test_probability_is_the_average_over_similar_respondents(predictor):
    answers = EXAMPLE_PROFILES["Getting by"]
    rows = predictor.build_rows(answers)
    expected = predictor.model.predict_proba(rows)[:, 1].mean()
    assert predictor.predict(answers).probability == pytest.approx(expected)


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


def test_no_answers_averages_over_all_respondents(predictor):
    blank = predictor.predict({})
    all_none = predictor.predict({f.key: None for f in INPUT_FIELDS})
    assert 0.0 <= blank.probability <= 1.0
    assert blank.n_answered == 0
    assert blank.probability == pytest.approx(all_none.probability)


def test_rare_answer_outweighs_common_one(predictor):
    # Few respondents often lack food; most have a savings account. The
    # similar respondents should be chosen mainly by the rare answer.
    result = predictor.predict({
        "food_sufficiency": "Often not enough to eat",
        "savings_account": "Yes",
    })
    assert result.is_fragile


def test_linked_features_follow_the_answer(predictor):
    rows = predictor.build_rows({
        "managing_financially": "Just getting by",
        "credit_card_balance": "I don't have a credit card",
    })
    assert (rows["atleast_okay"] == "No").all()
    assert (rows["C2A"] == "No").all()
    assert (rows["C4A"] == "__NOT_ASKED__").all()


@pytest.mark.parametrize(
    "answers, error",
    [
        ({"not_a_question": "Yes"}, KeyError),
        ({"savings_account": "Maybe"}, ValueError),
        ({"money_left_over": 3}, ValueError),
    ],
)
def test_invalid_input_is_rejected(predictor, answers, error):
    with pytest.raises(error):
        predictor.predict(answers)
