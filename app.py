"""Financial Resilience Map prediction form."""

import logging

import streamlit as st

from src.inference import EXAMPLE_PROFILES, INPUT_FIELDS, FragilityPredictor

logger = logging.getLogger(__name__)
SECTIONS = {"finances": "Your finances", "household": "Your household"}


@st.cache_resource(show_spinner="Loading the prediction model…")
def get_predictor():
    return FragilityPredictor.from_artifacts()


def load_profile():
    profile = EXAMPLE_PROFILES.get(st.session_state["profile"], {})
    for field in INPUT_FIELDS:
        answer = profile.get(field.key)
        if field.kind == "number":
            answer = "" if answer is None else str(answer)
        st.session_state[field.key] = answer


def validate_answers(raw_answers, predictor):
    answers = {}
    errors = []
    for field in INPUT_FIELDS:
        value = raw_answers[field.key]
        if value is None or (isinstance(value, str) and not value.strip()):
            errors.append(f"Answer this question: {field.label}")
            continue

        if field.kind == "number":
            low, high = predictor.number_range(field.key)
            try:
                value = int(value.strip())
            except ValueError:
                value = None
            if value is None or not low <= value <= high:
                errors.append(
                    f"{field.label} must be a whole number from {low} to {high}."
                )
                continue
        elif value not in predictor.options(field.key):
            errors.append(f"Choose a listed answer for: {field.label}")
            continue
        answers[field.key] = value
    return answers, errors

def main():
    st.set_page_config(page_title="Financial Resilience Map")
    st.title("Financial Resilience Map")
    st.write(
        "Estimate the probability that a household could not cover a $400 "
        "emergency expense with cash or its equivalent."
    )
    st.caption(f"Answer all {len(INPUT_FIELDS)} questions to request a prediction.")

    try:
        predictor = get_predictor()
    except Exception:
        logger.exception("Could not initialize the predictor")
        st.error(
            "The prediction model could not load. Check that you installed "
            "requirements.txt and that the project includes its processed "
            "survey data and model metadata. See the terminal for details."
        )
        return

    with st.sidebar:
        st.header("Sample profiles")
        st.selectbox(
            "Fill the questionnaire with an example",
            EXAMPLE_PROFILES,
            index=None,
            placeholder="Choose a sample profile",
            key="profile",
            on_change=load_profile,
        )
        st.button("Clear answers", on_click=clear_answers)

    raw_answers = {}
    with st.form("questionnaire"):
        for section, title in SECTIONS.items():
            st.subheader(title)
            for field in INPUT_FIELDS:
                if field.section != section:
                    continue
                if field.kind == "number":
                    low, high = predictor.number_range(field.key)
                    answer = st.text_input(
                        field.label,
                        placeholder=f"Whole number from {low} to {high}",
                        help=field.help or None,
                        key=field.key,
                    )
                else:
                    answer = st.selectbox(
                        field.label,
                        predictor.options(field.key),
                        index=None,
                        placeholder="Choose an answer",
                        help=field.help or None,
                        key=field.key,
                    )
                raw_answers[field.key] = answer
        submitted = st.form_submit_button("Predict", type="primary")

    if not submitted:
        return

    answers, errors = validate_answers(raw_answers, predictor)
    if errors:
        st.error("Complete or correct these answers, then select Predict again.")
        for error in errors:
            st.write(f"- {error}")
        return

    try:
        result = predictor.predict(answers)
    except (KeyError, ValueError):
        logger.exception("Prediction failed")
        st.error(
            "The model could not score these answers. Review your inputs "
            "and try again. If the error persists, see the terminal for details."
        )
        return

    st.header("Prediction")
    probability, classification = st.columns(2)
    probability.metric("Estimated probability of financial fragility",
                       f"{result.probability:.1%}")
    classification.metric("Classification", result.label)
    st.caption(f"The model classifies scores of {result.threshold:.1%} or higher "
               "as financially fragile.")
    st.caption("The model estimates additional survey inputs using similar "
               "training respondents.")


def clear_answers():
    st.session_state["profile"] = None
    load_profile()


if __name__ == "__main__":
    main()