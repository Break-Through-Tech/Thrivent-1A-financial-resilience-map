"""Streamlit app: predict financial fragility from a short questionnaire.

Run from the repository root:

    streamlit run app.py
"""

import streamlit as st

from src.inference import (
    EXAMPLE_PROFILES,
    INPUT_FIELDS,
    FragilityPredictor,
)


SKIP = "Prefer not to say"

st.set_page_config(page_title="Financial Resilience Map", page_icon="💵")


@st.cache_resource(show_spinner="Loading model...")
def get_predictor():
    return FragilityPredictor.from_artifacts()


def blank_answer(f):
    return None if f.kind == "number" else SKIP


def load_example():
    """Copy the chosen example profile into the form widgets."""
    profile = EXAMPLE_PROFILES.get(st.session_state["example"], {})
    for f in INPUT_FIELDS:
        st.session_state[f.key] = profile.get(f.key, blank_answer(f))


# Widgets read their starting values from session state, which also lets
# the example menu fill in the form.
for _f in INPUT_FIELDS:
    st.session_state.setdefault(_f.key, blank_answer(_f))


try:
    predictor = get_predictor()
except Exception as exc:  # show the reason instead of a blank page
    st.error(f"The model could not be loaded: {exc}")
    st.stop()


st.title("Financial Resilience Map")
st.write(
    "Answer a few questions to estimate how likely a household is to be "
    "**financially fragile**: unable to cover a $400 emergency expense with "
    "cash or its equivalent."
)

with st.sidebar:
    st.header("Try an example")
    st.selectbox(
        "Load a sample profile",
        ["(none)", *EXAMPLE_PROFILES],
        key="example",
        on_change=load_example,
    )
    st.header("About the model")
    st.write(
        f"**Model:** `{predictor.metadata['model']}`  \n"
        f"**Threshold:** {predictor.threshold:.2f}  \n"
        f"**Inputs used by the model:** {len(predictor.features)}"
    )
    st.caption(
        "Trained on the Federal Reserve's 2025 Survey of Household "
        "Economics and Decisionmaking (SHED). Questions you skip, and "
        "survey inputs the form does not ask about, are filled in from "
        "the training respondents whose answers are most like yours."
    )


answers = {}
with st.form("questionnaire"):
    sections = {"finances": "Your finances", "household": "Your household"}
    for section, title in sections.items():
        st.subheader(title)
        for f in INPUT_FIELDS:
            if f.section != section:
                continue
            if f.kind == "number":
                low, high = predictor.number_range(f.key)
                answers[f.key] = st.number_input(
                    f.label,
                    min_value=low,
                    max_value=high,
                    step=1,
                    placeholder=SKIP,
                    key=f.key,
                )
            else:
                choice = st.selectbox(
                    f.label,
                    [SKIP, *predictor.options(f.key)],
                    key=f.key,
                )
                answers[f.key] = None if choice == SKIP else choice

    submitted = st.form_submit_button("Predict", type="primary")


if submitted:
    try:
        result = predictor.predict(answers)
    except (KeyError, ValueError) as exc:
        st.error(f"Could not generate a prediction: {exc}")
        st.stop()

    st.header("Prediction")
    col1, col2 = st.columns(2)
    col1.metric("Probability of financial fragility",
                f"{result.probability:.0%}")
    col2.metric("Classification", result.label)
    st.progress(result.probability)

    if result.is_fragile:
        st.warning(
            f"The model's estimate is at or above the {result.threshold:.0%} "
            "threshold, so this household is classified as financially "
            "fragile."
        )
    else:
        st.success(
            f"The model's estimate is below the {result.threshold:.0%} "
            "threshold, so this household is classified as not financially "
            "fragile."
        )

    if result.n_answered == 0:
        st.info(
            "No questions were answered, so this is the prediction for a "
            "typical survey respondent."
        )

    with st.expander("Details"):
        st.write(
            f"You answered {result.n_answered} of {len(INPUT_FIELDS)} "
            f"questions. The model uses {result.n_features} survey inputs; "
            "the rest are filled in from the 100 training respondents whose "
            "answers are most like yours, then the saved preprocessing and "
            "model pipeline is applied."
        )
        st.json(result.answered)

    st.caption(
        "This is a statistical estimate from survey data, not financial "
        "advice or an assessment of any individual."
    )
