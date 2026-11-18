"""Financial Resilience Map UI.

Presentation-only redesign: inference logic and question definitions are unchanged.
"""
import logging
import sqlite3

import pandas as pd
import streamlit as st

from database import get_predictions, initialize_database, save_prediction
from src.inference import EXAMPLE_PROFILES, INPUT_FIELDS, FragilityPredictor

logger = logging.getLogger(__name__)
HISTORY_LIMIT = 10
CLASS_LABELS = {1: "Financially fragile", 0: "Not financially fragile"}
SECTIONS = {
    "finances": ("Your finances", ""),
    "household": ("Your household", ""),
}

CSS = """
<style>
:root { color-scheme: light; }
.stApp { background: #F7F8FA; color: #182B44; }
.block-container { max-width: 1050px; padding-top: 1.6rem; padding-bottom: 4rem; }
h1,h2,h3 { color: #172B45 !important; letter-spacing: -.025em; }
h1 { font-size: 2.5rem !important; }
[data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid #E5E9EF; }
[data-testid="stSidebar"] .block-container { padding-top: 2rem; }
[data-testid="stForm"] { background: white; border: 1px solid #E5E9EF; border-radius: 16px; padding: 1.6rem; box-shadow: 0 5px 24px rgba(25,47,73,.035); }
[data-testid="stWidgetLabel"] p { font-weight: 560; color: #25364E; }
[data-baseweb="select"] > div, [data-testid="stTextInput"] input { border-radius: 9px; background: #FCFCFD; }
.stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] button { border-radius: 9px; font-weight: 650; min-height: 2.8rem; }
[data-testid="stMetric"] { background: #fff; border: 1px solid #E5E9EF; border-radius: 14px; padding: 1.25rem; }
[data-testid="stMetricValue"] { color: #172B45; }
.hero { background: #172B45; border-radius: 18px; padding: 2.1rem 2.3rem; margin: 1rem 0 1.5rem; }
.hero h1 { color: white !important; font-size: 2.2rem !important; line-height:1.15; margin:.55rem 0 .7rem; }
.hero p { color:#E3EAF2; max-width:660px; margin:0; font-size:1.05rem; }
</style>
"""


@st.cache_resource(show_spinner="Loading the prediction model…")
def get_predictor():
    return FragilityPredictor.from_artifacts()


@st.cache_resource(show_spinner=False)
def setup_database():
    initialize_database()


def save_to_history(result):
    try:
        setup_database()
        save_prediction(int(result.is_fragile), result.probability)
    except (sqlite3.Error, ValueError):
        logger.exception("Could not save the prediction")
        st.error("The prediction could not be saved to the history. "
                 "It is still shown below. See the terminal for details.")


def render_history():
    st.divider()
    st.header("Recent predictions")
    try:
        setup_database()
        records = get_predictions(limit=HISTORY_LIMIT)
    except sqlite3.Error:
        logger.exception("Could not load the prediction history")
        st.error("The prediction history could not be loaded. See the terminal for details.")
        return
    if not records:
        st.caption("No predictions saved yet. Each prediction appears here after you select Predict.")
        return

    history = pd.DataFrame(records, columns=["ID", "prediction", "probability", "Saved at (UTC)"])
    history["Probability"] = history["probability"].map(
        lambda p: "Not recorded" if pd.isna(p) else f"{p:.1%}")
    history["Classification"] = history["prediction"].map(CLASS_LABELS)
    st.dataframe(history[["ID", "Saved at (UTC)", "Probability", "Classification"]],
                 hide_index=True, use_container_width=True)
    st.caption(f"Showing the last {HISTORY_LIMIT} predictions made in this app, newest first. "
               "Each entry stores only the probability, the classification, and the time, "
               "never your answers.")


def load_profile():
    profile = EXAMPLE_PROFILES.get(st.session_state.get("profile"), {})
    for field in INPUT_FIELDS:
        answer = profile.get(field.key)
        if field.kind == "number":
            answer = "" if answer is None else str(answer)
        st.session_state[field.key] = answer


def clear_answers():
    st.session_state["profile"] = None
    load_profile()
    st.session_state.pop("last_result", None)


def validate_answers(raw_answers, predictor):
    answers, errors = {}, []
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
                errors.append(f"{field.label} must be a whole number from {low} to {high}.")
                continue
        elif value not in predictor.options(field.key):
            errors.append(f"Choose a listed answer for: {field.label}")
            continue
        answers[field.key] = value
    return answers, errors


def render_field(field, predictor):
    if field.kind == "number":
        low, high = predictor.number_range(field.key)
        return st.text_input(field.label, placeholder=f"Whole number from {low} to {high}",
                             help=field.help or None, key=field.key)
    return st.selectbox(field.label, predictor.options(field.key), index=None,
                        placeholder="Choose an answer", help=field.help or None, key=field.key)


def main():
    st.set_page_config(page_title="Financial Resilience Map",
                       layout="wide", initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown('''<div class="hero"><h1>Financial Resilience Map</h1>
    <p>Estimate the probability that a household could not cover a $400
    emergency expense with cash or its equivalent.</p></div>''', unsafe_allow_html=True)

    try:
        predictor = get_predictor()
    except Exception:
        logger.exception("Could not initialize the predictor")
        st.error("The prediction model could not load. Verify the dependencies, processed data, "
                 "and model metadata. See the terminal for details.")
        return

    with st.sidebar:
        st.header("Sample profiles")
        st.selectbox("Fill the questionnaire with an example", EXAMPLE_PROFILES,
                     index=None, placeholder="Choose a sample profile",
                     key="profile", on_change=load_profile)
        st.button("Clear answers", on_click=clear_answers, use_container_width=True)

    st.caption(f"Answer all {len(INPUT_FIELDS)} questions to request a prediction.")

    raw_answers = {}
    with st.form("questionnaire"):
        for section, (title, description) in SECTIONS.items():
            fields = [f for f in INPUT_FIELDS if f.section == section]
            if not fields:
                continue
            st.markdown(f"### {title}")
            if description:
                st.caption(description)
            for i in range(0, len(fields), 2):
                cols = st.columns(2, gap="large")
                for col, field in zip(cols, fields[i:i+2]):
                    with col:
                        raw_answers[field.key] = render_field(field, predictor)
            st.divider()
        submitted = st.form_submit_button("Predict", type="primary",
                                          use_container_width=True)

    if submitted:
        answers, errors = validate_answers(raw_answers, predictor)
        if errors:
            st.error("Complete or correct these answers, then select Predict again.")
            for error in errors:
                st.write(f"- {error}")
            st.session_state.pop("last_result", None)
        else:
            try:
                st.session_state["last_result"] = predictor.predict(answers)
            except (KeyError, ValueError):
                logger.exception("Prediction failed")
                st.session_state.pop("last_result", None)
                st.error("The model could not score these answers. Review the inputs and retry.")
            else:
                # Saved only here, so reruns from other widgets do not save it again
                save_to_history(st.session_state["last_result"])

    result = st.session_state.get("last_result")
    if result is not None:
        st.divider()
        st.header("Prediction")
        left, right = st.columns(2, gap="medium")
        with left:
            st.metric("Estimated probability of financial fragility", f"{result.probability:.1%}")
        with right:
            st.metric("Classification", result.label)
        st.progress(float(max(0, min(1, result.probability))))
        st.caption(f"The model classifies scores of {result.threshold:.1%} or higher "
                   "as financially fragile.")
        st.caption("The model uses 425 survey inputs. For the ones not asked "
                   "here, the app takes the 100 most similar survey respondents, "
                   "scores each of them with your answers, and averages the "
                   "results.")

    render_history()


if __name__ == "__main__":
    main()
