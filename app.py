"""Streamlit interface for the saved email spam classifier."""

import streamlit as st

from src.predict import classify_message, load_artifacts

st.set_page_config(page_title="Email Spam Classifier", page_icon="✉", layout="centered")
st.title("Email Spam Classifier")
st.write("Check a message for spam using a machine-learning model trained on the UCI SMS Spam Collection.")


@st.cache_resource(show_spinner=False)
def get_artifacts():
    return load_artifacts()


if "message_text" not in st.session_state:
    st.session_state.message_text = ""
if "classification_result" not in st.session_state:
    st.session_state.classification_result = None


def clear_results() -> None:
    st.session_state.message_text = ""
    st.session_state.classification_result = None


st.text_area(
    "Email or message",
    key="message_text",
    height=240,
    placeholder="Paste the message you want to check...",
)
classify_column, clear_column = st.columns([1, 1])
with classify_column:
    classify_clicked = st.button("Classify message", type="primary", use_container_width=True)
with clear_column:
    st.button("Clear", on_click=clear_results, use_container_width=True)

if classify_clicked:
    try:
        model, vectorizer = get_artifacts()
        st.session_state.classification_result = classify_message(
            st.session_state.message_text,
            model,
            vectorizer,
        )
    except (FileNotFoundError, RuntimeError, ValueError, TypeError) as error:
        st.session_state.classification_result = None
        st.error(str(error))

result = st.session_state.classification_result
if result is not None:
    label, confidence = result
    if label == "SPAM":
        st.error("Prediction: SPAM")
    else:
        st.success("Prediction: NOT SPAM")
    if confidence is not None:
        st.metric("Model confidence", f"{confidence:.1%}")
    else:
        st.caption("This model does not provide a probability score.")
