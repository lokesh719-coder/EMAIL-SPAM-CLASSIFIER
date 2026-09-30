"""Load saved artifacts and classify a single message."""

from __future__ import annotations

from pathlib import Path

import joblib

from src.data_preprocessing import PROJECT_ROOT, preprocess_text

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "spam_classifier.pkl"
DEFAULT_VECTORIZER_PATH = PROJECT_ROOT / "models" / "tfidf_vectorizer.pkl"


def load_artifacts(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    vectorizer_path: str | Path = DEFAULT_VECTORIZER_PATH,
):
    """Load the trained classifier and its matching TF-IDF vectorizer."""
    classifier_file = Path(model_path)
    vectorizer_file = Path(vectorizer_path)
    if not classifier_file.is_file():
        raise FileNotFoundError(
            f"Saved classifier is missing: {classifier_file}. Run `python -m src.train_model` first."
        )
    if not vectorizer_file.is_file():
        raise FileNotFoundError(
            f"Saved TF-IDF vectorizer is missing: {vectorizer_file}. Run `python -m src.train_model` first."
        )
    try:
        return joblib.load(classifier_file), joblib.load(vectorizer_file)
    except (OSError, ValueError, EOFError) as error:
        raise RuntimeError(f"Could not load the saved model files: {error}") from error


def classify_message(message: str, model, vectorizer) -> tuple[str, float | None]:
    """Classify a message and return its label plus confidence when supported."""
    if not isinstance(message, str):
        raise ValueError("Please enter message text as a string.")
    if not message.strip():
        raise ValueError("Please enter a message before classifying it.")

    cleaned_message = preprocess_text(message)
    features = vectorizer.transform([cleaned_message])
    label = str(model.predict(features)[0]).upper()
    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        predicted_label = label.lower()
        class_labels = [str(class_name).lower() for class_name in model.classes_]
        confidence = float(probabilities[class_labels.index(predicted_label)])
    return label, confidence


def predict_message(
    message: str,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    vectorizer_path: str | Path = DEFAULT_VECTORIZER_PATH,
) -> tuple[str, float | None]:
    """Load the persisted artifacts and classify a new message."""
    model, vectorizer = load_artifacts(model_path, vectorizer_path)
    return classify_message(message, model, vectorizer)
