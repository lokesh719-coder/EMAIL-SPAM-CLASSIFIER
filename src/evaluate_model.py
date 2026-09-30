"""Model metrics and saved-model evaluation helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from src.data_preprocessing import PROJECT_ROOT, get_stopwords, load_dataset, preprocess_text


METRIC_COLUMNS = ["model", "accuracy", "precision", "recall", "f1_score"]


def evaluate_models(
    models: dict,
    features,
    labels,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Fit each candidate and return measured metrics and predictions."""
    records: list[dict[str, float | str]] = []
    predictions: dict[str, object] = {}
    for name, model in models.items():
        model.fit(features["train"], labels["train"])
        predicted = model.predict(features["test"])
        predictions[name] = predicted
        records.append(
            {
                "model": name,
                "accuracy": accuracy_score(labels["test"], predicted),
                "precision": precision_score(labels["test"], predicted, pos_label="spam", zero_division=0),
                "recall": recall_score(labels["test"], predicted, pos_label="spam", zero_division=0),
                "f1_score": f1_score(labels["test"], predicted, pos_label="spam", zero_division=0),
            }
        )
    return pd.DataFrame(records, columns=METRIC_COLUMNS), predictions


def save_confusion_matrices(
    labels: list[str],
    predictions: dict[str, object],
    output_path: Path,
) -> None:
    """Save a side-by-side confusion-matrix chart for all candidates."""
    figure, axes = plt.subplots(1, len(predictions), figsize=(5 * len(predictions), 4))
    if len(predictions) == 1:
        axes = [axes]
    for axis, (name, predicted) in zip(axes, predictions.items()):
        display = ConfusionMatrixDisplay(
            confusion_matrix(labels, predicted, labels=["ham", "spam"]),
            display_labels=["HAM", "SPAM"],
        )
        display.plot(ax=axis, cmap="Greens", colorbar=False, values_format="d")
        axis.set_title(name)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def evaluate_saved_model(
    dataset_path: str | Path | None = None,
    model_path: str | Path | None = None,
    vectorizer_path: str | Path | None = None,
) -> dict[str, float]:
    """Recreate the held-out split and report metrics for saved artifacts."""
    import joblib

    model_file = Path(model_path) if model_path else PROJECT_ROOT / "models" / "spam_classifier.pkl"
    vectorizer_file = (
        Path(vectorizer_path) if vectorizer_path else PROJECT_ROOT / "models" / "tfidf_vectorizer.pkl"
    )
    if not model_file.is_file():
        raise FileNotFoundError(f"Saved classifier not found: {model_file}. Train the model first.")
    if not vectorizer_file.is_file():
        raise FileNotFoundError(f"Saved TF-IDF vectorizer not found: {vectorizer_file}. Train the model first.")

    dataset = load_dataset(dataset_path)
    stop_words = get_stopwords()
    cleaned_text = dataset["text"].map(lambda message: preprocess_text(message, stop_words))
    train_text, test_text, train_labels, test_labels = train_test_split(
        cleaned_text,
        dataset["label"],
        test_size=0.2,
        random_state=42,
        stratify=dataset["label"],
    )
    del train_text, train_labels
    model = joblib.load(model_file)
    vectorizer = joblib.load(vectorizer_file)
    predicted = model.predict(vectorizer.transform(test_text))
    return {
        "accuracy": accuracy_score(test_labels, predicted),
        "precision": precision_score(test_labels, predicted, pos_label="spam", zero_division=0),
        "recall": recall_score(test_labels, predicted, pos_label="spam", zero_division=0),
        "f1_score": f1_score(test_labels, predicted, pos_label="spam", zero_division=0),
    }


def main() -> None:
    try:
        metrics = evaluate_saved_model()
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(pd.Series(metrics, name="saved classifier").to_string())


if __name__ == "__main__":
    main()
