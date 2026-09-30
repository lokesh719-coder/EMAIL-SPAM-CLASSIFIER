"""Train, compare, visualize, and persist the spam classifier."""

from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

from src.data_preprocessing import PROJECT_ROOT, get_stopwords, load_dataset, preprocess_text
from src.evaluate_model import evaluate_models, save_confusion_matrices

MODELS_DIR = PROJECT_ROOT / "models"
VISUALIZATIONS_DIR = PROJECT_ROOT / "visualizations"


def _feature_scores(model, feature_count: int) -> np.ndarray:
    """Return weights oriented so positive values favor the spam class."""
    classes = list(model.classes_)
    spam_index = classes.index("spam")
    ham_index = classes.index("ham")
    if hasattr(model, "feature_log_prob_"):
        scores = model.feature_log_prob_[spam_index] - model.feature_log_prob_[ham_index]
    elif hasattr(model, "coef_"):
        coefficients = model.coef_
        if hasattr(coefficients, "toarray"):
            coefficients = coefficients.toarray()
        scores = np.asarray(coefficients).reshape(-1)
        if spam_index == 0:
            scores = -scores
    else:
        raise TypeError("The selected classifier does not expose feature importance weights.")
    if len(scores) != feature_count:
        raise ValueError("Classifier feature weights do not match the fitted TF-IDF vocabulary.")
    return np.asarray(scores, dtype=float)


def save_feature_visualization(model, vectorizer: TfidfVectorizer, output_path: Path) -> None:
    """Plot the strongest spam- and ham-associated TF-IDF features."""
    feature_names = np.asarray(vectorizer.get_feature_names_out())
    scores = _feature_scores(model, len(feature_names))
    spam_indices = np.argsort(scores)[-15:]
    ham_indices = np.argsort(scores)[:15]

    figure, axes = plt.subplots(1, 2, figsize=(13, 7))
    axes[0].barh(feature_names[spam_indices], scores[spam_indices], color="#c2413b")
    axes[0].set_title("Features associated with SPAM")
    axes[0].set_xlabel("Relative model weight")
    axes[1].barh(feature_names[ham_indices], -scores[ham_indices], color="#247a62")
    axes[1].set_title("Features associated with HAM")
    axes[1].set_xlabel("Relative model weight")
    figure.suptitle("Most informative message features")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def train_and_save(dataset_path: str | Path | None = None) -> tuple[str, pd.DataFrame]:
    """Train three candidates and save the one with the best spam F1 score."""
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

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    try:
        train_features = vectorizer.fit_transform(train_text)
    except ValueError as error:
        raise ValueError(f"Unable to build TF-IDF features from the dataset: {error}") from error
    test_features = vectorizer.transform(test_text)

    models = {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Support Vector Machine": SVC(kernel="linear", probability=True, random_state=42),
    }
    features = {"train": train_features, "test": test_features}
    labels = {"train": train_labels, "test": test_labels}
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        results, predictions = evaluate_models(models, features, labels)

    # F1 is the selection target; remaining metrics make ties deterministic.
    ranking = results.sort_values(
        ["f1_score", "recall", "precision", "accuracy"], ascending=False
    ).reset_index(drop=True)
    selected_name = str(ranking.loc[0, "model"])
    selected_model = models[selected_name]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected_model, MODELS_DIR / "spam_classifier.pkl")
    joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.pkl")
    results.to_csv(VISUALIZATIONS_DIR / "model_comparison.csv", index=False)
    save_confusion_matrices(
        list(test_labels),
        predictions,
        VISUALIZATIONS_DIR / "confusion_matrices.png",
    )
    save_feature_visualization(
        selected_model,
        vectorizer,
        VISUALIZATIONS_DIR / "important_features.png",
    )

    print("Held-out test results (spam is the positive class):")
    print(ranking.to_string(index=False, formatters={
        column: "{:.4f}".format
        for column in ["accuracy", "precision", "recall", "f1_score"]
    }))
    print(f"\nSelected model by spam F1-score: {selected_name}")
    print(f"Saved classifier: {MODELS_DIR / 'spam_classifier.pkl'}")
    print(f"Saved vectorizer: {MODELS_DIR / 'tfidf_vectorizer.pkl'}")
    print(f"Charts and comparison table: {VISUALIZATIONS_DIR}")
    return selected_name, ranking


def main() -> None:
    try:
        train_and_save()
    except (FileNotFoundError, RuntimeError, ValueError, TypeError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
