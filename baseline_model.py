"""Train and evaluate the final TF-IDF + Logistic Regression genre model."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.multiclass import OneVsRestClassifier

from genres import Genre

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "mood_training_data.csv"
MODEL_DIR = BASE_DIR / "models"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
CLASSIFIER_PATH = MODEL_DIR / "logistic_genre_classifier.joblib"
LABELS_PATH = MODEL_DIR / "genre_labels.json"
SEED = 42


def load_training_data(data_path: Path = DATA_PATH) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load and validate labelled examples, returning texts and multi-hot labels."""
    if not data_path.exists():
        raise FileNotFoundError(f"Training data was not found: {data_path}")
    frame = pd.read_csv(data_path)
    if not {"text", "genres"}.issubset(frame.columns) or frame.empty:
        raise ValueError("Training CSV must contain non-empty 'text' and 'genres' columns.")
    if frame[["text", "genres"]].isna().any().any():
        raise ValueError("Training CSV contains missing text or genre labels.")

    texts = frame["text"].astype(str).to_numpy()
    labels = [genre.value for genre in Genre if genre is not Genre.UNKNOWN]
    index = {label: position for position, label in enumerate(labels)}
    encoded = np.zeros((len(frame), len(labels)), dtype=int)
    for row, value in enumerate(frame["genres"].astype(str)):
        for label in (part.strip() for part in value.split("|")):
            if label not in index:
                raise ValueError(f"Unknown genre label in training data: {label!r}")
            encoded[row, index[label]] = 1
    return texts, encoded, labels


def train_and_save_model() -> dict[str, float]:
    """Fit the final model, report held-out metrics, and save application artifacts."""
    texts, y, labels = load_training_data()
    indices = np.random.RandomState(SEED).permutation(len(texts))
    test_size = max(1, int(len(texts) * 0.2))
    test_indices, train_indices = indices[:test_size], indices[test_size:]

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, stop_words="english")
    x_train = vectorizer.fit_transform(texts[train_indices])
    x_test = vectorizer.transform(texts[test_indices])
    classifier = OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED))
    classifier.fit(x_train, y[train_indices])

    scores = classifier.predict_proba(x_test)
    predicted = np.zeros_like(y[test_indices])
    for row, selected in enumerate(np.argsort(scores, axis=1)[:, -2:]):
        predicted[row, selected] = 1
    metrics = {
        "top_2_micro_f1": f1_score(y[test_indices], predicted, average="micro", zero_division=0),
        "top_2_sample_f1": f1_score(y[test_indices], predicted, average="samples", zero_division=0),
        "top_2_micro_precision": precision_score(y[test_indices], predicted, average="micro", zero_division=0),
        "top_2_micro_recall": recall_score(y[test_indices], predicted, average="micro", zero_division=0),
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(classifier, CLASSIFIER_PATH)
    LABELS_PATH.write_text(json.dumps(labels, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    metrics = train_and_save_model()
    print("Saved TF-IDF vectorizer, Logistic Regression classifier, and genre labels to models/.")
    print("Held-out split comparison metrics (not an unbiased final benchmark):")
    for name, value in metrics.items():
        print(f"  {name.replace('_', ' ')}: {value:.3f}")


if __name__ == "__main__":
    main()
