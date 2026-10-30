"""Persisted-model genre prediction for the Mood-to-Movie application."""

import json
from pathlib import Path

import joblib
import numpy as np

from genres import Genre

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
CLASSIFIER_PATH = MODEL_DIR / "logistic_genre_classifier.joblib"
LABELS_PATH = MODEL_DIR / "genre_labels.json"


class ModelNotReadyError(RuntimeError):
    """Raised when the persisted recommendation model has not been trained."""


class GenrePredictor:
    """Load artifacts once and return the highest-scoring Genre values."""

    def __init__(self) -> None:
        missing = [path.name for path in (VECTORIZER_PATH, CLASSIFIER_PATH, LABELS_PATH) if not path.exists()]
        if missing:
            raise ModelNotReadyError("Genre model files are missing (" + ", ".join(missing) + "). Run `python baseline_model.py`.")
        self.vectorizer = joblib.load(VECTORIZER_PATH)
        self.classifier = joblib.load(CLASSIFIER_PATH)
        try:
            self.labels = [Genre(label) for label in json.loads(LABELS_PATH.read_text(encoding="utf-8"))]
        except ValueError as error:
            raise ModelNotReadyError("genre_labels.json contains an unsupported genre.") from error

    def predict(self, text: str, k: int = 2) -> list[Genre]:
        if not isinstance(text, str) or not text.strip():
            return []
        if k < 1:
            raise ValueError("k must be at least 1.")
        scores = self.classifier.predict_proba(self.vectorizer.transform([text]))[0]
        indices = np.argsort(scores)[-min(k, len(self.labels)):][::-1]
        return [self.labels[index] for index in indices]
