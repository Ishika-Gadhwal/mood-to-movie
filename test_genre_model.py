
import json
from pathlib import Path

import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "models" / "mood_genre_model.keras"
LABELS_FILE = BASE_DIR / "models" / "genre_labels.json"

model = tf.keras.models.load_model(MODEL_FILE)

with open(LABELS_FILE, encoding="utf-8") as file:
    genres = json.load(file)

test_inputs = [
    "I feel sad and lonely today.",
    "I want a terrifying movie with lots of suspense.",
    "I am happy and want something funny.",
    "I want to explore a magical world.",
    "I want to solve a complicated murder mystery.",
    "I want to watch a futuristic space adventure.",
]

for text in test_inputs:
    probabilities = model.predict(
        tf.constant([text]), verbose=1
    )[0]

    ranked = sorted(
        zip(genres, probabilities),
        key=lambda item: item[1],
        reverse=True,
    )

    print(f"\nInput: {text}")
    print("Top genre predictions:")

    for genre, score in ranked[:3]:
        print(f"  {genre}: {score:.3f}")
