
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

# Reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "mood_training_data.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_FILE = MODEL_DIR / "mood_genre_model.keras"
LABELS_FILE = MODEL_DIR / "genre_labels.json"

GENRES = [
    "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
    "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western",
]


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Training CSV not found: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    if list(df.columns) != ["text", "genres"]:
        raise ValueError(
            "CSV must contain exactly these columns: text, genres"
        )

    if df.empty or df.isna().any().any():
        raise ValueError("CSV is empty or contains missing values.")

    if df["text"].duplicated().any():
        raise ValueError("Training text contains duplicates.")

    # Convert each genre list into an 18-element multi-label vector.
    def encode_genres(value):
        labels = str(value).split("|")
        unknown = set(labels) - set(GENRES)

        if unknown or not labels:
            raise ValueError(f"Invalid genre labels: {value}")

        return [1.0 if genre in labels else 0.0 for genre in GENRES]

    x = df["text"].astype(str).to_numpy()
    y = np.array(
        [encode_genres(value) for value in df["genres"]],
        dtype=np.float32,
    )

    # Hold out 20% of the examples for final testing.
    indices = np.random.permutation(len(x))
    test_size = max(1, int(len(x) * 0.2))
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    x_train, y_train = x[train_indices], y[train_indices]
    x_test, y_test = x[test_indices], y[test_indices]

    # Convert text into numerical TF-IDF features.
    vectorizer = layers.TextVectorization(
        max_tokens=5000,
        output_mode="tf_idf",
        standardize="lower_and_strip_punctuation",
    )
    vectorizer.adapt(x_train)

    # One output probability per genre.
    inputs = tf.keras.Input(shape=(), dtype=tf.string)
    features = vectorizer(inputs)
    hidden = layers.Dense(64, activation="relu")(features)
    hidden = layers.Dropout(0.2)(hidden)
    outputs = layers.Dense(len(GENRES), activation="sigmoid")(hidden)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
        ],
    )

    print(f"Total examples: {len(x)}")
    print(f"Training examples: {len(x_train)}")
    print(f"Test examples: {len(x_test)}")
    print(f"Genres: {len(GENRES)}")
    print("\nTraining TensorFlow model...")

    model.fit(
        x_train,
        y_train,
        epochs=30,
        batch_size=16,
        validation_split=0.2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=4,
                restore_best_weights=True,
            )
        ],
        verbose=1,
    )

    print("\nEvaluation on unseen test examples:")
    results = model.evaluate(x_test, y_test, return_dict=True, verbose=0)

    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_FILE)

    with open(LABELS_FILE, "w", encoding="utf-8") as file:
        json.dump(GENRES, file, indent=2)

    print(f"\nSaved model: {MODEL_FILE}")
    print(f"Saved genre labels: {LABELS_FILE}")
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
