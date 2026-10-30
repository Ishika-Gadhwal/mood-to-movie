
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
    vectorizer = layers.TextVectorization( # TextVectorization is a TensorFlow/Keras layer that converts text into numerical representations.
        max_tokens=5000,  # This specifies the maximum vocabulary size. Vocabulary means the collection of unique words that the vectorizer learns from your training sentences.
        output_mode="tf_idf", # It tells TensorFlow to convert text into TF-IDF numerical values. TF-IDF stands for: TF = Term Frequency and IDF = Inverse Document Frequency. It gives importance to words based on their frequency and how distinctive they are across the dataset.
        standardize="lower_and_strip_punctuation", # Ex: I am scared! Is there a ghost? --> i am scared is there a ghost
    )
    vectorizer.adapt(x_train) # It tells the vectorizer to learn the vocabulary and calculate the IDF weights from your training text.

    # One output probability per genre.
    # output = LayerName(parameters)(previous_layer_output) -> syntax
    inputs = tf.keras.Input(shape=(), dtype=tf.string) # input layer
    features = vectorizer(inputs)                      # Text Vectorization layer
    hidden = layers.Dense(64, activation="relu")(features) # hidden layer 1 ->  This layer learns patterns from the numerical text features.  ReLU introduces non-linearity, allowing the network to learn more complex relationships between words and genres.
    hidden = layers.Dropout(0.2)(hidden) # Dropout hidden layer 2 -> Dropout is used to reduce overfitting. 64×0.2=12.8 means Approximately 13 neurons will be temporarily disabled during a training step
    outputs = layers.Dense(len(GENRES), activation="sigmoid")(hidden) # Output layer -> Creates 18 output neurons. Because each neuron represents one movie genre. Sigmoid converts each output into a probability between 0 and 1.

    model = tf.keras.Model(inputs=inputs, outputs=outputs) # connect all the layers together and create your complete neural network.

    # Before training the neural network, we need to configure how it should learn. (Compile the model)
    model.compile(
        optimizer="adam",  # The optimizer helps the model improve its predictions by adjusting the weights of its neurons.  Adam helps the model learn by updating its weights during training.
        loss="binary_crossentropy", # Binary crossentropy measures how far the predicted probabilities are from the correct 0/1 labels. If the prediction is close to the correct answer, the loss is low. If the prediction is far from the correct answer, the loss is high.
        metrics=[                    # Metrics are used to measure the model's performance.
            tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"), #  Binary Accuracy converts probabilities into 0 or 1 using a default threshold of 0.5. Then it compares these predictions with the actual labels.
        ],
    )

    print(f"Total examples: {len(x)}")
    print(f"Training examples: {len(x_train)}")
    print(f"Test examples: {len(x_test)}")
    print(f"Genres: {len(GENRES)}")
    print("\nTraining TensorFlow model...")

    model.fit( # fit() function is used to train a machine learning model.
        x_train, # Provides training sentences as input to the model
        y_train, # Provides the correct genre labels for those sentences
        epochs=30, # epoch means one complete pass through the entire training dataset. With repeated training, it adjusts its weights to improve its predictions. 30 is the maximum number of epochs here, not necessarily the number that will actually run. Early stopping can stop training sooner.
        batch_size=16, # Batch size determines how many examples the model learns from before updating its weights. After processing each batch, the model updates its weights using the optimizer (Adam).
        validation_split=0.2, # It means 20% of the training data will be reserved for validation. Training data: Teach the model, Validation data: Monitor learning and detect overfitting, Testing data: Evaluate final model performance
        callbacks=[ # Specifies additional functions to control the training process
            tf.keras.callbacks.EarlyStopping( # Automatically stops training when validation performance stops improving
                monitor="val_loss", # Tracks validation loss to check whether the model is improving
                patience=4, # Stops training if validation loss fails to improve for 4 consecutive epochs
                restore_best_weights=True, # Restores the model weights from the epoch with the lowest validation loss and highest val_binary_accuracy
            )
        ],
        verbose=1, # Displays a progress bar, training loss, and validation metrics, can take 3 values : 0: No training progress displayed , 1: Displays progress bar , 2: Displays one summary line per epoch
    )

    print("\nEvaluation on unseen test examples:")
    results = model.evaluate(x_test, y_test, return_dict=True, verbose=0)
    # model.evaluate() → Evaluates how well the trained model performs.
    # x_test → Provides unseen text sentences as input.
    # y_test → Provides the correct genre labels to compare against.
    # return_dict=True → Returns evaluation results in dictionary format (metric names and values).
    # verbose=0 → Hides the evaluation progress output.

    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_FILE) #  Saves trained neural network.

    with open(LABELS_FILE, "w", encoding="utf-8") as file:
        json.dump(GENRES, file, indent=2)  # Saves genre names in JSON format while preserving their order

    print(f"\nSaved model: {MODEL_FILE}")
    print(f"Saved genre labels: {LABELS_FILE}")
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
