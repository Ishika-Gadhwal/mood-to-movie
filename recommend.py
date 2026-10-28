
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds

from genre_router import route_to_genres


# Load the trained TensorFlow model and movie catalogue.
model = tf.keras.models.load_model("models/mood_movie_model.keras")
movies_df = pd.read_csv("movies.csv")

builder = tfds.builder("movielens/100k-ratings")
genre_feature = builder.info.features["movie_genres"].feature
num_genres = genre_feature.num_classes


def make_genre_vector(genre_ids):
    vector = np.zeros(num_genres, dtype=np.float32)

    if pd.isna(genre_ids) or not str(genre_ids).strip():
        return vector

    for genre_id in str(genre_ids).split(","):
        vector[int(genre_id)] = 1.0

    return vector


def recommend_movies(user_text, top_n=10):
    selected_genres = route_to_genres(user_text)

    if not selected_genres:
        print("Please describe your feelings or movie preference.")
        return

    wanted_genres = [genre.value for genre in selected_genres]

    # Match movies containing at least one selected genre.
    matching = movies_df[
        movies_df["genres"].apply(
            lambda genres: any(
                genre in str(genres).split(", ")
                for genre in wanted_genres
            )
        )
    ].copy()

    if matching.empty:
        print("No matching movies found.")
        return

    # Predict ratings using our trained TensorFlow model.
    X_movies = np.stack(
        matching["genre_ids"].apply(make_genre_vector).to_list()
    )

    matching["predicted_rating"] = model.predict(
        X_movies, verbose=0
    ).flatten()

    matching = matching.sort_values(
        "predicted_rating", ascending=False
    ).head(top_n)

    print(f"\nSelected genres: {', '.join(wanted_genres)}")
    print("\nMovie recommendations:\n")

    for _, movie in matching.iterrows():
        print(
            f"{movie['title']} | "
            f"Genres: {movie['genres']} | "
            f"Predicted rating: {movie['predicted_rating']:.2f}"
        )


if __name__ == "__main__":
    print("Mood-to-Movie is ready!")

    while True:
        user_text = input(
            "\nHow are you feeling, or what kind of movie do you want? "
            "(type 'exit' to stop): "
        )

        if user_text.strip().lower() == "exit":
            break

        recommend_movies(user_text)
