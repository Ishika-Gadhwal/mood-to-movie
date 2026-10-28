
import tensorflow_datasets as tfds
import pandas as pd

# Load the MovieLens dataset
dataset = tfds.load(
    "movielens/100k-ratings",
    split="train",
)

# Get genre ID-to-name mapping
builder = tfds.builder("movielens/100k-ratings")
genre_feature = builder.info.features["movie_genres"].feature

movies = {}

for record in tfds.as_numpy(dataset):
    movie_id = record["movie_id"].decode("utf-8")
    title = record["movie_title"].decode("utf-8")
    genre_ids = record["movie_genres"].tolist()

    # Keep one entry per movie
    if movie_id not in movies:
        genres = [
            genre_feature.int2str(int(genre_id))
            for genre_id in genre_ids
        ]

        movies[movie_id] = {
            "movie_id": movie_id,
            "title": title,
            "genres": ", ".join(genres),
            "genre_ids": ",".join(map(str, genre_ids)),
        }

# Save the unique movie catalogue
df = pd.DataFrame(movies.values())
df.to_csv("movies.csv", index=False, encoding="utf-8")

print("Movie catalogue created!")
print("Unique movies:", len(df))
print("\nSample movies:")
print(df[["title", "genres"]].head(10).to_string(index=False))
print("\nSaved to movies.csv")
