
import pandas as pd
import tensorflow_datasets as tfds

DATASET_NAME = "movielens/100k-ratings"
OUTPUT_FILE = "movies.csv"

print("Loading MovieLens dataset...")

ratings_df = tfds.as_dataframe(
    tfds.load(DATASET_NAME, split="train")
)

# Decode text columns if TFDS returns byte strings.
for column in ["movie_id", "movie_title"]:
    ratings_df[column] = ratings_df[column].apply(
        lambda value: value.decode("utf-8")
        if isinstance(value, bytes)
        else str(value)
    )

# Build a movie catalogue from the ratings dataset.
movies_df = ratings_df[
    ["movie_id", "movie_title", "movie_genres"]
].drop_duplicates(subset=["movie_id"]).copy()

# Convert genre IDs into readable genre names.
builder = tfds.builder(DATASET_NAME)
genre_feature = builder.info.features["movie_genres"].feature

def genre_names(genre_ids):
    names = []
    for genre_id in genre_ids:
        name = genre_feature.int2str(int(genre_id))
        if name == "no genres listed":
            continue
        names.append(name)
    return ", ".join(names)

movies_df["genres"] = movies_df["movie_genres"].apply(genre_names)
movies_df["genre_ids"] = movies_df["movie_genres"].apply(
    lambda ids: ",".join(str(int(i)) for i in ids)
)

# Calculate actual average ratings and rating counts per movie.
rating_stats = (
    ratings_df.groupby("movie_id")["user_rating"]
    .agg(average_rating="mean", rating_count="count")
    .reset_index()
)

movies_df = movies_df.merge(
    rating_stats,
    on="movie_id",
    how="left",
)

# Keep only the columns our recommendation system needs.
movies_df = movies_df[
    [
        "movie_id",
        "movie_title",
        "genres",
        "genre_ids",
        "average_rating",
        "rating_count",
    ]
].rename(columns={"movie_title": "title"})

movies_df["average_rating"] = movies_df["average_rating"].round(2)
movies_df["rating_count"] = movies_df["rating_count"].astype(int)

movies_df.to_csv(OUTPUT_FILE, index=False)

print(f"\nSaved {len(movies_df)} movies to {OUTPUT_FILE}")
print("\nSample movies:")
print(movies_df.head(5).to_string(index=False))

print("\nRating statistics:")
print(f"Average rating range: {movies_df['average_rating'].min():.2f}–"
      f"{movies_df['average_rating'].max():.2f}")
print(f"Rating count range: {movies_df['rating_count'].min()}–"
      f"{movies_df['rating_count'].max()}")
