
import tensorflow_datasets as tfds
import numpy as np

dataset = tfds.load(
    "movielens/100k-ratings",
    split="train",
)

ratings = []
movie_ids = set()
genres = set()

for record in tfds.as_numpy(dataset):
    ratings.append(float(record["user_rating"]))
    movie_ids.add(record["movie_id"].decode("utf-8"))
    genres.update(record["movie_genres"].tolist())

print("Total ratings:", len(ratings))
print("Unique movies:", len(movie_ids))
print("Unique genre IDs:", sorted(genres))
print("Average rating:", round(np.mean(ratings), 2))

print("\nRating distribution:")
for rating in sorted(set(ratings)):
    print(f"{rating:.0f} stars: {ratings.count(rating)}")
