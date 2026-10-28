
import tensorflow_datasets as tfds
import numpy as np

# 1. Load the MovieLens dataset
dataset = tfds.load(
    "movielens/100k-ratings",
    split="train",
)

# 2. Get the genre information
builder = tfds.builder("movielens/100k-ratings")
genre_feature = builder.info.features["movie_genres"].feature
num_genres = genre_feature.num_classes

# 3. Prepare input features (X) and target ratings (y)
X = []
y = []

for record in tfds.as_numpy(dataset):
    # Create a vector of zeros for all genres
    genre_vector = np.zeros(num_genres, dtype=np.float32)

    # Mark each genre present in this movie with 1
    for genre_id in record["movie_genres"]:
        genre_vector[genre_id] = 1.0

    X.append(genre_vector)
    y.append(float(record["user_rating"]))

# 4. Convert lists into NumPy arrays
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

print("Data preparation complete!")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("\nFirst movie's genre vector:")
print(X[0])
print("First rating:", y[0])

np.savez("prepared_data.npz", X=X, y=y)
print("Saved prepared data to prepared_data.npz")