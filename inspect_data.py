
import tensorflow_datasets as tfds

# # Load the MovieLens 100K ratings dataset
# dataset, info = tfds.load(
#     "movielens/100k-ratings",
#     split="train",
#     with_info=True,
#     as_supervised=False,
# )

# print("Dataset information:")
# print(info)

# print("\nFirst 3 rating records:")
# for record in dataset.take(3):
#     for key, value in record.items():
#         print(f"{key}: {value.numpy()}")
#     print("-" * 40)





# Load the dataset metadata
builder = tfds.builder("movielens/100k-ratings")
builder.download_and_prepare()

# Get the genre names from the dataset schema
genre_feature = builder.info.features["movie_genres"].feature

print("Genre ID -> Genre name")
for genre_id in range(genre_feature.num_classes):
    print(f"{genre_id} -> {genre_feature.int2str(genre_id)}")
