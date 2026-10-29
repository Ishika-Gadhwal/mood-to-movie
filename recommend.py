
import pandas as pd

from genre_router import route_to_genres

MOVIES_FILE = "movies.csv"


def load_movies():
    return pd.read_csv(MOVIES_FILE)


def recommend_movies(user_text: str, top_n: int = 10) -> dict:
    selected_genres = route_to_genres(user_text)

    if not selected_genres:
        return {
            "emotion": None,
            "genres": [],
            "movies": [],
            "message": "Please describe your mood or movie preference.",
        }

    wanted_genres = [genre.value for genre in selected_genres]
    movies_df = load_movies()

    # Convert each movie's genres into a set for matching.
    movies_df["genre_set"] = movies_df["genres"].fillna("").apply(
        lambda value: set(value.split(", "))
    )
    wanted_set = set(wanted_genres)

    # Prefer movies containing every selected genre.
    matching = movies_df[
        movies_df["genre_set"].apply(lambda genres: wanted_set.issubset(genres))
    ].copy()

    # If too few matches exist, include movies matching any selected genre.
    if len(matching) < top_n:
        matching = movies_df[
            movies_df["genre_set"].apply(lambda genres: bool(genres & wanted_set))
        ].copy()

    if matching.empty:
        return {
            "emotion": None,
            "genres": wanted_genres,
            "movies": [],
            "message": "No matching movies found.",
        }

    # Rank using actual ratings, with a small adjustment for rating reliability.
    global_mean = movies_df["average_rating"].mean()
    minimum_votes = 20

    matching["ranking_score"] = (
        matching["rating_count"] / (matching["rating_count"] + minimum_votes)
        * matching["average_rating"]
        + minimum_votes / (matching["rating_count"] + minimum_votes)
        * global_mean
    )

    # Prefer movies with at least 10 ratings when enough candidates exist.
    reliable = matching[matching["rating_count"] >= 10]

    if len(reliable) >= top_n:
        matching = reliable

    # Avoid recommending the same movie title more than once.
    matching = matching.sort_values(
        "ranking_score", ascending=False
    ).drop_duplicates(subset=["title"])

    matching = matching.head(top_n)

    return {
        "emotion": None,  # Emotion is printed by the existing genre router.
        "genres": wanted_genres,
        "movies": matching[
            ["title", "genres", "average_rating", "rating_count", "ranking_score"]
        ].to_dict(orient="records"),
        "message": None,
    }


if __name__ == "__main__":
    print("Mood-to-Movie is ready!")

    while True:
        user_text = input(
            "\nHow are you feeling, or what kind of movie do you want? "
            "(type 'exit' to stop): "
        )

        if user_text.strip().lower() == "exit":
            break

        result = recommend_movies(user_text)

        if result["message"]:
            print(result["message"])
            continue

        print("\nSelected genres:", ", ".join(result["genres"]))
        print("\nMovie recommendations:\n")

        for movie in result["movies"]:
            print(
                f"{movie['title']} | "
                f"Genres: {movie['genres']} | "
                f"Rating: {movie['average_rating']:.2f}/5 | "
                f"Ratings: {movie['rating_count']}"
            )
