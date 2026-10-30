"""MovieLens recommendation and command-line interface."""

from pathlib import Path

import pandas as pd

from genre_predictor import GenrePredictor

BASE_DIR = Path(__file__).resolve().parent
MOVIES_FILE = BASE_DIR / "movies.csv"


def load_movies() -> pd.DataFrame:
    if not MOVIES_FILE.exists():
        raise FileNotFoundError(f"Movie catalogue was not found: {MOVIES_FILE}")
    required = {"title", "genres", "average_rating", "rating_count"}
    movies = pd.read_csv(MOVIES_FILE)
    missing = required - set(movies.columns)
    if missing:
        raise ValueError(f"Movie catalogue is missing columns: {', '.join(sorted(missing))}")
    return movies


def recommend_movies(user_text: str, top_n: int = 10, predictor: GenrePredictor | None = None) -> dict:
    """Predict two genres, then filter and rank real MovieLens ratings."""
    if not isinstance(user_text, str) or not user_text.strip():
        return {"genres": [], "movies": [], "message": "Please describe your mood or movie preference."}
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")
    selected_genres = (predictor or GenrePredictor()).predict(user_text, k=2)
    wanted = {genre.value for genre in selected_genres}
    movies = load_movies().copy()
    movies["genre_set"] = movies["genres"].fillna("").map(lambda value: {part.strip() for part in value.split(",") if part.strip()})
    matching = movies[movies["genre_set"].map(wanted.issubset)].copy()
    match_type = "both genres"
    if len(matching) < top_n:
        matching = movies[movies["genre_set"].map(lambda genres: bool(genres & wanted))].copy()
        match_type = "either genre"
    if matching.empty:
        return {"genres": [genre.value for genre in selected_genres], "movies": [], "message": "No matching movies found."}

    global_mean, minimum_votes = movies["average_rating"].mean(), 20
    matching["ranking_score"] = (matching["rating_count"] / (matching["rating_count"] + minimum_votes) * matching["average_rating"] + minimum_votes / (matching["rating_count"] + minimum_votes) * global_mean)
    reliable = matching[matching["rating_count"] >= 10]
    if len(reliable) >= top_n:
        matching = reliable
    selected = matching.sort_values("ranking_score", ascending=False).drop_duplicates("title").head(top_n)
    return {"genres": [genre.value for genre in selected_genres], "movies": selected[["title", "genres", "average_rating", "rating_count", "ranking_score"]].to_dict(orient="records"), "message": None, "match_type": match_type}


def main() -> None:
    print("Mood-to-Movie is ready. Type 'exit' to stop.")
    while True:
        text = input("\nDescribe your mood or the movie you want: ")
        if text.strip().lower() == "exit":
            return
        try:
            result = recommend_movies(text)
        except (FileNotFoundError, ValueError, RuntimeError) as error:
            print(f"Unable to recommend movies: {error}")
            continue
        if result["message"]:
            print(result["message"])
            continue
        print("Predicted genres:", ", ".join(result["genres"]))
        for movie in result["movies"]:
            print(f"{movie['title']} | {movie['genres']} | {movie['average_rating']:.2f}/5 ({movie['rating_count']} ratings)")


if __name__ == "__main__":
    main()
