# Mood-to-Movie

Mood-to-Movie is a small machine-learning project that helps users find movies based on how they feel or what they want to watch.

The user writes a short message, such as: **"I want a terrifying movie with lots of suspense."** The app predicts the two most suitable genres, such as **Horror** and **Thriller**, and then suggests highly rated movies from the MovieLens catalogue.

This is an educational project. It does not fully understand human emotions, but it gives useful movie suggestions based on the words in the user's message.

## Features

- Accepts a mood or movie preference written in normal language.
- Predicts the top two movie genres.
- Finds movies that match both genres first.
- Uses movies matching either genre if there are not enough matches.
- Ranks movies using real MovieLens ratings and rating counts.
- Avoids duplicate movie titles.
- Includes a simple Streamlit web interface.

## How it works

```text
User message
    -> TF-IDF + Logistic Regression model
    -> Top 2 predicted genres
    -> MovieLens genre matching
    -> Rating-based movie ranking
    -> Movie recommendations
```

## Data and model

`mood_training_data.csv` contains example messages and their manually/AI-labelled movie genres. This file is used to train the genre prediction model.

`movies.csv` contains MovieLens movie titles, genres, average ratings, and rating counts. MovieLens does not contain mood labels. The app uses the predicted genres to filter movies, then uses the real MovieLens ratings to rank them.

The final model uses TF-IDF to turn text into numbers and One-vs-Rest Logistic Regression to predict more than one genre. The app uses the top two predicted genres for recommendations.

The trained model files are stored in the `models/` folder, so the app does not need to train the model every time it starts.

## Why Logistic Regression was selected

An earlier TensorFlow model was compared with Logistic Regression using the same 80/20 train-test split. Logistic Regression gave better results in this experiment.

| Metric | TensorFlow | Logistic Regression |
| --- | ---: | ---: |
| Best-threshold Micro F1 | 0.415 | 0.540 |
| Best-threshold Macro F1 | 0.320 | 0.474 |
| Best-threshold Sample F1 | 0.411 | 0.522 |
| Top-2 Micro F1 | 0.416 | 0.528 |
| Top-2 Sample F1 | 0.421 | 0.536 |

These numbers are from an experiment on one held-out split. They are useful for comparing the two models, but they are not a final benchmark.

## Movie ranking

The app does not create fake ratings. It uses the real average rating and rating count from `movies.csv`.

Movies with more ratings are usually more reliable, so the app uses a weighted rating formula. It also prefers movies with at least 10 ratings when enough options are available.

## Project structure

```text
mood-to-movie/
├── app.py                            # Streamlit web app
├── baseline_model.py                 # trains and saves the final model
├── genre_predictor.py                # loads the saved model and predicts genres
├── genres.py                         # Genre enum
├── recommend.py                      # movie matching, ranking, and CLI support
├── mood_training_data.csv            # labelled training examples
├── movies.csv                        # MovieLens movie catalogue
├── models/                           # saved model files
├── prepare_movies.py                 # creates movies.csv from MovieLens data
├── test_recommendation_flow.py       # automated tests
├── requirements.txt
└── requirements-data-preparation.txt # needed only to recreate movies.csv
```

## Installation and run (Windows PowerShell)

```powershell
cd "C:\fastapi-practice\Interview Prep\mood-to-movie"
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python baseline_model.py
streamlit run app.py
```

After running the last command, open the local link shown by Streamlit. It is usually `http://localhost:8501`.

## Other commands

Train the model again:

```powershell
python baseline_model.py
```

Run the tests:

```powershell
python -m unittest -v test_recommendation_flow.py
```

Run the command-line version:

```powershell
python recommend.py
```

Recreate `movies.csv` from MovieLens data:

```powershell
pip install -r requirements-data-preparation.txt
python prepare_movies.py
python baseline_model.py
```

## Example inputs

| User message | Predicted genres |
| --- | --- |
| I feel sad and lonely today. | Drama, Comedy |
| I want a terrifying movie with lots of suspense. | Horror, Thriller |
| I am happy and want something funny. | Comedy, Drama |
| I want to explore a magical world. | Fantasy, Adventure |
| I want to solve a complicated murder mystery. | Mystery, Thriller |
| I want to watch a futuristic space adventure. | Sci-Fi, Adventure |

## Limitations

- The training data is small and the labels are manually/AI assigned.
- Genre predictions are helpful suggestions, not a diagnosis of a user's emotion.
- Recommendations only use the titles, genres, ratings, and rating counts available in the local MovieLens catalogue.
