# Mood-to-Movie

Mood-to-Movie is an educational machine-learning prototype that turns a short natural-language mood or movie preference into two likely movie genres, then recommends highly rated matching movies from a local MovieLens catalogue.

For example, **“I want a terrifying movie with lots of suspense”** predicts **Horror** and **Thriller**, then ranks Horror/Thriller MovieLens titles using their real ratings and rating counts.

## Problem and features

Choosing a film from a large catalogue can be difficult when a person knows the feeling or experience they want rather than a title. The project provides:

- Natural-language, multi-label genre prediction with TF-IDF and Logistic Regression.
- Top-two genre selection for every non-empty request.
- Both-genre matching first, with either-genre fallback if too few movies match.
- Bayesian-style ranking using actual MovieLens average ratings and rating counts.
- Duplicate-title removal and a preference for titles with at least ten ratings.
- A simple Streamlit interface and a reusable command-line API.

This is not a production mood-understanding system and does not claim to understand human emotion perfectly.

## Architecture

```text
User text → TF-IDF vectorizer → One-vs-Rest Logistic Regression → top 2 Genre enum values
                                                                  ↓
MovieLens catalogue → genre match → weighted rating rank → Streamlit recommendations
```

## Data and model

`mood_training_data.csv` contains manually/AI-labelled text examples paired with one or more movie genres. It is used only to train the genre classifier. `movies.csv` is a prepared MovieLens catalogue containing genres and actual rating aggregates; MovieLens does not provide mood labels. Predicted genres filter that catalogue, and its real ratings/rating counts determine rank.

The final classifier is `TfidfVectorizer` (word unigrams and bigrams) plus `OneVsRestClassifier(LogisticRegression)`. It persists `models/tfidf_vectorizer.joblib`, `models/logistic_genre_classifier.joblib`, and `models/genre_labels.json`; the app loads these rather than retraining for each request.

An earlier TensorFlow experiment was compared on the same deterministic 80/20 held-out split before the final project was cleaned. It recorded the following comparison:

| Metric | TensorFlow | Logistic Regression |
| --- | ---: | ---: |
| Best-threshold Micro F1 | 0.415 | 0.540 |
| Best-threshold Macro F1 | 0.320 | 0.474 |
| Best-threshold Sample F1 | 0.411 | 0.522 |
| Top-2 Micro F1 | 0.416 | 0.528 |
| Top-2 Sample F1 | 0.421 | 0.536 |

These are configuration-comparison results on a held-out split, not an unbiased final benchmark. Logistic Regression was selected because it performed better in this experiment and keeps the application lightweight.

Movie ranking uses `v/(v+m) * R + m/(v+m) * C`, where `R` is a movie’s average rating, `v` its rating count, `C` the catalogue-wide mean, and `m=20`.

## Project structure

```text
mood-to-movie/
├── app.py                       # Streamlit UI
├── baseline_model.py            # trains and saves final model artifacts
├── genre_predictor.py           # persisted model loader/predictor
├── genres.py                    # shared Genre enum
├── recommend.py                 # matching, ranking, CLI
├── mood_training_data.csv
├── movies.csv
├── models/                      # final persisted model artifacts
├── prepare_movies.py            # regenerates MovieLens catalogue
├── test_recommendation_flow.py  # automated end-to-end checks
├── requirements.txt
└── requirements-data-preparation.txt # optional MovieLens catalogue dependencies
```

The final repository excludes the obsolete TensorFlow genre model, emotion router, diagnostics, and unrelated rating-preparation experiment.

## Install and run (Windows PowerShell)

```powershell
cd "C:\fastapi-practice\Interview Prep\mood-to-movie"
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python baseline_model.py
streamlit run app.py
```

Then open the local URL Streamlit prints (normally `http://localhost:8501`). The checked-in `movies.csv` and final model artifacts make the final command sufficient after installation. If model artifacts are missing, run `python baseline_model.py`.

To use the command line instead, run `python recommend.py`.

To regenerate the catalogue (requires a dataset download):

```powershell
pip install -r requirements-data-preparation.txt
python prepare_movies.py
python baseline_model.py
```

## Tests

```powershell
python -m unittest -v test_recommendation_flow.py
```

The tests cover all six example prompts, two `Genre` enum predictions, matching/ranking results, no duplicate titles, requested result counts, fallback behavior, and empty input.

## Example prompts

| Prompt | Predicted top genres |
| --- | --- |
| I feel sad and lonely today. | Drama, Comedy |
| I want a terrifying movie with lots of suspense. | Horror, Thriller |
| I am happy and want something funny. | Comedy, Drama |
| I want to explore a magical world. | Fantasy, Adventure |
| I want to solve a complicated murder mystery. | Mystery, Thriller |
| I want to watch a futuristic space adventure. | Sci-Fi, Adventure |

Actual movie recommendations vary only if the local catalogue is regenerated.

## Screenshots

No screenshots are included yet. To add genuine screenshots, run the Streamlit app, submit a prompt, save the browser capture under `screenshots/` (for example `screenshots/recommendations.png`), and add it here:

```markdown
![Mood-to-Movie recommendations](screenshots/recommendations.png)
```

## Limitations and future work

- The labelled training set is relatively small and its labels are manually/AI assigned.
- Genre predictions are a useful approximation, not a diagnosis of a user’s emotional state.
- Recommendations use only titles, genres, and rating aggregates available in the local catalogue.
- Future work could add user feedback, richer metadata, model calibration, data versioning, and a larger human-reviewed training set.
