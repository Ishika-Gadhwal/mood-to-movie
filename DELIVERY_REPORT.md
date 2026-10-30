# Mood-to-Movie delivery report

## Completed work

- Replaced the final recommendation path’s keyword/emotion router with a persisted TF-IDF + One-vs-Rest Logistic Regression model.
- Added model artifact generation in `baseline_model.py` and runtime loading in `genre_predictor.py`.
- Updated `recommend.py` to keep MovieLens matching, both-genre preference, either-genre fallback, weighted real-rating ranking, reliable-rating preference, duplicate removal, and CLI reuse.
- Added the Streamlit interface in `app.py`.
- Added focused automated flow tests and a professional README.
- Updated dependency and ignore files. Removed obsolete TensorFlow/emotion/rating-preparation experiments after confirming that the final application does not import them.

## Final structure

Key application files are `app.py`, `baseline_model.py`, `genre_predictor.py`, `recommend.py`, `genres.py`, `movies.csv`, `mood_training_data.csv`, `models/`, and `test_recommendation_flow.py`. See `README.md` for the complete structure.

## Files changed

- Created: `app.py`, `genre_predictor.py`, `test_recommendation_flow.py`, `requirements-data-preparation.txt`, `README.md`, `DELIVERY_REPORT.md`.
- Updated: `baseline_model.py`, `recommend.py`, `.gitignore`, `requirements.txt`, `models/genre_labels.json`.
- Generated: `models/tfidf_vectorizer.joblib`, `models/logistic_genre_classifier.joblib`.
- Deleted: `train_genre_model.py`, `test_genre_model.py`, `evaluate_genre_model.py`, `diagnose_genre_model.py`, `genre_router.py`, `test_emotion.py`, `prepare_data.py`, `inspect_data.py`, `inspect_genre_distribution.py`, `explore_data.py`, `requirements-experiments.txt`, `models/mood_genre_model.keras`, and `__pycache__/`.

## Tests run

- `python baseline_model.py`: succeeded; Top-2 Micro F1 **0.528**, Top-2 Sample F1 **0.536** on the deterministic held-out comparison split.
- `python -m unittest -v test_recommendation_flow.py`: **4 tests passed**.
- The six requested sample prompts each returned the expected genre pairs and ten recommendations.
- `streamlit run app.py --server.headless=true --server.port=8502`: started successfully; the Streamlit health endpoint returned **HTTP 200 (ok)**.

## Commands

Launch:

```powershell
streamlit run app.py
```

Retrain:

```powershell
python baseline_model.py
```

## Remaining manual work

No model connection or edits are required after installation. Add genuine UI screenshots manually if desired; README contains clearly labelled instructions and does not claim that screenshots exist.

## Known limitations

The classifier uses a small manually/AI-labelled dataset and is an educational genre-prediction prototype, not robust emotion understanding.
