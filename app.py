"""Streamlit interface for Mood-to-Movie."""

import streamlit as st

from genre_predictor import GenrePredictor, ModelNotReadyError
from recommend import recommend_movies

st.set_page_config(page_title="Mood-to-Movie", page_icon="🎬", layout="centered")


@st.cache_resource
def get_predictor() -> GenrePredictor:
    return GenrePredictor()


st.title("🎬 Mood-to-Movie")
st.write("Describe how you feel or the kind of movie you want, and discover MovieLens titles that fit.")
text = st.text_area("What would you like to watch?", placeholder="I'm feeling adventurous and want to explore a magical world...", height=120)

if st.button("Recommend Movies", type="primary"):
    if not text.strip():
        st.info("Please describe your mood or movie preference first.")
    else:
        try:
            with st.spinner("Finding movies for you..."):
                result = recommend_movies(text, predictor=get_predictor())
        except (ModelNotReadyError, FileNotFoundError, ValueError) as error:
            st.error(f"Mood-to-Movie is not ready: {error}")
        else:
            if result["message"]:
                st.info(result["message"])
            else:
                st.subheader("Predicted genres")
                st.write(" · ".join(result["genres"]))
                st.caption(f"Recommendations matched by {result['match_type']} and ranked with actual MovieLens ratings.")
                st.subheader("Recommendations")
                for movie in result["movies"]:
                    with st.container(border=True):
                        st.markdown(f"**{movie['title']}**")
                        st.write(movie["genres"])
                        st.caption(f"⭐ {movie['average_rating']:.2f}/5  •  {movie['rating_count']} ratings")
