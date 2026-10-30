"""Focused end-to-end tests for the persisted-model recommendation flow."""

import unittest

from genres import Genre
from genre_predictor import GenrePredictor
from recommend import recommend_movies


class RecommendationFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = GenrePredictor()

    def test_sample_inputs_return_two_enum_genres_and_movies(self):
        inputs = ["I feel sad and lonely today.", "I want a terrifying movie with lots of suspense.", "I am happy and want something funny.", "I want to explore a magical world.", "I want to solve a complicated murder mystery.", "I want to watch a futuristic space adventure."]
        for text in inputs:
            with self.subTest(text=text):
                genres = self.predictor.predict(text)
                self.assertEqual(len(genres), 2)
                self.assertTrue(all(isinstance(genre, Genre) for genre in genres))
                result = recommend_movies(text, top_n=5, predictor=self.predictor)
                self.assertIsNone(result["message"])
                self.assertEqual(len(result["movies"]), 5)
                self.assertEqual(len({movie["title"] for movie in result["movies"]}), 5)

    def test_empty_input_is_helpful(self):
        result = recommend_movies("   ", predictor=self.predictor)
        self.assertEqual(result["genres"], [])
        self.assertEqual(result["movies"], [])
        self.assertIn("Please describe", result["message"])

    def test_requested_count_and_fallback_are_respected(self):
        result = recommend_movies("I want a terrifying movie with lots of suspense.", top_n=3, predictor=self.predictor)
        self.assertEqual(len(result["movies"]), 3)
        self.assertIn(result["match_type"], {"both genres", "either genre"})

    def test_either_genre_fallback_when_no_title_has_both_genres(self):
        class FixedPredictor:
            def predict(self, text, k=2):
                return [Genre.ACTION, Genre.DOCUMENTARY]

        result = recommend_movies("anything", top_n=3, predictor=FixedPredictor())
        self.assertEqual(result["match_type"], "either genre")
        self.assertEqual(len(result["movies"]), 3)


if __name__ == "__main__":
    unittest.main()
