"""
Unit tests for DermaSense Feedback-Driven Adaptive Personalization Engine
"""

import os
import sys
import unittest
import tempfile
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.personalization.preference_manager import AdaptivePersonalizer


class TestAdaptivePersonalizer(unittest.TestCase):

    def setUp(self):
        # Use temporary file for isolated test state
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        self.temp_file.close()
        self.ap = AdaptivePersonalizer(storage_path=self.temp_file.name)

    def tearDown(self):
        if os.path.exists(self.temp_file.name):
            os.remove(self.temp_file.name)

    def test_cold_start_profile(self):
        """Tests that new users start with neutral multiplier 1.0."""
        profile = self.ap.get_or_create_profile("user_cold", skin_type="oily", concerns=["Acne & Blemish Control"])
        self.assertEqual(profile['interaction_count'], 0)
        
        mock_product = pd.Series({
            'product_name': 'Gentle Cleanser',
            'brand': 'CeraVe',
            'matched_actives': ['ceramide', 'hyaluronic acid']
        })
        mult = self.ap.compute_personalization_multiplier("user_cold", mock_product)
        self.assertEqual(mult, 1.0)

    def test_positive_feedback_adaptation(self):
        """Tests that liked products increase ingredient affinities and multiplier."""
        self.ap.record_feedback(
            user_id="user_pos",
            product_id="prod_101",
            feedback_type="like",
            brand="The Ordinary",
            actives=["niacinamide", "zinc pca"]
        )
        
        profile = self.ap.get_or_create_profile("user_pos")
        self.assertIn("niacinamide", profile['liked_ingredients'])
        self.assertGreater(profile['ingredient_affinities'].get('niacinamide', 0), 0)

        # Candidate containing liked active
        mock_product = pd.Series({
            'product_name': 'Niacinamide Serum',
            'brand': 'The Ordinary',
            'matched_actives': ['niacinamide']
        })
        mult = self.ap.compute_personalization_multiplier("user_pos", mock_product)
        self.assertGreater(mult, 1.0)

    def test_breakout_negative_avoidance(self):
        """Tests that reporting breakouts on a product penalizes and avoids its active ingredients."""
        self.ap.record_feedback(
            user_id="user_breakout",
            product_id="prod_irritant",
            feedback_type="dislike",
            reason="caused breakouts",
            brand="BrandX",
            actives=["salicylic acid"]
        )

        profile = self.ap.get_or_create_profile("user_breakout")
        self.assertIn("salicylic acid", profile['disliked_ingredients'])

        # Candidate containing breakout active
        mock_bad_product = pd.Series({
            'product_name': 'BHA Cleanser',
            'brand': 'BrandY',
            'matched_actives': ['salicylic acid']
        })
        mult = self.ap.compute_personalization_multiplier("user_breakout", mock_bad_product)
        self.assertLess(mult, 0.8)

    def test_routine_addition_implicit_positive(self):
        """Tests that adding a product to routine imparts positive affinity."""
        self.ap.record_feedback(
            user_id="user_routine",
            product_id="prod_moisturizer",
            feedback_type="routine_pm",
            brand="Kiehl's",
            actives=["squalane"]
        )
        profile = self.ap.get_or_create_profile("user_routine")
        self.assertIn("squalane", profile['liked_ingredients'])


if __name__ == '__main__':
    unittest.main()
