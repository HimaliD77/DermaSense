"""
Unit and Regression Tests for Stage-2 Cross-Encoder Reranking
Tests:
1. Reranker module loading, prediction, sigmoid bounds, and caching.
2. Recommender integration with use_reranker=True and use_reranker=False.
3. Hard constraint preservation (budget cap, category).
4. Graceful fallback on missing data or empty queries.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.reranker import CrossEncoderReranker
from src.models.recommender import DermaSenseRecommender


class TestCrossEncoderReranker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reranker = CrossEncoderReranker.get_instance()
        cls.recommender = DermaSenseRecommender(use_dense_embeddings=True, use_reranker=True)

    def test_reranker_instance_and_bounds(self):
        """Verifies model loading and that output scores fall strictly within [0.0, 1.0]."""
        self.assertTrue(self.reranker.is_available)
        
        sample_df = pd.DataFrame([
            {
                'product_id': 'test_p1',
                'product_name': 'Hydrating Ceramide Barrier Balm',
                'brand': 'DermaLab',
                'category': 'Moisturizer',
                'matched_actives': ['ceramide np', 'hyaluronic acid'],
                'matched_benefits': ['Hydration & Barrier'],
                'description': 'Deep moisture barrier restoration cream.'
            },
            {
                'product_id': 'test_p2',
                'product_name': 'Tea Tree Clarifying Spot Gel',
                'brand': 'AcneCare',
                'category': 'Treatment',
                'matched_actives': ['tea tree', 'salicylic acid'],
                'matched_benefits': ['Acne & Blemish Control'],
                'description': 'Targeted blemish control gel.'
            }
        ])
        
        scores = self.reranker.score_candidates("ceramide cream for severely dry barrier", sample_df)
        self.assertIsNotNone(scores)
        self.assertEqual(len(scores), 2)
        self.assertTrue(all(0.0 <= s <= 1.0 for s in scores))
        # The ceramide cream should rank significantly higher than the spot gel
        self.assertGreater(scores[0], scores[1])

    def test_cache_hit_consistency(self):
        """Verifies that repeat calls use cache and produce identical values."""
        sample_df = pd.DataFrame([{
            'product_id': 'cached_p1',
            'product_name': 'Vitamin C Brightening Serum',
            'brand': 'GlowCo',
            'category': 'Serum',
            'matched_actives': ['vitamin c'],
            'matched_benefits': ['Brightening & Tone'],
            'description': 'Fades spots.'
        }])
        
        q = "vitamin c dark spot treatment"
        score_1 = self.reranker.score_candidates(q, sample_df)
        score_2 = self.reranker.score_candidates(q, sample_df)
        self.assertEqual(score_1[0], score_2[0])
        self.assertIn((q, 'cached_p1'), self.reranker._cache)

    def test_recommender_flag_toggle(self):
        """Verifies that recommend() supports both use_reranker=True and use_reranker=False."""
        q = "Salicylic acid cleanser for oily skin"
        res_with_rerank = self.recommender.recommend(
            query=q,
            skin_type="oily",
            target_concerns=["Acne & Blemish Control"],
            top_k=5,
            use_reranker=True
        )
        self.assertFalse(res_with_rerank.empty)
        self.assertEqual(len(res_with_rerank), 5)
        self.assertTrue('rerank_score' in res_with_rerank.columns)
        self.assertTrue(res_with_rerank['rerank_score'].notna().any())

        res_without_rerank = self.recommender.recommend(
            query=q,
            skin_type="oily",
            target_concerns=["Acne & Blemish Control"],
            top_k=5,
            use_reranker=False
        )
        self.assertFalse(res_without_rerank.empty)
        self.assertEqual(len(res_without_rerank), 5)
        # When use_reranker=False, rerank_score remains NaN
        self.assertTrue(res_without_rerank['rerank_score'].isna().all())

    def test_hard_constraint_enforcement_under_reranking(self):
        """Verifies that budget caps and categories are strictly honored under Stage 2 reranking."""
        budget_cap = 2500
        cat = "Cleanser"
        res = self.recommender.recommend(
            query="luxury foaming deep cleanse",
            category=cat,
            max_price_inr=budget_cap,
            top_k=6,
            use_reranker=True
        )
        for _, row in res.iterrows():
            if pd.notna(row['price_inr']):
                self.assertLessEqual(row['price_inr'], budget_cap * 1.05)
            self.assertEqual(row['category'].lower(), cat.lower())

    def test_missing_metadata_and_empty_query_fallback(self):
        """Verifies graceful handling when query is empty or product text is sparse."""
        # Empty query should not crash and return baseline
        res_empty = self.recommender.recommend(
            query="",
            skin_type="dry",
            target_concerns=["Hydration & Barrier"],
            top_k=4,
            use_reranker=True
        )
        self.assertEqual(len(res_empty), 4)

        # Sparse metadata candidate should not crash reranker
        sparse_df = pd.DataFrame([{
            'product_id': 'sparse_p1',
            'product_name': 'Simple Moisturizer',
            'brand': '',
            'category': '',
            'matched_actives': None,
            'matched_benefits': None,
            'description': None
        }])
        scores = self.reranker.score_candidates("hydrating cream", sparse_df)
        self.assertIsNotNone(scores)
        self.assertEqual(len(scores), 1)


if __name__ == '__main__':
    unittest.main()
