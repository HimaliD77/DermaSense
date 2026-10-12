"""
Unit and Integration Tests for DermaSense Provenance-Aware Ingredient Knowledge Graph
Tests:
1. Graph construction, node types, and edge relationships.
2. Synonym normalization and entity resolution.
3. Provenance retrieval and clinical citation linkage.
4. Recommendation reasoning path traversal.
5. Clinical contraindication and layering conflict detection.
6. Explainer integration with graph evidence output.
"""

import os
import sys
import unittest
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.knowledge_graph.graph import IngredientKnowledgeGraph
from src.explainability.explainer import DermaSenseExplainer


class TestIngredientKnowledgeGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kg = IngredientKnowledgeGraph.get_instance()
        cls.explainer = DermaSenseExplainer()

    def test_graph_structure_and_counts(self):
        """Verifies graph scale and existence of required entity types."""
        self.assertGreaterEqual(self.kg.graph.number_of_nodes(), 200)
        self.assertGreaterEqual(self.kg.graph.number_of_edges(), 300)

        node_types = set(data.get('node_type') for _, data in self.kg.graph.nodes(data=True))
        expected_types = {'Ingredient', 'Synonym', 'CosmeticFunction', 'SkinConcern', 'EvidenceSource', 'CautionWarning'}
        for t in expected_types:
            self.assertIn(t, node_types)

    def test_synonym_entity_resolution(self):
        """Verifies multi-lingual and trade-name synonym resolution to canonical entities."""
        test_pairs = [
            ("Vitamin B3", "niacinamide"),
            ("Nicotinamide", "niacinamide"),
            ("L-Ascorbic Acid", "vitamin c"),
            ("Cica", "centella asiatica"),
            ("Tiger Grass", "centella asiatica"),
            ("Gotu Kola", "centella asiatica"),
            ("Sodium Hyaluronate", "hyaluronic acid"),
            ("HA", "hyaluronic acid"),
            ("BHA", "salicylic acid"),
            ("Pro-Vitamin B5", "panthenol"),
            ("AHA", "glycolic acid")
        ]
        for query_synonym, expected_canonical in test_pairs:
            resolved = self.kg.resolve_ingredient_entity(query_synonym)
            self.assertEqual(resolved, expected_canonical, f"Failed for '{query_synonym}'")

    def test_provenance_and_citations(self):
        """Verifies that canonical actives retain peer-reviewed clinical citations."""
        prov = self.kg.get_ingredient_provenance("niacinamide")
        self.assertIsNotNone(prov)
        self.assertEqual(prov['canonical_name'], "niacinamide")
        self.assertGreater(len(prov['cosmetic_functions']), 0)
        self.assertGreater(len(prov['evidence_citations']), 0)

        # Check citation details
        cit = prov['evidence_citations'][0]
        self.assertIn("PMID:", cit['citation_id'])
        self.assertTrue(len(cit['authors']) > 0)
        self.assertTrue(len(cit['finding']) > 0)

    def test_reasoning_paths_tracing(self):
        """Verifies reasoning path traversal from concern to active to citation."""
        paths = self.kg.trace_recommendation_reasoning(
            product_actives=['ceramide np', 'hyaluronic acid', 'glycerin'],
            user_concerns=['Hydration & Barrier']
        )
        self.assertGreater(len(paths), 0)
        for p in paths:
            self.assertTrue(p['is_ontology_verified'])
            self.assertIn('Hydration & Barrier', p['addressed_concerns'])

    def test_contraindication_detection(self):
        """Verifies chemical incompatibility alerts for known hostile pairings."""
        # Hostile pairing: Retinol + Glycolic Acid
        cautions = self.kg.detect_contraindications(['retinol', 'glycolic acid', 'squalane'])
        self.assertEqual(len(cautions), 1)
        self.assertEqual(cautions[0]['severity'], "High")
        self.assertIn("Retinoid + Direct AHA", cautions[0]['title'])

        # Benign pairing: Ceramide + Hyaluronic Acid
        benign_cautions = self.kg.detect_contraindications(['ceramide np', 'hyaluronic acid', 'glycerin'])
        self.assertEqual(len(benign_cautions), 0)

    def test_explainer_graph_integration(self):
        """Verifies that DermaSenseExplainer outputs graph evidence and citations."""
        sample_row = pd.Series({
            'product_id': 'test_p1',
            'product_name': 'Barrier Repair Ceramide Moisturizer',
            'brand': 'Dr. Jart+',
            'category': 'Moisturizer',
            'price_inr': 2400.0,
            'rating': 4.5,
            'review_count': 120,
            'matched_actives': '["ceramide np", "hyaluronic acid"]',
            'matched_benefits': '["Hydration & Barrier"]'
        })

        exp = self.explainer.explain_recommendation(
            sample_row,
            user_skin_type='dry',
            user_concerns=['Hydration & Barrier'],
            max_budget_inr=3000
        )
        self.assertIn('graph_evidence', exp)
        self.assertTrue(exp['graph_evidence']['has_graph_evidence'])
        self.assertGreater(exp['graph_evidence']['grounding_score'], 0.0)
        self.assertIn('Clinical Literature Evidence:', exp['full_narrative'])


if __name__ == '__main__':
    unittest.main()
