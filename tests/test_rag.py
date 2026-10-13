import os
import sys
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.knowledge_graph.graph import IngredientKnowledgeGraph
from src.rag.qa_engine import EvidenceGroundedQAEngine, MEDICAL_DISCLAIMER


class TestEvidenceGroundedQAEngine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.engine = EvidenceGroundedQAEngine.get_instance()

    def test_retrieve_context_for_active(self):
        """Tests that context retriever fetches PMIDs and mechanisms for active queries."""
        ctx = self.engine.retrieve_context(
            question="What are the clinical benefits of Niacinamide?",
            skin_type="oily",
            concerns=["Oil & Pore Control"]
        )
        self.assertIn("niacinamide", ctx['all_actives'])
        self.assertGreater(len(ctx['citations']), 0)
        # Verify PubMed citation format
        pmid_found = any("16029679" in c.get('citation_id', '') for c in ctx['citations'])
        self.assertTrue(pmid_found, "Expected Bissett et al. PMID:16029679 in Niacinamide citations")

    def test_contraindication_retrieval(self):
        """Tests that retriever flags hostile active combinations like Retinol and Glycolic Acid."""
        ctx = self.engine.retrieve_context(
            question="Can I layer my Retinol serum with Glycolic Acid toner?"
        )
        self.assertGreater(len(ctx['contraindications']), 0)
        rule_ids = [c['rule_id'] for c in ctx['contraindications']]
        self.assertIn('CAUTION_RETINOL_DIRECT_ACID', rule_ids)

    def test_deterministic_fallback_synthesis(self):
        """Tests zero-crash deterministic evidence generator when LLM is offline."""
        ctx = self.engine.retrieve_context(
            question="Can I layer Retinol with Glycolic Acid?"
        )
        fallback_ans = self.engine._generate_fallback_answer(
            question="Can I layer Retinol with Glycolic Acid?",
            context=ctx,
            is_medical=False
        )
        self.assertIn("Formulation Conflict", fallback_ans)
        self.assertIn("Retinoid + Direct AHA", fallback_ans)

    def test_medical_safety_refusal(self):
        """Tests that medical diagnosis questions receive the clinical disclaimer."""
        res = self.engine.answer_question(
            question="How do I cure my severe cystic acne infection with antibiotics?"
        )
        self.assertTrue(res['safety_compliant'])
        self.assertIn("dermatologist", res['answer'].lower())

    def test_groundedness_verification(self):
        """Tests claim-to-evidence verifier detects valid vs hallucinated citations."""
        mock_context = {
            'citations': [{'citation_id': 'PMID:16029679'}],
            'all_actives': ['niacinamide'],
            'products': []
        }
        
        # Valid answer
        valid_ans = "Niacinamide improves barrier function according to study [PMID:16029679]."
        v1 = self.engine.verify_groundedness(valid_ans, mock_context, is_medical=False)
        self.assertEqual(len(v1['citations_verified']), 1)
        self.assertEqual(len(v1['hallucinated_citations']), 0)
        self.assertGreaterEqual(v1['groundedness_score'], 0.9)

        # Hallucinated PMID
        bad_ans = "Niacinamide cures aging according to study [PMID:99999999]."
        v2 = self.engine.verify_groundedness(bad_ans, mock_context, is_medical=False)
        self.assertIn('PMID:99999999', v2['hallucinated_citations'])
        self.assertLess(v2['groundedness_score'], 0.8)

    def test_routine_advice_intent_and_synthesis(self):
        """Tests that asking for advice triggers ROUTINE_ADVICE intent and generates AM/PM regimen guidance."""
        from src.rag.qa_engine import classify_user_intent
        intent = classify_user_intent("What advice do you have for oily skin and acne?")
        self.assertEqual(intent, "ROUTINE_ADVICE")

        res = self.engine.answer_question(
            question="What advice do you have for my skincare routine?",
            skin_type="oily",
            concerns=["Acne & Blemish Control"]
        )
        self.assertIn("Morning (AM)", res['answer'])
        self.assertIn("Evening (PM)", res['answer'])
        self.assertIn("SPF", res['answer'])
        self.assertIn("patch-test", res['answer'].lower())


if __name__ == '__main__':
    unittest.main()
