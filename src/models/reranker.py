import os
import sys
import json
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

DERMASENSE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if DERMASENSE_DIR not in sys.path:
    sys.path.insert(0, DERMASENSE_DIR)

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

class CrossEncoderReranker:
    """
    Second-Stage Deep Learning Cross-Encoder Reranker for DermaSense.
    
    Refines candidate products retrieved by first-stage semantic search (MiniLM + FAISS)
    by computing full cross-attention interaction between the user query and the candidate formulation.
    
    Features:
    - Lazy loading & singleton instance capability
    - Sigmoid normalization into [0.0, 1.0] confidence score
    - In-memory query-product cache for instant repeat evaluations
    - Graceful fallback in case of hardware or model loading exceptions
    """

    _instance = None

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL, max_length: int = 256, auto_load: bool = True):
        self.model_name = model_name
        self.max_length = max_length
        self.model = None
        self._cache: Dict[Tuple[str, str], float] = {}
        self.is_available = False
        
        if auto_load:
            self.load_model()

    @classmethod
    def get_instance(cls, model_name: str = DEFAULT_RERANKER_MODEL):
        """Returns singleton instance to avoid repeated memory allocations."""
        if cls._instance is None:
            cls._instance = cls(model_name=model_name, auto_load=True)
        return cls._instance

    def load_model(self) -> bool:
        """Loads the CrossEncoder model into memory."""
        if self.model is not None:
            return True
        try:
            import torch
            from sentence_transformers import CrossEncoder
            device = "cuda" if torch.cuda.is_available() else "cpu"
            gpu_info = f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""
            print(f"Loading CrossEncoder reranker '{self.model_name}' on device: {device.upper()}{gpu_info}...")
            self.model = CrossEncoder(self.model_name, max_length=self.max_length, device=device)
            self.is_available = True
            print(f"✓ CrossEncoder reranker loaded successfully on {device.upper()}.")
            return True
        except Exception as e:
            print(f"⚠️ Warning: Failed to load CrossEncoder model '{self.model_name}': {e}")
            self.is_available = False
            self.model = None
            return False

    @staticmethod
    def build_candidate_document(row: pd.Series) -> str:
        """
        Synthesizes a structured formulation representation of the product candidate
        optimized for cross-attention matching against clinical queries.
        """
        pname = str(row.get('product_name', ''))
        brand = str(row.get('brand', ''))
        cat = str(row.get('category', ''))
        
        actives = row.get('matched_actives')
        if isinstance(actives, str):
            try:
                actives = json.loads(actives)
            except Exception:
                actives = [actives]
        elif isinstance(actives, (list, set, tuple, np.ndarray)):
            actives = [str(a) for a in actives if a]
        else:
            actives = []
        actives_str = ", ".join(actives) if actives else "General formulation"

        benefits = row.get('matched_benefits')
        if isinstance(benefits, str):
            try:
                benefits = json.loads(benefits)
            except Exception:
                benefits = [benefits]
        elif isinstance(benefits, (list, set, tuple, np.ndarray)):
            benefits = [str(b) for b in benefits if b]
        else:
            benefits = []
        benefits_str = ", ".join(benefits) if benefits else "General skincare"

        desc = str(row.get('description', ''))[:200].strip()
        
        doc = f"Product: {pname}. Brand: {brand}. Category: {cat}. Actives: {actives_str}. Benefits: {benefits_str}."
        if desc:
            doc += f" Details: {desc}"
        return doc

    def score_candidates(
        self,
        query: str,
        candidates_df: pd.DataFrame,
        batch_size: int = 32
    ) -> Optional[np.ndarray]:
        """
        Computes second-stage cross-encoder relevance scores for a DataFrame of candidates.
        
        Returns:
            np.ndarray of scores normalized in [0.0, 1.0] aligned with candidates_df,
            or None if the reranker is unavailable.
        """
        if not query or not query.strip() or len(candidates_df) == 0:
            return None

        if not self.is_available and not self.load_model():
            return None

        q_clean = query.strip()
        pairs_to_predict = []
        indices_to_predict = []
        scores = np.zeros(len(candidates_df), dtype=np.float32)

        # Check cache first
        for i, (_, row) in enumerate(candidates_df.iterrows()):
            pid = str(row.get('product_id', i))
            cache_key = (q_clean, pid)
            if cache_key in self._cache:
                scores[i] = self._cache[cache_key]
            else:
                doc = self.build_candidate_document(row)
                pairs_to_predict.append((q_clean, doc))
                indices_to_predict.append(i)

        # Predict uncached pairs in batches
        if pairs_to_predict:
            try:
                raw_scores = self.model.predict(pairs_to_predict, batch_size=batch_size, show_progress_bar=False)
                # Convert raw logits to sigmoid probabilities [0, 1]
                sigmoid_scores = 1.0 / (1.0 + np.exp(-np.clip(raw_scores, -20.0, 20.0)))
                
                for idx, sig_score in zip(indices_to_predict, sigmoid_scores):
                    row_pid = str(candidates_df.iloc[idx].get('product_id', idx))
                    self._cache[(q_clean, row_pid)] = float(sig_score)
                    scores[idx] = float(sig_score)
            except Exception as e:
                print(f"⚠️ Warning: CrossEncoder inference failed: {e}. Falling back.")
                return None

        return scores


if __name__ == "__main__":
    reranker = CrossEncoderReranker()
    print("Testing CrossEncoder Reranker...")
    sample_df = pd.DataFrame([
        {
            'product_id': 'p1',
            'product_name': 'Ceramidin Skin Barrier Moisturizing Cream',
            'brand': 'Dr. Jart+',
            'category': 'Moisturizer',
            'matched_actives': ['ceramide np', 'hyaluronic acid', 'glycerin'],
            'matched_benefits': ['Hydration & Barrier'],
            'description': 'Intense barrier relief moisturizer for dry compromised skin.'
        },
        {
            'product_id': 'p2',
            'product_name': 'Charcoal Pore Clarifying Clay Mask',
            'brand': 'Origins',
            'category': 'Mask',
            'matched_actives': ['charcoal', 'kaolin'],
            'matched_benefits': ['Oil & Pore Control'],
            'description': 'Deep pore purifying mask for oily and congested skin.'
        }
    ])
    
    test_q = "Ceramide barrier repair for dry dehydrated skin"
    res_scores = reranker.score_candidates(test_q, sample_df)
    print(f"Query: '{test_q}'")
    for idx, score in enumerate(res_scores):
        print(f"  {sample_df.iloc[idx]['product_name']} -> Score: {score:.4f}")
