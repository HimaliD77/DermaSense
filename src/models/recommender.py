import os
import sys
import json
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

DERMASENSE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if DERMASENSE_DIR not in sys.path:
    sys.path.insert(0, DERMASENSE_DIR)

PROCESSED_DIR = os.path.join(DERMASENSE_DIR, "data", "processed")
EMBEDDINGS_DIR = os.path.join(DERMASENSE_DIR, "data", "embeddings")

from src.models.embeddings import ProductEmbedder

class DermaSenseRecommender:
    """
    Personalized & Explainable Skincare Recommender Engine.
    Combines:
    1. Dense Semantic Vector Embeddings (Sentence Transformers + FAISS / Cosine Sim)
    2. Ingredient & Skin Concern Matching (Ontology Overlap)
    3. User Skin-Type Review Evidence (Skin-partitioned Ratings & Confidence)
    4. Price & Category Constraints
    """
    
    def __init__(self, data_path=None, use_dense_embeddings=True):
        if data_path is None:
            data_path = os.path.join(PROCESSED_DIR, "master_product_features.parquet")
            
        print(f"Initializing DermaSense Recommender with: {data_path}")
        self.df = pd.read_parquet(data_path)
        self.use_dense_embeddings = use_dense_embeddings
        self.embedder = None
        self.tfidf = None
        self.doc_matrix = None
        
        # Initialize Dense Embedder
        if self.use_dense_embeddings:
            try:
                self.embedder = ProductEmbedder(auto_load=True)
                if self.embedder.embeddings is None:
                    print("Precomputed embeddings not found. Falling back to TF-IDF retrieval.")
                    self.use_dense_embeddings = False
                else:
                    print(f"Dense Semantic Search initialized with {len(self.embedder.embeddings):,} vectors (384-dim).")
            except Exception as e:
                print(f"Failed to initialize dense embedder: {e}. Falling back to TF-IDF.")
                self.use_dense_embeddings = False
                
        # Initialize TF-IDF as backup or baseline
        if not self.use_dense_embeddings or self.doc_matrix is None:
            self.tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=15000, stop_words='english')
            self.doc_matrix = self.tfidf.fit_transform(self.df['semantic_document'].fillna(""))
            print(f"Loaded {len(self.df):,} products. TF-IDF Matrix shape: {self.doc_matrix.shape}")
        
    def recommend(self,
                  query="",
                  skin_type=None,
                  target_concerns=None,
                  category=None,
                  max_price_usd=None,
                  brand_preference=None,
                  weights=None,
                  top_k=5):
        """
        Recommends top-k products based on multi-factor personalized scoring.
        
        Parameters:
        - query: Natural language user query (e.g. "gentle soothing moisturizer for red acne skin")
        - skin_type: "oily", "dry", "combination", "normal", or None
        - target_concerns: List of concern strings (e.g. ["Hydration & Barrier", "Acne & Blemish Control"])
        - category: Product category filter ("Cleanser", "Moisturizer", "Treatment & Serum", "Sunscreen", etc.)
        - max_price_usd: Maximum budget in USD
        - brand_preference: Optional brand filter or search substring
        - weights: Dict of factor weights (default: semantic=0.35, concern=0.35, skin_evidence=0.25, base_rating=0.05)
        - top_k: Number of recommendations to return
        """
        candidates = self.df.copy()
        
        # 1. Hard Filters
        if category and category.lower() != 'all':
            candidates = candidates[candidates['category'].str.lower() == category.lower()]
            
        if max_price_usd is not None and max_price_usd > 0:
            candidates = candidates[(candidates['price_usd'].isna()) | (candidates['price_usd'] <= max_price_usd)]
            
        if brand_preference and brand_preference.strip():
            candidates = candidates[candidates['brand'].str.lower().str.contains(brand_preference.lower(), na=False)]
            
        if len(candidates) == 0:
            return pd.DataFrame()
            
        candidate_indices = candidates.index.values
        
        # 2. Semantic Similarity Score
        if query and query.strip():
            if self.use_dense_embeddings and self.embedder is not None and self.embedder.embeddings is not None:
                sim_scores = self.embedder.compute_similarity_for_candidates(query, candidate_indices)
            else:
                query_vec = self.tfidf.transform([query])
                sim_scores = cosine_similarity(query_vec, self.doc_matrix[candidate_indices]).flatten()
        else:
            sim_scores = np.ones(len(candidate_indices)) * 0.5
            
        # 3. Ingredient / Concern Match Score
        target_concerns = target_concerns or []
        concern_scores = []
        for idx in candidate_indices:
            row = self.df.iloc[idx]
            try:
                prod_benefits = json.loads(row.get('matched_benefits', '[]'))
            except Exception:
                prod_benefits = []
                
            if target_concerns:
                overlap = set(target_concerns).intersection(set(prod_benefits))
                score = len(overlap) / len(target_concerns)
            else:
                score = 0.5 if len(prod_benefits) > 0 else 0.0
            concern_scores.append(score)
            
        concern_scores = np.array(concern_scores)
        
        # 4. Skin-Type Review Evidence Score
        skin_type_scores = []
        st_clean = str(skin_type).lower().strip() if skin_type else ""
        
        for idx in candidate_indices:
            row = self.df.iloc[idx]
            if st_clean in ['combination', 'dry', 'normal', 'oily']:
                st_rating = row.get(f'{st_clean}_skin_avg_rating')
                st_count = row.get(f'{st_clean}_skin_review_count') or 0
                
                if pd.notna(st_rating) and st_count > 0:
                    # Normalized rating (1..5 -> 0..1)
                    rating_norm = (float(st_rating) - 1.0) / 4.0
                    # Confidence scaling based on review count (saturates around 50 reviews)
                    confidence = min(1.0, np.log1p(st_count) / np.log1p(50))
                    st_score = (0.7 * rating_norm) + (0.3 * confidence)
                else:
                    gen_rating = row.get('rating') or 4.0
                    st_score = ((float(gen_rating) - 1.0) / 4.0) * 0.7  # neutral prior with uncertainty penalty
            else:
                gen_rating = row.get('rating') or 4.0
                st_score = ((float(gen_rating) - 1.0) / 4.0)
                
            skin_type_scores.append(st_score)
            
        skin_type_scores = np.array(skin_type_scores)
        
        # 5. Composite Ranking Score
        w = weights or {
            'semantic': 0.35,
            'concern': 0.35,
            'skin_evidence': 0.25,
            'base_rating': 0.05
        }
        
        base_ratings = candidates['rating'].fillna(4.0).values / 5.0
        composite_score = (
            (w.get('semantic', 0.35) * sim_scores) +
            (w.get('concern', 0.35) * concern_scores) +
            (w.get('skin_evidence', 0.25) * skin_type_scores) +
            (w.get('base_rating', 0.05) * base_ratings)
        )
        
        candidates = candidates.copy()
        candidates['semantic_score'] = sim_scores.round(3)
        candidates['concern_match_score'] = concern_scores.round(3)
        candidates['skin_evidence_score'] = skin_type_scores.round(3)
        candidates['composite_score'] = composite_score.round(3)
        
        top_candidates = candidates.sort_values('composite_score', ascending=False).head(top_k)
        return top_candidates


if __name__ == "__main__":
    recommender = DermaSenseRecommender(use_dense_embeddings=True)
    
    # Test query
    results = recommender.recommend(
        query="gentle hydrating barrier repair cream with ceramides and squalane for dry flaky skin",
        skin_type="dry",
        target_concerns=["Hydration & Barrier", "Soothing & Anti-Redness"],
        category="Moisturizer",
        max_price_usd=50,
        top_k=3
    )
    
    print("\n" + "="*80)
    print("DENSE EMBEDDINGS RECOMMENDER EVALUATION TEST")
    print("="*80)
    for idx, row in results.iterrows():
        print(f"\nProduct: {row['product_name']} ({row['brand']})")
        print(f"  Category: {row['category']} | Price: ${row['price_usd']}")
        print(f"  Benefits: {row['matched_benefits']}")
        print(f"  Actives: {row['matched_actives']}")
        print(f"  Dry Skin Rating: {row['dry_skin_avg_rating']} ({row['dry_skin_review_count']} reviews)")
        print(f"  Scores -> Composite: {row['composite_score']} | Semantic: {row['semantic_score']} | Concern: {row['concern_match_score']} | Evidence: {row['skin_evidence_score']}")
