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

WEIGHT_PRESETS = {
    'balanced': {
        'semantic': 0.25,
        'concern': 0.25,
        'skin_evidence': 0.20,
        'sentiment': 0.15,
        'price': 0.05,
        'popularity': 0.10
    },
    'ingredient_focus': {
        'semantic': 0.20,
        'concern': 0.40,
        'skin_evidence': 0.15,
        'sentiment': 0.15,
        'price': 0.05,
        'popularity': 0.05
    },
    'review_focus': {
        'semantic': 0.15,
        'concern': 0.20,
        'skin_evidence': 0.35,
        'sentiment': 0.20,
        'price': 0.05,
        'popularity': 0.05
    },
    'budget_focus': {
        'semantic': 0.20,
        'concern': 0.25,
        'skin_evidence': 0.20,
        'sentiment': 0.10,
        'price': 0.20,
        'popularity': 0.05
    }
}

class DermaSenseRecommender:
    """
    Personalized & Explainable Skincare Recommender & Comparison Engine.
    Combines:
    1. Dense Semantic Vector Embeddings (Sentence Transformers + FAISS)
    2. Ingredient & Skin Concern Matching (Ontology Overlap)
    3. User Skin-Type Review Evidence (Skin-partitioned Ratings & Confidence)
    4. Review Sentiment Polarity & Aspect Highlights
    5. Price Alignment & Budget Value
    6. Popularity & Community Love Signal
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
                    print(f"Dense Semantic Search initialized with {len(self.embedder.embeddings):,} vectors.")
            except Exception as e:
                print(f"Failed to initialize dense embedder: {e}. Falling back to TF-IDF.")
                self.use_dense_embeddings = False
                
        # Initialize TF-IDF fallback
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
                  preset='balanced',
                  custom_weights=None,
                  top_k=6):
        """
        Recommends top-k products based on 6-factor personalized scoring.
        """
        candidates = self.df.copy()
        
        # 1. Hard Filters
        if category and category.lower() not in ['all', 'any', 'all categories']:
            candidates = candidates[candidates['category'].str.lower() == category.lower()]
            
        if max_price_usd is not None and max_price_usd > 0:
            # Keep products within budget (or products with missing price as neutral)
            candidates = candidates[(candidates['price_usd'].isna()) | (candidates['price_usd'] <= max_price_usd * 1.05)]
            
        if brand_preference and brand_preference.strip():
            candidates = candidates[candidates['brand'].str.lower().str.contains(brand_preference.lower(), na=False)]
            
        if len(candidates) == 0:
            return pd.DataFrame()
            
        candidate_indices = candidates.index.values
        
        # 2. Factor 1: Semantic Similarity Score
        if query and query.strip():
            if self.use_dense_embeddings and self.embedder is not None and self.embedder.embeddings is not None:
                sim_scores = self.embedder.compute_similarity_for_candidates(query, candidate_indices)
            else:
                query_vec = self.tfidf.transform([query])
                sim_scores = cosine_similarity(query_vec, self.doc_matrix[candidate_indices]).flatten()
        else:
            sim_scores = np.ones(len(candidate_indices)) * 0.5
            
        # 3. Factor 2: Ingredient Concern Match Score
        target_concerns = target_concerns or []
        concern_scores = []
        for idx in candidate_indices:
            row = self.df.iloc[idx]
            try:
                prod_benefits = json.loads(row.get('matched_benefits', '[]')) if isinstance(row.get('matched_benefits'), str) else (row.get('matched_benefits') or [])
            except Exception:
                prod_benefits = []
                
            if target_concerns:
                overlap = set(target_concerns).intersection(set(prod_benefits))
                score = len(overlap) / len(target_concerns)
            else:
                score = 0.5 if len(prod_benefits) > 0 else 0.0
            concern_scores.append(score)
            
        concern_scores = np.array(concern_scores)
        
        # 4. Factor 3: Skin-Type Review Evidence Score
        skin_type_scores = []
        st_clean = str(skin_type).lower().strip() if skin_type else ""
        
        for idx in candidate_indices:
            row = self.df.iloc[idx]
            if st_clean in ['combination', 'dry', 'normal', 'oily']:
                st_rating = row.get(f'{st_clean}_skin_avg_rating')
                st_count = row.get(f'{st_clean}_skin_review_count') or 0
                
                if pd.notna(st_rating) and st_count > 0:
                    rating_norm = (float(st_rating) - 1.0) / 4.0
                    confidence = min(1.0, np.log1p(st_count) / np.log1p(50))
                    st_score = (0.7 * rating_norm) + (0.3 * confidence)
                else:
                    gen_rating = row.get('rating') or 4.0
                    st_score = ((float(gen_rating) - 1.0) / 4.0) * 0.7
            else:
                gen_rating = row.get('rating') or 4.0
                st_score = ((float(gen_rating) - 1.0) / 4.0)
                
            skin_type_scores.append(st_score)
            
        skin_type_scores = np.array(skin_type_scores)
        
        # 5. Factor 4: Sentiment Score
        sentiment_scores = []
        for idx in candidate_indices:
            row = self.df.iloc[idx]
            if st_clean in ['combination', 'dry', 'normal', 'oily'] and pd.notna(row.get(f'{st_clean}_sentiment_score')):
                s_score = float(row.get(f'{st_clean}_sentiment_score'))
            else:
                s_score = float(row.get('sentiment_score') if pd.notna(row.get('sentiment_score')) else 0.70)
            sentiment_scores.append(s_score)
        sentiment_scores = np.array(sentiment_scores)
        
        # 6. Factor 5: Price Alignment Score
        price_scores = []
        for idx in candidate_indices:
            row = self.df.iloc[idx]
            p = row.get('price_usd')
            if pd.notna(p) and p > 0:
                if max_price_usd and max_price_usd > 0:
                    # Closer to 0 price relative to max_budget gets higher value score
                    p_score = 1.0 - min(1.0, float(p) / float(max_price_usd))
                else:
                    p_score = 0.5
            else:
                p_score = 0.5
            price_scores.append(p_score)
        price_scores = np.array(price_scores)
        
        # 7. Factor 6: Popularity & Base Quality Score
        base_ratings = candidates['rating'].fillna(4.0).values / 5.0
        review_counts = candidates['review_count'].fillna(0).values
        pop_confidence = np.clip(np.log1p(review_counts) / np.log1p(500), 0.0, 1.0)
        popularity_scores = (0.6 * base_ratings) + (0.4 * pop_confidence)
        
        # 8. Composite Linear Combination
        w = custom_weights or WEIGHT_PRESETS.get(preset, WEIGHT_PRESETS['balanced'])
        
        composite_score = (
            (w.get('semantic', 0.25) * sim_scores) +
            (w.get('concern', 0.25) * concern_scores) +
            (w.get('skin_evidence', 0.20) * skin_type_scores) +
            (w.get('sentiment', 0.15) * sentiment_scores) +
            (w.get('price', 0.05) * price_scores) +
            (w.get('popularity', 0.10) * popularity_scores)
        )
        
        candidates = candidates.copy()
        candidates['semantic_score'] = sim_scores.round(3)
        candidates['concern_match_score'] = concern_scores.round(3)
        candidates['skin_evidence_score'] = skin_type_scores.round(3)
        candidates['sentiment_score_calc'] = sentiment_scores.round(3)
        candidates['price_score_calc'] = price_scores.round(3)
        candidates['composite_score'] = composite_score.round(3)
        
        top_candidates = candidates.sort_values('composite_score', ascending=False).head(top_k)
        return top_candidates

    def compare_products(self, product_id_a, product_id_b, skin_type=None, concerns=None):
        """
        Performs in-depth algorithmic side-by-side comparison between two products.
        """
        row_a = self.df[self.df['product_id'] == product_id_a]
        row_b = self.df[self.df['product_id'] == product_id_b]
        
        if len(row_a) == 0 or len(row_b) == 0:
            return None
            
        p_a = row_a.iloc[0].to_dict()
        p_b = row_b.iloc[0].to_dict()
        
        # Parse ingredients & benefits
        def _parse(val):
            if isinstance(val, str):
                try: return json.loads(val)
                except: return []
            return val if isinstance(val, list) else []
            
        actives_a = set(_parse(p_a.get('matched_actives')))
        actives_b = set(_parse(p_b.get('matched_actives')))
        benefits_a = set(_parse(p_a.get('matched_benefits')))
        benefits_b = set(_parse(p_b.get('matched_benefits')))
        
        shared_actives = list(actives_a.intersection(actives_b))
        unique_actives_a = list(actives_a - actives_b)
        unique_actives_b = list(actives_b - actives_a)
        
        # Skin type metrics
        st_clean = str(skin_type).lower().strip() if skin_type else ""
        if st_clean in ['oily', 'dry', 'combination', 'normal']:
            rating_a = p_a.get(f'{st_clean}_skin_avg_rating') or p_a.get('rating') or 4.0
            rating_b = p_b.get(f'{st_clean}_skin_avg_rating') or p_b.get('rating') or 4.0
            rev_count_a = p_a.get(f'{st_clean}_skin_review_count') or 0
            rev_count_b = p_b.get(f'{st_clean}_skin_review_count') or 0
        else:
            rating_a = p_a.get('rating') or 4.0
            rating_b = p_b.get('rating') or 4.0
            rev_count_a = p_a.get('review_count') or 0
            rev_count_b = p_b.get('review_count') or 0
            
        # Determine winner for user profile
        concerns = concerns or []
        match_a = len(set(concerns).intersection(benefits_a))
        match_b = len(set(concerns).intersection(benefits_b))
        
        score_a = (0.4 * match_a) + (0.3 * (float(rating_a)/5.0)) + (0.3 * float(p_a.get('sentiment_score') or 0.7))
        score_b = (0.4 * match_b) + (0.3 * (float(rating_b)/5.0)) + (0.3 * float(p_b.get('sentiment_score') or 0.7))
        
        if score_a > score_b:
            verdict = f"**{p_a.get('product_name')}** is recommended for your profile due to stronger active ingredient alignment and skin-type satisfaction."
            winner_id = product_id_a
        elif score_b > score_a:
            verdict = f"**{p_b.get('product_name')}** is recommended for your profile due to stronger active ingredient alignment and skin-type satisfaction."
            winner_id = product_id_b
        else:
            verdict = "Both products are well-matched. Choose based on your preferred brand texture or budget."
            winner_id = "tie"
            
        return {
            'product_a': p_a,
            'product_b': p_b,
            'shared_actives': shared_actives,
            'unique_actives_a': unique_actives_a,
            'unique_actives_b': unique_actives_b,
            'benefits_a': list(benefits_a),
            'benefits_b': list(benefits_b),
            'rating_a': float(rating_a) if rating_a else None,
            'rating_b': float(rating_b) if rating_b else None,
            'reviews_a': int(rev_count_a) if rev_count_a else 0,
            'reviews_b': int(rev_count_b) if rev_count_b else 0,
            'winner_id': winner_id,
            'verdict': verdict
        }

    def search_catalog_by_name(self, search_term, limit=10):
        """Finds products by name or brand for autocomplete and manual search."""
        if not search_term or not search_term.strip():
            return []
        term = search_term.lower().strip()
        mask = (
            self.df['product_name'].str.lower().str.contains(term, na=False) |
            self.df['brand'].str.lower().str.contains(term, na=False)
        )
        matches = self.df[mask].head(limit)
        results = []
        for _, r in matches.iterrows():
            results.append({
                'product_id': r['product_id'],
                'product_name': r['product_name'],
                'brand': r['brand'],
                'category': r['category'],
                'price_usd': r['price_usd'],
                'rating': r['rating']
            })
        return results


if __name__ == "__main__":
    rec = DermaSenseRecommender()
    
    print("\n" + "="*80)
    print("TESTING 6-FACTOR COMPOSITE RECOMMENDATION")
    print("="*80)
    results = rec.recommend(
        query="lightweight daily sunscreen for sensitive combination skin without white cast",
        skin_type="combination",
        target_concerns=["Sun Protection", "Soothing & Anti-Redness"],
        category="Sunscreen",
        max_price_usd=40,
        preset='balanced',
        top_k=2
    )
    
    for idx, row in results.iterrows():
        print(f"\nProduct: {row['product_name']} ({row['brand']})")
        print(f"  Category: {row['category']} | Price: ${row['price_usd']} | Rating: {row['rating']}★")
        print(f"  Benefits: {row['matched_benefits']}")
        print(f"  Composite Score: {row['composite_score']} | Semantic: {row['semantic_score']} | Concern: {row['concern_match_score']} | Skin Evidence: {row['skin_evidence_score']} | Sentiment: {row['sentiment_score_calc']}")
