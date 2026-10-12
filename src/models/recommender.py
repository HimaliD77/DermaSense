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
from src.models.reranker import CrossEncoderReranker

USD_TO_INR_RATE = 86.5

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
    Operates in Indian Rupees (INR ₹) and incorporates:
    1. Dense Semantic Vector Embeddings (Sentence Transformers + FAISS)
    2. Deep-Learning Second-Stage Cross-Encoder Reranking
    3. Ingredient & Skin Concern Matching (Ontology Overlap)
    4. User Skin-Type Review Evidence (Skin-partitioned Ratings & Confidence)
    5. Review Sentiment Polarity & Aspect Highlights
    6. Price Alignment & Budget Value (INR ₹)
    7. Popularity & Community Love Signal
    """
    
    def __init__(self, data_path=None, use_dense_embeddings=True, use_reranker=True):
        if data_path is None:
            data_path = os.path.join(PROCESSED_DIR, "master_product_features.parquet")
            
        print(f"Initializing DermaSense Recommender with: {data_path}")
        self.df = pd.read_parquet(data_path)
        
        # Ensure robust INR pricing across all records
        if 'price_inr' not in self.df.columns or self.df['price_inr'].isna().all():
            self.df['price_inr'] = (self.df['price_usd'] * USD_TO_INR_RATE).round(0)
        else:
            self.df['price_inr'] = self.df['price_inr'].fillna(self.df['price_usd'] * USD_TO_INR_RATE).round(0)
            
        # Pre-parse benefits into list of sets for microsecond lookups
        def _parse_b(val):
            if isinstance(val, list):
                return set(val)
            if isinstance(val, str):
                try:
                    return set(json.loads(val))
                except Exception:
                    return set()
            return set()
        self.benefits_sets = [_parse_b(v) for v in self.df['matched_benefits']]

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

        # Initialize Cross-Encoder Reranker
        self.use_reranker = use_reranker
        self.reranker = None
        if self.use_reranker:
            try:
                self.reranker = CrossEncoderReranker.get_instance()
                print("Stage-2 Deep Learning Cross-Encoder Reranker initialized.")
            except Exception as e:
                print(f"⚠️ Could not load CrossEncoder reranker: {e}. Continuing without Stage-2 reranking.")
                self.use_reranker = False
        
    def recommend(self,
                  query="",
                  skin_type=None,
                  target_concerns=None,
                  category=None,
                  max_price_inr=None,
                  max_price_usd=None,
                  brand_preference=None,
                  preset='balanced',
                  custom_weights=None,
                  top_k=6,
                  use_reranker=None,
                  user_id=None):
        """
        Recommends top-k products based on 6-factor personalized scoring in INR ₹.
        """
        candidates = self.df.copy()
        
        # If max_price_usd is passed instead, convert to INR
        if max_price_inr is None and max_price_usd is not None and max_price_usd > 0:
            max_price_inr = max_price_usd * USD_TO_INR_RATE
            
        # 1. Hard Filters
        if category and category.lower() not in ['all', 'any', 'all categories']:
            candidates = candidates[candidates['category'].str.lower() == category.lower()]
            
        if max_price_inr is not None and max_price_inr > 0:
            candidates = candidates[(candidates['price_inr'].isna()) | (candidates['price_inr'] <= max_price_inr * 1.05)]
            
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
            
        # 3. Factor 2: Ingredient Concern Match Score (Optimized set intersection)
        target_concerns = target_concerns or []
        if target_concerns:
            target_set = set(target_concerns)
            target_len = len(target_concerns)
            concern_scores = np.array([
                len(target_set.intersection(self.benefits_sets[idx])) / target_len
                for idx in candidate_indices
            ])
        else:
            concern_scores = np.array([
                0.5 if len(self.benefits_sets[idx]) > 0 else 0.0
                for idx in candidate_indices
            ])
        
        # 4. Factor 3: Skin-Type Review Evidence Score (Vectorized)
        st_clean = str(skin_type).lower().strip() if skin_type else ""
        if st_clean in ['combination', 'dry', 'normal', 'oily']:
            st_ratings = candidates[f'{st_clean}_skin_avg_rating'].values
            st_counts = candidates[f'{st_clean}_skin_review_count'].fillna(0).values
            gen_ratings = candidates['rating'].fillna(4.0).values
            
            has_reviews = (~np.isnan(st_ratings)) & (st_counts > 0)
            rating_norm = np.where(has_reviews, (st_ratings - 1.0) / 4.0, 0.0)
            confidence = np.clip(np.log1p(st_counts) / np.log1p(50), 0.0, 1.0)
            st_score_with_rev = (0.7 * rating_norm) + (0.3 * confidence)
            st_score_fallback = ((gen_ratings - 1.0) / 4.0) * 0.7
            
            skin_type_scores = np.where(has_reviews, st_score_with_rev, st_score_fallback)
        else:
            gen_ratings = candidates['rating'].fillna(4.0).values
            skin_type_scores = (gen_ratings - 1.0) / 4.0
            
        # 5. Factor 4: Sentiment Score (Vectorized)
        if st_clean in ['combination', 'dry', 'normal', 'oily'] and f'{st_clean}_sentiment_score' in candidates.columns:
            st_senti = candidates[f'{st_clean}_sentiment_score'].values
            gen_senti = candidates['sentiment_score'].fillna(0.70).values
            sentiment_scores = np.where(~np.isnan(st_senti), st_senti, gen_senti)
        else:
            sentiment_scores = candidates['sentiment_score'].fillna(0.70).values
            
        # 6. Factor 5: Price Alignment Score in INR (Vectorized)
        p_vals = candidates['price_inr'].values
        if max_price_inr and max_price_inr > 0:
            price_scores = np.where(
                (~np.isnan(p_vals)) & (p_vals > 0),
                1.0 - np.clip(p_vals / float(max_price_inr), 0.0, 1.0),
                0.5
            )
        else:
            price_scores = np.full(len(candidates), 0.5)
        
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
        candidates['popularity_score_calc'] = popularity_scores.round(3)
        candidates['composite_score'] = composite_score.round(3)
        candidates['match_percentage'] = np.clip(np.nan_to_num(np.round(composite_score * 100), nan=80.0), 50, 99).astype(int)
        candidates['rerank_score'] = np.nan

        # Stage 2: Deep-Learning Cross-Encoder Reranking
        should_rerank = (
            (use_reranker if use_reranker is not None else self.use_reranker)
            and bool(query and query.strip())
            and (self.reranker is not None and self.reranker.is_available)
        )

        if should_rerank and len(candidates) > 0:
            try:
                # Retrieve top candidate pool based on Stage 1 multi-factor scores
                pool_size = min(len(candidates), max(top_k * 2, 16))
                top_pool_indices = candidates.sort_values('composite_score', ascending=False).head(pool_size).index
                pool_df = candidates.loc[top_pool_indices]

                rerank_scores = self.reranker.score_candidates(query, pool_df)
                if rerank_scores is not None and len(rerank_scores) == len(pool_df):
                    candidates.loc[top_pool_indices, 'rerank_score'] = rerank_scores.round(3)
                    # Refine semantic factor: fuse bi-encoder semantic score with cross-encoder relevance
                    refined_sem = (0.35 * candidates.loc[top_pool_indices, 'semantic_score'].values) + (0.65 * rerank_scores)
                    candidates.loc[top_pool_indices, 'semantic_score'] = refined_sem.round(3)
                    
                    # Recompute composite score with refined semantic factor (avoiding double-counting)
                    w_sem = w.get('semantic', 0.25)
                    pool_composite = (
                        (w_sem * refined_sem) +
                        (w.get('concern', 0.25) * candidates.loc[top_pool_indices, 'concern_match_score'].values) +
                        (w.get('skin_evidence', 0.20) * candidates.loc[top_pool_indices, 'skin_evidence_score'].values) +
                        (w.get('sentiment', 0.15) * candidates.loc[top_pool_indices, 'sentiment_score_calc'].values) +
                        (w.get('price', 0.05) * candidates.loc[top_pool_indices, 'price_score_calc'].values) +
                        (w.get('popularity', 0.10) * candidates.loc[top_pool_indices, 'popularity_score_calc'].values)
                    )
                    candidates.loc[top_pool_indices, 'composite_score'] = pool_composite.round(3)
                    candidates.loc[top_pool_indices, 'match_percentage'] = np.clip(
                        np.nan_to_num(np.round(pool_composite * 100), nan=80.0), 50, 99
                    ).astype(int)
            except Exception as e:
                print(f"⚠️ Reranker failed during inference: {e}. Gracefully falling back to Stage 1 ranking.")

        # Stage 3: Adaptive Feedback Personalization
        if user_id:
            try:
                from src.personalization.preference_manager import AdaptivePersonalizer
                ap = AdaptivePersonalizer.get_instance()
                pool_size = min(len(candidates), max(top_k * 10, 100))
                top_indices = candidates.sort_values('composite_score', ascending=False).head(pool_size).index
                pool_df = candidates.loc[top_indices]
                
                pers_multipliers = [ap.compute_personalization_multiplier(user_id, row) for _, row in pool_df.iterrows()]
                candidates.loc[top_indices, 'personalization_multiplier'] = np.array(pers_multipliers).round(3)
                personalized_composite = candidates.loc[top_indices, 'composite_score'].values * np.array(pers_multipliers)
                candidates.loc[top_indices, 'composite_score'] = personalized_composite.round(3)
                candidates.loc[top_indices, 'match_percentage'] = np.clip(
                    np.nan_to_num(np.round(personalized_composite * 100), nan=80.0), 50, 99
                ).astype(int)
            except Exception as e:
                pass

        top_candidates = candidates.sort_values('composite_score', ascending=False).head(top_k)
        return top_candidates

    def compare_products(self, product_id_a, product_id_b, skin_type=None, concerns=None):
        """
        Performs in-depth algorithmic side-by-side comparison between two products in INR.
        """
        row_a = self.df[self.df['product_id'] == product_id_a]
        row_b = self.df[self.df['product_id'] == product_id_b]
        
        if len(row_a) == 0 or len(row_b) == 0:
            return None
            
        p_a = row_a.iloc[0].to_dict()
        p_b = row_b.iloc[0].to_dict()
        
        # Ensure price_inr
        p_a['price_inr'] = p_a.get('price_inr') or (round(p_a.get('price_usd', 0) * USD_TO_INR_RATE) if p_a.get('price_usd') else None)
        p_b['price_inr'] = p_b.get('price_inr') or (round(p_b.get('price_usd', 0) * USD_TO_INR_RATE) if p_b.get('price_usd') else None)
        
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
            
        # Knowledge Graph Layering Compatibility Check
        layering_conflicts = []
        try:
            from src.knowledge_graph.graph import IngredientKnowledgeGraph
            kg = IngredientKnowledgeGraph.get_instance()
            combined_actives = list(actives_a.union(actives_b))
            layering_conflicts = kg.detect_contraindications(combined_actives)
        except Exception:
            layering_conflicts = []

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
            'verdict': verdict,
            'layering_conflicts': layering_conflicts
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
            p_inr = r.get('price_inr') or (round(r.get('price_usd', 0) * USD_TO_INR_RATE) if r.get('price_usd') else None)
            results.append({
                'product_id': r['product_id'],
                'product_name': r['product_name'],
                'brand': r['brand'],
                'category': r['category'],
                'price_inr': p_inr,
                'rating': r['rating']
            })
        return results
