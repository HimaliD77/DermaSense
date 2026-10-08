import os
import sys
import pandas as pd
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

DERMASENSE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROCESSED_DIR = os.path.join(DERMASENSE_DIR, "data", "processed")

def aggregate_review_features(df_reviews):
    """
    Computes product-level review sentiment and skin-type specific performance metrics.
    """
    print("\n--- Aggregating Review Signals & Skin Type Evidence ---")
    print(f"   Input reviews: {len(df_reviews):,}")
    
    # Clean rating and skin type
    df_rev = df_reviews.copy()
    df_rev['rating'] = pd.to_numeric(df_rev['rating'], errors='coerce')
    df_rev['is_pos'] = (df_rev['rating'] >= 4.0).astype(int)
    df_rev['text_length'] = df_rev['review_text'].fillna("").astype(str).str.len()
    
    # Overall product aggregations
    grouped = df_rev.groupby('product_id')
    
    overall_agg = grouped.agg(
        review_count_actual=('rating', 'count'),
        avg_rating_calculated=('rating', 'mean'),
        median_rating=('rating', 'median'),
        pct_positive_reviews=('is_pos', 'mean'),
        avg_review_char_length=('text_length', 'mean'),
        helpful_votes_sum=('helpful_votes', 'sum')
    ).reset_index()
    
    overall_agg['avg_rating_calculated'] = overall_agg['avg_rating_calculated'].round(2)
    overall_agg['pct_positive_reviews'] = (overall_agg['pct_positive_reviews'] * 100).round(1)
    overall_agg['avg_review_char_length'] = overall_agg['avg_review_char_length'].round(1)
    
    # Skin Type Specific aggregations
    skin_types = ['combination', 'dry', 'normal', 'oily']
    skin_type_dfs = []
    
    for st in skin_types:
        st_revs = df_rev[df_rev['user_skin_type'] == st]
        st_grp = st_revs.groupby('product_id').agg(
            **{
                f'{st}_skin_avg_rating': ('rating', 'mean'),
                f'{st}_skin_review_count': ('rating', 'count'),
                f'{st}_skin_pos_pct': ('is_pos', 'mean')
            }
        ).reset_index()
        
        st_grp[f'{st}_skin_avg_rating'] = st_grp[f'{st}_skin_avg_rating'].round(2)
        st_grp[f'{st}_skin_pos_pct'] = (st_grp[f'{st}_skin_pos_pct'] * 100).round(1)
        skin_type_dfs.append(st_grp)
        
    # Merge all skin type metrics
    review_features_df = overall_agg
    for st_df in skin_type_dfs:
        review_features_df = pd.merge(review_features_df, st_df, on='product_id', how='left')
        
    print(f"   Computed review signals for {len(review_features_df):,} distinct products.")
    return review_features_df

if __name__ == "__main__":
    revs_path = os.path.join(PROCESSED_DIR, "unified_reviews.parquet")
    if os.path.exists(revs_path):
        df_revs = pd.read_parquet(revs_path)
        df_rf = aggregate_review_features(df_revs)
        out_path = os.path.join(PROCESSED_DIR, "product_review_features.parquet")
        df_rf.to_parquet(out_path, index=False)
        print(f"   Saved to {out_path}")
