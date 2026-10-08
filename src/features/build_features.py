import os
import sys
import json
import pandas as pd
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

DERMASENSE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROCESSED_DIR = os.path.join(DERMASENSE_DIR, "data", "processed")

from ingredient_features import build_ingredient_features
from review_features import aggregate_review_features

def build_master_features():
    """Combines catalog, ingredient ontology, and review signals into master feature tables."""
    print("="*80)
    print("PHASE 3: BUILDING MASTER DERMASENSE FEATURE REPOSITORY")
    print("="*80)
    
    prods_path = os.path.join(PROCESSED_DIR, "unified_products.parquet")
    revs_path = os.path.join(PROCESSED_DIR, "unified_reviews.parquet")
    
    print(f"Loading products from: {prods_path}")
    df_prods = pd.read_parquet(prods_path)
    print(f"Loading reviews from: {revs_path}")
    df_revs = pd.read_parquet(revs_path)
    
    # 1. Ingredient & Benefit Features
    df_prods_ing = build_ingredient_features(df_prods)
    
    # 2. Review Features & Skin Type Aggregations
    df_rev_feat = aggregate_review_features(df_revs)
    
    # 3. Merge Features
    print("\n--- Merging Product & Review Signal Matrices ---")
    df_master = pd.merge(df_prods_ing, df_rev_feat, on='product_id', how='left')
    
    # Fill defaults for products without review partitions
    df_master['review_count_actual'] = df_master['review_count_actual'].fillna(0).astype(int)
    df_master['avg_rating_calculated'] = df_master['avg_rating_calculated'].fillna(df_master['rating'])
    
    # 4. Construct Semantic Text Document for Embedding / Retrieval
    print("--- Constructing Semantic Text Representations ---")
    semantic_docs = []
    for _, row in df_master.iterrows():
        pname = row.get('product_name', '')
        pbrand = row.get('brand', '')
        pcat = row.get('category', '')
        
        try:
            benefits = ", ".join(json.loads(row.get('matched_benefits', '[]')))
        except Exception:
            benefits = ""
            
        try:
            actives = ", ".join(json.loads(row.get('matched_actives', '[]')))
        except Exception:
            actives = ""
            
        desc = str(row.get('description', ''))[:300]
        st_target = str(row.get('skin_types_targeted', ''))
        
        doc = f"Product: {pname}. Brand: {pbrand}. Category: {pcat}. "
        if st_target and st_target != 'All':
            doc += f"Target Skin Type: {st_target}. "
        if benefits:
            doc += f"Skincare Benefits: {benefits}. "
        if actives:
            doc += f"Key Active Ingredients: {actives}. "
        if desc:
            doc += f"Details: {desc}."
            
        semantic_docs.append(doc.strip())
        
    df_master['semantic_document'] = semantic_docs
    
    # Save Master Features Table
    out_parquet = os.path.join(PROCESSED_DIR, "master_product_features.parquet")
    out_csv = os.path.join(PROCESSED_DIR, "master_product_features.csv")
    df_master.to_parquet(out_parquet, index=False)
    df_master.to_csv(out_csv, index=False, encoding='utf-8')
    
    print(f"\n Master Feature Table Created: {len(df_master):,} products × {len(df_master.columns)} features")
    print(f" Saved to: {out_parquet} and {out_csv}")
    print("="*80)

if __name__ == "__main__":
    build_master_features()
