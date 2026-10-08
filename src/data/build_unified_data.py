import os
import sys
import json
import pandas as pd
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

DERMASENSE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROCESSED_DIR = os.path.join(DERMASENSE_DIR, "data", "processed")

print("="*80)
print("PHASE 2: BUILDING STANDARDIZED SOURCE-AWARE UNIFIED DATASETS")
print("="*80)

def build_unified_products():
    """Combines cleaned products across Sephora, Amazon, and Flipkart into a unified catalog."""
    print("\n1. Standardizing and Combining Product Catalogs...")
    
    # 1. Sephora
    df_sep = pd.read_parquet(os.path.join(PROCESSED_DIR, "sephora_products.parquet"))
    df_sep_std = pd.DataFrame({
        'product_id': 'sephora_' + df_sep['product_id'].astype(str),
        'source': 'sephora',
        'source_product_id': df_sep['product_id'].astype(str),
        'product_name': df_sep['product_name'].astype(str),
        'brand': df_sep['brand_name'].astype(str),
        'category': df_sep['standardized_category'].astype(str),
        'subcategory': df_sep['secondary_category'].fillna("").astype(str),
        'description': df_sep['highlights_clean'].fillna("[]").astype(str),
        'ingredients_raw': df_sep['ingredients_raw'].fillna("").astype(str),
        'ingredients_list': df_sep['ingredients_list'].fillna("[]").astype(str),
        'price_usd': pd.to_numeric(df_sep['price_usd'], errors='coerce'),
        'price_inr': (pd.to_numeric(df_sep['price_usd'], errors='coerce') * 83.0).round(2),
        'rating': pd.to_numeric(df_sep['rating'], errors='coerce'),
        'review_count': pd.to_numeric(df_sep['reviews'], errors='coerce').fillna(0).astype(int),
        'skin_types_targeted': 'All'
    })
    
    # 2. Amazon
    df_amz = pd.read_parquet(os.path.join(PROCESSED_DIR, "amazon_products.parquet"))
    df_amz_std = pd.DataFrame({
        'product_id': 'amazon_' + df_amz['product_id'].astype(str),
        'source': 'amazon',
        'source_product_id': df_amz['product_id'].astype(str),
        'product_name': df_amz['product_name'].astype(str),
        'brand': df_amz['brand'].astype(str),
        'category': df_amz['standardized_category'].astype(str),
        'subcategory': df_amz['product_form'].fillna("").astype(str),
        'description': df_amz['description'].fillna("").astype(str),
        'ingredients_raw': '',
        'ingredients_list': '[]',
        'price_usd': pd.to_numeric(df_amz['price_usd'], errors='coerce'),
        'price_inr': (pd.to_numeric(df_amz['price_usd'], errors='coerce') * 83.0).round(2),
        'rating': pd.to_numeric(df_amz['rating'], errors='coerce'),
        'review_count': pd.to_numeric(df_amz['review_count'], errors='coerce').fillna(0).astype(int),
        'skin_types_targeted': df_amz['skin_types_targeted'].fillna("All").astype(str)
    })
    
    # 3. Flipkart
    df_fk = pd.read_parquet(os.path.join(PROCESSED_DIR, "flipkart_products.parquet"))
    df_fk_std = pd.DataFrame({
        'product_id': 'flipkart_' + df_fk['product_id'].astype(str),
        'source': 'flipkart',
        'source_product_id': df_fk['product_id'].astype(str),
        'product_name': df_fk['product_name'].astype(str),
        'brand': df_fk['brand'].astype(str),
        'category': df_fk['standardized_category'].astype(str),
        'subcategory': '',
        'description': '',
        'ingredients_raw': '',
        'ingredients_list': '[]',
        'price_usd': pd.to_numeric(df_fk['price_usd'], errors='coerce'),
        'price_inr': pd.to_numeric(df_fk['price_inr'], errors='coerce'),
        'rating': pd.to_numeric(df_fk['rating'], errors='coerce'),
        'review_count': pd.to_numeric(df_fk['review_count'], errors='coerce').fillna(0).astype(int),
        'skin_types_targeted': 'All'
    })
    
    # Concatenate unified products
    df_unified_prods = pd.concat([df_sep_std, df_amz_std, df_fk_std], ignore_index=True)
    print(f"   Unified Products Count: {len(df_unified_prods):,}")
    print(f"     - Sephora: {len(df_sep_std):,}")
    print(f"     - Amazon:  {len(df_amz_std):,}")
    print(f"     - Flipkart:{len(df_fk_std):,}")
    
    # Save unified products
    csv_out = os.path.join(PROCESSED_DIR, "unified_products.csv")
    parquet_out = os.path.join(PROCESSED_DIR, "unified_products.parquet")
    df_unified_prods.to_csv(csv_out, index=False, encoding='utf-8')
    df_unified_prods.to_parquet(parquet_out, index=False)
    print(f"   Saved unified products to {csv_out} and {parquet_out}")

def build_unified_reviews():
    """Combines cleaned reviews across Sephora, Amazon, and Flipkart into a unified repository."""
    print("\n2. Standardizing and Combining Reviews...")
    
    # 1. Sephora Reviews
    print("   Loading Sephora reviews...")
    df_sep_rev = pd.read_parquet(os.path.join(PROCESSED_DIR, "sephora_reviews.parquet"))
    df_sep_std = pd.DataFrame({
        'review_id': 'sephora_rev_' + df_sep_rev.index.astype(str),
        'source': 'sephora',
        'product_id': 'sephora_' + df_sep_rev['product_id'].astype(str),
        'source_product_id': df_sep_rev['product_id'].astype(str),
        'user_id': df_sep_rev['author_id'].astype(str),
        'rating': pd.to_numeric(df_sep_rev['rating'], errors='coerce'),
        'review_title': df_sep_rev['review_title'].fillna("").astype(str),
        'review_text': df_sep_rev['review_text'].fillna("").astype(str),
        'user_skin_type': df_sep_rev['skin_type'].replace({'': 'unknown', 'nan': 'unknown', 'None': 'unknown'}).fillna("unknown").astype(str),
        'user_skin_tone': df_sep_rev['skin_tone'].replace({'': 'unknown', 'nan': 'unknown', 'None': 'unknown'}).fillna("unknown").astype(str),
        'verified_purchase': pd.NA,
        'helpful_votes': pd.to_numeric(df_sep_rev['total_pos_feedback_count'], errors='coerce').fillna(0).astype(int),
        'timestamp': df_sep_rev['submission_time'].fillna("").astype(str)
    })
    
    # 2. Amazon Reviews
    print("   Loading Amazon reviews...")
    df_amz_rev = pd.read_parquet(os.path.join(PROCESSED_DIR, "amazon_reviews.parquet"))
    df_amz_std = pd.DataFrame({
        'review_id': 'amazon_rev_' + df_amz_rev.index.astype(str),
        'source': 'amazon',
        'product_id': 'amazon_' + df_amz_rev['product_id'].astype(str),
        'source_product_id': df_amz_rev['product_id'].astype(str),
        'user_id': df_amz_rev['user_id'].astype(str),
        'rating': pd.to_numeric(df_amz_rev['rating'], errors='coerce'),
        'review_title': df_amz_rev['review_title'].fillna("").astype(str),
        'review_text': df_amz_rev['review_text'].fillna("").astype(str),
        'user_skin_type': 'unknown',
        'user_skin_tone': 'unknown',
        'verified_purchase': df_amz_rev['verified_purchase'],
        'helpful_votes': pd.to_numeric(df_amz_rev['helpful_votes'], errors='coerce').fillna(0).astype(int),
        'timestamp': df_amz_rev['submission_time'].fillna("").astype(str)
    })
    
    # 3. Flipkart Reviews
    print("   Loading Flipkart reviews...")
    df_fk_rev = pd.read_parquet(os.path.join(PROCESSED_DIR, "flipkart_reviews.parquet"))
    df_fk_std = pd.DataFrame({
        'review_id': 'flipkart_rev_' + df_fk_rev.index.astype(str),
        'source': 'flipkart',
        'product_id': 'flipkart_' + df_fk_rev['product_id'].astype(str),
        'source_product_id': df_fk_rev['product_id'].astype(str),
        'user_id': df_fk_rev['user_id'].astype(str),
        'rating': pd.to_numeric(df_fk_rev['rating'], errors='coerce'),
        'review_title': df_fk_rev['review_title'].fillna("").astype(str),
        'review_text': df_fk_rev['review_text'].fillna("").astype(str),
        'user_skin_type': 'unknown',
        'user_skin_tone': 'unknown',
        'verified_purchase': pd.NA,
        'helpful_votes': 0,
        'timestamp': ""
    })
    
    # Concatenate unified reviews
    print("   Concatenating unified reviews...")
    df_unified_revs = pd.concat([df_sep_std, df_amz_std, df_fk_std], ignore_index=True)
    print(f"   Unified Reviews Count: {len(df_unified_revs):,}")
    print(f"     - Sephora: {len(df_sep_std):,}")
    print(f"     - Amazon:  {len(df_amz_std):,}")
    print(f"     - Flipkart:{len(df_fk_std):,}")
    
    # Save unified reviews
    csv_out = os.path.join(PROCESSED_DIR, "unified_reviews.csv")
    parquet_out = os.path.join(PROCESSED_DIR, "unified_reviews.parquet")
    df_unified_revs.to_csv(csv_out, index=False, encoding='utf-8')
    df_unified_revs.to_parquet(parquet_out, index=False)
    print(f"   Saved unified reviews to {csv_out} and {parquet_out}")

if __name__ == "__main__":
    build_unified_products()
    build_unified_reviews()
    print("\nSTANDARDIZED UNIFIED DATASETS CREATED SUCCESSFULLY!")
