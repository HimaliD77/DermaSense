import os
import sys
import glob
import ast
import json
import re
import pandas as pd
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

DERMASENSE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_SEPHORA_DIR = os.path.join(DERMASENSE_DIR, "data", "raw", "Sephora")
PROCESSED_DIR = os.path.join(DERMASENSE_DIR, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

print("="*80)
print("PHASE 2: CLEANING & FILTERING SEPHORA DATASET")
print("="*80)

def parse_ingredients(ing_str):
    """Parses stringified ingredient lists into clean, normalized ingredient tokens."""
    if pd.isna(ing_str) or not str(ing_str).strip():
        return [], ""
    
    raw_str = str(ing_str).strip()
    parsed_items = []
    
    # Try ast.literal_eval
    if raw_str.startswith('[') and raw_str.endswith(']'):
        try:
            items = ast.literal_eval(raw_str)
            if isinstance(items, list):
                parsed_items = [str(x).strip() for x in items if x]
        except Exception:
            # Fallback regex split
            parsed_items = re.findall(r"'([^']*)'", raw_str)
            if not parsed_items:
                parsed_items = re.findall(r'"([^"]*)"', raw_str)
    
    if not parsed_items:
        parsed_items = [x.strip() for x in raw_str.split(',') if x.strip()]
        
    cleaned_tokens = []
    for item in parsed_items:
        # Ignore marketing headings ending with colon or labels
        if item.endswith(':') or len(item) < 2 or 'clean at sephora' in item.lower():
            continue
        # Standardize common variations
        cleaned = re.sub(r'\s+', ' ', item).strip(" '\",.")
        if cleaned:
            cleaned_tokens.append(cleaned)
            
    raw_joined = ", ".join(cleaned_tokens)
    return cleaned_tokens, raw_joined

def map_sephora_category(row):
    """Maps Sephora hierarchical categories to standardized DermaSense category taxonomy."""
    primary = str(row.get('primary_category', '')).strip().lower()
    sec = str(row.get('secondary_category', '')).strip()
    tert = str(row.get('tertiary_category', '')).strip()
    
    sec_lower = sec.lower()
    
    if 'moisturiz' in sec_lower:
        return 'Moisturizer'
    elif 'treatment' in sec_lower or 'serum' in sec_lower:
        return 'Treatment & Serum'
    elif 'cleanser' in sec_lower or 'wash' in sec_lower:
        return 'Cleanser'
    elif 'eye' in sec_lower:
        return 'Eye Care'
    elif 'mask' in sec_lower:
        return 'Mask'
    elif 'sunscreen' in sec_lower or 'sun' in sec_lower:
        return 'Sunscreen'
    elif 'lip' in sec_lower:
        return 'Lip Care'
    elif 'self tanner' in sec_lower:
        return 'Self Tanner'
    elif 'wellness' in sec_lower:
        return 'Wellness'
    elif 'tool' in sec_lower or 'device' in sec_lower:
        return 'Skincare Device'
    elif 'gift' in sec_lower or 'set' in sec_lower:
        return 'Skincare Set'
    elif primary == 'skincare':
        return 'General Skincare'
    return 'Other'

def clean_sephora_products():
    """Cleans and filters Sephora product catalog for skincare domain."""
    prod_path = os.path.join(RAW_SEPHORA_DIR, "product_info.csv")
    print(f"\n1. Loading Sephora Products from: {prod_path}")
    df = pd.read_csv(prod_path)
    print(f"   Raw product count: {len(df):,}")
    
    # Filter for Skincare domain
    skincare_mask = (
        (df['primary_category'].str.lower() == 'skincare') |
        (df['secondary_category'].astype(str).str.lower().str.contains('moisturiz|cleanser|treatment|sunscreen|eye care|masks', na=False))
    )
    df_skin = df[skincare_mask].copy()
    print(f"   Filtered Skincare products: {len(df_skin):,} ({len(df_skin)/len(df)*100:.2f}% of catalog)")
    
    # Process Ingredients
    print("   Parsing and normalizing ingredients...")
    parsed_lists = []
    raw_joined_list = []
    for ing in df_skin['ingredients']:
        toks, raw_j = parse_ingredients(ing)
        parsed_lists.append(json.dumps(toks))
        raw_joined_list.append(raw_j)
        
    df_skin['ingredients_list'] = parsed_lists
    df_skin['ingredients_raw'] = raw_joined_list
    
    # Standardize Category
    df_skin['standardized_category'] = df_skin.apply(map_sephora_category, axis=1)
    
    # Standardize Pricing & Ratings
    df_skin['price_usd'] = pd.to_numeric(df_skin['price_usd'], errors='coerce')
    df_skin['rating'] = pd.to_numeric(df_skin['rating'], errors='coerce')
    df_skin['reviews'] = pd.to_numeric(df_skin['reviews'], errors='coerce')
    df_skin['loves_count'] = pd.to_numeric(df_skin['loves_count'], errors='coerce').fillna(0).astype(int)
    
    # Clean highlights
    df_skin['highlights_clean'] = df_skin['highlights'].apply(
        lambda x: json.dumps([i.strip(" '\"") for i in ast.literal_eval(x)]) if pd.notna(x) and str(x).startswith('[') else "[]"
    )
    
    # Select and organize columns
    clean_cols = [
        'product_id', 'product_name', 'brand_id', 'brand_name',
        'standardized_category', 'primary_category', 'secondary_category', 'tertiary_category',
        'price_usd', 'value_price_usd', 'sale_price_usd',
        'rating', 'reviews', 'loves_count',
        'ingredients_list', 'ingredients_raw',
        'size', 'variation_type', 'variation_value',
        'highlights_clean', 'online_only', 'out_of_stock', 'sephora_exclusive'
    ]
    df_clean = df_skin[[c for c in clean_cols if c in df_skin.columns]].copy()
    df_clean['source'] = 'sephora'
    
    # Save processed products
    csv_out = os.path.join(PROCESSED_DIR, "sephora_products.csv")
    parquet_out = os.path.join(PROCESSED_DIR, "sephora_products.parquet")
    df_clean.to_csv(csv_out, index=False, encoding='utf-8')
    df_clean.to_parquet(parquet_out, index=False)
    print(f"   Saved cleaned Sephora products: {len(df_clean):,} rows to {csv_out}")
    return set(df_clean['product_id'].astype(str).unique())

def clean_sephora_reviews(valid_product_ids):
    """Filters and standardizes Sephora reviews matching the cleaned skincare catalog."""
    review_files = sorted(glob.glob(os.path.join(RAW_SEPHORA_DIR, "reviews_*.csv")))
    print(f"\n2. Processing Sephora Reviews from {len(review_files)} files...")
    
    chunks = []
    total_raw = 0
    total_retained = 0
    
    for rf in review_files:
        fname = os.path.basename(rf)
        print(f"   Reading {fname}...")
        for chunk in pd.read_csv(rf, chunksize=100000, low_memory=False):
            total_raw += len(chunk)
            # Filter for products in our skincare catalog
            chunk_filtered = chunk[chunk['product_id'].astype(str).isin(valid_product_ids)].copy()
            if len(chunk_filtered) > 0:
                # Standardize skin_type
                chunk_filtered['skin_type'] = chunk_filtered['skin_type'].astype(str).str.lower().str.strip()
                chunk_filtered['skin_type'] = chunk_filtered['skin_type'].replace({
                    'nan': 'unknown', 'none': 'unknown', '': 'unknown'
                })
                # Standardize skin_tone
                chunk_filtered['skin_tone'] = chunk_filtered['skin_tone'].astype(str).str.lower().str.strip()
                chunk_filtered['skin_tone'] = chunk_filtered['skin_tone'].replace({
                    'nan': 'unknown', 'none': 'unknown', '': 'unknown'
                })
                # Clean text
                chunk_filtered['review_text'] = chunk_filtered['review_text'].fillna("").astype(str).str.strip()
                chunk_filtered['review_title'] = chunk_filtered['review_title'].fillna("").astype(str).str.strip()
                # Clean rating
                chunk_filtered['rating'] = pd.to_numeric(chunk_filtered['rating'], errors='coerce').fillna(0).astype(int)
                chunk_filtered['is_recommended'] = pd.to_numeric(chunk_filtered['is_recommended'], errors='coerce')
                
                # Keep essential columns
                keep_cols = [
                    'author_id', 'product_id', 'rating', 'is_recommended',
                    'helpfulness', 'total_feedback_count', 'total_pos_feedback_count', 'total_neg_feedback_count',
                    'submission_time', 'review_title', 'review_text',
                    'skin_type', 'skin_tone', 'eye_color', 'hair_color'
                ]
                chunks.append(chunk_filtered[keep_cols])
                total_retained += len(chunk_filtered)
                
    print(f"   Total raw reviews read: {total_raw:,}")
    print(f"   Retained skincare reviews: {total_retained:,} ({total_retained/total_raw*100:.2f}%)")
    
    if chunks:
        df_all_rev = pd.concat(chunks, ignore_index=True)
        # Explicit type casting for clean parquet/csv serialization
        for col in ['author_id', 'product_id', 'submission_time', 'review_title', 'review_text', 'skin_type', 'skin_tone', 'eye_color', 'hair_color']:
            df_all_rev[col] = df_all_rev[col].fillna("").astype(str)
        
        # Deduplicate
        initial_len = len(df_all_rev)
        df_all_rev.drop_duplicates(subset=['author_id', 'product_id', 'submission_time', 'rating'], inplace=True)
        print(f"   Deduplicated reviews: {len(df_all_rev):,} (removed {initial_len - len(df_all_rev):,} duplicates)")
        
        df_all_rev['source'] = 'sephora'
        
        csv_out = os.path.join(PROCESSED_DIR, "sephora_reviews.csv")
        parquet_out = os.path.join(PROCESSED_DIR, "sephora_reviews.parquet")
        df_all_rev.to_csv(csv_out, index=False, encoding='utf-8')
        df_all_rev.to_parquet(parquet_out, index=False)
        print(f"   Saved cleaned Sephora reviews: {len(df_all_rev):,} rows to {csv_out} and {parquet_out}")
    else:
        print("   WARNING: No reviews matched skincare products!")

if __name__ == "__main__":
    valid_prod_ids = clean_sephora_products()
    clean_sephora_reviews(valid_prod_ids)
    print("\nSEPHORA CLEANING & FILTERING COMPLETE!")
