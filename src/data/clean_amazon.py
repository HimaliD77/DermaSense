import os
import sys
import json
import re
from datetime import datetime, timezone
import pandas as pd
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

DERMASENSE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_DIR = os.path.join(DERMASENSE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(DERMASENSE_DIR, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

print("="*80)
print("PHASE 2: CLEANING & FILTERING AMAZON DATASET")
print("="*80)

SKINCARE_KEYWORDS = re.compile(
    r'\b(face wash|cleanser|cleansing|moisturiz\w*|serum|face cream|facial cream|sunscreen|spf\b|sunblock|'
    r'toner|eye cream|face mask|sheet mask|face scrub|anti-aging|anti aging|retinol|hyaluronic|'
    r'vitamin c|niacinamide|facial oil|acne|blemish|skin care|skincare|body lotion|facial lotion|'
    r'exfoliat\w*|collagen|peel|essence|centella|salicylic|glycolic)\b',
    re.IGNORECASE
)

EXCLUDE_KEYWORDS = re.compile(
    r'\b(leather|wig|wigs|hair extension|hair weave|clipper|clippers|tattoo|tattoos|'
    r'nail polish|nail clipper|nail art|shoe|shoes|industrial|hair dye|beard trimmer|'
    r'eyelash extension glue|lace front|braiding hair)\b',
    re.IGNORECASE
)

def parse_price(price_val):
    """Extracts numeric price float from raw string or number."""
    if pd.isna(price_val) or price_val is None:
        return np.nan
    if isinstance(price_val, (int, float)):
        return float(price_val) if price_val > 0 else np.nan
    val_str = str(price_val).strip()
    match = re.search(r'[\$]?\s*(\d+(\.\d{1,2})?)', val_str)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return np.nan
    return np.nan

def map_amazon_category(title, details, categories):
    """Categorizes Amazon products into standardized DermaSense taxonomy."""
    text = (str(title) + " " + " ".join(categories) + " " + str(details.get('Item Form', ''))).lower()
    
    if re.search(r'\b(moisturiz\w*|face cream|night cream|day cream|hydrating cream)\b', text):
        return 'Moisturizer'
    elif re.search(r'\b(serum|treatment|retinol|ampoule|essence|acid)\b', text):
        return 'Treatment & Serum'
    elif re.search(r'\b(cleanser|face wash|cleansing|foam cleanser|gel cleanser)\b', text):
        return 'Cleanser'
    elif re.search(r'\b(sunscreen|spf|sunblock|sun protection)\b', text):
        return 'Sunscreen'
    elif re.search(r'\b(eye cream|eye serum|under eye|eye gel)\b', text):
        return 'Eye Care'
    elif re.search(r'\b(mask|sheet mask|clay mask|peel off)\b', text):
        return 'Mask'
    elif re.search(r'\b(toner|mist|astringent)\b', text):
        return 'Toner & Mist'
    elif re.search(r'\b(scrub|exfoliat\w*|peeling)\b', text):
        return 'Exfoliator'
    elif re.search(r'\b(body lotion|body cream|body butter|body wash)\b', text):
        return 'Body Care'
    elif re.search(r'\b(lip balm|lip care|lip butter)\b', text):
        return 'Lip Care'
    return 'General Skincare'

def clean_amazon_products():
    """Filters and cleans Amazon metadata for skincare domain."""
    meta_path = os.path.join(RAW_DIR, "meta_All_Beauty.jsonl")
    print(f"\n1. Loading Amazon Metadata from: {meta_path}")
    
    cleaned_products = []
    total_count = 0
    
    with open(meta_path, 'r', encoding='utf-8') as f:
        for line in f:
            total_count += 1
            item = json.loads(line)
            
            title = str(item.get('title', '')).strip()
            details = item.get('details') or {}
            cats = item.get('categories') or []
            
            # Combine text for filtering
            search_blob = f"{title} {' '.join(cats)} {json.dumps(details)}"
            
            # Filter check
            if EXCLUDE_KEYWORDS.search(search_blob):
                continue
            if not SKINCARE_KEYWORDS.search(search_blob):
                continue
                
            p_asin = item.get('parent_asin')
            if not p_asin:
                continue
                
            # Extract attributes
            brand = item.get('store') or details.get('Brand') or details.get('Manufacturer') or "Generic"
            brand = str(brand).replace("Visit the ", "").replace(" Store", "").strip()
            
            skin_type = details.get('Skin Type') or details.get('Skin type') or "All"
            item_form = details.get('Item Form') or details.get('Item form') or ""
            unit_count = details.get('Unit Count') or details.get('Package Dimensions') or ""
            
            features = item.get('features') or []
            desc = item.get('description') or []
            full_desc = " ".join([str(x) for x in (features + desc) if str(x).strip()])
            
            price_usd = parse_price(item.get('price'))
            avg_rating = pd.to_numeric(item.get('average_rating'), errors='coerce')
            rating_num = pd.to_numeric(item.get('rating_number'), errors='coerce')
            
            category = map_amazon_category(title, details, cats)
            
            cleaned_products.append({
                'product_id': p_asin,
                'parent_asin': p_asin,
                'product_name': title,
                'brand': brand,
                'standardized_category': category,
                'product_form': str(item_form).strip(),
                'skin_types_targeted': str(skin_type).strip(),
                'size_info': str(unit_count).strip(),
                'price_usd': price_usd,
                'rating': avg_rating,
                'review_count': rating_num,
                'description': full_desc[:1000],  # trim for table
                'source': 'amazon'
            })
            
    df_prods = pd.DataFrame(cleaned_products)
    # Deduplicate on parent_asin
    df_prods.drop_duplicates(subset=['parent_asin'], inplace=True)
    
    print(f"   Total raw Amazon products: {total_count:,}")
    print(f"   Retained Skincare products: {len(df_prods):,} ({len(df_prods)/total_count*100:.2f}%)")
    print(f"   Category breakdown:\n{df_prods['standardized_category'].value_counts().to_string()}")
    
    csv_out = os.path.join(PROCESSED_DIR, "amazon_products.csv")
    parquet_out = os.path.join(PROCESSED_DIR, "amazon_products.parquet")
    df_prods.to_csv(csv_out, index=False, encoding='utf-8')
    df_prods.to_parquet(parquet_out, index=False)
    print(f"   Saved cleaned Amazon products to {csv_out} and {parquet_out}")
    
    return set(df_prods['parent_asin'].unique())

def clean_amazon_reviews(valid_asins):
    """Filters and standardizes Amazon reviews matching the cleaned skincare catalog."""
    rev_path = os.path.join(RAW_DIR, "All_Beauty.jsonl")
    print(f"\n2. Loading Amazon Reviews from: {rev_path}")
    
    cleaned_reviews = []
    total_count = 0
    retained_count = 0
    
    with open(rev_path, 'r', encoding='utf-8') as f:
        for line in f:
            total_count += 1
            item = json.loads(line)
            
            p_asin = item.get('parent_asin')
            if p_asin not in valid_asins:
                continue
                
            retained_count += 1
            uid = str(item.get('user_id', '')).strip()
            rating = pd.to_numeric(item.get('rating'), errors='coerce')
            title = str(item.get('title', '')).strip()
            text = str(item.get('text', '')).strip()
            
            # Timestamp conversion
            raw_ts = item.get('timestamp')
            iso_date = ""
            if raw_ts:
                try:
                    # ms timestamp
                    iso_date = datetime.fromtimestamp(raw_ts / 1000.0, tz=timezone.utc).strftime('%Y-%m-%d')
                except Exception:
                    iso_date = ""
                    
            helpful = pd.to_numeric(item.get('helpful_vote'), errors='coerce') or 0
            verified = bool(item.get('verified_purchase', False))
            
            cleaned_reviews.append({
                'user_id': uid,
                'product_id': p_asin,
                'parent_asin': p_asin,
                'asin': str(item.get('asin', '')).strip(),
                'rating': rating,
                'review_title': title,
                'review_text': text,
                'verified_purchase': verified,
                'helpful_votes': int(helpful),
                'submission_time': iso_date,
                'source': 'amazon'
            })
            
    print(f"   Total raw Amazon reviews: {total_count:,}")
    print(f"   Retained skincare reviews: {retained_count:,} ({retained_count/total_count*100:.2f}%)")
    
    df_rev = pd.DataFrame(cleaned_reviews)
    if len(df_rev) > 0:
        # Cast string columns
        for c in ['user_id', 'product_id', 'parent_asin', 'asin', 'review_title', 'review_text', 'submission_time', 'source']:
            df_rev[c] = df_rev[c].fillna("").astype(str)
            
        initial_len = len(df_rev)
        df_rev.drop_duplicates(subset=['user_id', 'parent_asin', 'submission_time', 'rating'], inplace=True)
        print(f"   Deduplicated reviews: {len(df_rev):,} (dropped {initial_len - len(df_rev):,} duplicates)")
        
        csv_out = os.path.join(PROCESSED_DIR, "amazon_reviews.csv")
        parquet_out = os.path.join(PROCESSED_DIR, "amazon_reviews.parquet")
        df_rev.to_csv(csv_out, index=False, encoding='utf-8')
        df_rev.to_parquet(parquet_out, index=False)
        print(f"   Saved cleaned Amazon reviews to {csv_out} and {parquet_out}")

if __name__ == "__main__":
    valid_asins = clean_amazon_products()
    clean_amazon_reviews(valid_asins)
    print("\nAMAZON CLEANING & FILTERING COMPLETE!")
