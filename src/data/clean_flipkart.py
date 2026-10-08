import os
import sys
import re
import hashlib
import pandas as pd
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

DERMASENSE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_DIR = os.path.join(DERMASENSE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(DERMASENSE_DIR, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

print("="*80)
print("PHASE 2: CLEANING & FILTERING FLIPKART DATASET")
print("="*80)

SKINCARE_KEYWORDS = re.compile(
    r'\b(cream|serum|moisturiz\w*|lotion|sunscreen|spf\b|face wash|cleanser|'
    r'skin|facial|toner|face scrub|face pack|face gel|acne|derma|blemish|'
    r'aloe vera|vitamin c|brightening|glow|kajal|hair oil|body oil)\b',
    re.IGNORECASE
)

EXCLUDE_KEYWORDS = re.compile(
    r'\b(cooler|theatre|speaker|tv|television|diaper|flask|dinner set|wallpaper|'
    r'sticker|tablet|electronic|toy|teddy|cooker|pot|table|desk|ear plug|tweeter|'
    r'hot water|stationery|crayon|pencil|casserole|fan|air cooler)\b',
    re.IGNORECASE
)

KNOWN_BRANDS = [
    'Nivea', 'MamaEarth', 'mCaffeine', 'Lakme', 'Lakm??', 'The Indie Earth',
    'Biotique', 'Ponds', "Pond's", 'Garnier', 'Himalaya', 'Plum', 'Minimalist',
    'Derma Co', 'Loreal', "L'Oreal", 'Cetaphil', 'Neutrogena', 'Dot & Key',
    'Khadi', 'Vaseline', 'Joy', 'Fair & Lovely', 'Glow & Lovely', 'HAIR & CARE'
]

def extract_brand(prod_name):
    """Extracts prominent brand name from Flipkart product title."""
    name = str(prod_name).strip()
    for b in KNOWN_BRANDS:
        if re.search(r'\b' + re.escape(b) + r'\b', name, re.IGNORECASE):
            if 'lakm' in b.lower():
                return 'Lakme'
            elif 'pond' in b.lower():
                return "Pond's"
            elif 'hair & care' in b.lower():
                return 'Hair & Care'
            return b
    # Fallback to first 2 words
    words = name.split()
    return " ".join(words[:2]) if len(words) >= 2 else (words[0] if words else "Generic")

def parse_inr_price(price_str):
    """Extracts numeric INR price float from messy string."""
    if pd.isna(price_str) or not str(price_str).strip():
        return np.nan
    clean_digits = re.sub(r'[^\d.]', '', str(price_str))
    try:
        val = float(clean_digits)
        return val if val > 0 else np.nan
    except ValueError:
        return np.nan

def map_flipkart_category(title):
    """Categorizes Flipkart skincare products."""
    t = str(title).lower()
    if 'moisturiz' in t or 'cream' in t:
        return 'Moisturizer'
    elif 'serum' in t:
        return 'Treatment & Serum'
    elif 'wash' in t or 'cleanser' in t:
        return 'Cleanser'
    elif 'sunscreen' in t or 'spf' in t:
        return 'Sunscreen'
    elif 'scrub' in t or 'pack' in t:
        return 'Exfoliator & Mask'
    elif 'toner' in t:
        return 'Toner & Mist'
    elif 'oil' in t:
        return 'Facial / Hair Oil'
    elif 'kajal' in t:
        return 'Eye Makeup'
    return 'General Skincare'

def generate_fk_product_id(name):
    """Generates a stable unique hash identifier for Flipkart product titles."""
    clean_name = re.sub(r'[^a-zA-Z0-9]', '', str(name).lower())
    short_hash = hashlib.md5(clean_name.encode('utf-8')).hexdigest()[:10]
    return f"fk_{short_hash}"

def clean_flipkart():
    """Filters, cleans, and standardizes Flipkart dataset."""
    fk_path = os.path.join(RAW_DIR, "Flipkart", "Dataset.csv")
    print(f"\n1. Loading Flipkart Dataset from: {fk_path}")
    
    # Read with latin1 encoding to handle non-utf8 characters
    df_raw = pd.read_csv(fk_path, encoding='latin1', low_memory=False)
    print(f"   Raw Flipkart records: {len(df_raw):,}")
    
    # Filter for Skincare / Personal Care
    names = df_raw['Product_name'].fillna("").astype(str)
    
    mask_include = names.apply(lambda x: bool(SKINCARE_KEYWORDS.search(x)))
    mask_exclude = names.apply(lambda x: bool(EXCLUDE_KEYWORDS.search(x)))
    
    df_filtered = df_raw[mask_include & ~mask_exclude].copy()
    print(f"   Retained Skincare/Cosmetic reviews: {len(df_filtered):,} ({len(df_filtered)/len(df_raw)*100:.2f}%)")
    print(f"   Unique skincare products identified: {df_filtered['Product_name'].nunique():,}")
    
    # Parse fields
    df_filtered['price_inr'] = df_filtered['Price'].apply(parse_inr_price)
    df_filtered['price_usd'] = (df_filtered['price_inr'] / 83.0).round(2)  # approximate conversion
    df_filtered['rating'] = pd.to_numeric(df_filtered['Rate'], errors='coerce').fillna(0).astype(int)
    
    df_filtered['review_title'] = df_filtered['Review'].fillna("").astype(str).str.strip()
    df_filtered['review_text'] = df_filtered['Summary'].fillna("").astype(str).str.strip()
    
    # Generate product ID and brand
    df_filtered['product_id'] = df_filtered['Product_name'].apply(generate_fk_product_id)
    df_filtered['brand'] = df_filtered['Product_name'].apply(extract_brand)
    df_filtered['standardized_category'] = df_filtered['Product_name'].apply(map_flipkart_category)
    df_filtered['source'] = 'flipkart'
    
    # Generate pseudo user_id based on review content hash
    df_filtered['user_id'] = df_filtered.apply(
        lambda r: "fk_user_" + hashlib.md5(f"{r['product_id']}_{r['rating']}_{r['review_text'][:30]}".encode('utf-8')).hexdigest()[:8],
        axis=1
    )
    
    # Build Flipkart Product Catalog
    print("\n2. Building aggregated Flipkart Product Catalog...")
    catalog_rows = []
    for prod_id, group in df_filtered.groupby('product_id'):
        first_row = group.iloc[0]
        avg_rating = round(float(group['rating'].mean()), 2)
        rev_count = len(group)
        med_price_inr = group['price_inr'].median()
        med_price_usd = group['price_usd'].median()
        
        catalog_rows.append({
            'product_id': prod_id,
            'source_product_id': prod_id,
            'product_name': str(first_row['Product_name']).strip(),
            'brand': first_row['brand'],
            'standardized_category': first_row['standardized_category'],
            'price_inr': med_price_inr,
            'price_usd': med_price_usd,
            'rating': avg_rating,
            'review_count': rev_count,
            'source': 'flipkart'
        })
        
    df_fk_prods = pd.DataFrame(catalog_rows)
    print(f"   Created Flipkart Product Catalog: {len(df_fk_prods):,} products")
    print(f"   Top Brands:\n{df_fk_prods['brand'].value_counts().head(10).to_string()}")
    
    # Organize Flipkart Reviews
    df_fk_reviews = df_filtered[[
        'user_id', 'product_id', 'rating', 'review_title', 'review_text',
        'price_inr', 'source'
    ]].copy()
    
    # Deduplicate reviews
    initial_len = len(df_fk_reviews)
    df_fk_reviews.drop_duplicates(subset=['product_id', 'rating', 'review_text'], inplace=True)
    print(f"   Deduplicated reviews: {len(df_fk_reviews):,} (dropped {initial_len - len(df_fk_reviews):,} duplicates)")
    
    # Save datasets
    prod_csv = os.path.join(PROCESSED_DIR, "flipkart_products.csv")
    prod_parquet = os.path.join(PROCESSED_DIR, "flipkart_products.parquet")
    rev_csv = os.path.join(PROCESSED_DIR, "flipkart_reviews.csv")
    rev_parquet = os.path.join(PROCESSED_DIR, "flipkart_reviews.parquet")
    
    df_fk_prods.to_csv(prod_csv, index=False, encoding='utf-8')
    df_fk_prods.to_parquet(prod_parquet, index=False)
    
    df_fk_reviews.to_csv(rev_csv, index=False, encoding='utf-8')
    df_fk_reviews.to_parquet(rev_parquet, index=False)
    
    print(f"   Saved cleaned Flipkart products to {prod_csv}")
    print(f"   Saved cleaned Flipkart reviews to {rev_csv}")

if __name__ == "__main__":
    clean_flipkart()
    print("\nFLIPKART CLEANING & FILTERING COMPLETE!")
