import os
import sys
import json
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

DERMASENSE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROCESSED_DIR = os.path.join(DERMASENSE_DIR, "data", "processed")

# Skincare Knowledge Ontology: Ingredient -> Target Benefit / Concern
SKIN_BENEFITS_MAP = {
    'Hydration & Barrier': [
        'hyaluronic acid', 'sodium hyaluronate', 'glycerin', 'squalane', 'ceramide',
        'ceramide np', 'ceramide ap', 'ceramide eop', 'panthenol', 'shea butter',
        'aloe barbadensis', 'aloe vera', 'urea', 'polyglutamic acid', 'sodium pca',
        'trehalose', 'beta-glucan', 'cholesterol', 'phytosphingosine', 'jojoba oil'
    ],
    'Acne & Blemish Control': [
        'salicylic acid', 'benzoyl peroxide', 'tea tree', 'melaleuca alternifolia',
        'willow bark', 'zinc pca', 'azelaic acid', 'sulfur', 'witch hazel',
        'hamamelis virginiana', 'neem'
    ],
    'Exfoliation & Texture': [
        'glycolic acid', 'lactic acid', 'mandelic acid', 'pha', 'gluconolactone',
        'lactobionic acid', 'papain', 'bromelain', 'citric acid', 'malic acid',
        'tartaric acid'
    ],
    'Brightening & Tone': [
        'niacinamide', 'vitamin c', 'ascorbic acid', 'tetrahexyldecyl ascorbate',
        'ascorbyl glucoside', 'magnesium ascorbyl phosphate', 'arbutin', 'alpha-arbutin',
        'kojic acid', 'tranexamic acid', 'licorice root', 'glycyrrhiza glabra',
        'glutathione', 'turmeric', 'curcuma longa'
    ],
    'Anti-Aging & Firming': [
        'retinol', 'retinal', 'retinoid', 'hydroxypinacolone retinoate', 'bakuchiol',
        'peptide', 'palmitoyl tripeptide', 'copper tripeptide', 'acetyl hexapeptide',
        'collagen', 'adenosine', 'resveratrol', 'ubiquinone', 'coq10', 'matrixyl'
    ],
    'Soothing & Anti-Redness': [
        'centella asiatica', 'madecassoside', 'asiaticoside', 'allantoin',
        'colloidal oatmeal', 'avena sativa', 'bisabolol', 'camellia sinensis',
        'green tea', 'calendula', 'artemisia', 'mugwort', 'chamomile', 'heartleaf'
    ],
    'Sun Protection': [
        'zinc oxide', 'titanium dioxide', 'avobenzone', 'homosalate',
        'octisalate', 'octocrylene', 'tinosorb', 'uvinul', 'mexoryl'
    ],
    'Oil & Pore Control': [
        'kaolin', 'bentonite', 'charcoal', 'zinc gluconate', 'silica'
    ]
}

def extract_ingredient_benefits(ingredients_list_json, description_text=""):
    """
    Analyzes ingredient tokens and description to identify matched skincare benefits and key actives.
    """
    try:
        tokens = json.loads(ingredients_list_json) if isinstance(ingredients_list_json, str) else []
    except Exception:
        tokens = []
        
    combined_text = (" ".join(tokens) + " " + str(description_text)).lower()
    
    matched_benefits = {}
    matched_actives = []
    
    for benefit_category, keywords in SKIN_BENEFITS_MAP.items():
        found = []
        for kw in keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', combined_text):
                found.append(kw)
                if kw not in matched_actives:
                    matched_actives.append(kw)
        if found:
            matched_benefits[benefit_category] = found
            
    return {
        'matched_benefits': list(matched_benefits.keys()),
        'matched_benefits_count': len(matched_benefits),
        'matched_actives': matched_actives,
        'matched_actives_count': len(matched_actives),
        'benefits_breakdown': matched_benefits
    }

def build_ingredient_features(df_products):
    """
    Extracts structured ingredient benefit features and TF-IDF representations.
    """
    print("\n--- Generating Ingredient & Benefit Features ---")
    
    benefits_col = []
    actives_col = []
    benefit_counts = []
    
    # Pre-create columns for each skin benefit category (multi-hot)
    benefit_flags = {k: [] for k in SKIN_BENEFITS_MAP.keys()}
    
    for idx, row in df_products.iterrows():
        ing_json = row.get('ingredients_list', '[]')
        desc = row.get('description', '')
        
        info = extract_ingredient_benefits(ing_json, desc)
        benefits_col.append(json.dumps(info['matched_benefits']))
        actives_col.append(json.dumps(info['matched_actives']))
        benefit_counts.append(info['matched_benefits_count'])
        
        for k in SKIN_BENEFITS_MAP.keys():
            benefit_flags[k].append(1 if k in info['matched_benefits'] else 0)
            
    df_feat = df_products.copy()
    df_feat['matched_benefits'] = benefits_col
    df_feat['matched_actives'] = actives_col
    df_feat['active_benefits_count'] = benefit_counts
    
    for k, flag_list in benefit_flags.items():
        clean_col = "benefit_" + re.sub(r'[^a-zA-Z0-9]', '_', k.lower()).strip('_')
        df_feat[clean_col] = flag_list
        
    print(f"   Processed {len(df_feat):,} products with ingredient benefits.")
    print("   Benefit Coverage across catalog:")
    for k in SKIN_BENEFITS_MAP.keys():
        clean_col = "benefit_" + re.sub(r'[^a-zA-Z0-9]', '_', k.lower()).strip('_')
        cnt = df_feat[clean_col].sum()
        pct = (cnt / len(df_feat)) * 100
        print(f"     - {k:25s}: {cnt:5,d} products ({pct:5.2f}%)")
        
    return df_feat

if __name__ == "__main__":
    prods_path = os.path.join(PROCESSED_DIR, "unified_products.parquet")
    if os.path.exists(prods_path):
        df_prods = pd.read_parquet(prods_path)
        df_feat = build_ingredient_features(df_prods)
        out_path = os.path.join(PROCESSED_DIR, "products_with_ingredient_features.parquet")
        df_feat.to_parquet(out_path, index=False)
        print(f"   Saved to {out_path}")
