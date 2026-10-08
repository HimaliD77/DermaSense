import os
import sys
import json
import re
import time
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pyarrow.parquet as pq

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

DERMASENSE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if DERMASENSE_DIR not in sys.path:
    sys.path.insert(0, DERMASENSE_DIR)

PROCESSED_DIR = os.path.join(DERMASENSE_DIR, "data", "processed")

POSITIVE_THEMES_PATTERNS = {
    'Deep Hydration': re.compile(r'\b(deeply hydrat\w*|hydrat\w+|moisturiz\w+|plump\w*|soft skin)\b', re.IGNORECASE),
    'Calming & Soothing': re.compile(r'\b(sooth\w+|calm\w+|reduces? redness|anti-redness|gentle)\b', re.IGNORECASE),
    'Acne & Blemish Control': re.compile(r'\b(cleared (?:my )?acne|reduces? breakouts?|fewer pimples|clears? blemishes?|cleared skin)\b', re.IGNORECASE),
    'Lightweight & Fast-Absorbing': re.compile(r'\b(lightweight|fast absorb\w*|absorbs quickly|non-greasy|not greasy|absorbs fast)\b', re.IGNORECASE),
    'Brightening & Radiance': re.compile(r'\b(brighten\w*|glowing|radiant|faded dark spots|even skin tone|glow)\b', re.IGNORECASE),
    'Texture Smoothing': re.compile(r'\b(smooth\w*|softens? skin|improved texture|silky skin|baby soft)\b', re.IGNORECASE),
    'Oil & Pore Control': re.compile(r'\b(controls? oil|mattifying|minimizes? pores?|unclogs? pores?|less oily|pore minimiz\w*)\b', re.IGNORECASE),
    'Barrier Repair': re.compile(r'\b(barrier repair|restored barrier|repaired barrier|healed skin)\b', re.IGNORECASE),
    'Great Under Makeup / No White Cast': re.compile(r'\b(no white cast|great under makeup|wears well under makeup|smooth base)\b', re.IGNORECASE)
}

NEGATIVE_THEMES_PATTERNS = {
    'Breakouts & Clogged Pores': re.compile(r'\b(broke me out|caused breakouts?|clogged (?:my )?pores?|cystic acne|caused pimples)\b', re.IGNORECASE),
    'Irritation & Redness': re.compile(r'\b(irritat\w+|burning|stinging|caused redness|allergic reaction|itch\w*|redness)\b', re.IGNORECASE),
    'Drying & Stripping': re.compile(r'\b(drying|stripped (?:my )?skin|flak\w+|tight feeling|dried out)\b', re.IGNORECASE),
    'Heavy & Greasy': re.compile(r'\b(too greasy|feels heavy|oily residue|sticky|tacky|greasy film)\b', re.IGNORECASE),
    'Strong Fragrance': re.compile(r'\b(strong scent|overpowering fragrance|smells bad|perfumed|strong smell)\b', re.IGNORECASE),
    'White Cast & Pilling': re.compile(r'\b(white cast|pilling|pills under makeup|chalky|white residue)\b', re.IGNORECASE),
    'Ineffective / No Results': re.compile(r'\b(did not work|waste of money|saw no difference|no results|disappointing)\b', re.IGNORECASE)
}


class ReviewSentimentAnalyzer:
    """
    Extracts VADER sentiment metrics and structured aspect themes from the 1.2M review corpus.
    Uses memory-efficient streaming batch processing to compute per-product and skin-type metrics.
    """

    def __init__(self, reviews_path=None):
        if reviews_path is None:
            reviews_path = os.path.join(PROCESSED_DIR, "unified_reviews.parquet")
        self.reviews_path = reviews_path
        self.sia = SentimentIntensityAnalyzer()

    def analyze_reviews(self, batch_size=50000):
        """Processes reviews in streaming batches and aggregates product sentiment stats."""
        print(f"Opening parquet table at {self.reviews_path} for streaming...")
        parquet_file = pq.ParquetFile(self.reviews_path)
        total_rows = parquet_file.metadata.num_rows
        print(f"Total reviews to stream: {total_rows:,}")
        
        # Product-level accumulation dictionaries
        # Key: product_id -> dict of accumulators
        prod_compounds = defaultdict(list)
        prod_pos_themes = defaultdict(list)
        prod_neg_themes = defaultdict(list)
        
        # Skin-type partitioned accumulators
        # Key: (product_id, skin_type) -> list
        st_compounds = defaultdict(list)
        st_pos_themes = defaultdict(list)
        st_neg_themes = defaultdict(list)
        
        processed_count = 0
        t0 = time.time()
        
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=['product_id', 'user_skin_type', 'rating', 'review_title', 'review_text']):
            df_batch = batch.to_pandas()
            
            pids = df_batch['product_id'].values
            skin_types = df_batch['user_skin_type'].fillna('unknown').str.lower().str.strip().values
            titles = df_batch['review_title'].fillna('').astype(str).values
            texts = df_batch['review_text'].fillna('').astype(str).values
            
            for i in range(len(df_batch)):
                pid = pids[i]
                if not pid:
                    continue
                    
                full_text = (titles[i] + " " + texts[i]).strip()
                if not full_text:
                    continue
                    
                st = skin_types[i]
                
                # Compound sentiment score (first 400 chars for high speed)
                compound = self.sia.polarity_scores(full_text[:400])['compound']
                prod_compounds[pid].append(compound)
                
                if st in ['oily', 'dry', 'combination', 'normal']:
                    st_compounds[(pid, st)].append(compound)
                    
                # Aspect extraction based on polarity
                if compound >= 0.05:
                    for theme_name, pattern in POSITIVE_THEMES_PATTERNS.items():
                        if pattern.search(full_text):
                            prod_pos_themes[pid].append(theme_name)
                            if st in ['oily', 'dry', 'combination', 'normal']:
                                st_pos_themes[(pid, st)].append(theme_name)
                                
                elif compound <= -0.05:
                    for theme_name, pattern in NEGATIVE_THEMES_PATTERNS.items():
                        if pattern.search(full_text):
                            prod_neg_themes[pid].append(theme_name)
                            if st in ['oily', 'dry', 'combination', 'normal']:
                                st_neg_themes[(pid, st)].append(theme_name)
                                
            processed_count += len(df_batch)
            elapsed = time.time() - t0
            rate = processed_count / max(0.1, elapsed)
            print(f"Processed {processed_count:,} / {total_rows:,} reviews ({processed_count/total_rows*100:.1f}%) - {rate:.0f} revs/sec")

        print(f"Aggregation across {len(prod_compounds):,} unique products...")
        rows = []
        for pid, comps in prod_compounds.items():
            comps_arr = np.array(comps)
            total_revs = len(comps_arr)
            norm_sentiment = float((np.mean(comps_arr) + 1.0) / 2.0)
            pos_pct = float(np.mean(comps_arr >= 0.05) * 100.0)
            neg_pct = float(np.mean(comps_arr <= -0.05) * 100.0)
            
            top_pos = [t for t, _ in Counter(prod_pos_themes[pid]).most_common(4)]
            top_neg = [t for t, _ in Counter(prod_neg_themes[pid]).most_common(3)]
            
            row = {
                'product_id': pid,
                'sentiment_score': round(norm_sentiment, 4),
                'sentiment_pos_pct': round(pos_pct, 2),
                'sentiment_neg_pct': round(neg_pct, 2),
                'top_positive_themes': json.dumps(top_pos),
                'top_negative_themes': json.dumps(top_neg),
                'sentiment_review_count': total_revs
            }
            
            for st in ['oily', 'dry', 'combination', 'normal']:
                st_comps = st_compounds.get((pid, st), [])
                if len(st_comps) >= 2:
                    st_arr = np.array(st_comps)
                    row[f'{st}_sentiment_score'] = round(float((np.mean(st_arr) + 1.0) / 2.0), 4)
                    row[f'{st}_top_positive'] = json.dumps([t for t, _ in Counter(st_pos_themes.get((pid, st), [])).most_common(3)])
                    row[f'{st}_top_negative'] = json.dumps([t for t, _ in Counter(st_neg_themes.get((pid, st), [])).most_common(2)])
                else:
                    row[f'{st}_sentiment_score'] = None
                    row[f'{st}_top_positive'] = json.dumps([])
                    row[f'{st}_top_negative'] = json.dumps([])
                    
            rows.append(row)
            
        df_sent = pd.DataFrame(rows)
        print(f"Generated sentiment profiles for {len(df_sent):,} products.")
        return df_sent

    def save_and_merge(self, df_sent, output_parquet=None, output_csv=None):
        """Saves sentiment features and merges with master_product_features."""
        if output_parquet is None:
            output_parquet = os.path.join(PROCESSED_DIR, "product_sentiment_features.parquet")
        if output_csv is None:
            output_csv = os.path.join(PROCESSED_DIR, "product_sentiment_features.csv")
            
        print(f"Saving sentiment features to {output_parquet} and {output_csv}...")
        df_sent.to_parquet(output_parquet, index=False)
        df_sent.to_csv(output_csv, index=False)
        
        # Merge with master_product_features
        master_path = os.path.join(PROCESSED_DIR, "master_product_features.parquet")
        master_csv_path = os.path.join(PROCESSED_DIR, "master_product_features.csv")
        
        if os.path.exists(master_path):
            print("Merging sentiment features into master_product_features.parquet...")
            df_master = pd.read_parquet(master_path)
            
            # Drop old sentiment columns if they exist
            sentiment_cols = [c for c in df_sent.columns if c != 'product_id']
            df_master = df_master.drop(columns=[c for c in sentiment_cols if c in df_master.columns], errors='ignore')
            
            df_merged = df_master.merge(df_sent, on='product_id', how='left')
            
            # Fill defaults for products without reviews
            df_merged['sentiment_score'] = df_merged['sentiment_score'].fillna(0.70)
            df_merged['sentiment_pos_pct'] = df_merged['sentiment_pos_pct'].fillna(75.0)
            df_merged['sentiment_neg_pct'] = df_merged['sentiment_neg_pct'].fillna(10.0)
            df_merged['top_positive_themes'] = df_merged['top_positive_themes'].fillna('[]')
            df_merged['top_negative_themes'] = df_merged['top_negative_themes'].fillna('[]')
            df_merged['sentiment_review_count'] = df_merged['sentiment_review_count'].fillna(0)
            
            df_merged.to_parquet(master_path, index=False)
            df_merged.to_csv(master_csv_path, index=False)
            print(f"Updated master features table! Total columns: {len(df_merged.columns):,} | Rows: {len(df_merged):,}")
            
        print("Sentiment pipeline completed successfully!")


if __name__ == "__main__":
    analyzer = ReviewSentimentAnalyzer()
    df_sent = analyzer.analyze_reviews()
    analyzer.save_and_merge(df_sent)
