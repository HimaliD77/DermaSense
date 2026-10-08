import os
import sys
import json
import pandas as pd
from flask import Flask, render_template, request, jsonify

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

DERMASENSE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if DERMASENSE_DIR not in sys.path:
    sys.path.insert(0, DERMASENSE_DIR)

from src.models.recommender import DermaSenseRecommender, WEIGHT_PRESETS
from src.explainability.explainer import DermaSenseExplainer, INGREDIENT_DESCRIPTIONS
from src.features.ingredient_features import SKIN_BENEFITS_MAP

app = Flask(__name__, template_folder="templates", static_folder="static")

print("Initializing DermaSense Engine for Web App...")
recommender = DermaSenseRecommender(use_dense_embeddings=True)
explainer = DermaSenseExplainer()
print("DermaSense Web App Engine Ready!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/stats', methods=['GET'])
def get_stats():
    total_products = len(recommender.df)
    total_brands = recommender.df['brand'].nunique()
    sources = recommender.df['source'].value_counts().to_dict()
    categories = recommender.df['category'].dropna().unique().tolist()
    return jsonify({
        'total_products': total_products,
        'total_brands': total_brands,
        'sources': sources,
        'categories': sorted([c for c in categories if c]),
        'concerns': list(SKIN_BENEFITS_MAP.keys())
    })

@app.route('/api/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json() or {}
        query = data.get('query', '')
        skin_type = data.get('skin_type')
        target_concerns = data.get('target_concerns', [])
        category = data.get('category', 'All')
        max_price = data.get('max_price')
        preset = data.get('preset', 'balanced')
        top_k = int(data.get('top_k', 6))
        
        if max_price:
            try: max_price = float(max_price)
            except: max_price = None

        results_df = recommender.recommend(
            query=query,
            skin_type=skin_type,
            target_concerns=target_concerns,
            category=category,
            max_price_usd=max_price,
            preset=preset,
            top_k=top_k
        )
        
        recommendations = []
        for _, row in results_df.iterrows():
            exp_data = explainer.explain_recommendation(
                row,
                user_skin_type=skin_type,
                user_concerns=target_concerns,
                max_budget_usd=max_price
            )
            
            # Helper to safely parse lists
            def _clean_list(val):
                if isinstance(val, str):
                    try: return json.loads(val)
                    except: return []
                return val if isinstance(val, list) else []

            recommendations.append({
                'product_id': str(row.get('product_id')),
                'product_name': str(row.get('product_name')),
                'brand': str(row.get('brand')),
                'category': str(row.get('category')),
                'price_usd': float(row.get('price_usd')) if pd.notna(row.get('price_usd')) else None,
                'rating': round(float(row.get('rating')), 2) if pd.notna(row.get('rating')) else 4.2,
                'review_count': int(row.get('review_count') or 0),
                'source': str(row.get('source')),
                'matched_benefits': _clean_list(row.get('matched_benefits')),
                'matched_actives': _clean_list(row.get('matched_actives')),
                'scores': {
                    'composite': float(row.get('composite_score', 0)),
                    'semantic': float(row.get('semantic_score', 0)),
                    'concern_match': float(row.get('concern_match_score', 0)),
                    'skin_evidence': float(row.get('skin_evidence_score', 0)),
                    'sentiment': float(row.get('sentiment_score_calc', 0))
                },
                'skin_type_metrics': {
                    'rating': float(row.get(f'{skin_type}_skin_avg_rating')) if skin_type and pd.notna(row.get(f'{skin_type}_skin_avg_rating')) else None,
                    'review_count': int(row.get(f'{skin_type}_skin_review_count') or 0) if skin_type else 0,
                    'pos_pct': float(row.get(f'{skin_type}_skin_pos_pct')) if skin_type and pd.notna(row.get(f'{skin_type}_skin_pos_pct')) else None
                },
                'explanation': exp_data
            })
            
        return jsonify({
            'status': 'success',
            'count': len(recommendations),
            'query_params': {
                'query': query,
                'skin_type': skin_type,
                'target_concerns': target_concerns,
                'category': category,
                'preset': preset
            },
            'recommendations': recommendations
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/compare', methods=['POST'])
def compare():
    try:
        data = request.get_json() or {}
        p_a = data.get('product_id_a')
        p_b = data.get('product_id_b')
        skin_type = data.get('skin_type')
        concerns = data.get('target_concerns', [])
        
        if not p_a or not p_b:
            return jsonify({'status': 'error', 'message': 'Two product IDs are required.'}), 400
            
        comp_result = recommender.compare_products(p_a, p_b, skin_type=skin_type, concerns=concerns)
        if not comp_result:
            return jsonify({'status': 'error', 'message': 'One or both products not found.'}), 404
            
        return jsonify({'status': 'success', 'comparison': comp_result})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/search', methods=['GET'])
def search():
    term = request.args.get('q', '')
    limit = int(request.args.get('limit', 8))
    results = recommender.search_catalog_by_name(term, limit=limit)
    return jsonify({'results': results})

@app.route('/api/ingredients', methods=['GET'])
def ingredients():
    category_map = {}
    for cat, actives in SKIN_BENEFITS_MAP.items():
        category_map[cat] = []
        for act in actives:
            desc = INGREDIENT_DESCRIPTIONS.get(act.lower(), "Supports targeted skin health and balance.")
            category_map[cat].append({
                'name': act.title(),
                'description': desc
            })
    return jsonify({'categories': category_map})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='127.0.0.1', port=port, debug=False)
