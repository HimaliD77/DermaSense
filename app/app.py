import os
import sys
import json
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory

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

# --- Page Routes ---

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static', 'images'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/')
def index():
    """Page 1: Skin Profile Intake Questionnaire."""
    return render_template('index.html', active_page='profile')

@app.route('/recommendations')
def recommendations_page():
    """Page 2: Dedicated Personalized Recommendations Results Page."""
    return render_template('recommendations.html', active_page='recommendations')

@app.route('/compare')
def compare_page():
    """Page 3: Side-by-Side Product Comparison Page."""
    return render_template('compare.html', active_page='compare')

@app.route('/ingredients')
def ingredients_page():
    """Page 4: Active Ingredient Knowledge Base."""
    return render_template('ingredients.html', active_page='ingredients')

# --- API Endpoints ---

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
        use_reranker = data.get('use_reranker', True)
        user_id = data.get('user_id')
        
        if max_price:
            try: max_price = float(max_price)
            except: max_price = None

        results_df = recommender.recommend(
            query=query,
            skin_type=skin_type,
            target_concerns=target_concerns,
            category=category,
            max_price_inr=max_price,
            preset=preset,
            top_k=top_k,
            use_reranker=use_reranker,
            user_id=user_id
        )
        
        recommendations = []
        for _, row in results_df.iterrows():
            exp_data = explainer.explain_recommendation(
                row,
                user_skin_type=skin_type,
                user_concerns=target_concerns,
                max_budget_inr=max_price
            )
            
            def _clean_list(val):
                if isinstance(val, str):
                    try: return json.loads(val)
                    except: return []
                return val if isinstance(val, list) else []

            p_inr = row.get('price_inr')
            if pd.isna(p_inr) and pd.notna(row.get('price_usd')):
                p_inr = round(row.get('price_usd') * 86.5)

            recommendations.append({
                'product_id': str(row.get('product_id')),
                'product_name': str(row.get('product_name')),
                'brand': str(row.get('brand')),
                'category': str(row.get('category')),
                'price_inr': float(p_inr) if pd.notna(p_inr) else None,
                'price_usd': float(row.get('price_usd')) if pd.notna(row.get('price_usd')) else None,
                'rating': round(float(row.get('rating')), 2) if pd.notna(row.get('rating')) else 4.2,
                'review_count': int(row.get('review_count')) if pd.notna(row.get('review_count')) else 0,
                'source': str(row.get('source')),
                'matched_benefits': _clean_list(row.get('matched_benefits')),
                'matched_actives': _clean_list(row.get('matched_actives')),
                'match_percentage': int(row.get('match_percentage') or min(99, max(50, round(float(row.get('composite_score', 0.8)) * 100)))),
                'scores': {
                    'composite': float(row.get('composite_score', 0)),
                    'semantic': float(row.get('semantic_score', 0)),
                    'rerank': float(row.get('rerank_score')) if pd.notna(row.get('rerank_score')) else None,
                    'concern_match': float(row.get('concern_match_score', 0)),
                    'skin_evidence': float(row.get('skin_evidence_score', 0)),
                    'sentiment': float(row.get('sentiment_score_calc', 0))
                },
                'skin_type_metrics': {
                    'rating': float(row.get(f'{skin_type}_skin_avg_rating')) if skin_type and pd.notna(row.get(f'{skin_type}_skin_avg_rating')) else None,
                    'review_count': int(row.get(f'{skin_type}_skin_review_count')) if skin_type and pd.notna(row.get(f'{skin_type}_skin_review_count')) else 0,
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
                'preset': preset,
                'max_price': max_price
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
    from src.knowledge_graph.graph import IngredientKnowledgeGraph
    kg = IngredientKnowledgeGraph.get_instance()
    category_map = {}
    for cat, actives in SKIN_BENEFITS_MAP.items():
        category_map[cat] = []
        for act in actives:
            desc = INGREDIENT_DESCRIPTIONS.get(act.lower(), "Supports targeted skin health and balance.")
            prov = kg.get_ingredient_provenance(act) or {}
            category_map[cat].append({
                'name': act.title(),
                'canonical': prov.get('canonical_name', act.lower()),
                'description': prov.get('description', desc),
                'synonyms': prov.get('synonyms', []),
                'mechanisms': [f['mechanism'] for f in prov.get('cosmetic_functions', [])],
                'citations': prov.get('evidence_citations', []),
                'cautions': prov.get('documented_cautions', [])
            })
    return jsonify({'categories': category_map})

@app.route('/api/graph/ingredient/<name>', methods=['GET'])
def graph_ingredient_provenance(name):
    try:
        from src.knowledge_graph.graph import IngredientKnowledgeGraph
        kg = IngredientKnowledgeGraph.get_instance()
        prov = kg.get_ingredient_provenance(name)
        if not prov:
            return jsonify({'status': 'not_found', 'message': f'Ingredient "{name}" not found in knowledge graph.'}), 404
        return jsonify({'status': 'success', 'provenance': prov})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/graph/check-conflicts', methods=['POST'])
def check_ingredient_conflicts():
    try:
        data = request.get_json() or {}
        actives = data.get('actives', [])
        from src.knowledge_graph.graph import IngredientKnowledgeGraph
        kg = IngredientKnowledgeGraph.get_instance()
        conflicts = kg.detect_contraindications(actives)
        return jsonify({
            'status': 'success',
            'has_conflicts': len(conflicts) > 0,
            'count': len(conflicts),
            'conflicts': conflicts
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/rag/ask', methods=['POST'])
def rag_ask():
    try:
        data = request.get_json() or {}
        question = data.get('question', '').strip()
        if not question:
            return jsonify({'status': 'error', 'message': 'Question is required.'}), 400

        product_id = data.get('product_id')
        product_id_b = data.get('product_id_b')
        skin_type = data.get('skin_type')
        concerns = data.get('concerns', [])

        from src.rag.qa_engine import EvidenceGroundedQAEngine
        qa_engine = EvidenceGroundedQAEngine.get_instance(recommender=recommender)
        result = qa_engine.answer_question(
            question=question,
            product_id=product_id,
            product_id_b=product_id_b,
            skin_type=skin_type,
            concerns=concerns
        )
        return jsonify({
            'status': 'success',
            'question': result['question'],
            'answer': result['answer'],
            'citations_verified': result['citations_verified'],
            'groundedness_score': result['groundedness_score'],
            'safety_compliant': result['safety_compliant'],
            'model_used': result['model_used'],
            'latency_ms': result['latency_ms']
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/personalization/feedback', methods=['POST'])
def personalization_feedback():
    try:
        data = request.get_json() or {}
        user_id = data.get('user_id')
        if not user_id:
            return jsonify({'status': 'error', 'message': 'user_id is required'}), 400
        
        product_id = data.get('product_id', '')
        feedback_type = data.get('feedback_type', 'like')
        rating = data.get('rating')
        reason = data.get('reason')
        brand = data.get('brand')
        actives = data.get('actives', [])

        from src.personalization.preference_manager import AdaptivePersonalizer
        ap = AdaptivePersonalizer.get_instance()
        profile = ap.record_feedback(
            user_id=user_id,
            product_id=product_id,
            feedback_type=feedback_type,
            rating=float(rating) if rating is not None else None,
            reason=reason,
            brand=brand,
            actives=actives
        )
        return jsonify({
            'status': 'success',
            'user_id': user_id,
            'interaction_count': profile['interaction_count'],
            'liked_ingredients': profile['liked_ingredients'],
            'disliked_ingredients': profile['disliked_ingredients'],
            'preferred_brands': profile['preferred_brands']
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/personalization/profile/<user_id>', methods=['GET'])
def get_personalization_profile(user_id):
    try:
        from src.personalization.preference_manager import AdaptivePersonalizer
        ap = AdaptivePersonalizer.get_instance()
        profile = ap.get_or_create_profile(user_id)
        return jsonify({'status': 'success', 'profile': profile})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='127.0.0.1', port=port, debug=False)
