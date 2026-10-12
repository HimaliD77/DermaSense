import os
import sys
import json
import pandas as pd

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

DERMASENSE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if DERMASENSE_DIR not in sys.path:
    sys.path.insert(0, DERMASENSE_DIR)

INGREDIENT_DESCRIPTIONS = {
    'hyaluronic acid': 'deeply hydrates and binds moisture to plump dry or dehydrated skin',
    'sodium hyaluronate': 'a smaller molecular weight form of hyaluronic acid that penetrates deeper skin layers',
    'ceramide': 'repairs and reinforces the skin protective lipid barrier to prevent moisture loss',
    'ceramide np': 'essential skin-identical lipid that restores moisture and soothes compromised skin barriers',
    'glycerin': 'effective humectant that draws water into the outer layer of skin for lasting softness',
    'squalane': 'lightweight, non-comedogenic emollient that locks in hydration without clogging pores',
    'niacinamide': 'regulates sebum production, improves uneven skin texture, and strengthens skin barrier',
    'salicylic acid': 'oil-soluble BHA that exfoliates deep inside pores to clear acne and blackheads',
    'glycolic acid': 'AHA that gently removes surface dead skin cells to reveal smoother, brighter skin',
    'lactic acid': 'gentle exfoliating AHA with natural humectant properties suitable for sensitive skin',
    'vitamin c': 'potent antioxidant that neutralizes free radicals, brightens hyperpigmentation, and boosts collagen',
    'ascorbic acid': 'pure form of Vitamin C that fades dark spots and evens skin tone',
    'retinol': 'stimulates cellular turnover and collagen synthesis to reduce fine lines and improve texture',
    'bakuchiol': 'plant-based gentle alternative to retinol that firms skin without irritation',
    'centella asiatica': 'calms irritation, reduces redness, and accelerates skin healing',
    'madecassoside': 'active compound from Centella that provides intense soothing relief for reactive skin',
    'allantoin': 'skin-protecting agent that soothes irritation and promotes tissue regeneration',
    'zinc oxide': 'mineral physical sunscreen that shields against UVA/UVB rays without chemical irritation',
    'titanium dioxide': 'gentle physical UV blocker suitable for sensitive and acne-prone skin',
    'colloidal oatmeal': 'clinically proven protectant that relieves itching, eczema, and skin barrier distress',
    'zinc pca': 'synergistic zinc salt that regulates excess oil production and calms blemish-causing microbes',
    'azelaic acid': 'gently targets redness, post-acne dark marks, and congestion with minimal irritation',
    'panthenol': 'Pro-Vitamin B5 that attracts water and accelerates epidermal tissue healing'
}

class DermaSenseExplainer:
    """
    Generates transparent, evidence-based explanations for DermaSense recommendations.
    Explains:
    1. Why the product matches user concerns
    2. Active ingredient mechanisms
    3. Community evidence & sentiment from matching skin-type reviewers
    4. Sentiment themes (praises and things to consider)
    5. Budget and category alignment
    """
    
    @staticmethod
    def explain_recommendation(product_row, user_skin_type=None, user_concerns=None, max_budget_inr=None, max_budget_usd=None):
        """Generates a structured explanation dictionary and readable narrative in INR."""
        pname = product_row.get('product_name', 'This product')
        brand = product_row.get('brand', '')
        category = product_row.get('category', 'Skincare')
        
        price_inr = product_row.get('price_inr')
        if pd.isna(price_inr) and pd.notna(product_row.get('price_usd')):
            price_inr = round(product_row.get('price_usd') * 86.5)
            
        if max_budget_inr is None and max_budget_usd is not None:
            max_budget_inr = max_budget_usd * 86.5
        
        try:
            prod_benefits = json.loads(product_row.get('matched_benefits', '[]')) if isinstance(product_row.get('matched_benefits'), str) else (product_row.get('matched_benefits') or [])
        except Exception:
            prod_benefits = []
            
        try:
            prod_actives = json.loads(product_row.get('matched_actives', '[]')) if isinstance(product_row.get('matched_actives'), str) else (product_row.get('matched_actives') or [])
        except Exception:
            prod_actives = []
            
        # Parse sentiment themes
        try:
            pos_themes = json.loads(product_row.get('top_positive_themes', '[]')) if isinstance(product_row.get('top_positive_themes'), str) else (product_row.get('top_positive_themes') or [])
        except Exception:
            pos_themes = []
            
        try:
            neg_themes = json.loads(product_row.get('top_negative_themes', '[]')) if isinstance(product_row.get('top_negative_themes'), str) else (product_row.get('top_negative_themes') or [])
        except Exception:
            neg_themes = []
            
        # 1. Matching Concerns
        user_concerns = user_concerns or []
        matched_user_concerns = [c for c in user_concerns if c in prod_benefits]
        
        # 2. Ingredient Rationales
        ingredient_explanations = []
        for act in prod_actives[:4]:
            act_clean = act.lower().strip()
            desc = INGREDIENT_DESCRIPTIONS.get(act_clean, f"active ingredient supporting {', '.join(prod_benefits[:2]) if prod_benefits else 'skin health'}")
            ingredient_explanations.append(f"**{act.title()}**: {desc}.")
            
        # 3. Skin Type Evidence & Sentiment Themes
        skin_type_evidence = ""
        st_clean = str(user_skin_type).lower().strip() if user_skin_type else ""
        st_pos_themes = []
        
        if st_clean in ['combination', 'dry', 'normal', 'oily']:
            st_rating = product_row.get(f'{st_clean}_skin_avg_rating')
            st_count = product_row.get(f'{st_clean}_skin_review_count')
            st_pos_pct = product_row.get(f'{st_clean}_skin_pos_pct')
            
            try:
                st_pos_themes = json.loads(product_row.get(f'{st_clean}_top_positive', '[]')) if isinstance(product_row.get(f'{st_clean}_top_positive'), str) else []
            except Exception:
                st_pos_themes = []
                
            if pd.notna(st_rating) and st_count and st_count > 0:
                skin_type_evidence = (
                    f"Users with **{st_clean.title()} Skin** rated this product **{st_rating:.2f} / 5.0** "
                    f"across {int(st_count):,} verified reviews ({st_pos_pct:.1f}% positive satisfaction)."
                )
            else:
                gen_rating = product_row.get('rating') or 4.2
                skin_type_evidence = f"Overall catalog rating of **{float(gen_rating):.2f} / 5.0** based on general community feedback."
        else:
            gen_rating = product_row.get('rating') or 4.2
            skin_type_evidence = f"Overall rating of **{float(gen_rating):.2f} / 5.0** across all customer reviews."
            
        # 4. Sentiment Theme Highlights
        sentiment_highlights = []
        relevant_pos = st_pos_themes if st_pos_themes else pos_themes
        if relevant_pos:
            sentiment_highlights.append(f"Most praised for: **{', '.join(relevant_pos[:3])}**")
        if neg_themes:
            sentiment_highlights.append(f"Aspects to consider: **{', '.join(neg_themes[:2])}**")
            
        # 5. Budget & Price Rationale
        budget_note = ""
        if pd.notna(price_inr) and price_inr > 0:
            if max_budget_inr and price_inr <= max_budget_inr:
                budget_note = f"Priced at **₹{price_inr:,.0f}**, which comfortably fits within your ₹{max_budget_inr:,.0f} budget."
            else:
                budget_note = f"Priced at **₹{price_inr:,.0f}**."
        else:
            budget_note = "Check platform for current pricing."
            
        # 6. Graph Evidence & Provenance Tracing
        graph_evidence = {
            'has_graph_evidence': False,
            'grounding_score': 0.0,
            'verified_paths_count': 0,
            'reasoning_paths': [],
            'contraindications': []
        }
        try:
            from src.knowledge_graph.graph import IngredientKnowledgeGraph
            kg = IngredientKnowledgeGraph.get_instance()
            graph_evidence = kg.get_graph_evidence_card(product_row, user_concerns)
        except Exception as e:
            pass

        # Summary narrative
        narrative_parts = [
            f"**{pname}** by **{brand}** is recommended as a top-performing **{category}**."
        ]
        if matched_user_concerns:
            narrative_parts.append(f"It directly targets your selected concerns: **{', '.join(matched_user_concerns)}**.")
        if ingredient_explanations:
            narrative_parts.append("Key Active Ingredients:\n" + "\n".join([f"  - {e}" for e in ingredient_explanations]))
        if skin_type_evidence:
            narrative_parts.append(f"Community Experience:\n  - {skin_type_evidence}")
        if sentiment_highlights:
            narrative_parts.append("Sentiment Insights:\n" + "\n".join([f"  - {h}" for h in sentiment_highlights]))
        if budget_note:
            narrative_parts.append(f"Pricing: {budget_note}")

        # Add clinical citations if available
        citations = []
        for path in graph_evidence.get('reasoning_paths', []):
            for cit in path.get('citations', []):
                citations.append(f"{cit.get('citation_id', 'Citation')}: {cit.get('title', '')} ({cit.get('journal', '')}, {cit.get('year', '')})")
        if citations:
            unique_citations = list(dict.fromkeys(citations))
            narrative_parts.append("Clinical Literature Evidence:\n" + "\n".join([f"  - {c}" for c in unique_citations[:2]]))

        # Add cautions if detected
        if graph_evidence.get('contraindications'):
            caution_lines = [f"⚠️ {c.get('title')}: {c.get('recommendation')}" for c in graph_evidence['contraindications']]
            narrative_parts.append("Usage Precautions:\n" + "\n".join([f"  - {cl}" for cl in caution_lines]))
            
        full_explanation_text = "\n\n".join(narrative_parts)
        
        return {
            'product_name': pname,
            'brand': brand,
            'category': category,
            'matched_concerns': matched_user_concerns,
            'active_ingredients': prod_actives,
            'ingredient_rationales': ingredient_explanations,
            'skin_type_evidence': skin_type_evidence,
            'sentiment_highlights': sentiment_highlights,
            'top_positive_themes': pos_themes,
            'top_negative_themes': neg_themes,
            'budget_note': budget_note,
            'graph_evidence': graph_evidence,
            'full_narrative': full_explanation_text
        }

if __name__ == "__main__":
    from src.models.recommender import DermaSenseRecommender
    
    rec = DermaSenseRecommender()
    top_prods = rec.recommend(
        query="oil-free gel moisturizer with niacinamide and zinc for oily acne skin",
        skin_type="oily",
        target_concerns=["Acne & Blemish Control", "Oil & Pore Control"],
        category="Moisturizer",
        max_price_usd=45,
        top_k=1
    )
    
    if len(top_prods) > 0:
        explainer = DermaSenseExplainer()
        exp = explainer.explain_recommendation(
            top_prods.iloc[0],
            user_skin_type="oily",
            user_concerns=["Acne & Blemish Control", "Oil & Pore Control"],
            max_budget_usd=45
        )
        print("\n" + "="*80)
        print("EXPLAINABILITY WITH SENTIMENT THEMES")
        print("="*80)
        print(exp['full_narrative'])
        print("="*80)
