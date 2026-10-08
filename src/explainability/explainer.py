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
    'titanium dioxide': 'gentle physical UV blocker suitable for sensitive and acne-prone skin'
}

class DermaSenseExplainer:
    """
    Generates transparent, evidence-based explanations for DermaSense recommendations.
    Explains:
    1. Why the product matches user concerns
    2. Active ingredient mechanism
    3. Community evidence from matching skin-type reviewers
    4. Budget and category alignment
    """
    
    @staticmethod
    def explain_recommendation(product_row, user_skin_type=None, user_concerns=None, max_budget_usd=None):
        """Generates a structured explanation dictionary and readable narrative."""
        pname = product_row.get('product_name', 'This product')
        brand = product_row.get('brand', '')
        category = product_row.get('category', 'Skincare')
        price_usd = product_row.get('price_usd')
        
        try:
            prod_benefits = json.loads(product_row.get('matched_benefits', '[]'))
        except Exception:
            prod_benefits = []
            
        try:
            prod_actives = json.loads(product_row.get('matched_actives', '[]'))
        except Exception:
            prod_actives = []
            
        # 1. Matching Concerns
        user_concerns = user_concerns or []
        matched_user_concerns = [c for c in user_concerns if c in prod_benefits]
        
        # 2. Ingredient Rationales
        ingredient_explanations = []
        for act in prod_actives[:4]:
            act_clean = act.lower().strip()
            desc = INGREDIENT_DESCRIPTIONS.get(act_clean, f"active ingredient supporting {', '.join(prod_benefits[:2])}")
            ingredient_explanations.append(f"**{act.title()}**: {desc}.")
            
        # 3. Skin Type Evidence
        skin_type_evidence = ""
        st_clean = str(user_skin_type).lower().strip() if user_skin_type else ""
        if st_clean in ['combination', 'dry', 'normal', 'oily']:
            st_rating = product_row.get(f'{st_clean}_skin_avg_rating')
            st_count = product_row.get(f'{st_clean}_skin_review_count')
            st_pos = product_row.get(f'{st_clean}_skin_pos_pct')
            
            if pd.notna(st_rating) and st_count and st_count > 0:
                skin_type_evidence = (
                    f"Users with **{st_clean.title()} Skin** rated this product **{st_rating:.2f} / 5.0** "
                    f"across {int(st_count):,} verified reviews ({st_pos:.1f}% positive satisfaction)."
                )
            else:
                gen_rating = product_row.get('rating')
                skin_type_evidence = f"Overall catalog rating of **{gen_rating:.2f} / 5.0** based on general customer feedback."
        else:
            gen_rating = product_row.get('rating')
            skin_type_evidence = f"Overall rating of **{gen_rating:.2f} / 5.0** across all skin profiles."
            
        # 4. Budget & Price Rationale
        budget_note = ""
        if pd.notna(price_usd) and price_usd > 0:
            if max_budget_usd and price_usd <= max_budget_usd:
                budget_note = f"Priced at **${price_usd:.2f}**, which is comfortably within your ${max_budget_usd:.2f} budget."
            else:
                budget_note = f"Priced at **${price_usd:.2f}**."
        else:
            budget_note = "Check platform for current pricing."
            
        # Summary narrative
        narrative_parts = [
            f"**{pname}** by **{brand}** is recommended because it is specifically formulated as a **{category}**."
        ]
        if matched_user_concerns:
            narrative_parts.append(f"It directly targets your selected concerns: **{', '.join(matched_user_concerns)}**.")
        if ingredient_explanations:
            narrative_parts.append("Key Active Ingredients:\n" + "\n".join([f"  - {e}" for e in ingredient_explanations]))
        if skin_type_evidence:
            narrative_parts.append(f"User Experience Evidence:\n  - {skin_type_evidence}")
        if budget_note:
            narrative_parts.append(f"Pricing: {budget_note}")
            
        full_explanation_text = "\n\n".join(narrative_parts)
        
        return {
            'product_name': pname,
            'brand': brand,
            'category': category,
            'matched_concerns': matched_user_concerns,
            'active_ingredients': prod_actives,
            'ingredient_rationales': ingredient_explanations,
            'skin_type_evidence': skin_type_evidence,
            'budget_note': budget_note,
            'full_narrative': full_explanation_text
        }

if __name__ == "__main__":
    from src.models.recommender import DermaSenseRecommender
    
    rec = DermaSenseRecommender()
    top_prods = rec.recommend(
        query="soothing moisturizer for oily acne prone skin with centella and niacinamide",
        skin_type="oily",
        target_concerns=["Acne & Blemish Control", "Brightening & Tone", "Soothing & Anti-Redness"],
        category="Moisturizer",
        top_k=1
    )
    
    if len(top_prods) > 0:
        explainer = DermaSenseExplainer()
        exp = explainer.explain_recommendation(
            top_prods.iloc[0],
            user_skin_type="oily",
            user_concerns=["Acne & Blemish Control", "Brightening & Tone", "Soothing & Anti-Redness"],
            max_budget_usd=50
        )
        print("\n" + "="*80)
        print("EXPLAINABILITY DEMONSTRATION")
        print("="*80)
        print(exp['full_narrative'])
        print("="*80)
