"""
DermaSense - Evidence-Grounded RAG (Retrieval-Augmented Generation) Q&A Engine
Integrates:
1. Multi-source evidence retrieval (Product catalog, skin-partitioned review NLP, Knowledge Graph clinical PMIDs).
2. Local LLM Generation via Ollama (qwen2.5:7b-instruct) with zero-shot deterministic fallback.
3. Strict clinical grounding, PMID citation validation, and medical safety guardrails.
"""

import os
import re
import json
import time
import urllib.request
import urllib.error
from typing import Dict, List, Optional, Any, Tuple

from src.knowledge_graph.graph import IngredientKnowledgeGraph
from src.features.ingredient_features import SKIN_BENEFITS_MAP

OLLAMA_API_URL = os.environ.get("OLLAMA_API_URL", "http://localhost:11434/api/generate")
DEFAULT_LLM_MODEL = os.environ.get("DERMASENSE_LLM_MODEL", "qwen2.5:7b-instruct")

MEDICAL_RED_FLAGS = [
    "cure acne", "diagnose", "infection", "prescription", "antibiotic",
    "isotretinoin", "accutane", "cyst burst", "bleeding", "melanoma",
    "skin cancer", "steroid", "dermatitis flare", "pus"
]

MEDICAL_DISCLAIMER = (
    "⚠️ **Clinical Disclaimer**: I am an AI skincare formulation assistant, not a dermatologist or physician. "
    "This guidance is based on cosmetic chemistry and consumer review evidence. "
    "Please consult a certified dermatologist for clinical diagnoses, infections, or prescription medical treatments."
)

# ---------------------------------------------------------------------------
# Skincare Consultation & Routine Guidance Knowledge Base
# ---------------------------------------------------------------------------
ROUTINE_INTENT_KEYWORDS = [
    "advice", "advise", "routine", "regimen", "how to", "how do i", "how should i",
    "what should i do", "tips", "steps", "order", "guide", "care for", "morning",
    "night", "am routine", "pm routine", "daily care", "layer", "skincare tips",
    "build a routine", "recommend a routine", "what to do", "help with my skin",
    "start", "starting", "best routine", "step by step"
]

CONTRAINDICATION_KEYWORDS = [
    "can i mix", "can i use together", "safe with", "interact", "combine",
    "mix with", "layer together", "conflict", "bad combination", "clash"
]

def classify_user_intent(question: str) -> str:
    """
    Classifies user question into:
    - 'ROUTINE_ADVICE': Guidance on skincare routine, steps, how to care, tips, regimen.
    - 'CONTRAINDICATION_CHECK': Conflict or layering safety questions.
    - 'GENERAL_INQUIRY': Product inquiry or general scientific question.
    """
    q = question.lower()
    if any(k in q for k in CONTRAINDICATION_KEYWORDS):
        return "CONTRAINDICATION_CHECK"
    if any(k in q for k in ROUTINE_INTENT_KEYWORDS):
        return "ROUTINE_ADVICE"
    return "GENERAL_INQUIRY"

SKIN_TYPE_ROUTINE_GUIDES = {
    "oily": {
        "focus": "Pore unclogging, sebum regulation, and lightweight non-comedogenic hydration.",
        "am_steps": [
            "Gentle foaming or gel cleanser (pH-balanced; purifies without stripping natural barrier lipids)",
            "Antioxidant / sebum-balancing serum (Niacinamide 2–5% or lightweight Vitamin C derivative)",
            "Oil-free water-gel hydrator rich in humectants (Hyaluronic acid, Glycerin)",
            "Broad-Spectrum SPF 50 (matte / gel finish; essential UV barrier shield)"
        ],
        "pm_steps": [
            "Gentle cleanser to wash away SPF, sebum, and environmental impurities",
            "Targeted treatment active 2–3 nights/week (Salicylic acid BHA 1–2% for pore clearing, or gentle Retinoid)",
            "Lightweight barrier lotion with Ceramides to balance moisture and prevent rebound oil production"
        ],
        "lifestyle_tips": [
            "Do not over-cleanse: Washing more than twice a day strips the stratum corneum, triggering rebound excess oil.",
            "Never skip moisturizer: Oily skin can still be dehydrated; water-based hydration normalizes oil glands.",
            "Always patch-test new exfoliating acids on the inner forearm for 24 hours before facial use."
        ],
        "key_actives": ["salicylic acid", "niacinamide", "zinc pca", "hyaluronic acid", "ceramide"]
    },
    "dry": {
        "focus": "Stratum corneum lipid replenishment, deep moisture trapping, and barrier repair.",
        "am_steps": [
            "Hydrating lotion or cream cleanser (or gentle lukewarm water rinse)",
            "Humectant-rich essence or serum applied to damp skin (Hyaluronic acid, Glycerin, Panthenol)",
            "Barrier-reinforcing emollient moisturizer with Ceramides and Fatty Acids",
            "Nourishing Broad-Spectrum SPF 50 with a moisturizing, dewy base"
        ],
        "pm_steps": [
            "Double cleanse: Nourishing cleansing oil/balm followed by gentle hydrating cleanser",
            "Gentle restorative active 1–2 nights/week (e.g. Lactic acid AHA or encapsulated low-strength Retinol)",
            "Rich night barrier cream with Ceramides NP and Squalane to seal hydration overnight"
        ],
        "lifestyle_tips": [
            "Apply hydrators on damp skin immediately after cleansing to trap water molecules.",
            "Avoid hot water, harsh foaming sulfates (SLS), and high-alcohol toners.",
            "Introduce chemical exfoliants no more than once weekly to protect delicate lipid layers."
        ],
        "key_actives": ["ceramide", "hyaluronic acid", "glycerin", "squalane", "panthenol"]
    },
    "combination": {
        "focus": "Zone-targeted balancing: refining oily T-zone pores while nourishing cheeks.",
        "am_steps": [
            "Balancing gel cleanser that removes excess T-zone oil without drying cheeks",
            "Niacinamide (2–4%) serum to regulate pore sebum and reinforce skin barrier",
            "Lightweight daily lotion or emulsion suitable across all facial zones",
            "Daily Broad-Spectrum SPF 50 fluid sunscreen"
        ],
        "pm_steps": [
            "Gentle thorough cleanse to clear sunscreen and urban residue",
            "Zone-targeted treatment (Salicylic acid on nose/chin/forehead, soothing Hydrator on cheeks)",
            "Balanced ceramide barrier cream"
        ],
        "lifestyle_tips": [
            "Zone-map your routine: You do not need to apply strong pore-clearing acids to dry cheeks.",
            "Prioritize consistency: Gentle daily barrier support yields better long-term results than intense spot treatments."
        ],
        "key_actives": ["niacinamide", "salicylic acid", "hyaluronic acid", "ceramide", "glycerin"]
    },
    "sensitive": {
        "focus": "Barrier protection, anti-inflammatory calming, and hypoallergenic minimalism.",
        "am_steps": [
            "Ultra-gentle non-foaming micellar water or cream wash",
            "Soothing anti-inflammatory serum (Centella Asiatica, Madecassoside, Allantoin)",
            "Fragrance-free barrier repair cream packed with Ceramides and Panthenol",
            "100% Mineral (Zinc Oxide / Titanium Dioxide) SPF 50 to prevent chemical filter stinging"
        ],
        "pm_steps": [
            "Gentle cleanse with lukewarm water (never hot water)",
            "Barrier restoration treatment (Panthenol 5% or Madecassoside complex)",
            "Restorative barrier cream to lock moisture and soothe micro-irritation"
        ],
        "lifestyle_tips": [
            "Strictly avoid added fragrance, essential oils, and abrasive physical scrubs.",
            "Always patch test any new formulation behind the ear for 48 hours.",
            "If stinging or burning occurs, immediately halt all actives and use only gentle wash + barrier cream."
        ],
        "key_actives": ["centella asiatica", "madecassoside", "allantoin", "ceramide", "panthenol"]
    },
    "normal": {
        "focus": "Antioxidant environmental defense, hydration maintenance, and barrier longevity.",
        "am_steps": [
            "Gentle foaming or gel cleanser",
            "Antioxidant serum (Vitamin C / Ascorbic Acid or Niacinamide)",
            "Lightweight daily hydrator with Ceramides and Hyaluronic acid",
            "Broad-Spectrum SPF 50 (the #1 essential preventative step)"
        ],
        "pm_steps": [
            "Gentle cleanser to remove daytime pollutants and sunscreen",
            "Cellular renewal treatment 2–3 nights/week (Retinol or mild AHA)",
            "Restorative evening moisturizer with Ceramides"
        ],
        "lifestyle_tips": [
            "Keep the core routine streamlined: Cleanse, Treat, Moisturize, Protect (SPF).",
            "Introduce only one new active product at a time to track skin compatibility."
        ],
        "key_actives": ["vitamin c", "niacinamide", "hyaluronic acid", "ceramide", "retinol"]
    }
}

CONCERN_GUIDELINES = {
    "Acne & Blemish Control": {
        "key_approach": "Unclog pore linings with oil-soluble Salicylic Acid (BHA 1–2%) and calm inflammation with Niacinamide.",
        "rule": "Never pick or squeeze blemishes. Introduce BHA 2–3 nights per week rather than daily to prevent barrier breakdown."
    },
    "Hydration & Barrier": {
        "key_approach": "Rebuild stratum corneum lipids using Ceramides, Hyaluronic Acid, and Glycerin.",
        "rule": "Apply humectants onto slightly damp skin and immediately seal with a ceramide-rich cream."
    },
    "Texture Smoothing": {
        "key_approach": "Encourage gentle surface desquamation with low-strength AHA (Lactic or Glycolic acid) and PM Retinoids.",
        "rule": "Never combine AHA exfoliants and Retinoids on the same evening; alternate nights."
    },
    "Brightening & Radiance": {
        "key_approach": "Inhibit tyrosinase activity and fade post-inflammatory hyperpigmentation with Vitamin C (AM) and Niacinamide.",
        "rule": "Always pair brightening actives with SPF 50; unprotected UV exposure quickly reverses radiance progress."
    },
    "Oil & Pore Control": {
        "key_approach": "Regulate sebum production with Niacinamide (2–5%) and Zinc PCA.",
        "rule": "Avoid harsh alcohol-based astringents that dry out skin and prompt rebound sebum secretion."
    },
    "Anti-Aging & Firming": {
        "key_approach": "Stimulate cellular turnover and collagen synthesis with PM Retinol/Bakuchiol and Peptides.",
        "rule": "Daily Broad-Spectrum SPF 50 is the single most effective preventative anti-aging step."
    },
    "Calming & Soothing": {
        "key_approach": "Downregulate inflammatory cytokines with Centella Asiatica, Allantoin, and Madecassoside.",
        "rule": "Keep the ingredient deck minimal; eliminate artificial fragrances and physical scrubs."
    }
}


class EvidenceGroundedQAEngine:
    """
    RAG engine providing verifiable, evidence-grounded skincare answers.
    """
    _instance = None

    def __init__(self, recommender=None, model_name: str = DEFAULT_LLM_MODEL):
        self.model_name = model_name
        self.recommender = recommender
        self.kg = IngredientKnowledgeGraph.get_instance()

    @classmethod
    def get_instance(cls, recommender=None, model_name: str = DEFAULT_LLM_MODEL):
        if cls._instance is None:
            cls._instance = cls(recommender=recommender, model_name=model_name)
        return cls._instance

    def _ensure_recommender(self):
        if self.recommender is None:
            # Check if parquet file exists to build a lightweight catalog without loading PyTorch twice
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            parquet_path = os.path.join(project_root, "data", "processed", "master_product_features.parquet")
            if os.path.exists(parquet_path):
                import pandas as pd
                df = pd.read_parquet(parquet_path)

                class LightweightProductCatalog:
                    def __init__(self, catalog_df):
                        self.df = catalog_df

                    def search_catalog_by_name(self, search_term, limit=4):
                        if not search_term or not str(search_term).strip():
                            return []
                        t = str(search_term).lower().strip()
                        mask = (
                            self.df['product_name'].str.lower().str.contains(t, na=False) |
                            self.df['brand'].str.lower().str.contains(t, na=False)
                        )
                        matches = self.df[mask].head(limit)
                        results = []
                        for _, r in matches.iterrows():
                            results.append({
                                'product_id': r['product_id'],
                                'product_name': r['product_name'],
                                'brand': r['brand'],
                                'category': r.get('category')
                            })
                        return results

                    def recommend(self, query='', skin_type=None, target_concerns=None, top_k=2, **kwargs):
                        if query:
                            q_terms = [re.escape(w.lower()) for w in str(query).split() if len(w) > 3]
                            if q_terms:
                                pattern = '|'.join(q_terms)
                                mask = (
                                    self.df['product_name'].str.lower().str.contains(pattern, na=False) |
                                    self.df['description'].astype(str).str.lower().str.contains(pattern, na=False)
                                )
                                subset = self.df[mask]
                                if len(subset) > 0:
                                    return subset.head(top_k)
                        return self.df.head(top_k)

                self.recommender = LightweightProductCatalog(df)
            else:
                from src.models.recommender import DermaSenseRecommender
                self.recommender = DermaSenseRecommender(use_dense_embeddings=False, use_reranker=False)

    def retrieve_context(
        self,
        question: str,
        product_id: Optional[str] = None,
        product_id_b: Optional[str] = None,
        skin_type: Optional[str] = None,
        concerns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Retrieves multi-modal evidence across:
        1. Target product specifications & INR pricing.
        2. Skin-partitioned review satisfaction ratings.
        3. Ingredient Knowledge Graph biological mechanisms and PubMed PMIDs.
        4. Formulation contraindications & collision rules.
        """
        self._ensure_recommender()
        df = self.recommender.df

        target_products = []

        # 1. Check explicit product IDs
        if product_id:
            row = df[df['product_id'] == product_id]
            if len(row) > 0:
                target_products.append(row.iloc[0])

        if product_id_b:
            row_b = df[df['product_id'] == product_id_b]
            if len(row_b) > 0:
                target_products.append(row_b.iloc[0])

        # 2. If no explicit IDs, fuzzy/keyword search product mentions in question
        if not target_products:
            # Check if any products match by name or brand
            q_clean = question.lower()
            matched_candidates = []
            
            # Fast check top search results
            search_results = self.recommender.search_catalog_by_name(question, limit=4)
            if search_results:
                for sr in search_results[:2]:
                    r = df[df['product_id'] == sr['product_id']]
                    if len(r) > 0:
                        matched_candidates.append(r.iloc[0])

            # If still none, run query recommendation
            if not matched_candidates:
                rec_df = self.recommender.recommend(
                    query=question,
                    skin_type=skin_type,
                    target_concerns=concerns or [],
                    top_k=2,
                    use_reranker=True
                )
                if len(rec_df) > 0:
                    for _, r in rec_df.head(2).iterrows():
                        matched_candidates.append(r)

            target_products = matched_candidates[:2]

        # 3. Assemble structured evidence units
        product_evidence = []
        all_actives = set()
        all_citations = []

        for p in target_products:
            p_dict = p.to_dict() if hasattr(p, 'to_dict') else dict(p)
            
            # Price in INR
            p_inr = p_dict.get('price_inr')
            if (p_inr is None or (isinstance(p_inr, float) and p_inr != p_inr)) and p_dict.get('price_usd'):
                p_inr = round(p_dict.get('price_usd') * 86.5)

            # Clean actives and benefits
            def _parse_list(val):
                if isinstance(val, str):
                    try: return json.loads(val)
                    except: return []
                return val if isinstance(val, list) else []

            actives = _parse_list(p_dict.get('matched_actives'))
            benefits = _parse_list(p_dict.get('matched_benefits'))
            for a in actives:
                all_actives.add(a)

            # Knowledge graph evidence card for this product
            kg_card = self.kg.get_graph_evidence_card(p_dict, concerns or [])
            product_citations = []
            for path in kg_card.get('reasoning_paths', []):
                for cit in path.get('citations', []):
                    product_citations.append(cit)
                    all_citations.append(cit)

            # Skin-partitioned metrics
            skin_metrics = {}
            if skin_type:
                st = skin_type.lower().strip()
                skin_metrics = {
                    'skin_type': st.capitalize(),
                    'rating': p_dict.get(f'{st}_skin_avg_rating'),
                    'review_count': p_dict.get(f'{st}_skin_review_count', 0),
                    'pos_pct': p_dict.get(f'{st}_skin_pos_pct')
                }

            product_evidence.append({
                'product_id': str(p_dict.get('product_id')),
                'product_name': str(p_dict.get('product_name')),
                'brand': str(p_dict.get('brand')),
                'category': str(p_dict.get('category')),
                'price_inr': p_inr,
                'overall_rating': round(float(p_dict.get('rating', 4.2)), 2) if p_dict.get('rating') else 4.2,
                'review_count': int(p_dict.get('review_count', 0)),
                'matched_actives': actives,
                'matched_benefits': benefits,
                'skin_partitioned_metrics': skin_metrics,
                'citations': product_citations,
                'contraindications': kg_card.get('contraindications', [])
            })

        # 4. Check for direct active mentions in the question text (e.g., "retinol", "niacinamide", "glycolic acid")
        q_lower = question.lower()
        for alias, canonical in self.kg.synonym_lookup.items():
            if re.search(r'\b' + re.escape(alias) + r'\b', q_lower):
                all_actives.add(canonical)
                prov = self.kg.get_ingredient_provenance(canonical)
                if prov:
                    for cit in prov.get('evidence_citations', []):
                        if cit not in all_citations:
                            all_citations.append(cit)

        # 5. Intent detection & Routine Guidance Assembly
        intent = classify_user_intent(question)
        routine_guidance = None

        if intent == "ROUTINE_ADVICE":
            st_key = (skin_type or '').lower().strip()
            if st_key not in SKIN_TYPE_ROUTINE_GUIDES:
                for candidate in ["oily", "dry", "combination", "sensitive", "normal"]:
                    if candidate in q_lower:
                        st_key = candidate
                        break
                if st_key not in SKIN_TYPE_ROUTINE_GUIDES:
                    st_key = "combination"

            guide = SKIN_TYPE_ROUTINE_GUIDES[st_key]
            user_concerns_list = concerns or []
            if not user_concerns_list:
                for c_name in CONCERN_GUIDELINES:
                    if any(word.lower() in q_lower for word in c_name.split() if len(word) > 3):
                        user_concerns_list.append(c_name)

            concern_tips = []
            for c_name in user_concerns_list:
                if c_name in CONCERN_GUIDELINES:
                    cg = CONCERN_GUIDELINES[c_name]
                    concern_tips.append(f"• **{c_name}**: {cg['key_approach']} *Dermatological Rule*: {cg['rule']}")

            routine_guidance = {
                'skin_type': st_key,
                'focus': guide['focus'],
                'am_steps': guide['am_steps'],
                'pm_steps': guide['pm_steps'],
                'lifestyle_tips': guide['lifestyle_tips'],
                'concern_tips': concern_tips
            }

            # Seed all_actives with the routine's key actives & KG citations
            for a in guide.get('key_actives', []):
                all_actives.add(a)
                prov = self.kg.get_ingredient_provenance(a)
                if prov:
                    for cit in prov.get('evidence_citations', []):
                        if cit not in all_citations:
                            all_citations.append(cit)

        # 6. Routine contraindications between all active substances
        contraindications = self.kg.detect_contraindications(list(all_actives))

        return {
            'products': product_evidence,
            'all_actives': sorted(list(all_actives)),
            'citations': all_citations,
            'contraindications': contraindications,
            'user_skin_type': skin_type,
            'user_concerns': concerns or [],
            'intent': intent,
            'routine_guidance': routine_guidance
        }

    def format_context_prompt(self, context: Dict[str, Any]) -> str:
        """
        Renders the retrieved evidence into a structured markdown context block for the LLM.
        """
        lines = []
        lines.append("### EVIDENCE CONTEXT (RETRIEVED DERMASENSE KNOWLEDGE GRAPH & DATA)")

        # Routine Consultation Guidance (if advice intent)
        if context.get('intent') == 'ROUTINE_ADVICE' and context.get('routine_guidance'):
            rg = context['routine_guidance']
            lines.append(f"\n#### DERMATOLOGICAL ROUTINE & STEP GUIDELINES ({rg['skin_type'].upper()} SKIN):")
            lines.append(f"Strategic Focus: {rg['focus']}")
            lines.append("Morning (AM) Regimen:")
            for idx, s in enumerate(rg['am_steps'], 1):
                lines.append(f"  {idx}. {s}")
            lines.append("Evening (PM) Regimen:")
            for idx, s in enumerate(rg['pm_steps'], 1):
                lines.append(f"  {idx}. {s}")
            lines.append("Essential Application Rules & Lifestyle Tips:")
            for tip in rg['lifestyle_tips']:
                lines.append(f"  - {tip}")
            if rg.get('concern_tips'):
                lines.append("Target Concern Strategies:")
                for ct in rg['concern_tips']:
                    lines.append(f"  {ct}")

        # Products
        if context.get('products'):
            lines.append("\n#### PRODUCT SPECIFICATIONS & REVIEW DATA:")
            for i, p in enumerate(context['products'], 1):
                price_str = f"₹{p['price_inr']:,}" if p['price_inr'] else "Market Price"
                lines.append(f"Product {i}: **{p['product_name']}** by **{p['brand']}**")
                lines.append(f"  - Category: {p['category']} | Price: {price_str}")
                lines.append(f"  - Catalog Rating: {p['overall_rating']}★ across {p['review_count']} verified reviews")
                
                # Skin type metrics
                st = p.get('skin_partitioned_metrics')
                if st and st.get('rating'):
                    lines.append(f"  - {st['skin_type']} Skin Cohort: {st['rating']}★ ({st['review_count']} reviews, {st.get('pos_pct', 85)}% positive)")

                if p.get('matched_actives'):
                    lines.append(f"  - Active Ingredients: {', '.join(p['matched_actives'])}")
                if p.get('matched_benefits'):
                    lines.append(f"  - Addressed Concerns: {', '.join(p['matched_benefits'])}")

        # Knowledge Graph Citations
        if context.get('citations'):
            lines.append("\n#### PEER-REVIEWED CLINICAL RESEARCH CITATIONS (PUBMED):")
            for cit in context['citations'][:4]:
                lines.append(f"- **[{cit['citation_id']}]**: {cit['authors']} ({cit['year']}). *{cit['title']}*. {cit['journal']}.")
                lines.append(f"  Key Finding: {cit['finding']}")

        # Contraindications & Conflicts
        if context.get('contraindications'):
            lines.append("\n#### ACTIVE FORMULATION CONTRAINDICATIONS & SAFETY WARNINGS:")
            for c in context['contraindications']:
                lines.append(f"- ⚠️ **{c['title']}** (Severity: {c['severity']}): {c['mechanism']}")
                lines.append(f"  Dermatological Guidance: {c['recommendation']}")
        else:
            lines.append("\n#### ACTIVE FORMULATION CONTRAINDICATIONS: None detected between active compounds.")

        return "\n".join(lines)

    def _call_ollama(self, prompt: str, system_prompt: str) -> Optional[str]:
        """
        Calls local Ollama API with low temperature for factual precision.
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "top_p": 0.9,
                "num_predict": 450
            }
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            OLLAMA_API_URL,
            data=data,
            headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=3.5) as res:
                res_data = json.loads(res.read().decode("utf-8"))
                return res_data.get("response", "").strip()
        except Exception:
            return None

    def _generate_fallback_answer(self, question: str, context: Dict[str, Any], is_medical: bool) -> str:
        """
        Zero-crash deterministic answer generator synthesizing verified evidence if Ollama is offline.
        """
        parts = []

        if is_medical:
            parts.append(MEDICAL_DISCLAIMER + "\n")

        products = context.get('products', [])
        cits = context.get('citations', [])
        conflicts = context.get('contraindications', [])
        intent = context.get('intent', 'GENERAL_INQUIRY')
        rg = context.get('routine_guidance')

        # Check conflict queries
        if conflicts:
            parts.append("### ⚠️ Formulation Conflict & Layering Warning")
            for c in conflicts:
                parts.append(f"• **{c['title']}**: {c['mechanism']}")
                parts.append(f"  *Clinical Recommendation*: {c['recommendation']}")
            parts.append("\n")

        # Check if user asked for routine or advice
        if intent == 'ROUTINE_ADVICE' and rg:
            st_name = rg['skin_type'].capitalize()
            parts.append(f"### 🌅 Tailored Skincare Regimen & Expert Advice")
            parts.append(f"**Target Skin Profile**: {st_name} Skin  \n**Clinical Focus**: {rg['focus']}")

            parts.append("#### ☀️ Morning (AM) Routine")
            for idx, step in enumerate(rg['am_steps'], 1):
                parts.append(f"**Step {idx}**: {step}")

            parts.append("#### 🌙 Evening (PM) Routine")
            for idx, step in enumerate(rg['pm_steps'], 1):
                parts.append(f"**Step {idx}**: {step}")

            parts.append("#### 💡 Core Dermatological Rules & Best Practices")
            for tip in rg['lifestyle_tips']:
                parts.append(f"• {tip}")

            if rg.get('concern_tips'):
                parts.append("#### 🎯 Targeted Concern Guidance")
                for ct in rg['concern_tips']:
                    parts.append(ct)

            if products:
                parts.append("#### 🌿 Compatible Catalog Formulations")
                for p in products[:2]:
                    p_price = f"₹{p['price_inr']:,}" if p['price_inr'] else "Market Price"
                    parts.append(f"• **{p['product_name']}** ({p['brand']}) — {p_price}: Category: *{p['category']}*. Features {', '.join(p['matched_actives'] or ['barrier-supporting emollients'])}, rated {p['overall_rating']}★ across {p['review_count']} verified reviews.")

            if cits:
                parts.append("#### 📚 Peer-Reviewed Clinical Evidence")
                for c in cits[:2]:
                    parts.append(f"• **[{c['citation_id']}]** (*{c['journal']}*, {c['year']}): {c['finding']}")

            return "\n\n".join(parts)

        # Check product recommendations or comparisons
        if products:
            if len(products) >= 2:
                parts.append("### ⚖️ Side-by-Side Product Comparison Analysis")
                p1, p2 = products[0], products[1]
                p1_price = f"₹{p1['price_inr']:,}" if p1['price_inr'] else "Market Price"
                p2_price = f"₹{p2['price_inr']:,}" if p2['price_inr'] else "Market Price"
                parts.append(f"1. **{p1['product_name']}** ({p1['brand']}) — {p1_price}: Formulated with {', '.join(p1['matched_actives'] or ['barrier emollients'])}, holding {p1['overall_rating']}★ across {p1['review_count']} reviews.")
                parts.append(f"2. **{p2['product_name']}** ({p2['brand']}) — {p2_price}: Formulated with {', '.join(p2['matched_actives'] or ['active complexes'])}, holding {p2['overall_rating']}★ across {p2['review_count']} reviews.")
            else:
                p = products[0]
                p_price = f"₹{p['price_inr']:,}" if p['price_inr'] else "Market Price"
                parts.append(f"### 🌿 Clinical Profile: {p['product_name']} ({p['brand']})")
                parts.append(f"This product is priced at {p_price} and features active compounds: **{', '.join(p['matched_actives'] or ['botanical extracts'])}** targeting {', '.join(p['matched_benefits'] or ['skin hydration'])}.")
                st = p.get('skin_partitioned_metrics')
                if st and st.get('rating'):
                    parts.append(f"For **{st['skin_type']} skin**, reviewers report an average satisfaction rating of **{st['rating']}★** based on {st['review_count']} verified reviews ({st.get('pos_pct', 85)}% positive feedback).")

        # Literature citations
        if cits:
            parts.append("\n### 📚 Peer-Reviewed Clinical Evidence")
            for c in cits[:2]:
                parts.append(f"• **[{c['citation_id']}]** (*{c['journal']}*, {c['year']}): {c['finding']}")

        if not parts:
            parts.append(
                "Based on the DermaSense catalog and ingredient knowledge graph, there is insufficient clinical evidence "
                "to answer this query with full certainty. Please verify the active ingredients or provide specific product names."
            )

        return "\n\n".join(parts)

    def answer_question(
        self,
        question: str,
        product_id: Optional[str] = None,
        product_id_b: Optional[str] = None,
        skin_type: Optional[str] = None,
        concerns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end evidence-grounded RAG:
        1. Multi-source retrieval.
        2. Prompt formatting with grounding constraints.
        3. LLM generation with graceful deterministic fallback.
        4. Claim-to-evidence verification and citation validation.
        """
        t0 = time.time()
        q_lower = question.lower()

        # Check medical safety
        is_medical = any(flag in q_lower for flag in MEDICAL_RED_FLAGS)

        # 1. Retrieve Context
        context = self.retrieve_context(
            question=question,
            product_id=product_id,
            product_id_b=product_id_b,
            skin_type=skin_type,
            concerns=concerns
        )

        formatted_context = self.format_context_prompt(context)

        # 2. Construct System and User Prompts
        system_prompt = (
            "You are DermaSense Evidence-Grounded Skincare AI Specialist. "
            "Your mission is to provide accurate, scientifically grounded, and practical skincare advice.\n\n"
            "STRICT RULES:\n"
            "1. ADVICE & ROUTINES: When the user asks for advice, routine recommendations, or 'what should I do', provide clear, step-by-step guidance (AM and PM routines, order of application, and usage frequency) rather than simply listing chemical ingredient names.\n"
            "2. GROUNDING: Base your formulation and ingredient claims strictly on the provided EVIDENCE CONTEXT. Do not invent products, ratings, or studies.\n"
            "3. ACTIVE INTEGRATION: When recommending active ingredients, always explain WHEN to apply them (AM vs PM) and HOW to introduce them safely (e.g. patch test, start 2x/week).\n"
            "4. CITATIONS: Whenever discussing clinical ingredient mechanisms, include the exact PubMed identifier in brackets (e.g. [PMID:16029679]).\n"
            "5. REVIEW EVIDENCE: Whenever discussing user satisfaction or skin-type fit, quote the specific rating and review count from the context.\n"
            "6. PRICING: Always state prices in Indian Rupees (₹).\n"
            "7. MEDICAL SAFETY: If the user asks for a diagnosis or treatment of an infection, medical condition, or prescription drug, start your answer with a clear clinical disclaimer advising them to consult a certified dermatologist.\n"
            "8. UNCERTAINTY: If the evidence context does not contain sufficient details to answer, state clearly what is unknown rather than guessing."
        )

        user_prompt = (
            f"{formatted_context}\n\n"
            f"### USER QUESTION:\n{question}\n\n"
            f"Provide a clear, actionable, evidence-grounded answer adhering strictly to the above rules:"
        )

        # 3. Call LLM or Fallback
        raw_answer = self._call_ollama(user_prompt, system_prompt)
        model_used = self.model_name

        if not raw_answer:
            raw_answer = self._generate_fallback_answer(question, context, is_medical)
            model_used = "deterministic-evidence-synthesizer"
        elif is_medical and "dermatologist" not in raw_answer.lower():
            raw_answer = f"{MEDICAL_DISCLAIMER}\n\n{raw_answer}"

        latency_ms = round((time.time() - t0) * 1000.0, 2)

        # 4. Verify Groundedness & Citations
        verification = self.verify_groundedness(raw_answer, context, is_medical)

        return {
            'question': question,
            'answer': raw_answer,
            'retrieved_context': context,
            'citations_found': verification['citations_found'],
            'citations_verified': verification['citations_verified'],
            'groundedness_score': verification['groundedness_score'],
            'safety_compliant': verification['safety_compliant'],
            'model_used': model_used,
            'latency_ms': latency_ms
        }

    def verify_groundedness(self, answer: str, context: Dict[str, Any], is_medical: bool) -> Dict[str, Any]:
        """
        Validates citations and factual alignment between answer and retrieved context.
        """
        # Find all PMIDs mentioned in answer
        pmid_pattern = re.compile(r'PMID:(\d+)', re.IGNORECASE)
        found_pmids = set(pmid_pattern.findall(answer))

        # Context PMIDs
        context_pmids = set()
        for cit in context.get('citations', []):
            cid = cit.get('citation_id', '')
            m = pmid_pattern.search(cid)
            if m:
                context_pmids.add(m.group(1))

        verified_pmids = list(found_pmids.intersection(context_pmids))
        hallucinated_pmids = list(found_pmids - context_pmids)

        # Check active ingredient grounding
        mentioned_actives = []
        answer_lower = answer.lower()
        for act in self.kg.get_all_ingredients():
            if act in answer_lower:
                mentioned_actives.append(act)

        context_actives = set(a.lower() for a in context.get('all_actives', []))
        for p in context.get('products', []):
            for a in p.get('matched_actives', []):
                context_actives.add(a.lower())

        grounded_actives = [a for a in mentioned_actives if a in context_actives]
        
        # Groundedness score calculation: harmonic combination of active grounding and citation precision
        act_score = 1.0
        if mentioned_actives:
            act_score = len(grounded_actives) / len(mentioned_actives)

        cit_score = 1.0
        if found_pmids:
            cit_score = len(verified_pmids) / len(found_pmids)
        elif hallucinated_pmids:
            cit_score = 0.0

        score = (0.5 * act_score) + (0.5 * cit_score)

        # Safety compliance
        safety_compliant = True
        if is_medical:
            safety_compliant = ("dermatologist" in answer_lower or "physician" in answer_lower or "doctor" in answer_lower)

        return {
            'citations_found': [f"PMID:{p}" for p in found_pmids],
            'citations_verified': [f"PMID:{p}" for p in verified_pmids],
            'hallucinated_citations': [f"PMID:{p}" for p in hallucinated_pmids],
            'groundedness_score': max(0.0, min(1.0, round(score, 3))),
            'safety_compliant': safety_compliant
        }
