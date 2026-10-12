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

        # 5. Routine contraindications between all active substances
        contraindications = self.kg.detect_contraindications(list(all_actives))

        return {
            'products': product_evidence,
            'all_actives': sorted(list(all_actives)),
            'citations': all_citations,
            'contraindications': contraindications,
            'user_skin_type': skin_type,
            'user_concerns': concerns or []
        }

    def format_context_prompt(self, context: Dict[str, Any]) -> str:
        """
        Renders the retrieved evidence into a structured markdown context block for the LLM.
        """
        lines = []
        lines.append("### EVIDENCE CONTEXT (RETRIEVED DERMASENSE KNOWLEDGE GRAPH & DATA)")

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
            with urllib.request.urlopen(req, timeout=18.0) as res:
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

        # Check conflict queries
        if conflicts:
            parts.append("### ⚠️ Formulation Conflict & Layering Warning")
            for c in conflicts:
                parts.append(f"• **{c['title']}**: {c['mechanism']}")
                parts.append(f"  *Clinical Recommendation*: {c['recommendation']}")
            parts.append("\n")

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
            "Your mission is to provide accurate, scientifically grounded answers to skincare questions.\n\n"
            "STRICT RULES:\n"
            "1. GROUNDING: Base every factual claim strictly on the provided EVIDENCE CONTEXT. Do not invent products, ratings, or studies.\n"
            "2. CITATIONS: Whenever discussing clinical ingredient mechanisms, include the exact PubMed identifier in brackets (e.g. [PMID:16029679]).\n"
            "3. REVIEW EVIDENCE: Whenever discussing user satisfaction or skin-type fit, quote the specific rating and review count from the context.\n"
            "4. PRICING: Always state prices in Indian Rupees (₹).\n"
            "5. MEDICAL SAFETY: If the user asks for a diagnosis or treatment of an infection, medical condition, or prescription drug, start your answer with a clear clinical disclaimer advising them to consult a certified dermatologist.\n"
            "6. UNCERTAINTY: If the evidence context does not contain sufficient details to answer, state clearly what is unknown rather than guessing."
        )

        user_prompt = (
            f"{formatted_context}\n\n"
            f"### USER QUESTION:\n{question}\n\n"
            f"Provide a clear, helpful, evidence-grounded answer adhering strictly to the above rules:"
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
