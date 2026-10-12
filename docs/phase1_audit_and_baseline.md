# DermaSense Upgrade: Phase 1 Technical Audit & Reproducible Baseline Report

**Project:** DermaSense — Personalized, Explainable AI Skincare Recommendation Engine  
**Phase:** Stage 1: Technical Audit & Baseline Verification  
**Date:** October 2026  
**Status:** Completed & Verified  

---

## 1. Executive Summary

As part of the DermaSense AI/ML Enhancement Initiative, this technical audit establishes the ground truth of the active codebase, verifies runtime environments, reproduces baseline evaluation metrics on 20 dermatological benchmarks, and defines the modular implementation roadmap for the scoped upgrades:
1. **Cross-Encoder Deep Learning Reranking**
2. **Provenance-Aware Ingredient Knowledge Graph**
3. **Evidence-Grounded Local RAG Skincare Q&A** (leveraging local GPU / Ollama)
4. **Adaptive Feedback-Driven Personalization**

---

## 2. Technical Audit of Existing Implementation

| Component / Subsystem | Current Code Location | Implementation Status & Key Observations |
| :--- | :--- | :--- |
| **1. Application Entry Point** | `app/app.py` | Flask 3.0+ web framework. Initializes `DermaSenseRecommender` and `DermaSenseExplainer` at startup. Hosts 4 page routes and 5 API endpoints (`/api/recommend`, `/api/compare`, `/api/search`, `/api/ingredients`, `/api/stats`). |
| **2. Recommendation Scoring Engine** | `src/models/recommender.py` | Vectorized NumPy multi-factor composite recommender with 6 factors ($S_{\text{sem}}, S_{\text{concern}}, S_{\text{skin}}, S_{\text{senti}}, S_{\text{price}}, S_{\text{pop}}$) and 4 preset weights (`balanced`, `ingredient_focus`, `review_focus`, `budget_focus`). Hard filters applied for category, brand, and INR budget cap ($1.05 \times B$). |
| **3. Embedding & Vector Search** | `src/models/embeddings.py` | Dense 384-dimensional embeddings generated with `sentence-transformers/all-MiniLM-L6-v2`. FAISS `IndexFlatIP` index loaded from `data/embeddings/product_index.faiss` (16,110 vectors). Microsecond candidate scoring via vectorized NumPy dot product. |
| **4. Sentiment Analysis Model** | `src/features/sentiment_features.py` | Rule/lexicon-based VADER (`SentimentIntensityAnalyzer`) streaming over review text + regex pattern matching for 9 positive and 7 negative dermatological themes. Skin-type partitioned sentiment aggregations precomputed in master parquet. |
| **5. Skincare Ingredient Ontology** | `src/features/ingredient_features.py` | Python dictionary `SKIN_BENEFITS_MAP` covering 8 primary skin benefit categories mapped to ~70 active compounds. Pre-extracted into multi-hot flags and matched lists. Currently lacks explicit graph topology, relationship edges, and scientific citations. |
| **6. Explainability (XAI) Mechanism** | `src/explainability/explainer.py` | Deterministic template-based clinical rationale generator. Explains concern overlap, active ingredient cellular mechanisms (from 22 descriptions), skin-partitioned satisfaction ratings, and budget compliance in INR ₹. |
| **7. Product & Review Catalogs** | `data/processed/` | `master_product_features.parquet` (16,110 products across Sephora US, Amazon Beauty, and Flipkart India; 63 columns). `unified_reviews.parquet` (1.2M user reviews with skin-type tags). Normalized Indian Rupee pricing (`price_inr`). |
| **8. UI Routes & Web Architecture** | `app/templates/`, `app/static/` | Semantic HTML5 + Vanilla CSS luxury aesthetic interface. Dedicated pages: Intake (`/`), Recommendations (`/recommendations`), Side-by-Side Compare (`/compare`), and Ingredient Dictionary (`/ingredients`). Includes sliding AM/PM routine drawer and active ingredient modals. |
| **9. Benchmark Evaluation Suite** | `results/evaluation_suite.py` | 20 clinical dermatological queries evaluating Precision@5, Precision@10, Budget Adherence %, Brand Diversity, Skin Evidence Score, and Sub-second Latency. Logs results to `results/evaluation_results.json` and `results/evaluation_summary.md`. |
| **10. Upgrade Readiness Status** | Proposed Upgrades 1–5 | • Upgrade 1 (Cross-Encoder): 0% built (MiniLM FAISS baseline ready)<br>• Upgrade 2 (ABSA): Baseline VADER/regex exists (deferred to future work)<br>• Upgrade 3 (Knowledge Graph): Baseline dictionary exists (graph structure 0% built)<br>• Upgrade 4 (RAG Q&A): 0% built (Ollama + `qwen2.5:7b-instruct` installed and ready)<br>• Upgrade 5 (Feedback Personalization): 0% built |

---

## 3. Hardware & Runtime Environment Verification

- **Operating System:** Windows 11
- **Host GPU:** NVIDIA GeForce RTX 4050 Laptop GPU (dedicated VRAM)
- **Python Environment:** Python 3.12 with `sentence-transformers`, `transformers`, `torch`, `faiss-cpu`, `scikit-learn`, `flask`, `pyarrow`
- **Graph Library:** `networkx` 3.7 installed and operational
- **Local LLM Engine:** Ollama v0.34.3 installed and active
  - Detected Local Models:
    - `qwen2.5:7b-instruct` (4.7 GB) — High-quality instruction-following LLM for evidence-grounded RAG
    - `qwen3-vl:8b` (6.1 GB)

---

## 4. Reproducible Baseline Quantitative Benchmark

The baseline evaluation suite was run on `results/evaluation_suite.py` across all 20 benchmark clinical dermatological test scenarios. The empirical baseline measurements are recorded below:

| Benchmark Metric | Target Standard | Live Measured Baseline | Evaluation Assessment |
| :--- | :--- | :--- | :--- |
| **Active Ingredient Precision@5** | $\ge 80.0\%$ | **93.0%** | 🟢 Exceeds Benchmark |
| **Active Ingredient Precision@10** | $\ge 75.0\%$ | **94.0%** | 🟢 Exceeds Benchmark |
| **Budget Constraint Compliance** | $\ge 95.0\%$ | **96.5%** | 🟢 Strict Bound Enforced |
| **Catalog Brand Diversity (Top-10)** | $\ge 5.0$ brands | **7.9 unique brands** | 🟢 High Diversity |
| **Mean Skin-Type Evidence Score** | $\ge 0.70$ | **0.907** | 🟢 Strong Concordance |
| **Mean Retrieval & Ranking Latency** | $< 150\text{ ms}$ | **91.1 ms** | ⚡ Instantaneous Real-Time |
| **95th Percentile Latency (p95)** | $< 250\text{ ms}$ | **135.2 ms** | ⚡ Instantaneous Real-Time |

---

## 5. Architectural Implementation Map

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       DERMASENSE MODULAR ARCHITECTURE                        │
└─────────────────────────────────────────────────────────────────────────────┘

 [User Input Profile: Skin Type + Stated Concerns + Budget (₹) + Exclusions]
                                    │
                                    ▼
 ┌───────────────────────────────────────────────────────────────────────────┐
 │ Stage 1: Candidate Pool Retrieval & Hard Filtering (src/models/)          │
 │ • Hard Constraint Validator: Category, INR Budget Cap, Verified Exclusions │
 │ • 1st-Stage Dense Semantic Vector Search: all-MiniLM-L6-v2 + FAISS Index   │
 └───────────────────────────────────────────────────────────────────────────┘
                                    │ (Top-K Candidate Pool, e.g., 30 items)
                                    ▼
 ┌───────────────────────────────────────────────────────────────────────────┐
 │ Stage 2: Deep-Learning Cross-Encoder Reranking [NEW: src/models/reranker] │
 │ • Model: cross-encoder/ms-marco-MiniLM-L-6-v2                             │
 │ • Inputs: (User Clinical Query, Candidate Product Formulation Document)   │
 │ • Fallback: Gracefully bypasses reranker if model is disabled/unavailable  │
 └───────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
 ┌───────────────────────────────────────────────────────────────────────────┐
 │ Stage 3: Multi-Evidence Composite Scoring & Knowledge Graph Traversal     │
 │ • Ingredient Knowledge Graph [NEW: src/knowledge_graph/graph.py]          │
 │   (Product ── Contains ──> Active ── Recommends ──> Concern + Citations)  │
 │ • Adaptive User Preferences [NEW: src/personalization/preference_manager] │
 │   (Soft bias learned from likes, dislikes, texture & fragrance feedback)  │
 │ • Review Evidence Score (Skin-type partitioned ratings & confidence)      │
 │ • Price Alignment Score (Normalized against INR budget)                   │
 └───────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
 ┌───────────────────────────────────────────────────────────────────────────┐
 │ Stage 4: Explainability & Grounded RAG Assistant [NEW: src/rag/qa_engine] │
 │ • Deterministic Clinical Rationale (src/explainability/explainer.py)      │
 │ • Evidence-Grounded Q&A (Ollama qwen2.5:7b-instruct + Verified Citations) │
 └───────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
 ┌───────────────────────────────────────────────────────────────────────────┐
 │ Stage 5: Web UI Presentation & Interactive Experience (app/app.py)        │
 │ • Interactive Match % Cards, AM/PM Regimen Drawer, Ingredient Modals      │
 │ • Interactive Q&A Assistant Drawer & Feedback Actions (Like/Dislike/Save) │
 └───────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Execution Roadmap & Next Steps

1. **Phase 1 (Done):** Audit codebase, establish environment readiness, verify reproducible baseline.
2. **Phase 2 (Next):** Implement **Cross-Encoder Deep Learning Reranking** (`src/models/reranker.py`), integrate into recommender pipeline with configurable flag, and measure relevance gains ($P@5$, latency).
3. **Phase 3:** Construct the **Provenance-Aware Ingredient Knowledge Graph** (`src/knowledge_graph/graph.py`) with clinical citations, chemical synonyms, and contraindication cautions.
4. **Phase 4:** Build the **Evidence-Grounded RAG Skincare Q&A Engine** (`src/rag/qa_engine.py`) backed by Ollama `qwen2.5:7b-instruct`.
5. **Phase 5:** Build **Feedback-Driven Adaptive Personalization** (`src/personalization/preference_manager.py`) with transparent SQLite persistence.
6. **Phase 6:** Full end-to-end integration, comparative ablation benchmarks, and project report update.
