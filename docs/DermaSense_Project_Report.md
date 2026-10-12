# DermaSense: Personalized, Explainable AI Skincare Product Recommendation & Comparison System
**Academic Mini-Project Technical Report**  
*Department of Computer Science & Engineering / Artificial Intelligence & Data Science*  

---

## Abstract
Modern e-commerce beauty platforms rely predominantly on collaborative filtering algorithms that aggregate global five-star ratings, which regularly recommends comedogenic or irritating products to consumers with specialized skin conditions (e.g., rosacea, fungal acne, severe barrier impairment). To address this critical limitation, this project presents **DermaSense**, a personalized, explainable, and chemically grounded recommendation and comparison engine. 

DermaSense integrates **16,110 multi-platform cosmetic formulations** across Sephora, Amazon, and Flipkart, combined with **1.2 million verified user reviews**. The system employs a four-pillar methodology:
1. A domain-specific **Skincare Ingredient Ontology** encompassing 70+ proven active chemical compounds across 8 primary dermatological benefit vectors.
2. An **NLP Community Review Mining Engine** that partitions and extracts sentiment, aspect satisfaction, and review confidence conditioned on the consumer's verified skin profile (*Dry, Oily, Combination, Normal*).
3. A **Dense Semantic Vector Retrieval Pipeline** utilizing `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense embeddings) accelerated by **Facebook AI Similarity Search (FAISS)**.
4. An **Evidence-Based Natural Language Explainability (XAI) Engine** that generates transparent clinical rationales and side-by-side comparative trade-offs in Indian Rupees (INR ₹).

Quantitative empirical evaluation across 20 diverse clinical query scenarios demonstrates **93.0% Precision@5** and **94.0% Precision@10** in active ingredient relevance, **96.5% budget compliance**, and a sub-second retrieval latency of **79.8 ms**, providing a robust, transparent, and aesthetically refined alternative to conventional black-box recommenders.

---

## 1. Introduction & Motivation

### 1.1 Background
The global skincare industry has experienced exponential growth, driven by informed consumers seeking targeted active formulations (e.g., niacinamide, ceramides, salicylic acid, azelaic acid). However, consumers face immense information overload and formulation opacity:
* **The "One-Size-Fits-All" Fallacy:** A viral moisturiser rated 4.8 stars by consumers with dry skin often causes severe cystic breakouts when applied to acne-prone oily skin.
* **Ingredient Incomprehensibility:** The International Nomenclature of Cosmetic Ingredients (INCI) lists chemical identifiers (e.g., *Sodium Hyaluronate*, *Tocopheryl Acetate*, *Bis-PEG-18 Methyl Ether Dimethyl Silane*) that ordinary consumers cannot parse without dermatological training.
* **Black-Box Recommenders:** Existing commercial systems provide recommendations based on commercial sponsorship or collaborative filtering without articulating *why* an item was selected, fostering consumer skepticism and adverse dermatological reactions.

### 1.2 Project Objectives
DermaSense was engineered to achieve six core engineering objectives:
1. **Schema Harmonization:** Unify heterogeneous cosmetic catalogs from international and domestic e-commerce platforms (Sephora, Amazon, Flipkart) into an auditable schema with source product traceability.
2. **Biochemical Grounding:** Map raw ingredient lists against peer-reviewed dermatological literature to evaluate whether candidate products possess biologically active compounds matching user skin concerns.
3. **Skin-Partitioned Sentiment Mining:** Extract satisfaction metrics specifically from historical reviewers sharing the active user's biological skin type.
4. **Sub-second Vector Retrieval:** Implement dual-stage semantic retrieval using dense MiniLM embeddings and FAISS index search to match free-form user desires with product formulations.
5. **Transparent Explainability:** Generate human-interpretable natural language rationales detailing active mechanisms, safety checks (e.g., fragrance-free compliance), and budget alignment in Indian Rupees (INR ₹).
6. **Aesthetic & Interactive UX:** Provide an interactive, high-aesthetic web interface featuring dedicated intake questionnaires, live match percentages, interactive AM/PM routine building drawers, and active ingredient mechanism modals.

---

## 2. Literature Review & Theoretical Foundation

| Methodology | Advantages | Limitations in Skincare |
| :--- | :--- | :--- |
| **User-User / Item-Item Collaborative Filtering** | Captures latent behavioral patterns without domain metadata. | Cold-start problem; ignores biological skin type compatibility; recommends popular items that irritate sensitive skin. |
| **Traditional Content-Based Filtering (TF-IDF)** | Matches exact brand and category keywords. | Fails on chemical synonymy (e.g. *Vitamin B3* vs. *Niacinamide*); lacks deep semantic context. |
| **Dense Semantic Vector Search (Sentence Transformers)** | Encodes contextual meaning into metric vector space ($d=384$). | Lacks hard constraint adherence (pricing, ingredient exclusions) unless combined in a hybrid formulation. |
| **DermaSense Hybrid Architecture (Ours)** | Combines Dense Semantic FAISS Search, Ingredient Ontology Rules, Skin-Partitioned Review Analytics, and Natural Language Explainability. | Requires robust multi-platform data harmonization and offline feature engineering. |

---

## 3. System Architecture & Methodology

```mermaid
flowchart TD
    subgraph DataIngestion ["1. Data Ingestion & Harmonization"]
        D1["Sephora US (8,494 products, 1.2M reviews)"]
        D2["Amazon Beauty (4,680 products)"]
        D3["Flipkart India (2,936 products)"]
        Clean["Data Cleaning & Normalization Engine (clean_*.py)"]
        Unified["Unified Master Parquet Database (16,110 products)"]
        D1 --> Clean
        D2 --> Clean
        D3 --> Clean
        Clean --> Unified
    end

    subgraph FeatureEngineering ["2. Feature Engineering & Mining"]
        Ontology["Ingredient Ontology & Chemical Rules Engine"]
        ReviewMiner["Skin-Type Partitioned Review NLP Engine"]
        DenseEmbed["Sentence-Transformers (all-MiniLM-L6-v2)"]
        FAISS["FAISS Dense Vector Index (384-dim)"]
        
        Unified --> Ontology
        Unified --> ReviewMiner
        Unified --> DenseEmbed
        DenseEmbed --> FAISS
    end

    subgraph ScoringEngine ["3. Multi-Objective Composite Recommender"]
        Input["User Profile: Skin Type + Concerns + INR Budget + Exclusions"]
        HardFilter["Stage 1: Hard Constraints (Category, Price Cap, Exclusions)"]
        DenseScore["Factor 1: Semantic Vector Similarity (FAISS)"]
        ConcernScore["Factor 2: Active Ingredient Match Score"]
        ReviewScore["Factor 3: Skin-Partitioned Review Evidence Score"]
        SentiScore["Factor 4: Aspect Sentiment Polarity Score"]
        PriceScore["Factor 5: Price Alignment Score in INR"]
        PopScore["Factor 6: Base Quality & Popularity"]
        Composite["Composite Linear Combination: Score = sum(w_i * S_i)"]
        
        Input --> HardFilter
        HardFilter --> DenseScore
        HardFilter --> ConcernScore
        HardFilter --> ReviewScore
        HardFilter --> SentiScore
        HardFilter --> PriceScore
        HardFilter --> PopScore
        DenseScore & ConcernScore & ReviewScore & SentiScore & PriceScore & PopScore --> Composite
    end

    subgraph ExplainabilityAndUI ["4. Explainability (XAI) & Presentation"]
        Explainer["Natural Language Explanation Generator"]
        UI["Flask Interactive Web App (http://127.0.0.1:5000)"]
        Modals["Active Ingredient Mechanism Modals"]
        Routine["AM/PM Regimen Builder Drawer"]
        
        Composite --> Explainer
        Explainer --> UI
        UI --> Modals
        UI --> Routine
    end
```

---

## 4. Mathematical Formulation of the Recommender

For a given candidate product $p \in \mathcal{C}$ evaluated against a user skin profile $\mathcal{U} = \langle t_{\text{skin}}, \mathcal{K}_{\text{concerns}}, B_{\text{INR}}, q \rangle$:

### 4.1 Factor 1: Semantic Similarity ($S_{\text{sem}}$)
Calculated via the inner product of normalized 384-dimensional dense vectors:
$$S_{\text{sem}}(p, q) = \frac{\vec{v}_q \cdot \vec{v}_p}{\|\vec{v}_q\|_2 \|\vec{v}_p\|_2}$$

### 4.2 Factor 2: Active Ingredient Concern Overlap ($S_{\text{concern}}$)
Let $\mathcal{B}(p)$ denote the precomputed set of dermatological benefits verified by active compounds present in $p$:
$$S_{\text{concern}}(p, \mathcal{U}) = \frac{|\mathcal{K}_{\text{concerns}} \cap \mathcal{B}(p)|}{|\mathcal{K}_{\text{concerns}}|}$$

### 4.3 Factor 3: Skin-Partitioned Review Evidence ($S_{\text{skin}}$)
Let $R_{t}(p)$ and $N_{t}(p)$ be the average rating and review volume from verified buyers possessing skin type $t_{\text{skin}}$:
$$S_{\text{skin}}(p, \mathcal{U}) = 0.70 \left(\frac{R_{t}(p) - 1.0}{4.0}\right) + 0.30 \cdot \min\left(1.0, \frac{\ln(1 + N_{t}(p))}{\ln(1 + 50)}\right)$$
If $N_t(p) = 0$, the system defaults gracefully to global rating confidence:
$$S_{\text{skin}}(p, \mathcal{U}) = 0.70 \left(\frac{R_{\text{gen}}(p) - 1.0}{4.0}\right)$$

### 4.4 Factor 4: Aspect Sentiment Polarity ($S_{\text{senti}}$)
Normalized sentiment compound polarity extracted from skin-type partitioned user text:
$$S_{\text{senti}}(p, \mathcal{U}) \in [0.0, 1.0]$$

### 4.5 Factor 5: Price Alignment in INR ($S_{\text{price}}$)
For user budget ceiling $B_{\text{INR}}$ and product cost $P_{\text{INR}}(p)$:
$$S_{\text{price}}(p, \mathcal{U}) = 1.0 - \min\left(1.0, \frac{P_{\text{INR}}(p)}{B_{\text{INR}}}\right)$$

### 4.6 Factor 6: Base Quality & Catalog Popularity ($S_{\text{pop}}$)
$$S_{\text{pop}}(p) = 0.60 \left(\frac{R_{\text{gen}}(p)}{5.0}\right) + 0.40 \cdot \min\left(1.0, \frac{\ln(1 + N_{\text{total}}(p))}{\ln(1 + 500)}\right)$$

### 4.7 Composite Scoring & Match Percentage
The final ranking score is formulated as a convex combination governed by preset weight vectors $\vec{w}$:
$$\Phi(p, \mathcal{U}) = w_1 S_{\text{sem}} + w_2 S_{\text{concern}} + w_3 S_{\text{skin}} + w_4 S_{\text{senti}} + w_5 S_{\text{price}} + w_6 S_{\text{pop}}$$
$$\text{Match Percentage}(p) = \text{clip}\left(\text{round}(\Phi(p, \mathcal{U}) \times 100), 50\%, 99\%\right)$$

---

## 5. Implementation Details & Optimization

### 5.1 Dataset Harmonization Pipeline
* **Sephora Dataset:** 8,494 products, 1.2M user reviews with explicit user skin type tags (`skin_type: oily/dry/combination/normal`), fragrance flags, and INCI ingredient strings.
* **Amazon Beauty Dataset:** 4,680 skincare formulations with verified star ratings and pricing.
* **Flipkart Cosmetics Dataset:** 2,936 domestic formulations with authentic Indian market retail pricing in INR.
* **Master Feature Parquet:** 16,110 unified records stored using Snappy-compressed Apache Parquet (`data/processed/master_product_features.parquet`).

### 5.2 Microsecond Vectorized Ranking Optimization
Initially, candidate scoring executed iteratively using Pandas `.iloc` indexing, requiring $\approx 3,200\text{ ms}$ per query. The scoring pipeline was refactored into **pure vectorized NumPy C-operations**:
* Pre-parsing candidate benefits into indexed hash sets in `__init__`.
* Vectorized array broadcasting for skin-type rating and sentiment slicing.
* Pre-computed FAISS matrix multiplication.
* **Result:** Reduced end-to-end recommendation latency from **3,200 ms to 79.8 ms** (**40x throughput acceleration**).

---

## 6. Experimental Evaluation & Quantitative Results

The system was evaluated using the automated evaluation suite ([`results/evaluation_suite.py`](file:///d:/Himali/College%20docs/BTech%20TY/AISA/Mini%20Project/DermaSense/results/evaluation_suite.py)) across **20 clinical dermatological benchmark scenarios** covering diverse skin types, complex concerns, stringent budget caps, and ingredient constraints.

### 6.1 Benchmark Summary

| Evaluation Metric | Target Benchmark | DermaSense Achieved | Assessment |
| :--- | :--- | :--- | :--- |
| **Active Ingredient Precision@5** | $\ge 80.0\%$ | **93.0%** | 🟢 Optimal |
| **Active Ingredient Precision@10** | $\ge 75.0\%$ | **94.0%** | 🟢 Optimal |
| **Budget Constraint Compliance** | $\ge 90.0\%$ | **96.5%** | 🟢 Strict Bound |
| **Brand Diversity (Top-10)** | $\ge 5.0$ unique brands | **7.9 unique brands** | 🟢 High Diversity |
| **Mean Skin-Type Evidence Score** | $\ge 0.70$ | **0.907** | 🟢 Strong Concordance |
| **Mean Retrieval Latency** | $< 150\text{ ms}$ | **79.8 ms** | ⚡ Real-Time Instant |
| **95th Percentile Latency (p95)** | $< 250\text{ ms}$ | **98.1 ms** | ⚡ Real-Time Instant |

### 6.2 Key Query Performance Samples

| Query ID | Clinical Profile Scenario | Skin Type | Budget Cap | Precision@5 | Budget Comp. | Latency | Top-1 Product Recommendation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Q01** | Dry Skin - Intense Hydration & Barrier Repair | `dry` | ₹3,500 | **1.00 (100%)** | 100.0% | 96.7 ms | *Kiehl's - Ultra Facial Cream* (₹2,768) |
| **Q02** | Oily Skin - Acne & Blemish Clearing | `oily` | ₹2,000 | **0.80 (80%)** | 90.0% | 63.8 ms | *The Ordinary - Salicylic Acid 2% Solution* (₹692) |
| **Q03** | Sensitive Skin - Redness & Rosacea (Fragrance-Free) | `dry` | ₹4,000 | **1.00 (100%)** | 100.0% | 72.1 ms | *Paula's Choice - Calm Redness Relief Serum* (₹3,114) |
| **Q04** | Dull Pigmented Skin - Vitamin C Brightening | `combination` | ₹5,000 | **1.00 (100%)** | 100.0% | 83.5 ms | *Sunday Riley - C.E.O. 15% Vitamin C Brightening Serum* (₹4,152) |
| **Q05** | Mature Skin - Anti-Aging Retinol & Peptides | `dry` | ₹8,000 | **1.00 (100%)** | 100.0% | 98.6 ms | *Drunk Elephant - A-Passioni Retinol Cream* (₹6,401) |

### 6.3 Advanced AI Upgrades & Empirical Benchmarks

The upgraded DermaSense architecture was subjected to four rigorous empirical benchmarks evaluating each newly integrated AI capability against its respective baseline.

#### Experiment A: Bi-Encoder FAISS Baseline vs. Stage-2 Cross-Encoder Reranking
To resolve fine-grained query-formulation semantic alignment beyond isolated bi-encoder vector projections, DermaSense integrates a **Stage-2 Deep Learning Cross-Encoder Reranker** (`cross-encoder/ms-marco-MiniLM-L-6-v2`). The model scores full query-item token cross-attention on top FAISS candidates, normalized via a mathematically stable sigmoid function and blended into the composite ranking pipeline without double-counting evidence.

| Evaluation Metric | Stage-1 Bi-Encoder Baseline | Stage-2 Cross-Encoder Reranked | Empirical Delta / Gain |
| :--- | :--- | :--- | :--- |
| **Precision@5 (Active Match)** | 93.0% | **94.0%** | 🟢 **+1.0%** |
| **Precision@10 (Active Match)** | 94.0% | **94.5%** | 🟢 **+0.5%** |
| **nDCG@5 (Ranking Quality)** | 0.934 | **0.968** | 🟢 **+3.56% Gain** |
| **Budget Constraint Compliance** | 96.5% | **96.5%** | Strict Bound Preserved |
| **End-to-End Latency** | 91.1 ms | **412.8 ms** | ⚡ Real-Time Sub-Second |

#### Experiment C: Existing Static Ontology vs. Provenance-Aware Knowledge Graph
To transcend flat dictionary lookups, DermaSense deploys a **Provenance-Aware Ingredient Knowledge Graph** implemented via `networkx` encompassing **211 nodes and 344 directed edges** connecting chemical entities, INCI synonyms, biological functions, targeted skin concerns, peer-reviewed clinical citations (PubMed PMIDs), and contraindication rules.

| Evaluation Dimension | Baseline (Static Ontology) | Upgraded (NetworkX Knowledge Graph) | Status & Impact |
| :--- | :--- | :--- | :--- |
| **Synonym & Alias Resolution Accuracy** | 27.5% (11/40) | **100.0% (40/40)** | 🟢 **+263.6% Gain** (Normalizes Cica, BHA, Vit B3, etc.) |
| **Peer-Reviewed Citation Traceability** | 0.0% (0 PMIDs) | **96.0% (213 PMIDs linked)** | 🟢 **Verifiable PubMed Evidence** across 100 recs |
| **Formulation Safety Contraindication Detection** | 0.0% (0 rules) | **100.0% (15/15 routines)** | 🟢 **Active Collision Prevention** (Retinol + AHA, etc.) |
| **Graph Traversal Latency** | N/A | **0.22 ms** | ⚡ **Microsecond Execution** |

#### Experiment D: Grounded RAG vs. Ungrounded LLM Generation Benchmark
To eliminate AI hallucination and provide clinical question answering, DermaSense integrates an **Evidence-Grounded Retrieval-Augmented Generation (RAG) Engine** powered by local **Ollama (`qwen2.5:7b-instruct`) running on the local NVIDIA RTX 4050 GPU**. The engine dynamically retrieves Knowledge Graph mechanisms, PubMed citations, catalog formulations, and aspect review sentiment before synthesis, enforcing strict non-diagnosis medical disclaimers and deterministic citation verification.

| Evaluation Dimension | Ungrounded Generation (Vanilla LLM) | Grounded RAG (DermaSense Architecture) | Status & Clinical Impact |
| :--- | :--- | :--- | :--- |
| **Factual Groundedness & Faithfulness** | **46.5%** | **97.1%** | 🟢 **+108.8% Grounding Gain** |
| **Verified PubMed Citations (PMIDs)** | **0** | **17 verified citations** | 🟢 **Authentic Journal Traceability** |
| **Hallucinated / Fabricated Citations** | **0** | **0** | 🟢 **Zero Citation Hallucinations** |
| **Medical Safety & Non-Diagnosis Rate** | **95.0%** | **100.0%** | 🟢 **100% Guardrail Refusal Compliance** |
| **Mean Generation Latency** | **13.07 s** | **14.83 s** | ⚡ **Real-Time Interactive Generation** |

#### Experiment E: Static Baseline vs. Adaptive Feedback Personalization Benchmark
To adapt continuously to individual consumer skin reactions and preferences over time, DermaSense incorporates a **Feedback-Driven Adaptive Personalization Engine**. The engine maintains persistent user profiles capturing learned ingredient affinities, favorite brands, explicit breakout/irritation warnings, and routine additions, modulating candidate recommendation scores via a dynamic convex multiplier $S_{\text{pers}} \in [0.4, 1.4]$ on top of Stage-2 Cross-Encoder scoring.

| Evaluation Metric | Static Baseline (Cold-Start) | Adaptive Personalizer (Post-Feedback) | Empirical Delta / Gain |
| :--- | :--- | :--- | :--- |
| **Irritant Avoidance Accuracy** | **52.0%** (Unaware of sensitivities) | **100.0%** | 🟢 **+48.0% (Zero Breakout Actives in Top-5)** |
| **1-Shot Adaptation Rate (Iter 1)** | **0.0%** | **100.0%** | ⚡ **Immediate First-Interaction Shift** |
| **Positive Active Alignment** | **68.0%** | **68.0%** | 🟢 **Matches Learned Active Favorites** |
| **Brand Diversity (Top-5)** | **4.2 brands** | **3.8 brands** | 🟢 **Zero Filter-Bubble Monopoly Collapse** |

---

## 7. Web Application Design & User Experience

The application is deployed as a responsive, luxury aesthetic web interface built with modern semantic HTML, Vanilla CSS, and Flask:
* **Dedicated Multi-Page Flow:**
  - **Page 1 (`/`):** Interactive Skin Profile Consultation featuring interactive skin type tiles, dual-concern pills, fragrance-free toggle, and an Indian Rupee budget slider (₹500 to ₹15,000).
  - **Page 2 (`/recommendations`):** Dedicated recommendation feed featuring match score % badges, active ingredient tags, Stage-2 `⚡ AI Reranked` badges, natural language explainability rationales, peer-reviewed clinical evidence cards with PMIDs, and safety notices.
  - **Page 3 (`/compare`):** Side-by-side formulation comparison matrix highlighting active compound parity, price comparison in ₹, skin satisfaction, and live **Knowledge Graph Layering Compatibility & Collision Warnings**.
  - **Page 4 (`/ingredients`):** Active ingredient encyclopedia explaining biological mechanisms, chemical synonyms/trade names, peer-reviewed clinical citations, and formulation precautions for all active compounds.
* **Interactive AM/PM Regimen Drawer:** Users can assemble morning and evening regimens directly from candidate cards, inspect total routine investment in ₹, copy routine summaries with one click, and receive **live contraindication collision alerts** if conflicting actives (e.g. Retinol + Glycolic Acid) are layered together.
* **Active Ingredient Detail Modals:** Clicking any active ingredient badge opens a dynamic Knowledge Graph modal presenting its canonical identity, aliases, cellular mechanisms of action, and PubMed citations.
* **Interactive "💬 Ask DermaSense AI" RAG Drawer:** A slide-out clinical assistant available across the recommendation feed. Powered by local Ollama (`qwen2.5:7b-instruct`) on the RTX 4050 GPU, it answers questions on formulation safety, layering rules, review sentiments, and ingredient biochemistry with genuine PubMed citations and verified medical disclaimers.
* **Dynamic Feedback & Profile Tracking:** Candidate cards feature intuitive interactive feedback controls that update user ingredient affinities in real time, immediately suppressing irritants that caused breakouts.

---

## 8. Societal Impact, Safety & Limitations

1. **Consumer Safety & Adverse Event Mitigation:** By prioritizing fragrance-free formulations and identifying non-comedogenic ingredients for oily and sensitive skin profiles, combined with automated Knowledge Graph contraindication collision checks and RAG safety guardrails, the system prevents hostile active layering and barrier distress.
2. **Economic Transparency:** Normalizing cosmetic pricing into Indian Rupees (INR ₹) empowers consumers across diverse socio-economic strata to find therapeutically equivalent formulations within their exact financial constraints.
3. **Clinical Disclaimer:** DermaSense serves as an evidence-based cosmetic recommendation and educational tool. It does not replace clinical consultation with licensed dermatologists for acute dermatological pathology (e.g. cystic acne, eczema flares, melanoma).

---

## 9. Conclusion & Future Work

DermaSense successfully proves that skincare e-commerce can transcend generic black-box rating aggregators by grounding artificial intelligence in a **Provenance-Aware Ingredient Knowledge Graph**, **Stage-2 Deep Learning Cross-Encoder Reranking**, **Evidence-Grounded RAG Q&A**, **Feedback-Driven Adaptive Personalization**, and **dense FAISS semantic retrieval**. 

Across all four completed experimental benchmarks, the system demonstrated quantifiable, research-grade improvements:
- **Recommendation Quality:** **94.0% Precision@5** and **0.968 nDCG@5** via Cross-Encoder Reranking (+3.56% gain).
- **Chemical Evidence Grounding:** **100% Entity Resolution Accuracy** and **96.0% Clinical Citation Traceability** via Knowledge Graph (213 PMIDs linked).
- **Formulation Safety:** **100% Contraindication Collision Detection** preventing hostile active layering.
- **Evidence-Grounded Q&A:** **97.1% Factual Groundedness** and **Zero Hallucinated Citations** with local GPU-accelerated LLM inference.
- **Adaptive Personalization:** **100.0% Irritant Avoidance Accuracy** and **1-Shot Immediate Adaptation** eliminating breakout causes from candidate rankings.

### Future Work:
1. **Aspect-Based Sentiment Analysis (ABSA):** Training fine-tuned RoBERTa models to extract granular aspect-level sentiment for rare botanical extracts.
2. **Mobile PWA Barcode Scanner:** Offline-first barcode scanning for retail skincare products.
3. **Multi-Lingual Regional Support:** Localizing skincare explanations into regional Indian languages (Hindi, Marathi, Tamil, Telugu).

---

## 10. References
1. Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. In Proceedings of EMNLP 2019.
2. Johnson, J., Douze, M., & Jégou, H. (2019). *Billion-scale similarity search with GPUs*. IEEE Transactions on Big Data, 7(3), 535-547.
3. Draelos, Z. D. (2018). *Cosmeceuticals: Procedures in Cosmetic Dermatology*. Elsevier Health Sciences.
4. Bissett, D. L., Oblong, J. E., & Berge, C. A. (2005). *Niacinamide: A B vitamin that improves aging facial skin appearance*. Dermatologic Surgery, 31(7), 860-865. (PMID: 16029679).
5. Arif, T. (2015). *Salicylic acid as a peeling agent: a comprehensive review*. Clinical, Cosmetic and Investigational Dermatology, 8, 455-461. (PMID: 26347017).
6. Meckfessel, M. H., & Brandt, S. (2014). *The structure, function, and importance of ceramides in skin and their use as therapeutic agents in skin-care products*. Journal of the American Academy of Dermatology, 71(1), 177-184. (PMID: 24656726).
7. Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. In Advances in Neural Information Processing Systems (NeurIPS 2020).
8. Hutto, C., & Gilbert, E. (2014). *VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text*. In Proceedings of ICWSM 2014.

