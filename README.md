# DermaSense 🌸
### Personalized, Explainable AI Skincare Product Recommendation and Comparison System

DermaSense is an AI-powered skincare recommendation system that combines ingredient ontology matching, skin-type community review evidence, dense semantic text embeddings, and evidence-based explainability to provide personalized skincare recommendations.

---

## 🌟 Key Features

1. **Multi-Platform Dataset Integration**: Harmonized schemas across Sephora, Amazon, and Flipkart with strict platform traceability (`source` + `source_product_id`).
2. **Skincare Ingredient Ontology**: Domain-specific rule engine covering 70+ active ingredients across 8 benefit categories (Hydration, Acne, Brightening, Anti-Aging, Redness, Sun Protection, Pore Control, Exfoliation).
3. **Skin-Type Review Evidence**: Review analytics partitioned across Oily, Dry, Combination, and Normal skin profiles across 1.2M user reviews.
4. **Dense Semantic Embeddings**: Powered by `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors) and fast nearest-neighbor search with **FAISS**.
5. **Transparent Explainability Engine**: Dynamic evidence-based narratives explaining *why* a product is recommended based on chemical actives, skin-type satisfaction, and budget.

---

## 📁 Repository Structure

```
DermaSense/
├── app/                            # Web application (Flask / UI)
├── data/
│   ├── raw/                        # Raw platform datasets (Sephora, Amazon, Flipkart)
│   ├── processed/                  # Unified & feature-engineered parquet datasets
│   └── embeddings/                 # 384-dim dense vectors & FAISS index
├── docs/                           # Project documentation & design reports
├── notebooks/                      # Exploratory Data Analysis & visual notebooks
├── results/                        # Evaluation scenarios & reports
├── src/
│   ├── data/                       # Data cleaning and harmonization pipeline
│   │   ├── clean_sephora.py
│   │   ├── clean_amazon.py
│   │   ├── clean_flipkart.py
│   │   └── build_unified_data.py
│   ├── features/                   # Ingredient ontology & review aggregation
│   │   ├── ingredient_features.py
│   │   ├── review_features.py
│   │   └── build_features.py
│   ├── models/                     # Recommender algorithms & vector search
│   │   ├── embeddings.py           # SentenceTransformer + FAISS indexer
│   │   └── recommender.py          # Multi-factor composite recommendation engine
│   └── explainability/             # Evidence-based explanation generator
│       └── explainer.py
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Web Application (Interactive UI)
```bash
python app/app.py
```
Open your browser at **`http://127.0.0.1:5000`** to access:
- **🌸 Skin Profile Consultation:** Questionnaire with skin type, concern tags, and INR (₹) budget slider.
- **✨ Recommendations:** AI-ranked products with match score %, clickable ingredient chemistry modals, and interactive AM/PM routine drawer.
- **⚖️ Compare:** Side-by-side product formulation and review comparison.
- **🌿 Ingredient Dictionary:** Searchable database of 70+ proven skincare actives.

---

## 🧪 CLI Testing & Evaluation

- **Test Recommender Engine:**
  ```bash
  python src/models/recommender.py
  ```
- **Test Explainability Generator:**
  ```bash
  python src/explainability/explainer.py
  ```
- **Launch Jupyter Notebook for EDA:**
  ```bash
  jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
  ```

