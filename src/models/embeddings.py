import os
import sys
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

DERMASENSE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROCESSED_DIR = os.path.join(DERMASENSE_DIR, "data", "processed")
EMBEDDINGS_DIR = os.path.join(DERMASENSE_DIR, "data", "embeddings")

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_EMBEDDINGS_PATH = os.path.join(EMBEDDINGS_DIR, "product_embeddings.npy")
DEFAULT_INDEX_PATH = os.path.join(EMBEDDINGS_DIR, "product_index.faiss")

class ProductEmbedder:
    """
    Handles dense semantic text embeddings using Sentence Transformers (all-MiniLM-L6-v2)
    and fast vector similarity search using FAISS (IndexFlatIP).
    """

    def __init__(self, model_name=DEFAULT_MODEL_NAME, auto_load=True):
        self.model_name = model_name
        self.model = None
        self.embeddings = None
        self.faiss_index = None
        self.dimension = 384  # all-MiniLM-L6-v2 embedding dimension
        
        if auto_load:
            self._load_model()
            self.load_index_if_available()

    def _load_model(self):
        """Loads the SentenceTransformer model into memory."""
        if self.model is None:
            print(f"Loading SentenceTransformer model '{self.model_name}'...")
            self.model = SentenceTransformer(self.model_name)
            print("SentenceTransformer loaded successfully.")

    def load_index_if_available(self, embeddings_path=DEFAULT_EMBEDDINGS_PATH, index_path=DEFAULT_INDEX_PATH):
        """Loads precomputed embeddings and FAISS index if they exist on disk."""
        if os.path.exists(embeddings_path):
            print(f"Loading cached embeddings from {embeddings_path}...")
            self.embeddings = np.load(embeddings_path).astype(np.float32)
            print(f"Loaded embeddings matrix: {self.embeddings.shape}")

        if os.path.exists(index_path):
            print(f"Loading FAISS index from {index_path}...")
            self.faiss_index = faiss.read_index(index_path)
            print(f"Loaded FAISS index containing {self.faiss_index.ntotal} vectors.")
        elif self.embeddings is not None:
            print("Building FAISS index from loaded embeddings matrix...")
            self.build_faiss_index(self.embeddings)

    def encode(self, texts, batch_size=64, show_progress_bar=True, normalize=True):
        """
        Encodes a list or Series of texts into dense vectors.
        Normalizes vectors by default so that Inner Product equals Cosine Similarity.
        """
        self._load_model()
        if isinstance(texts, str):
            texts = [texts]
        elif isinstance(texts, pd.Series):
            texts = texts.fillna("").tolist()
            
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=normalize
        ).astype(np.float32)
        
        return embeddings

    def build_faiss_index(self, embeddings):
        """
        Builds a FAISS IndexFlatIP (Inner Product) index for cosine similarity search.
        Assumes embeddings are already L2-normalized.
        """
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        dim = embeddings.shape[1]
        self.dimension = dim
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        self.faiss_index = index
        self.embeddings = embeddings
        return self.faiss_index

    def save_artifacts(self, embeddings_path=DEFAULT_EMBEDDINGS_PATH, index_path=DEFAULT_INDEX_PATH):
        """Saves embeddings and FAISS index to disk."""
        os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
        if self.embeddings is not None:
            np.save(embeddings_path, self.embeddings)
            print(f"Saved embeddings to {embeddings_path} ({self.embeddings.shape})")
            
        if self.faiss_index is not None:
            faiss.write_index(self.faiss_index, index_path)
            print(f"Saved FAISS index to {index_path}")

    def compute_product_embeddings(self, data_path=None, save=True):
        """
        Computes dense embeddings for all products in master_product_features.parquet.
        """
        if data_path is None:
            data_path = os.path.join(PROCESSED_DIR, "master_product_features.parquet")
            
        print(f"Reading product features from {data_path}...")
        df = pd.read_parquet(data_path)
        
        print(f"Generating dense embeddings for {len(df):,} products using {self.model_name}...")
        texts = df['semantic_document'].fillna("").tolist()
        
        embeddings = self.encode(texts, batch_size=128, show_progress_bar=True, normalize=True)
        self.embeddings = embeddings
        
        print("Building FAISS index...")
        self.build_faiss_index(embeddings)
        
        if save:
            self.save_artifacts()
            
        return embeddings

    def search_query(self, query, top_k=50):
        """
        Searches top-k products globally across the entire catalog using FAISS.
        Returns (indices, similarity_scores).
        """
        if self.faiss_index is None:
            raise ValueError("FAISS index is not initialized. Run compute_product_embeddings() or load_index_if_available().")
            
        query_vec = self.encode([query], show_progress_bar=False, normalize=True)
        scores, indices = self.faiss_index.search(query_vec, top_k)
        return indices[0], scores[0]

    def compute_similarity_for_candidates(self, query, candidate_indices):
        """
        Computes cosine similarity between a single query and a specific subset of candidate products.
        Uses fast NumPy dot product on normalized vectors.
        """
        if self.embeddings is None:
            raise ValueError("Embeddings matrix is not loaded.")
            
        query_vec = self.encode([query], show_progress_bar=False, normalize=True).flatten()
        # Matrix multiplication of (1, 384) x (N_candidates, 384).T -> (N_candidates,)
        sub_embeddings = self.embeddings[candidate_indices]
        sim_scores = np.dot(sub_embeddings, query_vec)
        # Clip to [0, 1] range for stability
        sim_scores = np.clip(sim_scores, 0.0, 1.0)
        return sim_scores


if __name__ == "__main__":
    embedder = ProductEmbedder(auto_load=False)
    embeddings = embedder.compute_product_embeddings(save=True)
    
    # Test query
    test_query = "gentle foaming cleanser for oily acne prone skin with salicylic acid"
    print(f"\n--- Testing Semantic Search Query: '{test_query}' ---")
    indices, scores = embedder.search_query(test_query, top_k=5)
    
    df = pd.read_parquet(os.path.join(PROCESSED_DIR, "master_product_features.parquet"))
    for rank, (idx, score) in enumerate(zip(indices, scores), start=1):
        prod = df.iloc[idx]
        print(f"\n{rank}. [{score:.4f}] {prod['product_name']} ({prod['brand']})")
        print(f"   Category: {prod['category']} | Benefits: {prod['matched_benefits']}")
