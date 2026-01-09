from __future__ import annotations

import os
import pickle
from typing import List, Any, Dict

import faiss
import numpy as np

from src.embedding import EmbeddingPipeline


class FaissVectorStore:
    """Simple FAISS-backed vector store for LangChain Documents.

    Features:
    - Build an index from a list of LangChain `Document` objects
    - Persist/load FAISS index and metadata
    - Query by text (returns metadata and similarity scores)
    """

    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2"):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.index_path = os.path.join(self.persist_dir, "faiss.index")
        self.meta_path = os.path.join(self.persist_dir, "metadata.pkl")

        # Embedding pipeline (SentenceTransformer wrapper)
        self.emb_pipeline = EmbeddingPipeline(model_name=embedding_model)

        # FAISS structures
        self.index: faiss.Index | None = None
        self.metadatas: List[Dict[str, Any]] = []

    # ---- Utilities ----
    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    # ---- Build / Persist ----
    def build_from_documents(self, documents: List[Any]) -> None:
        """Chunk, embed and build a FAISS index from LangChain Documents."""
        print(f"[INFO] Building vectorstore from {len(documents)} documents...")
        chunks = self.emb_pipeline.chunk_documents(documents)
        texts = [c.page_content for c in chunks]

        if not texts:
            raise ValueError("No text chunks to index")

        embeddings = self.emb_pipeline.model.encode(texts, show_progress_bar=True)
        embeddings = np.asarray(embeddings, dtype="float32")
        embeddings = self._normalize(embeddings)

        dim = embeddings.shape[1]
        # Use inner-product on normalized vectors = cosine similarity
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        # Build metadata list aligned with index
        self.metadatas = []
        for c in chunks:
            md = {"text": c.page_content}
            # include existing document metadata if present
            if hasattr(c, "metadata") and isinstance(c.metadata, dict):
                md.update(c.metadata)
            self.metadatas.append(md)

        self.index = index
        self.save()
        print(f"[INFO] Built index with {index.ntotal} vectors.")

    def save(self) -> None:
        """Persist FAISS index and metadata to disk."""
        if self.index is None:
            raise RuntimeError("No index to save")
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadatas, f)
        print(f"[INFO] Saved FAISS index to {self.index_path} and metadata to {self.meta_path}")

    def load(self) -> None:
        """Load FAISS index and metadata from disk."""
        if not os.path.exists(self.index_path) or not os.path.exists(self.meta_path):
            raise FileNotFoundError("Index or metadata files not found in persist directory")
        self.index = faiss.read_index(self.index_path)
        with open(self.meta_path, "rb") as f:
            self.metadatas = pickle.load(f)
        print(f"[INFO] Loaded FAISS index ({self.index.ntotal} vectors) and {len(self.metadatas)} metadata entries")

    # ---- Query ----
    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Return top_k nearest chunks for `query_text` as list of dicts with `score` and `metadata`."""
        if self.index is None:
            # try loading if persisted
            try:
                self.load()
            except Exception as e:
                raise RuntimeError("Index not built or loaded") from e

        q_emb = self.emb_pipeline.model.encode([query_text])
        q_emb = np.asarray(q_emb, dtype="float32")
        q_emb = self._normalize(q_emb)

        # FAISS expects (nq, d)
        D, I = self.index.search(q_emb, top_k)
        scores = D[0].tolist()
        indices = I[0].tolist()

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores, indices):
            if idx < 0 or idx >= len(self.metadatas):
                continue
            results.append({"score": float(score), "metadata": self.metadatas[idx]})
        return results


if __name__ == "__main__":
    # quick smoke test
    from src.data_loader import load_all_documents

    docs = load_all_documents("data")
    store = FaissVectorStore(persist_dir="faiss_store_test")
    store.build_from_documents(docs)
    res = store.query("machine learning basics", top_k=3)
    print("Top results:")
    for r in res:
        print(r["score"], r["metadata"].get("text", ""))
