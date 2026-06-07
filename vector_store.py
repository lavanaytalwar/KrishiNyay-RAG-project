"""
KrishiNyay — vector_store.py
Clean interface for querying ChromaDB.
Used by rag_chain.py and validate_corpus.py.

Usage:
    from vector_store import VectorStore
    vs = VectorStore()
    results = vs.query("PM-KISAN ke liye kaun eligible hai", n=3)
"""

import json, pickle, logging
from pathlib import Path
from typing import Optional
import chromadb

log = logging.getLogger("krishinyay.vector_store")

ROOT       = Path(__file__).resolve().parent
CHROMA_DIR = ROOT / "chroma_db"
CHUNKS_DIR = ROOT / "data" / "chunks"
COLLECTION = "krishinyay_v1"


class VectorStore:
    def __init__(self):
        self.client     = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = self.client.get_collection(COLLECTION)
        self.embed_fn   = self._load_embedder()
        count = self.collection.count()
        log.info(f"VectorStore ready — {count} chunks indexed")

    def _load_embedder(self):
        """Load multilingual model or TF-IDF fallback."""
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                local_files_only=True,
            )
            def embed(text: str) -> list[float]:
                return model.encode([text], normalize_embeddings=True)[0].tolist()
            log.info("Using multilingual SentenceTransformer for queries")
            return embed
        except Exception:
            # TF-IDF fallback — load saved vectorizer
            vec_path = CHROMA_DIR / "tfidf_vectorizer.pkl"
            if not vec_path.exists():
                raise RuntimeError(
                    "No embedding model available. "
                    "Run chunk_and_embed.py first."
                )
            import pickle, numpy as np
            with open(vec_path, "rb") as f:
                vec = pickle.load(f)

            def embed(text: str) -> list[float]:
                mat  = vec.transform([text]).toarray().astype(float)
                norm = float(np.linalg.norm(mat))
                if norm > 0:
                    mat = mat / norm
                return mat[0].tolist()

            log.info("Using TF-IDF fallback for queries")
            return embed

    def query(
        self,
        question: str,
        n: int = 4,
        category: Optional[str] = None,
        state: Optional[str] = None,
        priority: Optional[str] = None,
    ) -> list[dict]:
        """
        Query the vector store. Returns list of result dicts with:
          text, source, display, url, category, state, score
        """
        embedding = self.embed_fn(question)
        where     = self._build_filter(category, state, priority)

        kwargs = dict(
            query_embeddings=[embedding],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        if where:
            kwargs["where"] = where

        try:
            raw = self.collection.query(**kwargs)
        except Exception as e:
            log.warning(f"Query failed: {e} — retrying without filters")
            raw = self.collection.query(
                query_embeddings=[embedding],
                n_results=n,
                include=["documents", "metadatas", "distances"],
            )

        results = []
        docs      = raw["documents"][0]
        metas     = raw["metadatas"][0]
        distances = raw["distances"][0]

        for doc, meta, dist in zip(docs, metas, distances):
            # Chroma cosine distance → similarity: 0=identical, 2=opposite
            similarity = round(1 - (dist / 2), 4)
            results.append({
                "text":       doc,
                "source":     meta.get("source", ""),
                "display":    meta.get("display", ""),
                "url":        meta.get("url", ""),
                "category":   meta.get("category", ""),
                "state":      meta.get("state", ""),
                "similarity": similarity,
            })

        return results

    def _build_filter(self, category, state, priority) -> Optional[dict]:
        filters = []
        if category:
            filters.append({"category": {"$eq": category}})
        if state:
            filters.append({"state": {"$eq": state}})
        if priority:
            filters.append({"priority": {"$eq": priority}})
        if not filters:
            return None
        if len(filters) == 1:
            return filters[0]
        return {"$and": filters}

    def stats(self) -> dict:
        count = self.collection.count()
        return {"total_chunks": count, "collection": COLLECTION}
