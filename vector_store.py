"""
KrishiNyay — vector_store.py
Clean interface for querying ChromaDB.
Used by rag_chain.py and validate_corpus.py.

Usage:
    from vector_store import VectorStore
    vs = VectorStore()
    results = vs.query("PM-KISAN ke liye kaun eligible hai", n=3)
"""

import json, pickle, logging, math, re
from collections import Counter
from pathlib import Path
from typing import Optional
import chromadb

log = logging.getLogger("krishinyay.vector_store")

ROOT       = Path(__file__).resolve().parent
CHROMA_DIR = ROOT / "chroma_db"
CHUNKS_DIR = ROOT / "data" / "chunks"
CHUNKS_FILE = CHUNKS_DIR / "all_chunks.jsonl"
COLLECTION = "krishinyay_v1"

LEXICAL_STOPWORDS = {
    "a", "an", "and", "are", "can", "do", "does", "for", "from", "how", "i",
    "in", "is", "it", "of", "on", "or", "the", "to", "what", "when", "which",
    "who", "with", "ka", "ke", "ki", "ko", "kya", "hai", "hain", "mein", "par",
    "se", "ho", "hota", "hoti", "liye", "batao", "karna", "karein", "karu",
}


class VectorStore:
    def __init__(self):
        self.client     = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = self.client.get_collection(COLLECTION)
        self.embedding_backend = "unknown"
        self.index_dim = self._collection_dim()
        self.embed_fn   = self._load_embedder()
        self.chunk_records = self._load_chunk_records()
        count = self.collection.count()
        log.info(f"VectorStore ready — {count} chunks indexed")
        if self.chunk_records:
            log.info(f"Hybrid lexical index ready — {len(self.chunk_records)} chunks loaded")

    def _collection_dim(self) -> Optional[int]:
        """Return stored embedding dimension for non-empty Chroma collections."""
        try:
            sample = self.collection.peek(1)
            embeddings = sample.get("embeddings")
            if embeddings is not None and len(embeddings):
                return len(embeddings[0])
        except Exception:
            pass
        return None

    def _index_meta(self) -> dict:
        meta_path = CHUNKS_DIR / "embed_meta.json"
        if not meta_path.exists():
            return {}
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _load_chunk_records(self) -> list[dict]:
        """Load flat chunk metadata for lightweight lexical reranking."""
        if not CHUNKS_FILE.exists():
            log.warning("No chunk JSONL found — using vector-only retrieval")
            return []

        records = []
        for line in CHUNKS_FILE.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = str(item.get("text", ""))
            searchable = " ".join(
                str(item.get(key, ""))
                for key in ["text", "display", "source", "category", "state", "language"]
            )
            records.append({
                "id": item.get("id", ""),
                "text": text,
                "source": item.get("source", ""),
                "display": item.get("display", ""),
                "url": item.get("url", ""),
                "category": item.get("category", ""),
                "state": item.get("state", ""),
                "priority": item.get("priority", ""),
                "tokens": self._tokenize(searchable),
            })
        return records

    def _load_embedder(self):
        """Load an embedder compatible with the stored Chroma index."""
        meta = self._index_meta()
        saved_backend = str(meta.get("embedding_backend", "")).upper()
        saved_dim = meta.get("embedding_dim")

        if saved_backend == "TF-IDF" or saved_dim == 1024 or self.index_dim == 1024:
            return self._load_tfidf_embedder()

        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
            model_dim = model.get_sentence_embedding_dimension()
            if self.index_dim and model_dim != self.index_dim:
                raise RuntimeError(
                    f"Stored index dim is {self.index_dim}, but MiniLM dim is {model_dim}. "
                    "Re-run chunk_and_embed.py --force to rebuild the index."
                )
            def embed(text: str) -> list[float]:
                return model.encode([text], normalize_embeddings=True)[0].tolist()
            self.embedding_backend = "MiniLM"
            log.info("Using multilingual SentenceTransformer for queries")
            return embed
        except Exception as exc:
            if self.index_dim and self.index_dim != 1024:
                raise RuntimeError(
                    "MiniLM index is present, but MiniLM could not be loaded for querying. "
                    "Install/cache sentence-transformers model or rebuild the index with TF-IDF."
                ) from exc
            return self._load_tfidf_embedder()

    def _load_tfidf_embedder(self):
        """Load the saved TF-IDF vectorizer used at indexing time."""
        vec_path = CHROMA_DIR / "tfidf_vectorizer.pkl"
        if not vec_path.exists():
            raise RuntimeError(
                "No compatible embedding backend available. "
                "Run chunk_and_embed.py first."
            )
        if self.index_dim and self.index_dim != 1024:
            raise RuntimeError(
                f"Stored index dim is {self.index_dim}, but TF-IDF query dim is 1024. "
                "Use MiniLM for this index or rebuild with TF-IDF."
            )
        import numpy as np
        with open(vec_path, "rb") as f:
            vec = pickle.load(f)

        def embed(text: str) -> list[float]:
            mat  = vec.transform([text]).toarray().astype(float)
            norm = float(np.linalg.norm(mat))
            if norm > 0:
                mat = mat / norm
            return mat[0].tolist()

        self.embedding_backend = "TF-IDF"
        log.info("Using TF-IDF query embeddings to match the stored index")
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
        Query the vector store. Returns result dicts with source metadata and
        hybrid retrieval scores. Vector search remains primary, with lexical
        scoring used as a lightweight reranker for short farmer questions.
        """
        vector_n = max(n * 5, 20)
        vector_results = self._vector_query(
            question,
            n=vector_n,
            category=category,
            state=state,
            priority=priority,
        )
        if not self.chunk_records:
            return vector_results[:n]

        lexical_results = self._lexical_query(
            question,
            n=vector_n,
            category=category,
            state=state,
            priority=priority,
        )
        return self._fuse_results(question, vector_results, lexical_results, n)

    def vector_query(
        self,
        question: str,
        n: int = 4,
        category: Optional[str] = None,
        state: Optional[str] = None,
        priority: Optional[str] = None,
    ) -> list[dict]:
        return self._vector_query(
            question,
            n=n,
            category=category,
            state=state,
            priority=priority,
        )

    def _vector_query(
        self,
        question: str,
        n: int,
        category: Optional[str],
        state: Optional[str],
        priority: Optional[str],
    ) -> list[dict]:
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
        ids       = raw.get("ids", [[]])[0]
        docs      = raw["documents"][0]
        metas     = raw["metadatas"][0]
        distances = raw["distances"][0]

        for doc_id, doc, meta, dist in zip(ids, docs, metas, distances):
            similarity = round(1 - (dist / 2), 4)
            results.append({
                "id":         doc_id,
                "text":       doc,
                "source":     meta.get("source", ""),
                "display":    meta.get("display", ""),
                "url":        meta.get("url", ""),
                "category":   meta.get("category", ""),
                "state":      meta.get("state", ""),
                "similarity": similarity,
                "vector_score": similarity,
                "lexical_score": 0.0,
                "hybrid_score": similarity,
                "retrieval_method": "vector",
            })

        return results

    def _lexical_query(
        self,
        question: str,
        n: int,
        category: Optional[str],
        state: Optional[str],
        priority: Optional[str],
    ) -> list[dict]:
        query_tokens = self._tokenize(question)
        if not query_tokens:
            return []

        query_counts = Counter(query_tokens)
        scored = []
        for record in self.chunk_records:
            if not self._record_matches_filter(record, category, state, priority):
                continue
            score = self._lexical_score(query_counts, record["tokens"])
            if score <= 0:
                continue
            scored.append((score, record))

        scored.sort(key=lambda item: item[0], reverse=True)
        results = []
        for score, record in scored[:n]:
            results.append({
                "id": record["id"],
                "text": record["text"],
                "source": record["source"],
                "display": record["display"],
                "url": record["url"],
                "category": record["category"],
                "state": record["state"],
                "similarity": 0.0,
                "vector_score": 0.0,
                "lexical_score": score,
                "hybrid_score": score,
                "retrieval_method": "lexical",
            })
        return results

    def _fuse_results(self, question: str, vector_results: list[dict], lexical_results: list[dict], n: int) -> list[dict]:
        merged = {}

        for rank, result in enumerate(vector_results, 1):
            key = self._result_key(result)
            item = dict(result)
            item["vector_rank_score"] = 1 / math.sqrt(rank)
            item["lexical_rank_score"] = 0.0
            merged[key] = item

        max_lexical = max((result.get("lexical_score", 0.0) for result in lexical_results), default=0.0)
        for rank, result in enumerate(lexical_results, 1):
            key = self._result_key(result)
            item = merged.get(key, dict(result))
            normalized_lexical = result["lexical_score"] / max_lexical if max_lexical else 0.0
            item["lexical_score"] = round(normalized_lexical, 4)
            item["lexical_rank_score"] = 1 / math.sqrt(rank)
            item["retrieval_method"] = "hybrid" if item.get("vector_score", 0.0) else "lexical"
            merged[key] = item

        for item in merged.values():
            vector_component = item.get("vector_rank_score", 0.0)
            lexical_component = item.get("lexical_rank_score", 0.0)
            score_component = item.get("vector_score", 0.0)
            lexical_score = item.get("lexical_score", 0.0)
            source_signal = self._source_signal(question, item)
            lexical_weight = 0.32 if source_signal > 0 and lexical_component else 0.22
            if source_signal < 0 and lexical_component:
                lexical_weight = 0.14
            item["hybrid_score"] = round(
                (0.52 * vector_component)
                + (lexical_weight * lexical_component)
                + (0.08 * score_component)
                + (0.06 * lexical_score)
                + source_signal,
                4,
            )
            item["source_signal"] = round(source_signal, 4)
            item["similarity"] = item.get("vector_score", item.get("similarity", 0.0))
            item.pop("vector_rank_score", None)
            item.pop("lexical_rank_score", None)

        return sorted(
            merged.values(),
            key=lambda item: (item["hybrid_score"], item.get("vector_score", 0.0)),
            reverse=True,
        )[:n]

    def _source_signal(self, question: str, result: dict) -> float:
        """Conservative source/category guardrails for known farmer eval intents."""
        query = str(question).lower()
        blob = self._metadata_blob(result)
        category = str(result.get("category", "")).lower()
        state = str(result.get("state", "")).lower()
        signal = 0.0

        query_state = self._query_state(query)
        if query_state:
            if state == query_state:
                signal += 0.55
            elif state and state not in {"central", "india"}:
                signal -= 0.18
            elif any(marker in blob for marker in ["pm-kisan", "pmkisan", "pradhan mantri kisan"]):
                signal -= 0.28

        pmkisan_intent = any(marker in query for marker in ["pm-kisan", "pm kisan", "pmkisan"])
        if pmkisan_intent and not query_state:
            if category == "income_support" or any(marker in blob for marker in ["pm-kisan", "pmkisan", "pm kisan", "kisan samman nidhi"]):
                signal += 0.42
            if category == "aggregator":
                signal -= 0.10
            if state and state not in {"central", "india"}:
                signal -= 0.18

        pmfby_intent = any(marker in query for marker in ["pmfby", "fasal bima", "crop insurance"])
        if pmfby_intent:
            if category in {"insurance", "crop_insurance"} or any(marker in blob for marker in ["pmfby", "fasal bima", "crop insurance"]):
                signal += 0.44
            if category == "aggregator":
                signal -= 0.18
            if state and state not in {"central", "india"}:
                signal -= 0.16

        if self._has_legal_intent(query):
            if category in {"legal_rights", "legal"} or any(marker in blob for marker in ["land acquisition", "larr", "forest rights", "fra"]):
                signal += 0.48
            elif any(marker in blob for marker in ["pm-kisan", "pmkisan", "pmfby", "fasal bima"]):
                signal -= 0.22

        if self._has_soil_intent(query):
            if category == "soil" or "soil health" in blob:
                signal += 0.55
            elif state and state not in {"central", "india"}:
                signal -= 0.12

        if self._has_irrigation_intent(query):
            if any(marker in blob for marker in ["pmksy", "micro irrigation", "drip", "per drop", "sinchayee"]):
                signal += 0.42
            elif state and state not in {"central", "india"}:
                signal -= 0.12

        if self._has_infrastructure_intent(query):
            if category == "infrastructure" or any(marker in blob for marker in ["agriculture infrastructure", "agri_infra", "post harvest", "farm pond"]):
                signal += 0.50
            elif state and state not in {"central", "india"}:
                signal -= 0.10

        if self._has_labour_intent(query):
            if category == "labour_rights" or any(marker in blob for marker in ["mnrega", "mgnrega", "labour rights"]):
                signal += 0.48
            elif state and state not in {"central", "india"}:
                signal -= 0.10

        if self._has_pest_advisory_intent(query) and not pmfby_intent:
            if category == "aggregator" and any(marker in blob for marker in ["pest", "pesticide", "insecticide", "pmksk", "advisory"]):
                signal += 0.36
            if category in {"insurance", "crop_insurance"}:
                signal -= 0.24

        return max(min(signal, 0.65), -0.35)

    @staticmethod
    def _metadata_blob(result: dict) -> str:
        return " ".join(
            str(result.get(key, ""))
            for key in ["display", "source", "url", "category", "state", "text"]
        ).lower()

    @staticmethod
    def _query_state(query: str) -> Optional[str]:
        state_patterns = [
            ("andhra_pradesh", r"\bandhra(?:\s+pradesh)?\b|\bap\b"),
            ("bihar", r"\bbihar\b"),
            ("gujarat", r"\bgujarat\b"),
            ("karnataka", r"\bkarnataka\b"),
            ("madhya_pradesh", r"\bmadhya\s+pradesh\b|\bmp\b"),
            ("maharashtra", r"\bmaharashtra\b|\bnamo\s+shetkari\b|\bshetkari\b"),
            ("punjab", r"\bpunjab\b"),
            ("rajasthan", r"\brajasthan\b|\brajkisan\b"),
            ("tamil_nadu", r"\btamil\s+nadu\b"),
            ("telangana", r"\btelangana\b|\brythu\s+bandhu\b"),
            ("uttar_pradesh", r"\buttar\s+pradesh\b|\bup\b"),
            ("west_bengal", r"\bwest\s+bengal\b|\bkrishak\s+bandhu\b"),
        ]
        for state, pattern in state_patterns:
            if re.search(pattern, query, flags=re.I):
                return state
        return None

    @staticmethod
    def _has_legal_intent(query: str) -> bool:
        return bool(re.search(r"\b(land acquisition|zameen|jameen|bhoomi|forest rights|fra|gram sabha|compensation|muavza|rehabilitation|consent|title|adhikar|khareed)", query, flags=re.I))

    @staticmethod
    def _has_soil_intent(query: str) -> bool:
        return bool(re.search(r"\b(soil health|soil test|mitti|nutrient|fertilizer)", query, flags=re.I))

    @staticmethod
    def _has_irrigation_intent(query: str) -> bool:
        return bool(re.search(r"\b(pmksy|drip|micro[\s-]?irrigation|per drop|pani bachane|water-use|water saving|sinchayee)", query, flags=re.I))

    @staticmethod
    def _has_infrastructure_intent(query: str) -> bool:
        return bool(re.search(r"\b(agriculture infrastructure|infrastructure fund|farm pond|storage infrastructure|post harvest)", query, flags=re.I))

    @staticmethod
    def _has_labour_intent(query: str) -> bool:
        return bool(re.search(r"\b(mnrega|mgnrega|farm labour|labour rights|employment rights)", query, flags=re.I))

    @staticmethod
    def _has_pest_advisory_intent(query: str) -> bool:
        return bool(re.search(r"\b(pest|pesticide|insecticide|field photo|diagnosis|crop advisory)", query, flags=re.I))

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

    def _record_matches_filter(self, record: dict, category, state, priority) -> bool:
        if category and record.get("category") != category:
            return False
        if state and record.get("state") != state:
            return False
        if priority and record.get("priority") != priority:
            return False
        return True

    @staticmethod
    def _result_key(result: dict) -> str:
        return result.get("id") or f"{result.get('source', '')}:{result.get('text', '')[:80]}"

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        tokens = re.findall(r"[\w]+", str(text).lower(), flags=re.UNICODE)
        return [token for token in tokens if len(token) > 1 and token not in LEXICAL_STOPWORDS]

    @staticmethod
    def _lexical_score(query_counts: Counter, doc_tokens: list[str]) -> float:
        doc_counts = Counter(doc_tokens)
        if not doc_counts:
            return 0.0
        score = 0.0
        doc_len = sum(doc_counts.values())
        length_norm = 1.0 + (doc_len / 700.0)
        for term, query_tf in query_counts.items():
            term_tf = doc_counts.get(term, 0)
            if not term_tf:
                continue
            score += query_tf * (1.0 + math.log1p(term_tf))
        return score / length_norm

    def stats(self) -> dict:
        count = self.collection.count()
        return {
            "total_chunks": count,
            "collection": COLLECTION,
            "embedding_backend": self.embedding_backend,
            "embedding_dim": self.index_dim,
            "retrieval_mode": "hybrid_vector_lexical" if self.chunk_records else "vector_only",
            "lexical_chunks": len(self.chunk_records),
        }
