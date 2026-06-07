"""
KrishiNyay — chunk_and_embed.py
Reads clean docs → splits into chunks → embeds → stores in ChromaDB.

EMBEDDING:
  Your machine : paraphrase-multilingual-MiniLM-L12-v2 (Hindi+English, 384-dim)
  Sandbox      : TF-IDF sklearn fallback (no GPU/torch needed, proves pipeline)

Run:
  python chunk_and_embed.py
  python chunk_and_embed.py --source pmkisan
  python chunk_and_embed.py --force
"""

import json, re, argparse, logging, os, sys, pickle
from pathlib import Path
from datetime import datetime
from typing import Optional

ROOT       = Path(__file__).resolve().parent
CLEAN_DIR  = ROOT / "data" / "processed" / "clean"
CHUNKS_DIR = ROOT / "data" / "chunks"
CHROMA_DIR = ROOT / "chroma_db"
LOGS_DIR   = ROOT / "logs"
for d in [CHUNKS_DIR, CHROMA_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / f"embed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("krishinyay.embedder")

CHUNK_SIZE    = 500
CHUNK_OVERLAP = 60
COLLECTION    = "krishinyay_v1"
SEPARATORS    = ["\n\n", "\n", "।", ".", "?", "!", " ", ""]


# ── Text splitter ──────────────────────────────────────────────────────────
def split_text(text: str) -> list:
    if len(text) <= CHUNK_SIZE:
        return [text.strip()] if text.strip() else []
    for sep in SEPARATORS:
        if sep and sep in text:
            parts, chunks, current = text.split(sep), [], ""
            for part in parts:
                candidate = current + (sep if current else "") + part
                if len(candidate) <= CHUNK_SIZE:
                    current = candidate
                else:
                    if current.strip():
                        chunks.append(current.strip())
                    overlap_start = max(0, len(current) - CHUNK_OVERLAP)
                    current = current[overlap_start:] + (sep if current else "") + part
            if current.strip():
                chunks.append(current.strip())
            return [c for c in chunks if len(c) > 30]
    return [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP)]


# ── Embedders — deferred imports inside functions ──────────────────────────
def try_sentence_transformer():
    """Try loading multilingual SentenceTransformer. Returns (fn, dim) or (None, None)."""
    try:
        # Import inside function to avoid module-level torch crash
        from sentence_transformers import SentenceTransformer  # noqa
        model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            local_files_only=True,
        )
        dim = model.get_sentence_embedding_dimension()
        log.info(f"✓ Multilingual SentenceTransformer loaded (dim={dim})")
        def embed(texts):
            return model.encode(texts, normalize_embeddings=True).tolist()
        return embed, dim
    except Exception as e:
        log.warning(f"SentenceTransformer unavailable: {type(e).__name__}: {str(e)[:80]}")
        return None, None


def build_tfidf(all_texts: list):
    """TF-IDF fallback — works in any environment, no GPU needed."""
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(max_features=1024, ngram_range=(1, 2), sublinear_tf=True)
    vec.fit(all_texts)
    dim = 1024
    log.info(f"✓ TF-IDF fallback fitted (dim={dim}, docs={len(all_texts)})")
    log.info("  On your machine: pip install sentence-transformers for Hindi embeddings")

    def embed(texts):
        import numpy as np
        mat   = vec.transform(texts).toarray().astype(float)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return (mat / norms).tolist()

    # Persist vectorizer for vector_store.py to reuse
    vec_path = CHROMA_DIR / "tfidf_vectorizer.pkl"
    with open(vec_path, "wb") as f:
        pickle.dump(vec, f)
    log.info(f"  TF-IDF vectorizer saved → {vec_path}")
    return embed, dim


# ── Doc helpers ────────────────────────────────────────────────────────────
def doc_to_text(doc: dict) -> str:
    if "pages" in doc and doc["pages"]:
        return "\n\n".join(p["text"] for p in doc["pages"] if p.get("text","").strip())
    return doc.get("text", "").strip()


def safe_str(v) -> str:
    if isinstance(v, list):
        return ",".join(str(x) for x in v)
    return str(v) if v is not None else ""


# ── ChromaDB ───────────────────────────────────────────────────────────────
def get_collection(force=False):
    import chromadb
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if force:
        try:
            client.delete_collection(COLLECTION)
            log.info(f"Deleted existing collection '{COLLECTION}'")
        except Exception:
            pass
    return client.get_or_create_collection(
        name=COLLECTION, metadata={"hnsw:space": "cosine"}
    )


def already_indexed(col, doc_id: str) -> bool:
    try:
        return len(col.get(where={"doc_id": doc_id}, limit=1)["ids"]) > 0
    except Exception:
        return False


# ── Main ───────────────────────────────────────────────────────────────────
def run(source_filter=None, force=False):
    log.info("=" * 60)
    log.info("  CHUNK & EMBED  —  KrishiNyay Phase 2")
    log.info("=" * 60)

    # 1. Discover clean docs
    search_dirs = [
        CLEAN_DIR,
        ROOT / "data" / "raw" / "schemes",        # fallback to raw if clean not yet run
        ROOT / "data" / "raw" / "state_schemes",
    ]
    all_files = []
    for d in search_dirs:
        if d.exists():
            all_files.extend(d.glob("*.json"))

    if source_filter:
        all_files = [f for f in all_files if source_filter in f.stem]

    # Deduplicate by stem
    seen, unique = set(), []
    for f in all_files:
        if f.stem not in seen:
            seen.add(f.stem)
            unique.append(f)

    if not unique:
        log.error(f"No docs found. Run Phase 1 first: python scripts/run_phase1.py")
        return

    log.info(f"Found {len(unique)} doc(s)")

    # 2. Load + chunk all docs
    all_records = []
    for f in unique:
        try:
            doc    = json.loads(f.read_text(encoding="utf-8"))
            text   = doc_to_text(doc)
            source = doc.get("name", f.stem)
            doc_id = doc.get("id") or f.stem
            chunks = split_text(text)
            for i, chunk in enumerate(chunks):
                all_records.append({
                    "id":   f"{doc_id}_c{i}",
                    "text": chunk,
                    "meta": {
                        "doc_id":     safe_str(doc_id),
                        "source":    source,
                        "display":   safe_str(doc.get("display","")),
                        "url":       safe_str(doc.get("url","")),
                        "category":  safe_str(doc.get("category","scheme")),
                        "state":     safe_str(doc.get("state","central")),
                        "priority":  safe_str(doc.get("priority","medium")),
                        "language":  safe_str(doc.get("language","english")),
                        "chunk_idx": str(i),
                    },
                })
            log.info(f"  Chunked: {source:40s} → {len(chunks)} chunks")
        except Exception as e:
            log.warning(f"  Skipping {f.name}: {e}")

    log.info(f"  Total chunks: {len(all_records)}")

    # 3. Load embedder
    all_texts = [r["text"] for r in all_records]
    embed_fn, dim = try_sentence_transformer()
    if embed_fn is None:
        embed_fn, dim = build_tfidf(all_texts)

    # 4. Get collection
    col = get_collection(force=force)

    # 5. Embed + upsert
    from itertools import groupby
    sorted_records = sorted(all_records, key=lambda x: x["meta"]["doc_id"])
    upserted = skipped = 0
    BATCH = 64

    for doc_id, grp in groupby(sorted_records, key=lambda x: x["meta"]["doc_id"]):
        records = list(grp)
        source = records[0]["meta"]["source"]
        if not force and already_indexed(col, doc_id):
            log.info(f"  ↩  Skip (indexed): {doc_id:40s} ({len(records)} chunks)")
            skipped += len(records)
            continue
        for i in range(0, len(records), BATCH):
            batch = records[i:i+BATCH]
            texts = [r["text"] for r in batch]
            embs  = embed_fn(texts)
            col.upsert(
                ids        = [r["id"]   for r in batch],
                embeddings = embs,
                documents  = texts,
                metadatas  = [r["meta"] for r in batch],
            )
            upserted += len(batch)
        log.info(f"  ✓ Embedded: {doc_id:40s} ({len(records)} chunks, source={source})")

    # 6. Save JSONL for Phase 3 fine-tuning
    jsonl_path = CHUNKS_DIR / "all_chunks.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps({"id":r["id"],"text":r["text"],**r["meta"]},
                               ensure_ascii=False) + "\n")

    (CHUNKS_DIR / "embed_meta.json").write_text(json.dumps({
        "collection":COLLECTION, "chunk_size":CHUNK_SIZE,
        "chunk_overlap":CHUNK_OVERLAP, "total_chunks":len(all_records),
        "upserted":upserted, "skipped":skipped,
        "embedded_at":datetime.now().isoformat(),
    }, indent=2))

    log.info(f"\n  Upserted : {upserted}")
    log.info(f"  Skipped  : {skipped}")
    log.info(f"  JSONL    → {jsonl_path}")
    log.info(f"  ChromaDB → {CHROMA_DIR}")
    log.info(f"\n  Done. Run: python validate_corpus.py")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", default=None)
    p.add_argument("--force",  action="store_true")
    args = p.parse_args()
    run(source_filter=args.source, force=args.force)
