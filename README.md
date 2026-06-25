# KrishiNyay AI — Source-Grounded RAG Assistant for Indian Farmers

KrishiNyay AI is an India-focused retrieval-augmented generation (RAG) project for answering farmer questions about government schemes, crop insurance, farm credit, legal rights, and agriculture support using trusted documents.

The current goal is a small, runnable, demo-ready RAG system before adding advanced agents, voice, OCR, WhatsApp, or vision features.

## Project Status

Current baseline:
- Embedding backend: MiniLM (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`)
- Embedding dimension: 384
- Local indexed corpus: 1,748 chunks
- Retrieval validation: 15/15 passing on the current RAG + dynamic routing smoke set

Completed phases:
- Phase 0 — repo hygiene and truthful MiniLM baseline.
- Phase 1 — manual PDF ingestion workflow and curated official PDF ingestion.
- Phase 2 — retrieval normalization, dynamic live-data routing, health metadata, and validation upgrade.

What remains:
- Phase 3 — frontend/FastAPI demo polish, animations, better UX states, and deployment-ready configuration.
- Phase 4 — stronger farmer-facing evaluation set from public sources such as government FAQs, farmer forums, YouTube comments, NGOs, and help sites.
- Phase 5 — retrieval quality upgrades such as hybrid search, reranking, category/state filters, and richer source ranking.
- Later phases — OCR for scanned PDFs, live mandi/weather API integrations, LangGraph workflows, voice/WhatsApp channels, and optional fine-tuning only after enough validated data exists.

Implemented:
- Data ingestion scripts for government scheme pages and PDFs
- Text cleaning and Hindi-aware normalization
- Character chunking with Hindi sentence separator support
- ChromaDB indexing and retrieval
- MiniLM-first embeddings with TF-IDF fallback
- FastAPI backend with `/health`, `/query`, and `/ingest`
- Static web UI for chat, source inspection, ingestion, and system status
- Offline validation and RAGAS-style evaluation scripts
- Tracked `sample_data/` so a fresh clone has a small demo corpus
- Manifest-based manual PDF ingestion for curated official PDFs

In progress:
- Additional curated official PDFs for stronger demo coverage
- Retrieval quality tuning by category and state
- README screenshots and demo GIF
- More focused tests

Planned:
- Hybrid search with BM25 plus vector search
- Cross-encoder reranking
- OCR for scanned PDFs
- Voice input with Indic ASR/TTS
- LangGraph agent workflows
- Live mandi/weather APIs
- WhatsApp or IVR interface

## Architecture

```text
Documents / PDFs / sample JSON
        ↓
Text extraction and cleaning
        ↓
Hindi-aware chunking
        ↓
MiniLM embeddings, or TF-IDF fallback
        ↓
ChromaDB vector store
        ↓
Retriever returns top source chunks
        ↓
LLM or offline template answer
        ↓
Answer with sources
```

MiniLM is the preferred retrieval backend because it handles semantic similarity across English, Hindi, and Hinglish better than keyword search. TF-IDF remains as a fallback for offline or constrained environments.

Important: the index and query backend must match. If ChromaDB was built with TF-IDF, queries must use the saved TF-IDF vectorizer. If it was built with MiniLM, queries must use MiniLM. The code records backend metadata and exposes the active backend in `/health`.

Phase 2 adds lightweight query normalization before retrieval. Common farmer phrasing such as `fasal bima`, `zameen`, `bhav`, `barbaad`, `KCC`, and `PMFBY` is expanded with retrieval hints while preserving the original question for the final answer prompt.

Live-changing questions are routed away from static RAG:

```text
PM-KISAN beneficiary/payment status → official PM-KISAN portal guidance
mandi price / bhav questions       → eNAM live price guidance
weather / spraying questions       → IMD/local advisory guidance
```

## Repository Structure

```text
.
├── app.py                  # FastAPI app and static UI routes
├── rag_chain.py            # Retrieval + prompt + LLM/template answer
├── vector_store.py         # ChromaDB query interface
├── chunk_and_embed.py      # Chunking, embedding, ChromaDB indexing
├── clean_text.py           # Text cleaning and Indic normalization
├── ingest_manual_pdfs.py   # Manifest-driven manual PDF ingestion
├── scrape_schemes.py       # Government scheme scraping
├── download_pdfs.py        # PDF download and text extraction
├── validate_corpus.py      # Retrieval smoke validation
├── evaluate_rag.py         # RAGAS-style evaluation
├── sample_data/            # Small tracked demo corpus
├── corpus/pdfs/            # Tracked templates and future curated PDF corpus
├── web/                    # Static frontend
├── eval/                   # Evaluation output
├── requirements.txt        # Direct runtime dependencies
└── requirements-full.txt   # Full local environment freeze
```

Ignored local artifacts:

```text
data/
chroma_db/
logs/
krishinyay-env/
```

## PDF Source Policy

Curated official PDFs may be tracked later when they are small, stable, and important for reproducible demos. Bulk download folders, staging PDFs, extracted JSON, logs, and generated vector indexes should stay ignored.

Phase 1 uses this split:

```text
corpus/pdfs/official/        tracked curated source PDFs, added deliberately
data/manual_pdfs/staging/    ignored working area for uploads and experiments
data/processed/              ignored generated extracted JSON
chroma_db/                   ignored generated vector index
```

The first demo source shortlist is tracked in `docs/first_demo_sources.md`. It separates official grounding documents from farmer-language/evaluation sources so the RAG corpus stays trustworthy.

Manual PDF ingestion now uses a local manifest:

```bash
mkdir -p data/manual_pdfs/staging
cp corpus/pdfs/manual_manifest.example.json data/manual_pdfs/manifest.json
```

Then place your PDFs in `data/manual_pdfs/staging/`, edit `data/manual_pdfs/manifest.json`, and run:

```bash
python ingest_manual_pdfs.py --manifest data/manual_pdfs/manifest.json
python clean_text.py --force
python chunk_and_embed.py --force
python validate_corpus.py
```

Manifest entries require `name`, `display`, `file`, `category`, `state`, `language`, and `priority`, plus either `url` or `source_note`. Invalid files are rejected if they are not real PDFs, are HTML/error pages saved as `.pdf`, have duplicate names, or contain too little extractable text.

## Setup

```bash
python3 -m venv krishinyay-env
source krishinyay-env/bin/activate
pip install -r requirements.txt
```

For MiniLM retrieval, allow `sentence-transformers` to download:

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"
```

If MiniLM is unavailable, indexing falls back to TF-IDF so the pipeline can still be demonstrated offline.

## Build The Demo Index

Use the tracked sample corpus:

```bash
python chunk_and_embed.py --sample-only --force
python validate_corpus.py
```

Use the full local corpus after running ingestion:

```bash
python ingest_manual_pdfs.py --manifest data/manual_pdfs/manifest.json
python scrape_schemes.py
python download_pdfs.py
python clean_text.py
python chunk_and_embed.py --force
python validate_corpus.py
```

## Run The App

```bash
uvicorn app:app --reload --port 8000
```

Open:

```text
http://localhost:8000
```

API endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /health` | Shows indexed chunks, collection, embedding backend, embedding dimension, and LLM provider |
| `POST /query` | Answers a question using dynamic routing or RAG, with `route` and source metadata |
| `POST /ingest` | Adds a text document to the live vector store |

## LLM Modes

The project works without a paid LLM by returning a template answer from the top retrieved chunk. For richer generation, set one of:

```bash
export GEMINI_API_KEY=...
export OPENROUTER_API_KEY=...
export ANTHROPIC_API_KEY=...
```

Or run Ollama locally:

```bash
ollama pull llama3.1:8b
```

## Sample Questions

```text
Who is eligible for PM-KISAN?
PM-KISAN mein kitne paise milte hain?
PMFBY claim kitne time mein file karna hota hai?
KCC ke liye kya documents chahiye?
FRA forest rights kya hota hai?
Maharashtra mein farmers ko kya extra scheme milti hai?
```

## Evaluation

Run retrieval validation:

```bash
python validate_corpus.py
```

The validation script checks both static RAG retrieval and dynamic routing for live data questions.

Run RAGAS-style evaluation:

```bash
python evaluate_rag.py
```

Outputs are written under `eval/`.

## Current Limitations

- The local PMFBY PDF copy may be an HTML/error page saved with a `.pdf` extension; crop-insurance coverage should be repaired with a valid official document or sample source.
- Live beneficiary status, payment status, mandi prices, and weather should not be answered from static RAG context; the app routes these to official/live sources where possible.
- `data/` and `chroma_db/` are ignored intentionally, so reproducible demos should use `sample_data/` or re-run ingestion locally.

## Resume Bullet

Built an India-focused RAG assistant for farmer scheme navigation using government/PDF ingestion, Hindi-aware cleaning and chunking, MiniLM multilingual embeddings, ChromaDB retrieval, FastAPI, source-grounded answer generation, and RAGAS-style evaluation.
