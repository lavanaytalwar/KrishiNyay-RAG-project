# KrishiNyay AI — Source-Grounded RAG Assistant for Indian Farmers

KrishiNyay AI is an India-focused retrieval-augmented generation (RAG) project for answering farmer questions about government schemes, crop insurance, farm credit, legal rights, and agriculture support using trusted documents.

The current goal is a small, runnable, demo-ready RAG system before adding advanced agents, voice, WhatsApp, or vision features.

## Project Status

Current baseline:
- Embedding backend: MiniLM (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`)
- Embedding dimension: 384
- Local indexed corpus: 1,748 chunks
- Retrieval validation: 15/15 passing on the current RAG + dynamic routing smoke set
- Farmer eval gate: 100 farmer-facing questions with Phase 5 hybrid retrieval validation

Completed phases:
- Phase 0 — repo hygiene and truthful MiniLM baseline.
- Phase 1 — manual PDF ingestion workflow and curated official PDF ingestion.
- Phase 2 — retrieval normalization, dynamic live-data routing, health metadata, and validation upgrade.
- Phase 3 — FastAPI/frontend demo polish, CSS motion UI, better UX states, and deployment configuration.
- Phase 4 — farmer-facing eval dataset with route, source-type, language, and topic coverage.
- Phase 5 — hybrid MiniLM + lexical retrieval baseline with full eval regression gate.

What remains:
- Phase 6 — OCR for scanned PDFs in manual ingestion.
- Later phases — live mandi/weather API integrations, LangGraph workflows, voice/WhatsApp channels, and optional fine-tuning only after enough validated data exists.

Implemented:
- Data ingestion scripts for government scheme pages and PDFs
- Text cleaning and Hindi-aware normalization
- Character chunking with Hindi sentence separator support
- ChromaDB indexing and retrieval
- MiniLM-first embeddings with TF-IDF fallback
- FastAPI backend with `/health`, `/query`, and `/ingest`
- Static web UI for chat, source inspection, ingestion, and system status
- Demo UI polish with phase cards, route trace, loading states, and CSS/HTML motion UI
- Offline validation and RAGAS-style evaluation scripts
- Tracked `sample_data/` so a fresh clone has a small demo corpus
- Manifest-based manual PDF ingestion for curated official PDFs
- Hybrid MiniLM + lexical retrieval with source-aware guardrails
- Optional OCR hooks for scanned PDF pages in manual ingestion

In progress:
- Indic OCR language-pack validation on real Hindi/Marathi scanned official PDFs

Planned:
- Cross-encoder reranking
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
├── web/media/              # Ignored reference media for motion design
├── eval/                   # Evaluation output
├── .env.example            # Local/deployment environment template
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

For scanned or image-only official PDFs, install the optional OCR dependencies and the system Tesseract binary, then run:

```bash
python ingest_manual_pdfs.py --manifest data/manual_pdfs/manifest.json --ocr
python validate_ocr_pipeline.py
```

OCR is disabled by default. When enabled, ingestion first uses `pdfplumber` and only sends low-text pages to Tesseract. Processed JSON records include OCR metadata such as `ocr_enabled`, `ocr_engine`, `ocr_pages_attempted`, and `ocr_pages_extracted`.

The Phase 6 validator creates an image-only PDF fixture and confirms that OCR extracts known text when Tesseract is installed. The core Homebrew `tesseract` package includes English data; install `tesseract-lang` separately before validating Hindi, Marathi, Punjabi, or other Indic scanned PDFs.

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
| `GET /demo-config` | Shows completed phases, remaining work, demo questions, and motion UI slots |
| `POST /query` | Answers a question using dynamic routing or RAG, with `route` and source metadata |
| `POST /ingest` | Adds a text document to the live vector store |

## Frontend Motion UI

The frontend uses CSS/HTML motion UI instead of embedding raw MP4 files. This keeps the demo lightweight, easier to deploy, and less dependent on large generated media.

Current motion concepts:

```text
Farmer + AI assistant hero
RAG pipeline flow
Dynamic live-data router
Sticky-note retrieval
Source/citation stack
```

MP4 files can still be kept locally in `web/media/` as design references, but video files are ignored by git and are not loaded by the app.

## Deployment Notes

Copy the environment template when preparing a hosted demo:

```bash
cp .env.example .env
```

Build and run with Docker:

```bash
docker build -t krishinyay-rag .
docker run --env-file .env -p 8000:8000 krishinyay-rag
```

For platforms that provide `PORT`, the included `Dockerfile` runs:

```bash
uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}
```

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
- OCR requires local system Tesseract plus Python OCR packages; core English OCR is validated, while Indic OCR requires extra Tesseract language data.
- `data/` and `chroma_db/` are ignored intentionally, so reproducible demos should use `sample_data/` or re-run ingestion locally.

## Resume Bullet

Built an India-focused RAG assistant for farmer scheme navigation using government/PDF ingestion, Hindi-aware cleaning and chunking, MiniLM multilingual embeddings, ChromaDB retrieval, FastAPI, source-grounded answer generation, and RAGAS-style evaluation.
