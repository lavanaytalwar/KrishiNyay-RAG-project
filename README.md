# KrishiNyay AI — Phase 1: Data Pipeline

Complete data collection pipeline for all Indian government scheme portals,
state-wise schemes, crop science guides, and legal acts.

## What this collects

| Category | Sources | Method |
|---|---|---|
| Central govt schemes | PM-KISAN, PMFBY, eNAM, KCC, PMKSY, RKVY, MIDH, AIF, MNREGA | requests + BS4 |
| State schemes | 12 states: MH, UP, RJ, MP, AP, TS, KA, PB, GJ, BR, TN, WB | requests + BS4 |
| JS-rendered portals | Rajkisan, Rythu Bandhu, UP Agriculture | Playwright |
| PDFs | PM-KISAN guidelines, PMFBY guidelines, KCC scheme, FRA 2006, Land Acquisition Act 2013, ICAR crop guides | pdfplumber |
| Agri guides (Hindi) | Vikaspedia agriculture section (Hindi) | requests + BS4 |

## Setup

```bash
# 1. Clone and enter project
git clone https://github.com/yourusername/krishinyay
cd krishinyay

# 2. Create virtual environment
python -m venv krishinyay-env
source krishinyay-env/bin/activate   # Mac/Linux
# krishinyay-env\Scripts\activate    # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install Playwright browser (for JS sites)
playwright install chromium

# 5. Create .env (add any API keys later)
cp .env.example .env
```

## Run the full pipeline

```bash
# Run everything (recommended first time)
python3 run_phase1.py

# Or run steps individually:
python3 scrape_schemes.py          # all portals
python3 scrape_schemes.py --type central   # only central
python3 scrape_schemes.py --type state     # only state
python3 scrape_schemes.py --name pmkisan   # one source only

python3 download_pdfs.py          # PDFs + text extraction
python3 clean_text.py             # clean + normalise all text

# Then embed (Phase 2):
python3 chunk_and_embed.py
python3 validate_corpus.py
```

## Phase 3 web app

Phase 3 turns the RAG scripts into a usable web service with a multi-page UI.

```bash
# install dependencies
pip install -r requirements.txt

# run the FastAPI app
uvicorn app:app --reload --port 8000
```

Open `http://localhost:8000`.

### API endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | Shows indexed chunk count, embedding backend, LLM provider, and router status |
| `POST /query` | Runs dynamic-query routing first, then RAG over ChromaDB |
| `POST /ingest` | Adds a new text document to the live vector store without rebuilding the full corpus |

### LLM modes

Set one hosted key for deployed generation, or run without a key for offline template answers.

```bash
export GEMINI_API_KEY=...
# or
export OPENROUTER_API_KEY=...
# local-only fallback
ollama pull llama3.1:8b
```

For Hugging Face Spaces, use the included `Dockerfile`, set the Space SDK to Docker,
and add API keys as Space secrets.

## Output structure

```
data/
├── raw/
│   ├── schemes/          ← central govt scheme JSON files
│   ├── state_schemes/    ← state scheme JSON files
│   └── pdfs/             ← downloaded PDFs
├── processed/
│   ├── *.json            ← PDF-extracted text docs
│   └── clean/            ← all cleaned, normalised docs
├── chunks/
│   └── all_chunks.jsonl  ← final chunked corpus
├── scrape_manifest.json  ← what was scraped and when
└── ocr_queue.json        ← scanned PDFs needing OCR (Phase 5)

chroma_db/                ← vector store (Phase 2)
logs/                     ← timestamped run logs
```

## Each JSON document looks like

```json
{
  "id": "pmkisan_a1b2c3",
  "name": "pmkisan",
  "display": "PM-KISAN",
  "url": "https://pmkisan.gov.in/faq.aspx",
  "category": "income_support",
  "state": "central",
  "priority": "high",
  "language": ["hindi", "english"],
  "text": "PM-KISAN scheme provides income support...",
  "char_count": 4821,
  "scraped_at": "2026-05-21T10:30:00"
}
```

## Common issues

| Problem | Fix |
|---|---|
| Empty text from a site | Site may be JS-rendered or moved — check the URL in `sources.json` |
| PDF text garbled | PDF is scanned — goes to `ocr_queue.json` for Phase 5 |
| `indic-nlp-library` error | `pip install indic-nlp-library` |
| 403 / 429 errors | Increase `--delay` flag: `python3 scrape_schemes.py --delay 5` |

## Resume bullet (after this phase)

> Built automated data pipeline scraping 23 central + 12 state Indian government 
> scheme portals using requests, BeautifulSoup, and Playwright; extracted 
> structured text from 15+ PDFs using pdfplumber; applied Hindi Unicode 
> normalisation via IndicNLP; produced clean 500+ chunk corpus for RAG ingestion.
