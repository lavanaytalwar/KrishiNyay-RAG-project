# First Demo Acquisition Status

Generated source artifacts live under `data/`, which is intentionally ignored by git.

## Local Outputs

- `data/first_demo_acquisition/acquisition_report.json`: machine-readable download report.
- `data/first_demo_acquisition/pdfs/`: downloaded PDFs.
- `data/first_demo_acquisition/raw_html/`: downloaded raw HTML pages.
- `data/first_demo_acquisition/raw_data/`: downloaded structured JSON/API payloads.
- `data/first_demo_acquisition/raw_text/`: extracted text from PDFs, HTML, and JSON.
- `data/manual_pdfs/manifest.json`: generated manifest for extractable official PDFs.
- `data/manual_pdfs/staging/`: staged PDFs referenced by the manual ingestion manifest.

## Current Counts

- Source entries attempted: 27.
- Successful downloads: 23.
- Failed downloads: 4.
- Ingest-ready official PDFs: 6.
- Manual PDF ingestion result: 6 processed, 0 skipped, 0 failed.

## Ingested Official PDFs

| Source | Extracted chars | Status |
| --- | ---: | --- |
| PM-KISAN Operational Guidelines | 27,421 | Ingested |
| PM-KISAN Revised FAQ | 1,838 | Ingested |
| PMFBY Operational Guidelines | 599,399 | Ingested |
| Forest Rights Act, Rules and Guidelines | 97,390 | Ingested |
| LARR / Land Acquisition Act 2013 | 166,585 | Ingested |
| e-NAM Operational Guidelines | 45,238 | Ingested |

## Captured Non-PDF Sources

- e-NAM FAQ.
- Soil Health Card structured JSON from the public myScheme API.
- Punjab Agriculture Department portal.
- Haryana Agriculture Department portal.
- Maharashtra Agriculture Department portal.
- Access Agriculture turmeric pages in English, Hindi, and Marathi.
- Digital Green public homepage.
- CABI PlantwisePlus public page.
- AgriGov paper and KisanQRS paper as research references.

## Remaining Gaps

| Source | Current result | Next action |
| --- | --- | --- |
| PMKSY official PDFs | `pmksy.gov.in` returns HTTP 503 to CLI and Chrome reports `ERR_BLOCKED_BY_CLIENT`. | Manually download in normal Chrome/Safari if needed, then place PDFs in `data/manual_pdfs/staging/`. |
| CABI PlantwisePlus Knowledge Bank | Knowledge Bank root returns HTTP 403 to automation. | Public CABI PlantwisePlus page is captured; use Knowledge Bank only after manual/license review. |
| aAQUA farmer Q&A | Legacy host does not resolve; `aaqua.org` is currently a parked domain. | Replace with another public farmer Q&A/eval source unless a working archive is found. |
| PM-KISAN KCC form | PDF downloaded but has 0 extractable characters. | Defer to OCR/manual transcription. |
| Vikaspedia scheme page | Downloaded page is only a thin shell. | Use existing scraper outputs or browser-based capture for specific Vikaspedia article pages. |

## Commands

```bash
python acquire_first_demo_sources.py
python ingest_manual_pdfs.py --manifest data/manual_pdfs/manifest.json
```
