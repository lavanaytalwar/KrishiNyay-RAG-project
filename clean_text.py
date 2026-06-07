"""
KrishiNyay — Text cleaner & normaliser
Cleans all scraped HTML text and extracted PDF text.
Handles Hindi Unicode normalisation via IndicNLP.

Run: python scripts/clean_text.py
"""

import json
import re
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional

try:
    from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
    INDIC_NLP = True
    normalizer = IndicNormalizerFactory().get_normalizer("hi")
except ImportError:
    INDIC_NLP = False
    print("Warning: indic-nlp-library not installed. Hindi normalisation skipped.")
    print("Fix: pip install indic-nlp-library")

ROOT      = Path(__file__).resolve().parent
RAW_DIRS  = [
    ROOT / "data" / "raw" / "schemes",
    ROOT / "data" / "raw" / "state_schemes",
    ROOT / "data" / "processed",         # already-extracted PDFs
]
CLEAN_DIR = ROOT / "data" / "processed" / "clean"
CLEAN_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR  = ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / f"clean_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("krishinyay.cleaner")


# ── Cleaning functions ─────────────────────────────────────────────────────

def remove_boilerplate(text: str) -> str:
    """Remove common govt website boilerplate phrases."""
    boilerplate = [
        r"Skip to main content.*?\n",
        r"Screen Reader Access.*?\n",
        r"A-\s*A\+.*?\n",
        r"Last Updated.*?\n",
        r"Copyright.*?Government of India.*?\n",
        r"This site is best viewed.*?\n",
        r"Designed.*?National Informatics Centre.*?\n",
        r"Terms & Conditions.*?\n",
        r"Disclaimer.*?\n",
        r"Site Map.*?\n",
        r"Feedback.*?\n",
        r"Contact Us.*?\n",
        r"\bjavascript\b.*?\n",
        r"Please enable JavaScript.*?\n",
        r"You need JavaScript.*?\n",
    ]
    for pattern in boilerplate:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return text


def clean_text(text: str, source_name: str = "") -> str:
    """Full cleaning pipeline for a text block."""
    if not text:
        return ""

    # Remove URLs and emails
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    text = re.sub(r'\S+@\S+\.\S+', '', text)

    # Remove boilerplate
    text = remove_boilerplate(text)

    # Remove non-printable characters while preserving major Indic scripts.
    text = re.sub(
        r'[^\x20-\x7E'
        r'\u0900-\u097F'  # Devanagari
        r'\u0980-\u09FF'  # Bengali
        r'\u0A00-\u0A7F'  # Gurmukhi
        r'\u0A80-\u0AFF'  # Gujarati
        r'\u0B00-\u0B7F'  # Odia
        r'\u0B80-\u0BFF'  # Tamil
        r'\u0C00-\u0C7F'  # Telugu
        r'\u0C80-\u0CFF'  # Kannada
        r'\u0D00-\u0D7F'  # Malayalam
        r'\n\r\t]',
        '',
        text,
    )

    # Collapse whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\t+', ' ', text)

    # Hindi Unicode normalisation (fixes encoding inconsistencies)
    if INDIC_NLP:
        try:
            text = normalizer.normalize(text)
        except Exception:
            pass   # if normaliser fails on a chunk, carry on

    # Remove very short lines (navigation artifacts, single chars)
    lines = text.split('\n')
    lines = [l for l in lines if len(l.strip()) > 3]
    text  = '\n'.join(lines)

    return text.strip()


def deduplicate_paragraphs(text: str) -> str:
    """Remove duplicate paragraphs (common in scraped govt pages)."""
    seen   = set()
    result = []
    for para in text.split('\n\n'):
        key = para.strip().lower()[:100]
        if key and key not in seen:
            seen.add(key)
            result.append(para.strip())
    return '\n\n'.join(result)


# ── Process one JSON file ──────────────────────────────────────────────────

def process_json(json_path: Path) -> Optional[dict]:
    """Load, clean, deduplicate, and save one JSON doc."""
    try:
        with open(json_path, encoding="utf-8") as f:
            doc = json.load(f)
    except Exception as e:
        log.error(f"  Failed to read {json_path.name}: {e}")
        return None

    source_name = doc.get("name", json_path.stem)

    # Handle both flat-text docs (scraped HTML) and paged docs (PDFs)
    if "pages" in doc and doc["pages"]:
        # PDF format — clean each page
        for page in doc["pages"]:
            if "text" in page:
                page["text"] = deduplicate_paragraphs(
                    clean_text(page["text"], source_name)
                )
        doc["text"] = "\n\n".join(p["text"] for p in doc["pages"] if p.get("text"))
    elif doc.get("full_text"):
        doc["text"] = deduplicate_paragraphs(
            clean_text(doc["full_text"], source_name)
        )
    elif "text" in doc:
        # Flat HTML format
        doc["text"] = deduplicate_paragraphs(
            clean_text(doc["text"], source_name)
        )

    doc["char_count"] = len(doc.get("text", ""))
    doc["cleaned_at"] = datetime.now().isoformat()

    return doc


# ── Main ───────────────────────────────────────────────────────────────────

def output_is_current(source_path: Path, out_path: Path) -> bool:
    """Return True when an existing clean file is newer than its raw source."""
    return out_path.exists() and out_path.stat().st_mtime >= source_path.stat().st_mtime


def main():
    parser = argparse.ArgumentParser(description="Clean and normalise scraped text")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-clean all files even if clean outputs already exist",
    )
    args = parser.parse_args()

    log.info(f"{'═'*60}")
    log.info("  TEXT CLEANER & NORMALISER")
    log.info(f"{'═'*60}")
    if not INDIC_NLP:
        log.warning("  Hindi normalisation disabled (indic-nlp-library not installed)")

    all_json_files = []
    for raw_dir in RAW_DIRS:
        if raw_dir.exists():
            files = list(raw_dir.glob("*.json"))
            # Skip files already in clean/ subdir
            files = [f for f in files if "clean" not in str(f)]
            all_json_files.extend(files)

    # Deduplicate by stem
    seen_stems = set()
    unique_files = []
    for f in all_json_files:
        if f.stem not in seen_stems:
            seen_stems.add(f.stem)
            unique_files.append(f)

    log.info(f"  Found {len(unique_files)} JSON files to clean")

    results = {"cleaned": 0, "skipped": 0, "failed": 0, "total_chars": 0}

    for json_path in unique_files:
        out_path = CLEAN_DIR / json_path.name

        if not args.force and output_is_current(json_path, out_path):
            log.info(f"  ↩ Already clean: {json_path.name}")
            results["skipped"] += 1
            continue

        doc = process_json(json_path)
        if not doc:
            results["failed"] += 1
            continue

        char_count = doc.get("char_count", 0)

        if char_count < 50:
            log.warning(f"  ⚠ Very short after cleaning ({char_count} chars), skipping: {json_path.name}")
            if out_path.exists():
                out_path.unlink()
            results["skipped"] += 1
            continue

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)

        results["cleaned"]     += 1
        results["total_chars"] += char_count
        log.info(f"  ✓ {json_path.name:50s} {char_count:>8,} chars")

    log.info(f"\n{'═'*60}")
    log.info(f"  Cleaned  : {results['cleaned']}")
    log.info(f"  Skipped  : {results['skipped']}")
    log.info(f"  Failed   : {results['failed']}")
    log.info(f"  Total chars: {results['total_chars']:,}")
    log.info(f"  Output   → {CLEAN_DIR}")
    log.info("  Run chunk_and_embed.py next.")


if __name__ == "__main__":
    main()
