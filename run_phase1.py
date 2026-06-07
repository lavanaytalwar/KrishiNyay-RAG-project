"""
KrishiNyay — Phase 1 master runner
Runs the complete data pipeline in order:
  1. Scrape HTML scheme portals (central + state)
  2. Download + extract PDFs
  3. Clean + normalise all text

Run: python scripts/run_phase1.py

Flags:
  --skip-scrape     skip HTML scraping (use existing raw files)
  --skip-pdfs       skip PDF download/extraction
  --skip-clean      skip text cleaning
  --only-central    scrape only central schemes
  --only-state      scrape only state schemes
"""

import subprocess
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime

ROOT     = Path(__file__).resolve().parent
LOGS_DIR = ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / f"phase1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("krishinyay.phase1")

PYTHON = sys.executable
SCRIPTS = Path(__file__).parent


def run_script(script: str, args: list[str] | None = None) -> bool:
    """Run a script as a subprocess. Returns True if successful."""
    args = args or []
    cmd = [PYTHON, str(SCRIPTS / script)] + args
    log.info(f"\n{'═'*60}")
    log.info(f"  Running: {' '.join(cmd)}")
    log.info(f"{'═'*60}")
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        log.error(f"  ✗ {script} failed with code {result.returncode}")
        return False
    log.info(f"  ✓ {script} completed successfully")
    return True


def check_requirements() -> bool:
    """Check that required packages are installed."""
    log.info("Checking requirements...")
    required = [
        ("requests",             "requests"),
        ("bs4",                  "beautifulsoup4"),
        ("pdfplumber",           "pdfplumber"),
        ("langchain",            "langchain langchain-community"),
        ("sentence_transformers","sentence-transformers"),
        ("chromadb",             "chromadb"),
        ("tqdm",                 "tqdm"),
    ]
    missing = []
    for module, pkg in required:
        try:
            __import__(module)
        except ImportError:
            missing.append(pkg)

    if missing:
        log.error(f"Missing packages: {' '.join(missing)}")
        log.error(f"Fix: pip install {' '.join(missing)}")
        return False

    log.info("  ✓ All required packages present")
    return True


def print_phase1_summary():
    """Print what was produced by Phase 1."""
    log.info(f"\n{'═'*60}")
    log.info("  PHASE 1 COMPLETE — what you now have")
    log.info(f"{'═'*60}")

    clean_dir = ROOT / "data" / "processed" / "clean"
    chunks_dir = ROOT / "data" / "chunks"
    chroma_dir = ROOT / "chroma_db"

    def count_files(d, pattern="*.json"):
        return len(list(d.glob(pattern))) if d.exists() else 0

    def total_chars(d):
        import json
        total = 0
        if not d.exists():
            return 0
        for f in d.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                total += data.get("char_count", 0)
            except Exception:
                pass
        return total

    log.info(f"  Clean docs     : {count_files(clean_dir)} files")
    log.info(f"  Total chars    : {total_chars(clean_dir):,}")
    log.info(f"  Chunks JSONL   : {'✓' if (chunks_dir / 'all_chunks.jsonl').exists() else '✗'}")
    log.info(f"  ChromaDB       : {'✓' if chroma_dir.exists() else '✗'}")
    log.info(f"\n  Next step: run  python scripts/chunk_and_embed.py")
    log.info(f"  Then validate:  python scripts/validate_corpus.py")


def main():
    parser = argparse.ArgumentParser(description="KrishiNyay Phase 1 runner")
    parser.add_argument("--skip-scrape",   action="store_true")
    parser.add_argument("--skip-pdfs",     action="store_true")
    parser.add_argument("--skip-clean",    action="store_true")
    parser.add_argument("--only-central",  action="store_true")
    parser.add_argument("--only-state",    action="store_true")
    args = parser.parse_args()

    log.info(f"\n{'█'*60}")
    log.info(f"  KRISHINYAY — PHASE 1 DATA PIPELINE")
    log.info(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log.info(f"{'█'*60}")

    if not check_requirements():
        sys.exit(1)

    success = True

    # Step 1: Scrape HTML portals
    if not args.skip_scrape:
        scrape_args = []
        if args.only_central:
            scrape_args += ["--type", "central"]
        elif args.only_state:
            scrape_args += ["--type", "state"]
        if not run_script("scrape_schemes.py", scrape_args):
            log.warning("Scraping had errors — continuing with what we got")

    # Step 2: Download + extract PDFs
    if not args.skip_pdfs:
        if not run_script("download_pdfs.py"):
            log.warning("PDF download had errors — continuing")

    # Step 3: Clean all text
    if not args.skip_clean:
        if not run_script("clean_text.py"):
            log.error("Cleaning failed — stopping")
            success = False

    if success:
        print_phase1_summary()
    else:
        log.error("\nPhase 1 finished with errors. Check logs/")
        sys.exit(1)


if __name__ == "__main__":
    main()
