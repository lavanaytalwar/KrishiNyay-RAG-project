"""
KrishiNyay — manual PDF ingester.

Reads a local manifest of manually collected official PDFs, extracts text with
pdfplumber, validates basic PDF quality, and writes structured JSON documents
to data/processed/ for clean_text.py and chunk_and_embed.py.

Run:
  python ingest_manual_pdfs.py --manifest data/manual_pdfs/manifest.json
  python ingest_manual_pdfs.py --manifest data/manual_pdfs/manifest.json --force
"""

import argparse
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pdfplumber


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "data" / "manual_pdfs" / "manifest.json"
PROC_DIR = ROOT / "data" / "processed"
LOGS_DIR = ROOT / "logs"

REQUIRED_FIELDS = {
    "name",
    "display",
    "category",
    "state",
    "language",
    "priority",
}


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[
            logging.FileHandler(
                LOGS_DIR / f"manual_pdfs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            ),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("krishinyay.manual_pdfs")


def load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {path}\n"
            "Copy corpus/pdfs/manual_manifest.example.json to data/manual_pdfs/manifest.json "
            "and fill in your local PDF file paths."
        )

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        documents = data.get("documents", [])
    elif isinstance(data, list):
        documents = data
    else:
        raise ValueError("Manifest must be a JSON object with documents[] or a list of documents.")

    if not documents:
        raise ValueError("Manifest contains no documents.")

    return documents


def resolve_pdf_path(raw_path: str, manifest_path: Path) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate
    manifest_relative = manifest_path.parent / candidate
    if manifest_relative.exists():
        return manifest_relative.resolve()
    return (ROOT / candidate).resolve()


def validate_item(item: dict[str, Any], seen_names: set[str]) -> list[str]:
    errors = []
    missing = sorted(REQUIRED_FIELDS - set(item))
    if missing:
        errors.append(f"missing required fields: {', '.join(missing)}")

    name = str(item.get("name", "")).strip()
    if not name:
        errors.append("name is empty")
    elif not re.fullmatch(r"[a-z0-9][a-z0-9_\-]*", name):
        errors.append("name must use lowercase letters, numbers, underscores, or hyphens")
    elif name in seen_names:
        errors.append(f"duplicate name: {name}")
    else:
        seen_names.add(name)

    if not item.get("url") and not item.get("source_note"):
        errors.append("provide either url or source_note")

    if not item.get("file"):
        errors.append("missing file path")

    return errors


def reject_non_pdf(pdf_path: Path) -> None:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    head = pdf_path.read_bytes()[:512].lstrip()
    if head.startswith(b"<!DOCTYPE html") or head.startswith(b"<html"):
        raise ValueError("file looks like HTML, not a PDF")
    if not head.startswith(b"%PDF-"):
        raise ValueError("file does not start with a PDF header")


def extract_pdf_text(pdf_path: Path) -> dict[str, Any]:
    pages_data = []
    scanned_pages = 0

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        for idx, page in enumerate(pdf.pages, 1):
            text = (page.extract_text() or "").strip()
            if len(text) < 30:
                scanned_pages += 1
                continue
            text = re.sub(r"\n{3,}", "\n\n", text)
            text = re.sub(r" {2,}", " ", text)
            pages_data.append({
                "page": idx,
                "text": text,
                "char_count": len(text),
            })

    full_text = "\n\n".join(page["text"] for page in pages_data)
    return {
        "pages": pages_data,
        "total_pages": total_pages,
        "scanned_pages": scanned_pages,
        "text": full_text,
        "char_count": len(full_text),
    }


def process_item(
    item: dict[str, Any],
    manifest_path: Path,
    force: bool,
    min_chars: int,
    log: logging.Logger,
) -> tuple[str, str]:
    name = item["name"]
    pdf_path = resolve_pdf_path(item["file"], manifest_path)
    out_path = PROC_DIR / f"{name}.json"

    if out_path.exists() and not force:
        log.info(f"  ↩ Already processed: {name}")
        return name, "skipped"

    reject_non_pdf(pdf_path)
    extracted = extract_pdf_text(pdf_path)
    if extracted["char_count"] < min_chars:
        raise ValueError(
            f"too little extractable text ({extracted['char_count']} chars); "
            "likely scanned or unsuitable for text RAG"
        )

    doc = {
        "id": name,
        "name": name,
        "display": item["display"],
        "url": item.get("url", ""),
        "source_note": item.get("source_note", ""),
        "category": item.get("category", "scheme"),
        "state": item.get("state", "central"),
        "priority": item.get("priority", "medium"),
        "language": item.get("language", "english"),
        "source_type": "manual_pdf",
        "file_name": pdf_path.name,
        "pages": extracted["pages"],
        "total_pages": extracted["total_pages"],
        "scanned_pages": extracted["scanned_pages"],
        "text": extracted["text"],
        "char_count": extracted["char_count"],
        "ingested_at": datetime.now().isoformat(),
    }

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(
        f"  ✓ {name:35s} {extracted['total_pages']:>4} pages · "
        f"{extracted['char_count']:>8,} chars → {out_path.name}"
    )
    if extracted["scanned_pages"]:
        log.warning(f"    {extracted['scanned_pages']} page(s) had little/no extractable text")
    return name, "processed"


def run(manifest: Path, force: bool = False, min_chars: int = 500) -> dict[str, int]:
    log = setup_logging()
    manifest = manifest.expanduser().resolve()

    log.info("=" * 60)
    log.info("  MANUAL PDF INGESTION — KrishiNyay Phase 1")
    log.info("=" * 60)
    log.info(f"Manifest: {manifest}")

    documents = load_manifest(manifest)
    seen_names: set[str] = set()
    results = {"processed": 0, "skipped": 0, "failed": 0}

    for item in documents:
        errors = validate_item(item, seen_names)
        if errors:
            results["failed"] += 1
            log.error(f"  ✗ {item.get('name', '<unknown>')}: {'; '.join(errors)}")
            continue

        try:
            _, status = process_item(item, manifest, force, min_chars, log)
            results[status] += 1
        except Exception as exc:
            results["failed"] += 1
            log.error(f"  ✗ {item.get('name')}: {exc}")

    log.info("")
    log.info("=" * 60)
    log.info(f"  Processed: {results['processed']}")
    log.info(f"  Skipped  : {results['skipped']}")
    log.info(f"  Failed   : {results['failed']}")
    log.info("  Next: python clean_text.py --force")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest manually collected official PDFs")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--min-chars", type=int, default=500)
    args = parser.parse_args()

    results = run(args.manifest, force=args.force, min_chars=args.min_chars)
    if results["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
