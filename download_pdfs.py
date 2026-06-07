"""
KrishiNyay — PDF downloader
Downloads PDFs from ICAR, CACP MSP notifications, legal acts (indiacode).
Then extracts text with pdfplumber.

Run: python scripts/download_pdfs.py
"""

import requests
from bs4 import BeautifulSoup
import pdfplumber
import json
import time
import logging
import re
from pathlib import Path
from datetime import datetime
from urllib.parse import urljoin, urlparse

ROOT      = Path(__file__).resolve().parent
PDF_DIR   = ROOT / "data" / "raw" / "pdfs"
PROC_DIR  = ROOT / "data" / "processed"
LOGS_DIR  = ROOT / "logs"

for d in [PDF_DIR, PROC_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / f"pdfs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("krishinyay.pdfs")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept-Language": "en-IN,en;q=0.9",
}

# ── Direct PDF URLs ────────────────────────────────────────────────────────
# These are stable, well-known government PDFs — download directly.
DIRECT_PDFS = [
    # Central scheme operational guidelines
    {
        "name": "pmkisan_guidelines",
        "display": "PM-KISAN Operational Guidelines",
        "url": "https://pmkisan.gov.in/Documents/Pradhan_Mantri_Kisan_Samman_Nidhi_Yojana_Operational_Guidelines.pdf",
        "category": "central_scheme"
    },
    {
        "name": "pmfby_guidelines",
        "display": "PMFBY Revised Operational Guidelines 2020",
        "url": "https://pmfby.gov.in/pdf/GOI_Revised_Operational_Guidelines_PMFBY_RWBCIS_2020.pdf",
        "category": "central_scheme"
    },
    {
        "name": "kcc_scheme_guidelines",
        "display": "Kisan Credit Card Revised Scheme",
        "url": "https://www.nabard.org/auth/writereaddata/tender/1608180417KCC%20Revised%20Scheme.pdf",
        "category": "credit"
    },
    # Legal acts
    {
        "name": "fra_2006",
        "display": "Forest Rights Act 2006",
        "url": "https://tribal.nic.in/downloads/FRA/FRAact2006.pdf",
        "category": "legal"
    },
    {
        "name": "fra_rules_2008",
        "display": "Forest Rights Rules 2008",
        "url": "https://tribal.nic.in/downloads/FRA/FRA_Rules2008.pdf",
        "category": "legal"
    },
    {
        "name": "land_acquisition_act_2013",
        "display": "Right to Fair Compensation Act 2013",
        "url": "https://indiacode.nic.in/bitstream/123456789/2081/1/201330.pdf",
        "category": "legal"
    },
    # MSP circulars (most recent available)
    {
        "name": "msp_kharif_2024",
        "display": "MSP Kharif Crops 2024-25 CACP",
        "url": "https://cacp.dacnet.nic.in/ViewContents.aspx?Input=1&PageId=36&KeyId=0",
        "category": "msp",
        "is_page": True   # need to find PDF links on this page
    },
    # ICAR crop guides
    {
        "name": "icar_wheat_guide",
        "display": "ICAR Wheat Production Technology",
        "url": "https://icar.org.in/files/Wheat-and-Barley.pdf",
        "category": "crop_science"
    },
    {
        "name": "icar_rice_guide",
        "display": "ICAR Rice Production Technology",
        "url": "https://icar.org.in/files/Rice.pdf",
        "category": "crop_science"
    },
    {
        "name": "icar_cotton_guide",
        "display": "ICAR Cotton Production",
        "url": "https://icar.org.in/files/Cotton.pdf",
        "category": "crop_science"
    },
    {
        "name": "icar_soybean",
        "display": "ICAR Soybean Production",
        "url": "https://icar.org.in/files/Soybean.pdf",
        "category": "crop_science"
    },
]


def download_pdf(url: str, out_path: Path, retries: int = 3) -> bool:
    """Download a PDF to out_path. Returns True on success."""
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30, stream=True)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            if "pdf" not in content_type and not url.endswith(".pdf"):
                log.warning(f"    Content-type '{content_type}' — may not be a PDF")
            with open(out_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            if out_path.stat().st_size == 0:
                log.warning(f"    Downloaded empty file: {out_path.name}")
                out_path.unlink(missing_ok=True)
                return False
            size_kb = out_path.stat().st_size // 1024
            log.info(f"  ✓ Downloaded {size_kb} KB → {out_path.name}")
            return True
        except Exception as e:
            log.warning(f"  Attempt {attempt} failed: {e}")
            if attempt < retries:
                time.sleep(3)
    return False


def extract_pdf_text(pdf_path: Path) -> dict:
    """Extract text from a PDF using pdfplumber. Returns structured doc dict."""
    pages_data = []
    scanned_pages = 0

    try:
        with pdfplumber.open(pdf_path) as pdf:
            total = len(pdf.pages)
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                text = text.strip()

                if len(text) < 30:
                    scanned_pages += 1
                    continue

                # Basic Hindi/English text cleaning
                text = re.sub(r'\n{3,}', '\n\n', text)
                text = re.sub(r' {2,}', ' ', text)

                pages_data.append({
                    "page":       i + 1,
                    "text":       text,
                    "char_count": len(text),
                })

        full_text = "\n\n".join(p["text"] for p in pages_data)

        return {
            "pages":         pages_data,
            "total_pages":   total,
            "scanned_pages": scanned_pages,  # image-only pages → need OCR later
            "full_text":     full_text,
            "char_count":    len(full_text),
        }

    except Exception as e:
        log.error(f"  pdfplumber failed on {pdf_path.name}: {e}")
        return {"pages": [], "total_pages": 0, "scanned_pages": 0,
                "full_text": "", "char_count": 0, "error": str(e)}


def process_pdf(item: dict) -> dict:
    """Download + extract one PDF. Returns complete doc dict."""
    name    = item["name"]
    display = item["display"]
    url     = item["url"]

    pdf_path  = PDF_DIR  / f"{name}.pdf"
    proc_path = PROC_DIR / f"{name}.json"

    # Skip if already processed
    if proc_path.exists():
        log.info(f"  ↩ Already processed: {name}")
        with open(proc_path, encoding="utf-8") as f:
            return json.load(f)

    log.info(f"{'─'*60}")
    log.info(f"Processing: {display}")
    log.info(f"  URL: {url}")

    # Download
    if not pdf_path.exists():
        success = download_pdf(url, pdf_path)
        if not success:
            log.error(f"  ✗ Could not download {url}")
            return {"name": name, "error": "download_failed", "char_count": 0}
    else:
        log.info(f"  ↩ PDF already on disk: {pdf_path.name}")

    # Extract text
    log.info(f"  Extracting text...")
    extracted = extract_pdf_text(pdf_path)

    if extracted["scanned_pages"] > 0:
        log.warning(
            f"  ⚠ {extracted['scanned_pages']} scanned pages found — "
            f"add to OCR queue (Phase 5)"
        )

    doc = {
        "id":          name,
        "name":        name,
        "display":     display,
        "url":         url,
        "category":    item.get("category", "scheme"),
        "source_type": "pdf",
        "pages":       extracted["pages"],
        "total_pages": extracted["total_pages"],
        "scanned_pages": extracted["scanned_pages"],
        "text":        extracted["full_text"],
        "char_count":  extracted["char_count"],
        "scraped_at":  datetime.now().isoformat(),
    }

    with open(proc_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)

    log.info(
        f"  ✓ {extracted['total_pages']} pages · "
        f"{extracted['char_count']:,} chars → {proc_path.name}"
    )
    return doc


def scrape_pdf_links_from_page(url: str, base_name: str, category: str) -> list:
    """
    Some pages (like CACP) list PDFs as links. Find and download them.
    """
    log.info(f"  Finding PDFs on page: {url}")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        pdf_links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if ".pdf" in href.lower():
                full_url  = urljoin(url, href)
                link_text = a.get_text(" ", strip=True)[:60] or Path(urlparse(full_url).path).stem
                safe_name = re.sub(r'[^\w\-]+', '_', link_text).strip("_")[:40] or "document"
                pdf_links.append({
                    "name":     f"{base_name}_{safe_name}",
                    "display":  link_text,
                    "url":      full_url,
                    "category": category,
                })
        log.info(f"  Found {len(pdf_links)} PDF links")
        return pdf_links[:10]  # cap at 10 per page to avoid runaway downloads
    except Exception as e:
        log.error(f"  Failed to scan page for PDFs: {e}")
        return []


def main():
    log.info(f"{'═'*60}")
    log.info(f"  PDF DOWNLOADER  ({len(DIRECT_PDFS)} configured sources)")
    log.info(f"{'═'*60}")

    all_pdfs  = []
    all_docs  = []

    # Separate direct PDFs from page-scrape PDFs
    for item in DIRECT_PDFS:
        if item.get("is_page"):
            links = scrape_pdf_links_from_page(
                item["url"], item["name"], item["category"]
            )
            all_pdfs.extend(links)
        else:
            all_pdfs.append(item)

    log.info(f"  Total PDFs to process: {len(all_pdfs)}")

    for item in all_pdfs:
        doc = process_pdf(item)
        all_docs.append(doc)
        time.sleep(2)

    # Summary
    good        = [d for d in all_docs if d.get("char_count", 0) > 500]
    scanned     = [d for d in all_docs if d.get("scanned_pages", 0) > 0]
    failed      = [d for d in all_docs if d.get("char_count", 0) <= 500]
    total_chars = sum(d.get("char_count", 0) for d in all_docs)

    log.info(f"\n{'═'*60}")
    log.info(f"  PDF SUMMARY")
    log.info(f"{'═'*60}")
    log.info(f"  Processed  : {len(all_docs)}")
    log.info(f"  Good text  : {len(good)}")
    log.info(f"  Has scanned pages (need OCR) : {len(scanned)}")
    log.info(f"  Failed / too short : {len(failed)}")
    log.info(f"  Total chars: {total_chars:,}")

    if scanned:
        log.info("\n  PDFs with scanned pages (add these to OCR queue in Phase 5):")
        for d in scanned:
            log.info(f"    {d['name']}: {d['scanned_pages']} scanned pages")

    # Save OCR queue for Phase 5
    ocr_queue = [
        {"name": d["name"], "pdf_path": str(PDF_DIR / f"{d['name']}.pdf")}
        for d in scanned
    ]
    with open(ROOT / "data" / "ocr_queue.json", "w") as f:
        json.dump(ocr_queue, f, indent=2)
    log.info(f"\n  OCR queue saved → data/ocr_queue.json")
    log.info("  Run clean_text.py next.")


if __name__ == "__main__":
    main()
