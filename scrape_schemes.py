"""
KrishiNyay — Scheme Portal Scraper
Scrapes all central and state government scheme portals.
Run: python scripts/scrape_schemes.py
Run only central: python scripts/scrape_schemes.py --type central
Run only states:  python scripts/scrape_schemes.py --type state
Run one source:   python scripts/scrape_schemes.py --name pmkisan
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import logging
import argparse
import hashlib
import re
from pathlib import Path
from datetime import datetime
from typing import Optional
from urllib.parse import urljoin, urlparse

import pdfplumber

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent
SOURCES_FILE = Path(__file__).parent / "sources.json"
RAW_CENTRAL  = ROOT / "data" / "raw" / "schemes"
RAW_STATE    = ROOT / "data" / "raw" / "state_schemes"
PDF_DIR      = ROOT / "data" / "raw" / "pdfs"
LOGS_DIR     = ROOT / "logs"

for d in [RAW_CENTRAL, RAW_STATE, PDF_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────────
log_file = LOGS_DIR / f"scrape_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("krishinyay.scraper")

# ── HTTP session ───────────────────────────────────────────────────────────
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-IN,en;q=0.9,hi;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
})

# ── Helpers ────────────────────────────────────────────────────────────────

def fetch_response(url: str, retries: int = 3, timeout: int = 30) -> Optional[requests.Response]:
    """Fetch a URL with retries. Returns a response or None."""
    for attempt in range(1, retries + 1):
        try:
            resp = SESSION.get(url, timeout=timeout, allow_redirects=True)
            resp.raise_for_status()
            return resp
        except requests.exceptions.HTTPError as e:
            log.warning(f"  HTTP {e.response.status_code} on {url} (attempt {attempt})")
        except requests.exceptions.ConnectionError:
            log.warning(f"  Connection error on {url} (attempt {attempt})")
        except requests.exceptions.Timeout:
            log.warning(f"  Timeout on {url} (attempt {attempt})")
        except Exception as e:
            log.warning(f"  Unexpected error on {url}: {e} (attempt {attempt})")
        if attempt < retries:
            time.sleep(3 * attempt)   # exponential back-off
    return None


def is_pdf_response(url: str, resp: requests.Response) -> bool:
    """Detect PDFs from URL, content type, or file signature."""
    content_type = resp.headers.get("content-type", "").lower()
    path = urlparse(resp.url or url).path.lower()
    return (
        "pdf" in content_type
        or path.endswith(".pdf")
        or resp.content[:4] == b"%PDF"
    )


def extract_text(html: str, selectors: list[str]) -> str:
    """
    Parse HTML and extract text from the given CSS tag names.
    Falls back to full body text if selectors yield nothing useful.
    """
    soup = BeautifulSoup(html, "html.parser")
    metadata_text = extract_metadata_text(soup)
    jsonld_text = extract_jsonld_text(soup)

    # Remove noise elements
    for tag in soup(["script", "style", "nav", "footer",
                     "header", "noscript", "iframe", "form"]):
        tag.decompose()

    blocks = []
    for sel in selectors:
        for el in soup.select(sel):
            text = el.get_text(separator=" ", strip=True)
            if len(text) > 40:          # skip trivially short strings
                blocks.append(text)

    if blocks:
        deduped_blocks = []
        seen = set()
        for block in blocks:
            key = re.sub(r"\s+", " ", block).strip().lower()[:200]
            if key not in seen:
                seen.add(key)
                deduped_blocks.append(block)
        blocks = deduped_blocks

    full_text = "\n\n".join(blocks)

    fallback_texts = [text for text in (jsonld_text, metadata_text) if text]
    if len(full_text.strip()) < 200 and fallback_texts:
        full_text = "\n\n".join(fallback_texts)

    # Fallback: grab main content or the full body when selectors miss the page.
    if len(full_text.strip()) < 200:
        content = (
            soup.find("main")
            or soup.find("article")
            or soup.find(id=re.compile(r"(content|main)", re.I))
            or soup.find(class_=re.compile(r"(content|main)", re.I))
            or soup.find("body")
        )
        if content:
            full_text = content.get_text(separator="\n", strip=True)

    # Collapse excess whitespace
    full_text = re.sub(r'\n{3,}', '\n\n', full_text)
    full_text = re.sub(r' {2,}', ' ', full_text)
    return full_text.strip()


def extract_metadata_text(soup: BeautifulSoup) -> str:
    """Extract useful static text from page metadata when visible HTML is sparse."""
    texts = []
    seen = set()

    for tag in soup.find_all(["title", "h1", "h2", "h3"]):
        text = tag.get_text(separator=" ", strip=True)
        if text and text.lower() not in seen:
            seen.add(text.lower())
            texts.append(text)

    for tag in soup.find_all("meta"):
        name = (tag.get("name") or tag.get("property") or "").lower()
        if name not in {"description", "keywords", "og:title", "og:description"}:
            continue
        content = tag.get("content", "").strip()
        if len(content) > 20 and content.lower() not in seen:
            seen.add(content.lower())
            texts.append(content)

    return "\n\n".join(texts).strip()


def extract_jsonld_text(soup: BeautifulSoup) -> str:
    """Extract article/folder content from JSON-LD used by Vikaspedia."""
    texts = []

    def walk(value):
        if isinstance(value, dict):
            article_body = value.get("articleBody")
            if article_body:
                article_soup = BeautifulSoup(str(article_body), "html.parser")
                texts.append(article_soup.get_text(separator="\n", strip=True))
            description = value.get("description")
            if description and len(str(description)) > 40:
                texts.append(str(description))
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    for script in soup.find_all("script", type="application/ld+json"):
        raw_json = script.string or script.get_text()
        if not raw_json:
            continue
        try:
            walk(json.loads(raw_json))
        except json.JSONDecodeError:
            continue

    return "\n\n".join(texts).strip()


def discover_child_links(html: str, page_url: str, max_links: int = 20) -> list[str]:
    """Find same-folder content links from a Vikaspedia folder/index page."""
    soup = BeautifulSoup(html, "html.parser")
    base_path = urlparse(page_url).path.rstrip("/")
    links = []
    seen = set()

    for anchor in soup.find_all("a", href=True):
        child_url = urljoin(page_url, anchor["href"])
        maybe_add_child_link(child_url, page_url, base_path, seen, links)
        if len(links) >= max_links:
            break

    for match in re.findall(r'https?://[^"\s<>]+/viewcontent/[^"\s<>]+|/viewcontent/[^"\s<>]+', html):
        child_url = urljoin(page_url, match.replace("\\/", "/"))
        maybe_add_child_link(child_url, page_url, base_path, seen, links)
        if len(links) >= max_links:
            break

    for match in re.findall(r'"context_path"\s*:\s*"([^"]+)"', html):
        context_path = match.replace("\\/", "/")
        if not context_path.startswith("/"):
            continue
        child_url = urljoin(page_url, f"/viewcontent{context_path}")
        if urlparse(page_url).query and not urlparse(child_url).query:
            child_url = f"{child_url}?{urlparse(page_url).query}"
        maybe_add_child_link(child_url, page_url, base_path, seen, links)
        if len(links) >= max_links:
            break

    return links


def maybe_add_child_link(
    child_url: str,
    page_url: str,
    base_path: str,
    seen: set[str],
    links: list[str],
) -> None:
    """Add a discovered same-folder child link when it is useful."""
    parsed = urlparse(child_url)
    page_netloc = urlparse(page_url).netloc
    if parsed.netloc and parsed.netloc != page_netloc:
        return
    child_path = parsed.path.rstrip("/")
    if not child_path.startswith(base_path + "/"):
        return
    if child_path == base_path or child_url in seen:
        return
    if any(skip in child_url.lower() for skip in ("login", "edit", "download", "#")):
        return

    seen.add(child_url)
    links.append(child_url)


def extract_pdf_text(pdf_path: Path) -> tuple[str, int, int]:
    """Extract readable text from a downloaded PDF."""
    pages = []
    scanned_pages = 0

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        for page in pdf.pages:
            text = (page.extract_text() or "").strip()
            if len(text) < 30:
                scanned_pages += 1
                continue
            text = re.sub(r'\n{3,}', '\n\n', text)
            text = re.sub(r' {2,}', ' ', text)
            pages.append(text)

    return "\n\n".join(pages).strip(), total_pages, scanned_pages


def save_pdf_response(resp: requests.Response, name: str, suffix: str) -> Path:
    """Persist a PDF response for traceability and pdfplumber extraction."""
    pdf_path = PDF_DIR / f"{name}_{suffix}.pdf"
    if not pdf_path.exists():
        pdf_path.write_bytes(resp.content)
    return pdf_path


def url_id(url: str) -> str:
    """Short hash suffix to deduplicate multi-URL sources."""
    return hashlib.md5(url.encode()).hexdigest()[:6]


def already_scraped(out_path: Path) -> bool:
    """Skip if a usable file exists and was written today."""
    if not out_path.exists():
        return False
    mtime = datetime.fromtimestamp(out_path.stat().st_mtime)
    if mtime.date() != datetime.today().date():
        return False
    try:
        with open(out_path, encoding="utf-8") as f:
            doc = json.load(f)
        return doc.get("char_count", 0) >= 100
    except Exception:
        return False


def existing_char_count(out_path: Path) -> Optional[int]:
    """Read a previous scrape char count when the JSON is parseable."""
    if not out_path.exists():
        return None
    try:
        with open(out_path, encoding="utf-8") as f:
            doc = json.load(f)
        return int(doc.get("char_count", 0))
    except Exception:
        return None


def remove_stale_short_output(out_path: Path, char_count: int) -> None:
    """Delete short stale outputs so downstream cleaning does not ingest them."""
    if char_count >= 100 or not out_path.exists():
        return
    try:
        out_path.unlink()
        log.info(f"  Removed short output: {out_path.name}")
    except OSError as e:
        log.warning(f"  Could not remove short output {out_path.name}: {e}")


def prune_short_raw_outputs() -> None:
    """Remove previous failed scrape JSON files before building a new manifest."""
    for raw_dir in (RAW_CENTRAL, RAW_STATE):
        for out_path in raw_dir.glob("*.json"):
            char_count = existing_char_count(out_path)
            if char_count is not None and char_count < 100:
                remove_stale_short_output(out_path, char_count)


# ── Core scraper ───────────────────────────────────────────────────────────

def scrape_source(source: dict, out_dir: Path, delay: float = 2.5) -> list[dict]:
    """
    Scrape all URLs for a single source entry.
    Returns list of doc dicts written (one per URL).
    """
    name      = source["name"]
    display   = source.get("display", name)
    selectors = source.get("selectors", ["p", "li"])
    urls      = source["urls"]
    results   = []

    log.info(f"{'─'*60}")
    log.info(f"Scraping: {display}  ({len(urls)} URL(s))")

    for url in urls:
        suffix   = url_id(url)
        out_path = out_dir / f"{name}_{suffix}.json"

        if already_scraped(out_path):
            log.info(f"  ↩  Skipping (already scraped today): {out_path.name}")
            with open(out_path, encoding="utf-8") as f:
                results.append(json.load(f))
            continue
        previous_char_count = existing_char_count(out_path)
        if previous_char_count is not None and previous_char_count < 100:
            remove_stale_short_output(out_path, previous_char_count)

        log.info(f"  → {url}")
        resp = fetch_response(url)

        if not resp:
            log.error(f"  ✗ Failed to fetch: {url}")
            continue

        source_type = "html"
        total_pages = None
        scanned_pages = None

        if is_pdf_response(url, resp):
            source_type = "pdf"
            pdf_path = save_pdf_response(resp, name, suffix)
            try:
                text, total_pages, scanned_pages = extract_pdf_text(pdf_path)
            except Exception as e:
                log.error(f"  ✗ Failed to extract PDF text from {pdf_path.name}: {e}")
                text = ""
        else:
            resp.encoding = resp.apparent_encoding or "utf-8"
            text = extract_text(resp.text, selectors)

            if source.get("follow_links"):
                child_texts = []
                child_urls = discover_child_links(
                    resp.text,
                    resp.url or url,
                    max_links=int(source.get("max_child_links", 20)),
                )
                log.info(f"  Found {len(child_urls)} child scheme link(s)")

                for child_url in child_urls:
                    log.info(f"    ↳ {child_url}")
                    child_resp = fetch_response(child_url, retries=2)
                    if not child_resp:
                        continue
                    if is_pdf_response(child_url, child_resp):
                        child_suffix = url_id(child_url)
                        pdf_path = save_pdf_response(child_resp, name, child_suffix)
                        try:
                            child_text, _, _ = extract_pdf_text(pdf_path)
                        except Exception as e:
                            log.warning(f"    PDF child extraction failed: {e}")
                            continue
                    else:
                        child_resp.encoding = child_resp.apparent_encoding or "utf-8"
                        child_text = extract_text(child_resp.text, selectors)
                    if len(child_text) >= 100:
                        child_texts.append(f"Source: {child_url}\n{child_text}")
                    time.sleep(delay)

                if child_texts:
                    text = "\n\n".join([text, *child_texts]).strip()

        doc = {
            "id":          f"{name}_{suffix}",
            "name":        name,
            "display":     display,
            "url":         url,
            "category":    source.get("category", "scheme"),
            "state":       source.get("state", "central"),
            "priority":    source.get("priority", "medium"),
            "language":    source.get("language", ["english"]),
            "source_type":  source_type,
            "text":        text,
            "char_count":  len(text),
            "scraped_at":  datetime.now().isoformat(),
        }
        if total_pages is not None:
            doc["total_pages"] = total_pages
            doc["scanned_pages"] = scanned_pages

        if len(text) < 100:
            log.warning(f"  ⚠ Very short text ({len(text)} chars) — not saving corpus file")
            remove_stale_short_output(out_path, len(text))
            results.append(doc)
            continue

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)

        log.info(f"  ✓ {len(text):,} chars → {out_path.name}")
        results.append(doc)

        time.sleep(delay)   # be polite to govt servers

    return results


# ── Batch runners ──────────────────────────────────────────────────────────

def run_central(sources: list, only: Optional[str] = None, delay: float = 2.5) -> list:
    all_docs = []
    log.info(f"\n{'═'*60}")
    log.info(f"  CENTRAL SCHEME SCRAPERS  ({len(sources)} sources)")
    log.info(f"{'═'*60}")
    for s in sources:
        if only and s["name"] != only:
            continue
        docs = scrape_source(s, RAW_CENTRAL, delay=delay)
        all_docs.extend(docs)
    return all_docs


def run_state(sources: list, only: Optional[str] = None, delay: float = 2.5) -> list:
    all_docs = []
    log.info(f"\n{'═'*60}")
    log.info(f"  STATE SCHEME SCRAPERS  ({len(sources)} states)")
    log.info(f"{'═'*60}")
    for s in sources:
        if only and s["name"] != only:
            continue
        docs = scrape_source(s, RAW_STATE, delay=delay)
        all_docs.extend(docs)
    return all_docs


# ── Summary reporter ───────────────────────────────────────────────────────

def print_summary(all_docs: list):
    log.info(f"\n{'═'*60}")
    log.info("  SCRAPE SUMMARY")
    log.info(f"{'═'*60}")

    total_chars = sum(d.get("char_count", 0) for d in all_docs)
    failed      = [d for d in all_docs if d.get("char_count", 0) < 100]
    good        = [d for d in all_docs if d.get("char_count", 0) >= 100]

    log.info(f"  Total documents scraped : {len(all_docs)}")
    log.info(f"  Successful (≥100 chars) : {len(good)}")
    log.info(f"  Likely failed (<100 ch) : {len(failed)}")
    log.info(f"  Total chars collected   : {total_chars:,}")
    log.info(f"  Approx words            : {total_chars // 6:,}")

    if failed:
        log.warning("\n  Sites that may need Playwright (JS-rendered):")
        for d in failed:
            log.warning(f"    ✗ {d['display']:40s} → {d['url']}")

    # Save a manifest
    manifest_path = ROOT / "data" / "scrape_manifest.json"
    manifest = {
        "scraped_at":  datetime.now().isoformat(),
        "total_docs":  len(all_docs),
        "total_chars": total_chars,
        "documents":   [
            {
                "id":         d["id"],
                "name":       d["name"],
                "display":    d["display"],
                "state":      d.get("state", "central"),
                "char_count": d.get("char_count", 0),
                "url":        d["url"],
            }
            for d in all_docs
        ]
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    log.info(f"\n  Manifest saved → {manifest_path}")
    log.info(f"  Full log      → {log_file}")


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="KrishiNyay scheme scraper")
    parser.add_argument(
        "--type",
        choices=["central", "state", "all"],
        default="all",
        help="Which scraper set to run (default: all)"
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Scrape only a specific source by name, e.g. --name pmkisan"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.5,
        help="Seconds between requests (default: 2.5)"
    )
    args = parser.parse_args()

    with open(SOURCES_FILE, encoding="utf-8") as f:
        sources = json.load(f)

    prune_short_raw_outputs()

    all_docs = []

    if args.type in ("central", "all"):
        docs = run_central(sources["central_schemes"], only=args.name, delay=args.delay)
        all_docs.extend(docs)

    if args.type in ("state", "all"):
        docs = run_state(sources["state_schemes"], only=args.name, delay=args.delay)
        all_docs.extend(docs)

    print_summary(all_docs)
    log.info("\nDone. Run download_pdfs.py next for extra PDF-only sources.")


if __name__ == "__main__":
    main()
