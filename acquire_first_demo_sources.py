#!/usr/bin/env python3
"""
Download the first-demo source shortlist into the ignored local data folder.

Outputs:
- data/first_demo_acquisition/pdfs/*.pdf
- data/first_demo_acquisition/raw_html/*.html
- data/first_demo_acquisition/raw_text/*.txt
- data/first_demo_acquisition/acquisition_report.json
- data/manual_pdfs/manifest.json

The script intentionally keeps downloaded source artifacts under data/, which is
git-ignored. It is meant for reproducible local acquisition, not for committing
third-party or generated content.
"""

from __future__ import annotations

import json
import re
import shutil
import ssl
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "first_demo_acquisition"
PDF_DIR = OUT / "pdfs"
HTML_DIR = OUT / "raw_html"
DATA_DIR = OUT / "raw_data"
TEXT_DIR = OUT / "raw_text"
MANUAL_PDF_DIR = ROOT / "data" / "manual_pdfs" / "staging"
MANUAL_MANIFEST = ROOT / "data" / "manual_pdfs" / "manifest.json"
REPORT_PATH = OUT / "acquisition_report.json"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
)
MYSCHEME_API_KEY = "tYTy5eEhlu9rFjyxuCr7ra7ACp4dv1RH8gWuHTDc"


@dataclass(frozen=True)
class Source:
    name: str
    display: str
    url: str
    kind: str
    category: str
    state: str
    language: str
    priority: int
    use: str
    source_note: str = ""


SOURCES: list[Source] = [
    Source(
        "pmkisan_operational_guidelines",
        "PM-KISAN Operational Guidelines",
        "https://pmkisan.gov.in/Documents/RevisedPM-KISANOperationalGuidelines%28English%29.pdf",
        "pdf",
        "scheme",
        "central",
        "en",
        0,
        "rag_grounding",
    ),
    Source(
        "pmkisan_revised_faq",
        "PM-KISAN Revised FAQ",
        "https://pmkisan.gov.in/Documents/RevisedFAQ.pdf",
        "pdf",
        "scheme",
        "central",
        "en",
        0,
        "rag_grounding",
    ),
    Source(
        "pmkisan_kcc_form",
        "PM-KISAN KCC Form",
        "https://pmkisan.gov.in/Documents/Kcc.pdf",
        "pdf",
        "credit",
        "central",
        "en",
        0,
        "rag_grounding",
    ),
    Source(
        "pmfby_operational_guidelines",
        "PMFBY Operational Guidelines",
        "https://pmfby.amnex.co.in/pmfby/pdf/operational_guidelines_pmfby.pdf",
        "pdf",
        "insurance",
        "central",
        "en",
        0,
        "rag_grounding",
    ),
    Source(
        "fra_act_rules_guidelines",
        "Forest Rights Act, Rules and Guidelines Booklet",
        "https://tribal.nic.in/downloads/FRA/FRAActnRulesBook.pdf",
        "pdf",
        "legal_rights",
        "central",
        "en",
        0,
        "rag_grounding",
    ),
    Source(
        "land_acquisition_larr_act_2013",
        "Land Acquisition, Rehabilitation and Resettlement Act, 2013",
        "https://www.indiacode.nic.in/bitstream/123456789/2121/1/A2013-30.pdf",
        "pdf",
        "legal_rights",
        "central",
        "en",
        1,
        "rag_grounding",
    ),
    Source(
        "pmksy_micro_irrigation_guidelines",
        "PMKSY Per Drop More Crop Micro Irrigation Guidelines",
        "https://pmksy.gov.in/MicroIrrigation/Archive/GuidelinesMIRevised250817.pdf",
        "pdf",
        "irrigation",
        "central",
        "en",
        1,
        "rag_grounding",
    ),
    Source(
        "pmksy_faq",
        "PMKSY FAQ",
        "https://pmksy.gov.in/pdfLinks/FAQ.pdf",
        "pdf",
        "irrigation",
        "central",
        "en",
        1,
        "rag_grounding",
    ),
    Source(
        "pmksy_pdmc_myscheme",
        "myScheme - PMKSY Per Drop More Crop",
        "https://www.myscheme.gov.in/schemes/pmksy-pdmc",
        "html",
        "irrigation",
        "central",
        "en",
        1,
        "rag_grounding",
        "Fallback raw text source while pmksy.gov.in returns HTTP 503 to CLI and Chrome.",
    ),
    Source(
        "enam_operational_guidelines",
        "e-NAM Operational Guidelines",
        "https://enam.gov.in/web/assest/download/Revised-Operational-Guidelines-of-e-NAM.pdf",
        "pdf",
        "market",
        "central",
        "en",
        1,
        "rag_grounding",
    ),
    Source(
        "enam_faq",
        "e-NAM Farmer FAQ",
        "https://enam.gov.in/web/resources/FAQs-of-eNam",
        "html",
        "market",
        "central",
        "en",
        1,
        "rag_grounding",
    ),
    Source(
        "soil_health_portal",
        "Soil Health Card Portal",
        "https://soilhealth.dac.gov.in/",
        "html",
        "soil_health",
        "central",
        "en",
        1,
        "rag_grounding",
    ),
    Source(
        "soil_health_card_myscheme",
        "myScheme - Soil Health Card",
        "https://www.myscheme.gov.in/schemes/shc",
        "html",
        "soil_health",
        "central",
        "en",
        1,
        "rag_grounding",
        "Fallback raw text source while soilhealth.dac.gov.in serves mostly app shell text.",
    ),
    Source(
        "soil_health_card_myscheme_json",
        "myScheme API - Soil Health Card",
        "https://api.myscheme.gov.in/schemes/v6/public/schemes?slug=shc&lang=en",
        "json",
        "soil_health",
        "central",
        "en",
        1,
        "rag_grounding",
        "Structured public myScheme API payload.",
    ),
    Source(
        "vikaspedia_agriculture_schemes",
        "Vikaspedia Agriculture Policies and Schemes",
        "https://vikaspedia.in/agriculture/policies-and-schemes",
        "html",
        "scheme",
        "central",
        "en",
        0,
        "eval_and_supplemental",
    ),
    Source(
        "punjab_agriculture_portal",
        "Punjab Agriculture Department Portal",
        "https://agri.punjab.gov.in/",
        "html",
        "state_scheme",
        "punjab",
        "en",
        1,
        "rag_grounding",
    ),
    Source(
        "haryana_agriculture_portal",
        "Haryana Agriculture Department Portal",
        "https://agriharyana.gov.in/",
        "html",
        "state_scheme",
        "haryana",
        "en",
        1,
        "rag_grounding",
    ),
    Source(
        "maharashtra_agriculture_portal",
        "Maharashtra Agriculture Department Portal",
        "https://krishi.maharashtra.gov.in/",
        "html",
        "state_scheme",
        "maharashtra",
        "mr",
        1,
        "rag_grounding",
    ),
    Source(
        "access_agriculture_turmeric_en",
        "Access Agriculture - Growing and Processing Turmeric",
        "https://www.accessagriculture.org/growing-and-processing-turmeric",
        "html",
        "crop_advisory",
        "central",
        "en",
        1,
        "eval_only",
        "Access Agriculture marks the page all rights reserved; keep out of the committed corpus until license review.",
    ),
    Source(
        "access_agriculture_turmeric_hi",
        "Access Agriculture - Turmeric Hindi",
        "https://www.accessagriculture.org/hi/growing-and-processing-turmeric",
        "html",
        "crop_advisory",
        "central",
        "hi",
        1,
        "eval_only",
        "Access Agriculture marks the page all rights reserved; keep out of the committed corpus until license review.",
    ),
    Source(
        "access_agriculture_turmeric_mr",
        "Access Agriculture - Turmeric Marathi",
        "https://www.accessagriculture.org/mr/growing-and-processing-turmeric",
        "html",
        "crop_advisory",
        "maharashtra",
        "mr",
        1,
        "eval_only",
        "Access Agriculture marks the page all rights reserved; keep out of the committed corpus until license review.",
    ),
    Source(
        "digital_green_home",
        "Digital Green",
        "https://www.digitalgreen.org/",
        "html",
        "crop_advisory",
        "central",
        "en",
        2,
        "research_reference",
        "Public website only. Do not assume access to datasets or transcripts without permission.",
    ),
    Source(
        "cabi_plantwiseplus_public",
        "CABI PlantwisePlus Public Page",
        "https://www.cabi.org/plantwiseplus/",
        "html",
        "crop_advisory",
        "central",
        "en",
        2,
        "research_reference",
        "Accessible public CABI page. The PlantwisePlus Knowledge Bank itself may still block automated access.",
    ),
    Source(
        "cabi_plantwiseplus_knowledge_bank",
        "CABI PlantwisePlus Knowledge Bank",
        "https://plantwiseplusknowledgebank.org/",
        "html",
        "crop_advisory",
        "central",
        "en",
        2,
        "research_reference",
        "May block automated access or require license review.",
    ),
    Source(
        "aaqua_forum",
        "aAQUA Farmer Q&A Forum",
        "https://aaqua.persistent.co.in/aaqua/forum/index",
        "html",
        "farmer_questions",
        "central",
        "mixed",
        2,
        "eval_only",
        "Use only if accessible and license permits reuse.",
    ),
    Source(
        "agrigov_paper",
        "AgriGov Dataset Paper",
        "https://arxiv.org/pdf/2606.08272",
        "pdf",
        "scheme",
        "central",
        "en",
        2,
        "research_reference",
        "Paper only unless the actual dataset download and license are found.",
    ),
    Source(
        "kisanqrs_kcc_research_paper",
        "KisanQRS Kisan Call Centre Research Paper",
        "https://arxiv.org/pdf/2411.08883",
        "pdf",
        "farmer_questions",
        "central",
        "en",
        2,
        "research_reference",
        "Paper describes KCC call-log experiments; raw KCC logs are not assumed public.",
    ),
]


class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.skip_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self.skip_depth += 1
        if tag in {"p", "br", "li", "h1", "h2", "h3", "h4", "tr"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self.skip_depth:
            self.skip_depth -= 1
        if tag in {"p", "li", "h1", "h2", "h3", "h4", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self.skip_depth:
            return
        stripped = re.sub(r"\s+", " ", data).strip()
        if stripped:
            self.parts.append(stripped)

    def text(self) -> str:
        text = "\n".join(self.parts)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def safe_suffix(url: str, content_type: str | None, kind: str) -> str:
    if kind == "pdf":
        return ".pdf"
    if kind == "json":
        return ".json"
    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix
    if suffix and len(suffix) < 8:
        return suffix
    if content_type and "json" in content_type:
        return ".json"
    return ".html"


def fetch(url: str, timeout: int = 45) -> tuple[bytes, dict[str, str], str]:
    headers = {"User-Agent": USER_AGENT}
    if "api.myscheme.gov.in" in url:
        headers["x-api-key"] = MYSCHEME_API_KEY
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=timeout) as response:
            data = response.read()
            headers = {k.lower(): v for k, v in response.headers.items()}
            return data, headers, response.geturl()
    except URLError as exc:
        reason = str(getattr(exc, "reason", exc))
        if "CERTIFICATE_VERIFY_FAILED" not in reason:
            raise
        context = ssl._create_unverified_context()
        with urlopen(request, timeout=timeout, context=context) as response:
            data = response.read()
            headers = {k.lower(): v for k, v in response.headers.items()}
            return data, headers, response.geturl()


def extract_pdf_text(path: Path) -> str:
    try:
        import pdfplumber  # type: ignore
    except Exception:
        return ""

    pages: list[str] = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                pages.append(page.extract_text() or "")
    except Exception:
        return ""
    return "\n\n".join(page for page in pages if page.strip()).strip()


def extract_html_text(data: bytes) -> str:
    html = data.decode("utf-8", errors="replace")
    parser = TextExtractor()
    parser.feed(html)
    return parser.text()


def extract_json_text(data: bytes) -> str:
    try:
        payload = json.loads(data.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return ""

    parts: list[str] = []

    def walk(value: object) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                if isinstance(nested, (str, int, float, bool)) and nested not in {"", None}:
                    parts.append(f"{key}: {nested}")
                else:
                    walk(nested)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, str) and value.strip():
            parts.append(value.strip())

    walk(payload)
    text = "\n".join(parts)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def ensure_dirs() -> None:
    for directory in [PDF_DIR, HTML_DIR, DATA_DIR, TEXT_DIR, MANUAL_PDF_DIR, MANUAL_MANIFEST.parent]:
        directory.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def acquire_source(source: Source) -> dict[str, object]:
    record: dict[str, object] = {
        **asdict(source),
        "status": "pending",
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        data, headers, final_url = fetch(source.url)
        content_type = headers.get("content-type", "")
        suffix = safe_suffix(final_url, content_type, source.kind)
        if source.kind == "pdf":
            target_dir = PDF_DIR
        elif source.kind == "json":
            target_dir = DATA_DIR
        else:
            target_dir = HTML_DIR
        raw_path = target_dir / f"{source.name}{suffix}"
        raw_path.write_bytes(data)

        is_pdf = data[:5] == b"%PDF-"
        if source.kind == "pdf" and not is_pdf:
            record.update(
                {
                    "status": "failed",
                    "error": "Downloaded content is not a PDF",
                    "content_type": content_type,
                    "final_url": final_url,
                    "raw_path": str(raw_path.relative_to(ROOT)),
                    "bytes": len(data),
                }
            )
            return record

        if source.kind == "pdf":
            text = extract_pdf_text(raw_path)
        elif source.kind == "json":
            text = extract_json_text(data)
        else:
            text = extract_html_text(data)
        text_path = TEXT_DIR / f"{source.name}.txt"
        text_path.write_text(text, encoding="utf-8")

        record.update(
            {
                "status": "ok",
                "content_type": content_type,
                "final_url": final_url,
                "raw_path": str(raw_path.relative_to(ROOT)),
                "text_path": str(text_path.relative_to(ROOT)),
                "bytes": len(data),
                "text_chars": len(text),
            }
        )
        return record
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        record.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        return record


def build_manual_manifest(records: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for record in records:
        if record.get("status") != "ok" or record.get("kind") != "pdf":
            continue
        if int(record.get("text_chars") or 0) < 500:
            continue
        use = str(record.get("use", ""))
        if use not in {"rag_grounding"}:
            continue
        source_path = ROOT / str(record["raw_path"])
        staged_path = MANUAL_PDF_DIR / source_path.name
        shutil.copy2(source_path, staged_path)
        entries.append(
            {
                "name": record["name"],
                "display": record["display"],
                "file": str(staged_path.relative_to(ROOT)),
                "url": record.get("final_url") or record.get("url"),
                "category": record["category"],
                "state": record["state"],
                "language": record["language"],
                "priority": record["priority"],
            }
        )
    return entries


def main() -> int:
    ensure_dirs()
    records: list[dict[str, object]] = []
    for index, source in enumerate(SOURCES, start=1):
        print(f"[{index:02d}/{len(SOURCES)}] {source.name} ...", flush=True)
        record = acquire_source(source)
        records.append(record)
        print(f"  -> {record['status']} ({record.get('text_chars', 0)} chars)", flush=True)
        time.sleep(0.5)

    manual_manifest = build_manual_manifest(records)
    write_json(MANUAL_MANIFEST, manual_manifest)
    write_json(
        REPORT_PATH,
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_sources": len(SOURCES),
            "ok": sum(1 for record in records if record.get("status") == "ok"),
            "failed": sum(1 for record in records if record.get("status") != "ok"),
            "manual_manifest": str(MANUAL_MANIFEST.relative_to(ROOT)),
            "manual_manifest_entries": len(manual_manifest),
            "records": records,
        },
    )
    print(f"\nReport: {REPORT_PATH.relative_to(ROOT)}")
    print(f"Manual PDF manifest: {MANUAL_MANIFEST.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
