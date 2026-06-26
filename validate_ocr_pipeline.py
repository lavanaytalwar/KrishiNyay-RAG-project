"""
Validate the Phase 6 OCR ingestion hooks.

This validator does not require Tesseract to be installed. It checks that the
normal PDF path still works, scanned/low-text pages are detected, and OCR
dependency status is reported deterministically.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from pypdf import PdfWriter

from ingest_manual_pdfs import extract_pdf_text, reject_non_pdf
from ocr_utils import check_ocr_dependencies


def make_blank_pdf(path: Path) -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=144, height=144)
    with path.open("wb") as handle:
        writer.write(handle)


def main() -> int:
    deps = check_ocr_dependencies()
    failures = []

    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = Path(tmpdir) / "blank_scan_like.pdf"
        make_blank_pdf(pdf_path)

        try:
            reject_non_pdf(pdf_path)
        except Exception as exc:
            failures.append(f"valid PDF rejected: {exc}")

        extracted = extract_pdf_text(pdf_path, use_ocr=False)
        if extracted["total_pages"] != 1:
            failures.append(f"expected 1 page, got {extracted['total_pages']}")
        if extracted["scanned_pages"] != 1:
            failures.append(f"expected 1 scanned page, got {extracted['scanned_pages']}")
        if extracted["char_count"] != 0:
            failures.append(f"expected blank PDF char_count=0, got {extracted['char_count']}")
        if extracted["ocr_enabled"]:
            failures.append("OCR should be disabled by default")

        if deps["available"]:
            ocr_extracted = extract_pdf_text(pdf_path, use_ocr=True, ocr_max_pages=1)
            if not ocr_extracted["ocr_enabled"]:
                failures.append("OCR flag was not preserved")
            if ocr_extracted["ocr_pages_attempted"] != 1:
                failures.append("OCR did not attempt the scanned page")

    print("OCR dependencies:")
    print(f"  available : {deps['available']}")
    print(f"  renderer  : {deps['renderer'] or 'missing'}")
    print(f"  engine    : {deps['engine'] or 'missing'}")
    if deps["missing"]:
        print(f"  missing   : {', '.join(deps['missing'])}")
    if not deps["available"]:
        print("  note      : OCR execution skipped; ingestion still detects scanned pages")

    if failures:
        print("\nFAILURES:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("\nOK: OCR ingestion hooks validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
