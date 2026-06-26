"""
Validate the Phase 6 OCR ingestion path.

The validator always checks scanned/low-text page detection. When Tesseract is
available, it also creates an image-only PDF fixture and asserts that actual OCR
extracts known text from it.
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


def make_image_only_pdf(path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (1800, 1200), "white")
    draw = ImageDraw.Draw(image)
    font = load_test_font(size=72)
    lines = [
        "KRISHINYAY OCR TEST",
        "FARMER CREDIT NOTICE 2026",
        "THIS PAGE IS IMAGE ONLY",
    ]
    y = 220
    for line in lines:
        draw.text((180, y), line, fill="black", font=font)
        y += 140
    image.save(path, "PDF", resolution=200.0)


def load_test_font(size: int):
    from PIL import ImageFont

    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


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
            fixture_path = Path(tmpdir) / "image_only_ocr_fixture.pdf"
            make_image_only_pdf(fixture_path)
            ocr_extracted = extract_pdf_text(
                fixture_path,
                use_ocr=True,
                ocr_language="eng",
                ocr_max_pages=1,
            )
            if not ocr_extracted["ocr_enabled"]:
                failures.append("OCR flag was not preserved")
            if ocr_extracted["ocr_pages_attempted"] != 1:
                failures.append("OCR did not attempt the scanned page")
            text = ocr_extracted["text"].upper()
            required_tokens = ["KRISHINYAY", "OCR", "FARMER", "CREDIT"]
            missing_tokens = [token for token in required_tokens if token not in text]
            if missing_tokens:
                failures.append(
                    "OCR fixture text missing token(s): "
                    + ", ".join(missing_tokens)
                    + f"; extracted={text[:200]!r}"
                )

    print("OCR dependencies:")
    print(f"  available : {deps['available']}")
    print(f"  renderer  : {deps['renderer'] or 'missing'}")
    print(f"  engine    : {deps['engine'] or 'missing'}")
    if deps["missing"]:
        print(f"  missing   : {', '.join(deps['missing'])}")
    if deps["available"]:
        print("  fixture   : image-only PDF OCR assertion ran")
    else:
        print("  note      : OCR execution skipped; ingestion still detects scanned pages")

    if failures:
        print("\nFAILURES:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("\nOK: OCR ingestion pipeline validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
