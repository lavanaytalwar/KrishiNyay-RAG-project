"""
Optional OCR helpers for scanned PDF pages.

The project can ingest normal text PDFs without OCR dependencies. OCR is only
used when manual ingestion is run with --ocr and the local environment has both
a PDF renderer and a Tesseract engine available.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def check_ocr_dependencies() -> dict[str, Any]:
    renderer_available = _module_available("pypdfium2")
    pytesseract_available = _module_available("pytesseract")
    tesseract_path = shutil.which("tesseract")

    engine = ""
    if pytesseract_available:
        engine = "pytesseract"
    elif tesseract_path:
        engine = "tesseract_cli"

    return {
        "available": renderer_available and bool(engine),
        "renderer": "pypdfium2" if renderer_available else "",
        "renderer_available": renderer_available,
        "engine": engine,
        "pytesseract_available": pytesseract_available,
        "tesseract_cli": tesseract_path or "",
        "missing": _missing_dependencies(renderer_available, engine),
    }


def ocr_pdf_pages(
    pdf_path: Path,
    page_numbers: list[int],
    language: str = "eng+hin",
    dpi: int = 200,
) -> dict[str, Any]:
    deps = check_ocr_dependencies()
    if not deps["available"]:
        missing = ", ".join(deps["missing"])
        raise RuntimeError(
            f"OCR dependencies unavailable ({missing}). Install pypdfium2, pytesseract, "
            "and the system Tesseract binary, then rerun with --ocr."
        )

    import pypdfium2 as pdfium

    pages = []
    with pdfium.PdfDocument(str(pdf_path)) as pdf:
        total_pages = len(pdf)
        for page_number in page_numbers:
            if page_number < 1 or page_number > total_pages:
                continue
            image = _render_page(pdf, page_number, dpi)
            text = _ocr_image(image, deps["engine"], language)
            text = _normalize_ocr_text(text)
            pages.append({
                "page": page_number,
                "text": text,
                "char_count": len(text),
                "extraction_method": "ocr_tesseract",
                "ocr_language": language,
                "ocr_dpi": dpi,
            })

    return {
        "pages": pages,
        "ocr_engine": deps["engine"],
        "ocr_renderer": deps["renderer"],
        "ocr_language": language,
        "ocr_dpi": dpi,
    }


def _module_available(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except Exception:
        return False


def _missing_dependencies(renderer_available: bool, engine: str) -> list[str]:
    missing = []
    if not renderer_available:
        missing.append("pypdfium2")
    if not engine:
        missing.append("pytesseract or tesseract CLI")
    return missing


def _render_page(pdf: Any, page_number: int, dpi: int):
    page = pdf[page_number - 1]
    bitmap = page.render(scale=dpi / 72)
    image = bitmap.to_pil().convert("RGB")
    try:
        page.close()
    except Exception:
        pass
    return image


def _ocr_image(image: Any, engine: str, language: str) -> str:
    if engine == "pytesseract":
        import pytesseract

        return pytesseract.image_to_string(image, lang=language, config="--psm 6")

    with tempfile.NamedTemporaryFile(suffix=".png") as image_file:
        image.save(image_file.name)
        completed = subprocess.run(
            ["tesseract", image_file.name, "stdout", "-l", language, "--psm", "6"],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout


def _normalize_ocr_text(text: str) -> str:
    text = re.sub(r"\r\n?", "\n", text or "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()
