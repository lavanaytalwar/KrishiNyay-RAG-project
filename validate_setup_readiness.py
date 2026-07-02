"""
Validate local setup readiness for the hardened KrishiNyay demo.

Required checks fail. Optional field-readiness checks report clear setup actions
without blocking the baseline demo unless strict flags are used.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
from pathlib import Path

from ocr_utils import check_ocr_dependencies, installed_tesseract_languages


ROOT = Path(__file__).resolve().parent

REQUIRED_MODULES = [
    "fastapi",
    "pydantic",
    "requests",
    "chromadb",
    "sentence_transformers",
    "sklearn",
    "pypdf",
    "pypdfium2",
    "pytesseract",
    "PIL",
]

REQUIRED_PATHS = [
    ROOT / "eval" / "farmer_questions.jsonl",
    ROOT / "sample_data",
    ROOT / "web" / "index.html",
    ROOT / "web" / "app.js",
    ROOT / "web" / "styles.css",
    ROOT / "demo_chroma_db" / "chroma.sqlite3",
    ROOT / "demo_data" / "chunks" / "all_chunks.jsonl",
]

INDIC_OCR_LANGS = {"hin", "mar", "pan"}


def module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def ollama_model_available(model: str) -> tuple[bool, str]:
    try:
        import requests

        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        response.raise_for_status()
        models = [item["name"] for item in response.json().get("models", [])]
    except Exception as exc:
        return False, f"Ollama not reachable: {exc}"
    if model not in models:
        return False, f"Ollama running but missing {model}; run: ollama pull {model}"
    return True, f"Ollama model available: {model}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-indic-ocr", action="store_true", help="Fail if Hindi/Marathi/Punjabi OCR language data is missing")
    parser.add_argument("--strict-live-keys", action="store_true", help="Fail if the Agmarknet/Data.gov.in key is missing")
    parser.add_argument("--strict-ollama", action="store_true", help="Fail if local Ollama llama3.1:8b is unavailable")
    args = parser.parse_args()

    failures = []
    warnings = []

    for module_name in REQUIRED_MODULES:
        if not module_available(module_name):
            failures.append(f"missing Python module: {module_name}")

    for path in REQUIRED_PATHS:
        if not path.exists():
            failures.append(f"missing required path: {path.relative_to(ROOT)}")

    ocr = check_ocr_dependencies()
    if not ocr["available"]:
        failures.append(f"OCR baseline unavailable: {', '.join(ocr['missing'])}")

    languages = set(installed_tesseract_languages())
    missing_indic = sorted(INDIC_OCR_LANGS - languages)
    if missing_indic:
        message = (
            "Indic OCR language data missing: "
            + ", ".join(missing_indic)
            + ". Install Homebrew tesseract-lang or add the tessdata files."
        )
        if args.strict_indic_ocr:
            failures.append(message)
        else:
            warnings.append(message)

    key_configured = bool(os.environ.get("DATA_GOV_IN_API_KEY") or os.environ.get("AGMARKNET_API_KEY"))
    if not key_configured:
        message = "Mandi API key not configured; live mandi route will use safe official-portal fallback."
        if args.strict_live_keys:
            failures.append(message)
        else:
            warnings.append(message)

    live_ingest_enabled = os.environ.get("ENABLE_LIVE_INGEST", "").strip().lower() in {"1", "true", "yes", "on"}
    public_demo_enabled = os.environ.get("DEMO_PUBLIC", "").strip().lower() in {"1", "true", "yes", "on"}
    live_ingest_token = bool(os.environ.get("LIVE_INGEST_TOKEN", "").strip())
    if live_ingest_enabled and not live_ingest_token:
        warnings.append(
            "Live ingest is enabled without LIVE_INGEST_TOKEN; use only in trusted local demo/admin environments."
        )
    if public_demo_enabled and live_ingest_enabled:
        failures.append("DEMO_PUBLIC=true requires ENABLE_LIVE_INGEST=false")

    gemini_key_configured = bool(os.environ.get("GEMINI_API_KEY", "").strip())
    llm_provider = os.environ.get("LLM_PROVIDER", "auto").strip().lower() or "auto"
    if public_demo_enabled and llm_provider == "gemini" and not gemini_key_configured:
        failures.append("DEMO_PUBLIC=true with LLM_PROVIDER=gemini requires GEMINI_API_KEY")
    elif not gemini_key_configured:
        warnings.append("Gemini API key not configured; hosted public demo generation is not launch-ready.")

    model = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
    ollama_ok, ollama_message = ollama_model_available(model)
    if not ollama_ok:
        if args.strict_ollama:
            failures.append(ollama_message)
        else:
            warnings.append(ollama_message)

    if not shutil.which("node"):
        failures.append("node executable not found; frontend syntax gate cannot run")

    print("Setup readiness:")
    print(f"  repo          : {ROOT}")
    print(f"  OCR baseline  : {'available' if ocr['available'] else 'missing'}")
    print(f"  Tesseract langs: {', '.join(sorted(languages)) or 'none'}")
    print(f"  Mandi API key : {'configured' if key_configured else 'missing'}")
    print(f"  Gemini key    : {'configured' if gemini_key_configured else 'missing'}")
    print(f"  LLM provider  : {llm_provider}")
    print(f"  Public demo   : {'enabled' if public_demo_enabled else 'disabled'}")
    print(f"  Live ingest   : {'enabled' if live_ingest_enabled else 'disabled'}")
    print(f"  Ingest token  : {'configured' if live_ingest_token else 'missing'}")
    print(f"  Ollama        : {ollama_message}")
    print(f"  Node          : {subprocess.getoutput('node --version') if shutil.which('node') else 'missing'}")

    if warnings:
        print("\nWARNINGS:")
        for warning in warnings:
            print(f"  - {warning}")

    if failures:
        print("\nFAILURES:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("\nOK: setup readiness gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
