"""
Validate answer generation on top of the existing retrieval pipeline.

This gate is intentionally separate from retrieval validation. It requires a
real local Ollama model and fails with setup instructions instead of silently
accepting template fallback output.

Run:
    python validate_generation.py --provider ollama
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class GenerationCase:
    question: str
    language: str
    expected_source_markers: tuple[str, ...]


GENERATION_CASES = [
    GenerationCase(
        question="PM-KISAN ke liye kaun eligible hai?",
        language="hinglish",
        expected_source_markers=("pm-kisan", "pmkisan", "kisan samman nidhi"),
    ),
    GenerationCase(
        question="How do I claim PMFBY crop insurance after flood damage?",
        language="english",
        expected_source_markers=("pmfby", "fasal bima", "crop insurance"),
    ),
    GenerationCase(
        question="Zameen khareedne par kisan ke kya adhikar hain?",
        language="hinglish",
        expected_source_markers=("land acquisition", "larr", "rehabilitation", "forest rights"),
    ),
]


def ollama_models() -> list[str]:
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        response.raise_for_status()
    except Exception as exc:
        raise RuntimeError(
            "Ollama is not reachable at http://localhost:11434. "
            "Install/start Ollama, then run: ollama pull llama3.1:8b"
        ) from exc
    return [model["name"] for model in response.json().get("models", [])]


def fail_setup(message: str) -> int:
    print("GENERATION SETUP REQUIRED")
    print(message)
    print()
    print("Expected local setup:")
    print("  1. Install/start Ollama: https://ollama.com")
    print("  2. Pull the model: ollama pull llama3.1:8b")
    print("  3. Re-run: python validate_generation.py --provider ollama")
    return 2


def contains_template_fallback(answer: str) -> bool:
    lowered = answer.lower()
    return "[retrieved from:" in lowered or "template answer" in lowered


def source_blob(sources: list[dict]) -> str:
    return " ".join(
        str(source.get(key, ""))
        for source in sources
        for key in ["display", "source", "category", "state", "text"]
    ).lower()


def answer_looks_language_compatible(answer: str, language: str) -> bool:
    if language == "english":
        devanagari_chars = sum(1 for char in answer if "\u0900" <= char <= "\u097F")
        return devanagari_chars <= max(4, len(answer) // 20)
    return True


def validate_answer(case: GenerationCase, result: dict, max_answer_chars: int) -> list[str]:
    answer = str(result.get("answer", "")).strip()
    sources = result.get("sources", [])
    errors = []

    if result.get("llm_provider") == "template":
        errors.append("llm_provider is template")
    if contains_template_fallback(answer):
        errors.append("answer used template fallback")
    if len(answer) < 60:
        errors.append(f"answer too short ({len(answer)} chars)")
    if len(answer) > max_answer_chars:
        errors.append(f"answer too long ({len(answer)} chars)")
    if not sources:
        errors.append("no retrieved sources returned")
    if not answer_looks_language_compatible(answer, case.language):
        errors.append("answer language does not match question")

    support = source_blob(sources)
    if not any(marker in support for marker in case.expected_source_markers):
        errors.append("retrieved sources do not match expected topic markers")

    return errors


def run_ollama_gate(model: str, max_answer_chars: int) -> int:
    try:
        models = ollama_models()
    except RuntimeError as exc:
        return fail_setup(str(exc))

    if model not in models:
        return fail_setup(
            f"Ollama is running, but model '{model}' is not installed. "
            f"Available models: {models or 'none'}"
        )

    os.environ["OLLAMA_MODEL"] = model

    sys.path.insert(0, str(ROOT))
    from rag_chain import RAGChain

    chain = RAGChain(n_results=4)
    if chain.llm_provider != f"ollama:{model}":
        print(f"FAIL: expected llm_provider=ollama:{model}, got {chain.llm_provider}")
        return 1

    failures = []
    for case in GENERATION_CASES:
        result = chain.ask(case.question)
        errors = validate_answer(case, result, max_answer_chars)
        if errors:
            failures.append((case.question, errors, result.get("answer", "")))
        else:
            print(f"✓ {case.question}")
            print(f"  provider={result['llm_provider']} sources={len(result['sources'])}")

    if failures:
        print()
        print("FAILURES:")
        for question, errors, answer in failures:
            print(f"  - {question}")
            print(f"    errors: {', '.join(errors)}")
            print(f"    answer: {answer[:300]!r}")
        return 1

    print()
    print(f"OK: Ollama generation gate passed with {model}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate local model answer generation")
    parser.add_argument("--provider", choices=["ollama"], default="ollama")
    parser.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", "llama3.1:8b"))
    parser.add_argument("--max-answer-chars", type=int, default=1200)
    args = parser.parse_args()

    if args.provider == "ollama":
        return run_ollama_gate(args.model, args.max_answer_chars)
    raise AssertionError(f"Unsupported provider: {args.provider}")


if __name__ == "__main__":
    raise SystemExit(main())
