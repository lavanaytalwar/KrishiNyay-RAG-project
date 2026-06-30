"""Turn-level answer language policy for KrishiNyay.

The workflow may carry topic memory across turns, but answer language should
come from the latest user turn when that turn is explicit. Ambiguous short
follow-ups can inherit the previous turn's language.
"""

from __future__ import annotations

import re
from typing import Optional


ANSWER_LANGUAGE_ENGLISH = "english"
ANSWER_LANGUAGE_HINGLISH = "hinglish"

STRONG_HINGLISH_MARKERS = {
    "aaj",
    "aur",
    "baarish",
    "barish",
    "batao",
    "chahiye",
    "hai",
    "hain",
    "hoga",
    "hogi",
    "ka",
    "kab",
    "kaise",
    "kal",
    "karu",
    "kare",
    "kaun",
    "ke",
    "kya",
    "liye",
    "milega",
    "mera",
    "meri",
    "mujhe",
    "paus",
    "sakta",
    "zameen",
}

CONTEXTUAL_HINGLISH_MARKERS = {
    "aloo",
    "bhav",
    "chana",
    "dhan",
    "fasal",
    "gehu",
    "gehun",
    "kapas",
    "kisan",
    "makka",
    "mandi",
    "mausam",
    "pyaz",
    "sarson",
}

ENGLISH_MARKERS = {
    "after",
    "application",
    "can",
    "claim",
    "crop",
    "damage",
    "documents",
    "do",
    "does",
    "eligible",
    "eligibility",
    "flood",
    "how",
    "insurance",
    "land",
    "please",
    "price",
    "rain",
    "rate",
    "rights",
    "should",
    "spray",
    "spraying",
    "tell",
    "today",
    "tomorrow",
    "weather",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
}


def normalise_answer_language(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    normalized = value.strip().lower().replace("_", "-")
    if normalized in {"english", "en", "english-only"}:
        return ANSWER_LANGUAGE_ENGLISH
    if normalized in {"hindi", "hinglish", "hi", "hindi-hinglish", "roman-hindi"}:
        return ANSWER_LANGUAGE_HINGLISH
    return None


def _has_devanagari(text: str) -> bool:
    return any("\u0900" <= char <= "\u097F" for char in text)


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower()))


def detect_answer_language(question: str, fallback: Optional[str] = None) -> str:
    normalized_fallback = normalise_answer_language(fallback)
    if _has_devanagari(question):
        return ANSWER_LANGUAGE_HINGLISH

    words = _words(question)
    if words & STRONG_HINGLISH_MARKERS:
        return ANSWER_LANGUAGE_HINGLISH
    if words & ENGLISH_MARKERS:
        return ANSWER_LANGUAGE_ENGLISH
    if words & CONTEXTUAL_HINGLISH_MARKERS:
        return normalized_fallback or ANSWER_LANGUAGE_HINGLISH
    return normalized_fallback or ANSWER_LANGUAGE_ENGLISH


def is_hinglish_language(answer_language: Optional[str]) -> bool:
    return normalise_answer_language(answer_language) == ANSWER_LANGUAGE_HINGLISH


def prompt_language_instruction(answer_language: str) -> str:
    normalized = normalise_answer_language(answer_language) or ANSWER_LANGUAGE_ENGLISH
    if normalized == ANSWER_LANGUAGE_ENGLISH:
        return "English only"
    return "Hindi/Hinglish in simple Roman Hindi, matching the user's wording"


def requires_english_answer(answer_language: Optional[str]) -> bool:
    return normalise_answer_language(answer_language) == ANSWER_LANGUAGE_ENGLISH
