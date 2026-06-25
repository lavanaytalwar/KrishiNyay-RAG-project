"""
Query utilities shared by FastAPI, RAG retrieval, and validation.
"""

import re


NORMALIZATION_RULES = [
    (r"\baadhar\b", "aadhaar"),
    (r"\bpm[\s_]?kisan\b", "pm-kisan"),
    (r"\bpmfby\b", "pradhan mantri fasal bima yojana pmfby crop insurance"),
    (r"\bfasal bima\b", "pradhan mantri fasal bima yojana crop insurance"),
    (r"\bkcc\b", "kisan credit card kcc"),
    (r"\bbhav\b", "mandi price bhav"),
    (r"\bmuavza\b", "compensation muavza"),
    (r"\bzameen\b", "land zameen"),
    (r"\bbhoomi\b", "land bhoomi"),
    (r"\bjameen\b", "land jameen"),
    (r"\bbarbad\b", "damage loss barbad"),
    (r"\bbarbaad\b", "damage loss barbaad"),
    (r"\bpaise\b", "payment money paise"),
    (r"\bpaisa\b", "payment money paisa"),
]


def normalize_query(question: str) -> str:
    """Add lightweight Hindi/Hinglish retrieval hints without changing intent."""
    normalized = re.sub(r"\s+", " ", question.strip())
    expanded = normalized.lower()
    additions = []
    for pattern, replacement in NORMALIZATION_RULES:
        if re.search(pattern, expanded, flags=re.I):
            additions.append(replacement)
    if not additions:
        return normalized
    return f"{normalized} {' '.join(dict.fromkeys(additions))}"


def canonical_for_routing(question: str) -> str:
    """Normalize only spelling/spacing for route detection, not retrieval hints."""
    normalized = re.sub(r"\s+", " ", question.strip().lower())
    normalized = re.sub(r"\baadhar\b", "aadhaar", normalized)
    normalized = re.sub(r"\bpm[\s_]?kisan\b", "pm-kisan", normalized)
    return normalized
