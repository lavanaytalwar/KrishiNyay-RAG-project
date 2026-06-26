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
    (r"\bsoil health card\b", "soil health card soil nutrient fertilizer recommendation"),
    (r"\bmitti\b", "soil mitti nutrient soil health card fertilizer recommendation"),
    (r"\bsoil test\b", "soil health card soil test nutrient fertilizer recommendation"),
    (r"\bfertilizer\b", "soil nutrient fertilizer recommendation"),
    (r"\bpmksy\b", "pradhan mantri krishi sinchayee yojana pmksy micro irrigation per drop more crop"),
    (r"\bdrip\b", "drip irrigation micro irrigation per drop more crop pmksy"),
    (r"\bmicro[\s-]?irrigation\b", "micro irrigation per drop more crop pmksy"),
    (r"\bpani bachane\b", "water saving micro irrigation drip pmksy per drop more crop"),
    (r"\bfarm pond\b", "agriculture infrastructure fund farm pond storage infrastructure"),
    (r"\bstorage infrastructure\b", "agriculture infrastructure fund storage post harvest infrastructure"),
    (r"\bagriculture infrastructure\b", "agriculture infrastructure fund post harvest infrastructure"),
    (r"\bmnrega\b", "mgnrega mnrega farm labour employment rights"),
    (r"\bmgnrega\b", "mgnrega mnrega farm labour employment rights"),
    (r"\bfarm labour\b", "mgnrega farm labour employment rights"),
    (r"\bpest\b", "pest pesticide insecticide crop advisory"),
    (r"\bpesticide\b", "pest pesticide insecticide crop advisory"),
    (r"\bbhav\b", "mandi price bhav"),
    (r"\bmuavza\b", "compensation muavza"),
    (r"\bzameen\b", "land zameen"),
    (r"\bbhoomi\b", "land bhoomi"),
    (r"\bjameen\b", "land jameen"),
    (r"\badhikar\b", "rights entitlement compensation rehabilitation"),
    (r"\bkhareed\w*\b", "acquisition purchase land rights compensation"),
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
