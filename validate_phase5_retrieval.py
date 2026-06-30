"""
Validate the Phase 5 hybrid retrieval baseline against the Phase 4 eval set.

Run:
    python validate_phase5_retrieval.py
"""

import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent
MIN_DYNAMIC_ROUTE_ACCURACY = 1.00
MIN_STATIC_TOP1_SOURCE_TYPE = 0.70
MIN_STATIC_TOP3_SOURCE_TYPE = 0.95
MIN_TOPIC_TOP1_SOURCE_TYPE = 0.60

STATE_MARKERS = {
    "maharashtra": ("maharashtra", "namo shetkari", "shetkari"),
    "punjab": ("punjab",),
    "bihar": ("bihar",),
    "rajasthan": ("rajasthan",),
    "telangana": ("telangana", "rythu bandhu", "rythu"),
    "gujarat": ("gujarat",),
    "west bengal": ("west bengal", "krishak bandhu"),
    "madhya pradesh": ("madhya pradesh", "mp agriculture"),
}

TOPIC_MARKERS = {
    "pm_kisan": ("pm-kisan", "pmkisan", "kisan samman nidhi"),
    "pmfby": ("pmfby", "fasal bima", "crop insurance"),
    "kcc_credit": ("kisan credit card", "kcc"),
    "land_fra_legal": ("forest rights", "fra", "land acquisition", "larr", "rehabilitation"),
}

SPECIFIC_CHECKS = [
    {
        "label": "English land acquisition rights top-3",
        "question": "What rights does a land acquisition affected family have?",
        "expect_any": ["larr", "land acquisition", "forest rights", "fra"],
        "top_k": 3,
    },
    {
        "label": "Hinglish land acquisition rights top-1",
        "question": "Zameen khareedne par kisan ke kya adhikar hain?",
        "expect_any": ["larr", "land acquisition", "forest rights", "fra"],
        "top_k": 1,
    },
    {
        "label": "Maharashtra Namo Shetkari top-3",
        "question": "Maharashtra mein PM-KISAN ke upar extra paisa kaunsi scheme deti hai?",
        "expect_any": ["maharashtra", "namo shetkari"],
        "top_k": 3,
    },
    {
        "label": "Punjab state scheme not PM-KISAN top-1",
        "question": "Punjab de kisan nu state agriculture scheme vich kedi madad mil sakdi hai?",
        "expect_not_top1": ["pm-kisan", "pmkisan"],
        "expect_any": ["punjab"],
        "top_k": 3,
    },
    {
        "label": "PMFBY claim documents prefer PMFBY top-3",
        "question": "What documents should a farmer keep for a PMFBY claim?",
        "expect_any": ["pmfby", "fasal bima", "crop insurance"],
        "top_k": 3,
    },
]


def load_eval_items() -> list[dict[str, str]]:
    from validate_farmer_eval import load_eval_items as load_all_eval_items

    return load_all_eval_items(include_phase9=True)


def infer_source_type(result: dict[str, Any]) -> str:
    text = " ".join(
        str(result.get(key, ""))
        for key in ["display", "source", "url", "category", "state"]
    ).lower()
    category = str(result.get("category", "")).lower()
    state = str(result.get("state", "")).lower()

    if "live" in text or "portal" in category:
        return "live_portal"
    if category in {"soil", "infrastructure", "labour_rights", "market", "market_linkage"}:
        return "official_portal"
    if state and state not in {"central", "india", ""}:
        return "state_scheme"
    if "vikaspedia_schemes" in text and "schemesall/schemes-for-farmers" in text:
        return "official_portal"
    if any(marker in text for marker in ["soil health", "pmksy", "infrastructure fund", "mnrega"]):
        return "official_portal"
    if any(marker in text for marker in ["vikaspedia", "kisan credit card", "sample_kcc", "kcc"]):
        return "vikaspedia"
    if category in {"legal_rights", "insurance", "crop_insurance", "income_support", "legal"}:
        return "official_pdf"
    if any(marker in text for marker in ["pm-kisan", "pmkisan", "pmfby", "fasal bima", "larr", "land acquisition", "forest rights", "guidelines", "faq"]):
        return "official_pdf"
    if category in {"market", "market_linkage"}:
        return "official_portal"
    return "official_portal"


def text_blob(result: dict[str, Any]) -> str:
    return " ".join(
        str(result.get(key, ""))
        for key in ["display", "source", "url", "category", "state", "text"]
    ).lower()


def sig(result: Optional[dict[str, Any]]) -> tuple[str, str, str, str]:
    if not result:
        return ("", "", "", "")
    return (
        str(result.get("display", "")),
        str(result.get("source", "")),
        str(result.get("category", "")),
        str(result.get("state", "")),
    )


def source_type_hit(results: list[dict[str, Any]], expected: str, top_k: int) -> bool:
    return any(infer_source_type(result) == expected for result in results[:top_k])


def expected_state_markers(question: str) -> tuple[str, ...]:
    lowered = question.lower()
    for state, markers in STATE_MARKERS.items():
        if state in lowered or any(marker in lowered for marker in markers):
            return markers
    return ()


def top_k_blob(results: list[dict[str, Any]], top_k: int = 3) -> str:
    return " ".join(text_blob(result) for result in results[:top_k])


def source_guardrail_miss(item: dict[str, str], results: list[dict[str, Any]]) -> Optional[str]:
    blob = top_k_blob(results, 3)
    topic = item["topic"]
    markers = TOPIC_MARKERS.get(topic, ())
    if markers and not any(marker in blob for marker in markers):
        return f"{topic} top-3 missing topic marker {markers}"

    if topic == "state_schemes":
        markers = expected_state_markers(item["question"])
        if markers and not any(marker in blob for marker in markers):
            return f"state_schemes top-3 missing state marker {markers}"

    if item["expected_route"] == "rag" and item["expected_source_type"] != "live_portal" and results:
        top1_blob = text_blob(results[0])
        if any(marker in top1_blob for marker in ("beneficiary status", "payment status", "live_status")) and topic != "pm_kisan":
            return "static RAG top-3 was displaced by live-status portal content"

    return None


def main() -> int:
    sys.path.insert(0, str(ROOT))

    from app import route_dynamic_query
    from query_utils import normalize_query
    from vector_store import VectorStore

    items = load_eval_items()
    store = VectorStore()

    dynamic_total = dynamic_correct = 0
    static_total = top1_hits = top3_hits = 0
    topic_totals = Counter()
    topic_top1 = Counter()
    misses_by_topic = defaultdict(list)
    top1_changed = []
    improved = []
    regressed = []
    severe_regressions = []
    guardrail_failures = []

    for item in items:
        question = item["question"]
        expected_route = item["expected_route"]
        expected_source_type = item["expected_source_type"]
        topic = item["topic"]

        dynamic = route_dynamic_query(question)
        if expected_route == "dynamic_router":
            dynamic_total += 1
            if dynamic and dynamic.get("route") == "dynamic_router":
                dynamic_correct += 1
            else:
                misses_by_topic[topic].append((question, "route_miss", None, None))
            continue

        if dynamic:
            static_total += 1
            topic_totals[topic] += 1
            misses_by_topic[topic].append((question, "false_dynamic", None, None))
            continue

        static_total += 1
        topic_totals[topic] += 1
        normalized = normalize_query(question)
        vector_results = store.vector_query(normalized, n=3)
        hybrid_results = store.query(normalized, n=3)

        vector_top1_hit = source_type_hit(vector_results, expected_source_type, 1)
        hybrid_top1_hit = source_type_hit(hybrid_results, expected_source_type, 1)
        hybrid_top3_hit = source_type_hit(hybrid_results, expected_source_type, 3)

        if sig(vector_results[0] if vector_results else None) != sig(hybrid_results[0] if hybrid_results else None):
            top1_changed.append((question, sig(vector_results[0] if vector_results else None), sig(hybrid_results[0] if hybrid_results else None)))

        if hybrid_top1_hit:
            top1_hits += 1
            topic_top1[topic] += 1
        if hybrid_top3_hit:
            top3_hits += 1
        else:
            misses_by_topic[topic].append((question, expected_source_type, sig(hybrid_results[0] if hybrid_results else None), [infer_source_type(result) for result in hybrid_results]))

        guardrail_miss = source_guardrail_miss(item, hybrid_results)
        if guardrail_miss:
            guardrail_failures.append((question, guardrail_miss))
            misses_by_topic[topic].append((question, guardrail_miss, sig(hybrid_results[0] if hybrid_results else None), [sig(result) for result in hybrid_results[:3]]))

        if not vector_top1_hit and hybrid_top1_hit:
            improved.append((question, sig(vector_results[0] if vector_results else None), sig(hybrid_results[0] if hybrid_results else None)))
        if vector_top1_hit and not hybrid_top1_hit:
            regressed.append((question, sig(vector_results[0] if vector_results else None), sig(hybrid_results[0] if hybrid_results else None)))
            severe_regressions.append((question, expected_source_type, sig(vector_results[0] if vector_results else None), sig(hybrid_results[0] if hybrid_results else None)))

    dynamic_accuracy = dynamic_correct / dynamic_total if dynamic_total else 1.0
    top1_rate = top1_hits / static_total if static_total else 0.0
    top3_rate = top3_hits / static_total if static_total else 0.0

    print("=" * 70)
    print("  PHASE 5 RETRIEVAL VALIDATION")
    print("=" * 70)
    print(f"Items                     : {len(items)}")
    print(f"Dynamic route accuracy    : {dynamic_correct}/{dynamic_total} ({dynamic_accuracy:.0%})")
    print(f"Static top-1 source type  : {top1_hits}/{static_total} ({top1_rate:.0%})")
    print(f"Static top-3 source type  : {top3_hits}/{static_total} ({top3_rate:.0%})")
    print(f"Hybrid changed top-1      : {len(top1_changed)}")
    print(f"Hybrid improved top-1     : {len(improved)}")
    print(f"Hybrid regressed top-1    : {len(regressed)}")
    print()
    print("Topic top-1 source-type hit rates:")
    for topic, total in topic_totals.items():
        hits = topic_top1[topic]
        print(f"  {topic:22s} {hits:2d}/{total:<2d} ({hits / total:.0%})")

    if improved:
        print()
        print("Improved examples:")
        for question, before, after in improved[:5]:
            print(f"  + {question}")
            print(f"    vector={before}")
            print(f"    hybrid={after}")
    if regressed:
        print()
        print("Regressed examples:")
        for question, before, after in regressed[:5]:
            print(f"  - {question}")
            print(f"    vector={before}")
            print(f"    hybrid={after}")

    failures = []
    if dynamic_accuracy < MIN_DYNAMIC_ROUTE_ACCURACY:
        failures.append("dynamic route accuracy below 100%")
    if top1_rate < MIN_STATIC_TOP1_SOURCE_TYPE:
        failures.append("static top-1 source-type hit below 70%")
    if top3_rate < MIN_STATIC_TOP3_SOURCE_TYPE:
        failures.append("static top-3 source-type hit below 95%")
    for topic, total in topic_totals.items():
        if total and topic_top1[topic] / total < MIN_TOPIC_TOP1_SOURCE_TYPE:
            failures.append(f"topic {topic} top-1 below 60%")
    if severe_regressions:
        failures.append(f"{len(severe_regressions)} severe vector-to-hybrid top-1 regression(s)")
    if guardrail_failures:
        failures.append(f"{len(guardrail_failures)} source/category guardrail failure(s)")

    specific_failures = []
    for check in SPECIFIC_CHECKS:
        results = store.query(normalize_query(check["question"]), n=max(check.get("top_k", 3), 3))
        top_results = results[:check.get("top_k", 3)]
        blobs = [text_blob(result) for result in top_results]
        if any(token in text_blob(results[0]) for token in check.get("expect_not_top1", [])):
            specific_failures.append(f"{check['label']}: forbidden top-1 {sig(results[0])}")
            continue
        expected = check.get("expect_any", [])
        if expected and not any(any(token in blob for token in expected) for blob in blobs):
            specific_failures.append(f"{check['label']}: expected one of {expected}, got {[sig(result) for result in top_results]}")
    failures.extend(specific_failures)

    if failures:
        print()
        print("FAILURES:")
        for failure in failures:
            print(f"  - {failure}")
        print()
        print("Miss examples by topic:")
        for topic, misses in misses_by_topic.items():
            if misses:
                print(f"  {topic}:")
                for miss in misses[:4]:
                    print(f"    {miss}")
        return 1

    print()
    print("OK: Phase 5 retrieval gate passed")
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
