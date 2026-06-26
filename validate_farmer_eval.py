"""
Validate the Phase 4 farmer-facing evaluation dataset.

Run:
    python validate_farmer_eval.py
    python validate_farmer_eval.py --spot-check
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "eval" / "farmer_questions.jsonl"

REQUIRED_KEYS = {
    "question",
    "language",
    "topic",
    "expected_route",
    "expected_source_type",
    "reference_answer",
    "source_basis",
}

EXPECTED_TOPIC_COUNTS = {
    "pm_kisan": 16,
    "pmfby": 16,
    "kcc_credit": 12,
    "land_fra_legal": 12,
    "state_schemes": 16,
    "mandi_weather_live": 14,
    "crop_soil_advisory": 14,
}

EXPECTED_LANGUAGE_COUNTS = {
    "hinglish": 45,
    "english": 35,
    "regional_romanized": 20,
}

ALLOWED_ROUTES = {"rag", "dynamic_router"}
ALLOWED_SOURCE_TYPES = {
    "official_pdf",
    "official_portal",
    "vikaspedia",
    "state_scheme",
    "live_portal",
}

MIN_ITEMS = 100
SPOT_CHECK_TARGET = 15


def load_items(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise ValueError(f"Missing dataset: {path}")

    items = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"Line {line_no}: blank lines are not allowed")
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Line {line_no}: invalid JSON: {exc}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"Line {line_no}: expected a JSON object")
        items.append(item)
    return items


def validate_shape(items: list[dict[str, Any]]) -> None:
    if len(items) < MIN_ITEMS:
        raise ValueError(f"Expected at least {MIN_ITEMS} items, found {len(items)}")

    questions = Counter()
    topics = Counter()
    languages = Counter()
    routes = Counter()
    source_types = Counter()

    for index, item in enumerate(items, 1):
        keys = set(item)
        missing = REQUIRED_KEYS - keys
        extra = keys - REQUIRED_KEYS
        if missing:
            raise ValueError(f"Item {index}: missing keys: {sorted(missing)}")
        if extra:
            raise ValueError(f"Item {index}: unexpected keys: {sorted(extra)}")

        for key in REQUIRED_KEYS:
            value = item[key]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Item {index}: {key} must be a non-empty string")

        topic = item["topic"]
        language = item["language"]
        route = item["expected_route"]
        source_type = item["expected_source_type"]
        question_key = " ".join(item["question"].casefold().split())

        if topic not in EXPECTED_TOPIC_COUNTS:
            raise ValueError(f"Item {index}: unknown topic {topic!r}")
        if language not in EXPECTED_LANGUAGE_COUNTS:
            raise ValueError(f"Item {index}: unknown language {language!r}")
        if route not in ALLOWED_ROUTES:
            raise ValueError(f"Item {index}: unknown route {route!r}")
        if source_type not in ALLOWED_SOURCE_TYPES:
            raise ValueError(f"Item {index}: unknown source type {source_type!r}")
        if route == "dynamic_router" and source_type != "live_portal":
            raise ValueError(f"Item {index}: dynamic_router items must use live_portal")
        if route == "rag" and source_type == "live_portal":
            raise ValueError(f"Item {index}: rag items must not use live_portal")

        questions[question_key] += 1
        topics[topic] += 1
        languages[language] += 1
        routes[route] += 1
        source_types[source_type] += 1

    duplicates = [question for question, count in questions.items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate questions found: {duplicates[:5]}")

    if dict(topics) != EXPECTED_TOPIC_COUNTS:
        raise ValueError(
            f"Topic counts mismatch. Expected {EXPECTED_TOPIC_COUNTS}, found {dict(topics)}"
        )
    if dict(languages) != EXPECTED_LANGUAGE_COUNTS:
        raise ValueError(
            f"Language counts mismatch. Expected {EXPECTED_LANGUAGE_COUNTS}, found {dict(languages)}"
        )

    print(f"Items      : {len(items)}")
    print(f"Topics     : {dict(topics)}")
    print(f"Languages  : {dict(languages)}")
    print(f"Routes     : {dict(routes)}")
    print(f"Source type: {dict(source_types)}")


def _spot_check_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    seen_topics = Counter()

    for item in items:
        topic = item["topic"]
        if seen_topics[topic] < 2:
            selected.append(item)
            seen_topics[topic] += 1
        if len(selected) >= SPOT_CHECK_TARGET:
            return selected

    for item in items:
        if item not in selected:
            selected.append(item)
        if len(selected) >= SPOT_CHECK_TARGET:
            break

    return selected


def run_spot_check(items: list[dict[str, Any]]) -> None:
    sys.path.insert(0, str(ROOT))

    from app import route_dynamic_query
    from query_utils import normalize_query
    from vector_store import VectorStore

    selected = _spot_check_items(items)
    if len(selected) < SPOT_CHECK_TARGET:
        raise ValueError(f"Expected {SPOT_CHECK_TARGET} spot-check items, found {len(selected)}")

    store = VectorStore()
    failures = []

    for index, item in enumerate(selected, 1):
        question = item["question"]
        expected_route = item["expected_route"]
        dynamic = route_dynamic_query(question)

        if expected_route == "dynamic_router":
            ok = bool(dynamic and dynamic.get("route") == "dynamic_router")
            if not ok:
                failures.append(f"[{index}] expected dynamic_router: {question}")
            continue

        if dynamic:
            failures.append(f"[{index}] unexpected dynamic route for static case: {question}")
            continue

        results = store.query(normalize_query(question), n=1)
        if not results:
            failures.append(f"[{index}] no retrieval result for static case: {question}")

    if failures:
        raise ValueError("Spot-check failures:\n" + "\n".join(failures))

    print(f"Spot check : {len(selected)} router/retriever cases passed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spot-check", action="store_true", help="Run 15 router/retriever checks")
    args = parser.parse_args()

    try:
        items = load_items(DATASET)
        validate_shape(items)
        if args.spot_check:
            run_spot_check(items)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("OK: farmer eval dataset is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
