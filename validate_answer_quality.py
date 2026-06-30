"""
Validate farmer-facing answer quality at the API boundary.

This gate is model-optional. It checks route/language/evidence metadata and
basic answer safety without requiring a live external weather or mandi call.
The separate generation gate still validates real Ollama synthesis.
"""

from __future__ import annotations

import contextlib
from typing import Any, Callable
from unittest.mock import patch

from fastapi.testclient import TestClient

import live_data
from app import app


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def post_query(client: TestClient, payload: dict[str, Any]) -> dict[str, Any]:
    response = client.post("/query", json=payload)
    assert_true(response.status_code == 200, f"HTTP {response.status_code}: {response.text}")
    return response.json()


def fake_weather_request(url: str, params: dict[str, Any], timeout: int = 12) -> dict[str, Any]:
    assert_true("latitude" in params, "weather request should include latitude")
    return {
        "current": {
            "temperature_2m": 31.5,
            "precipitation": 0.0,
            "wind_speed_10m": 9.2,
        },
        "daily": {
            "time": ["2026-06-29", "2026-06-30", "2026-07-01"],
            "precipitation_probability_max": [12, 18, 22],
            "precipitation_sum": [0.0, 0.3, 0.5],
        },
    }


@contextlib.contextmanager
def mocked_live_weather(handler: Callable[..., dict[str, Any]]):
    with patch.object(live_data, "_request_json", side_effect=handler):
        yield


def source_blob(result: dict[str, Any]) -> str:
    return " ".join(
        str(source.get(key, ""))
        for source in result.get("sources", [])
        for key in ["display", "source", "category", "state", "text", "url"]
    ).lower()


def answer_text(result: dict[str, Any]) -> str:
    return str(result.get("answer", "")).strip()


def has_devanagari(text: str) -> bool:
    return any("\u0900" <= char <= "\u097F" for char in text)


def assert_common_quality(result: dict[str, Any], *, route: str, language: str) -> None:
    answer = answer_text(result)
    assert_true(result.get("route") == route, f"expected route={route}, got {result.get('route')}")
    assert_true(result.get("answer_language") == language, f"expected answer_language={language}")
    assert_true(len(answer) >= 40, "answer is too short to be useful")
    assert_true(len(answer) <= 1600, "answer should stay concise")
    assert_true(result.get("intent") is not None, "intent metadata is missing")
    assert_true("evidence_verifier" in result, "evidence verifier metadata is missing")
    assert_true("workflow_context" in result, "workflow context metadata is missing")
    if language == "english":
        assert_true(not has_devanagari(answer), "English answer should not contain Devanagari")


def test_static_english_pmfby(client: TestClient) -> None:
    result = post_query(client, {"question": "How do I claim PMFBY crop insurance after flood damage?"})
    assert_common_quality(result, route="rag", language="english")
    assert_true(result.get("evidence_verified"), "static RAG evidence should verify")
    assert_true("pmfby" in source_blob(result) or "crop insurance" in source_blob(result), "PMFBY source marker missing")


def test_static_hinglish_pmkisan(client: TestClient) -> None:
    result = post_query(client, {"question": "PM-KISAN ke liye kaun eligible hai?"})
    assert_common_quality(result, route="rag", language="hinglish")
    assert_true(result.get("evidence_verified"), "PM-KISAN evidence should verify")
    assert_true("pm-kisan" in source_blob(result) or "pmkisan" in source_blob(result), "PM-KISAN source marker missing")


def test_static_legal_rights(client: TestClient) -> None:
    result = post_query(client, {"question": "What rights does a land acquisition affected family have?"})
    assert_common_quality(result, route="rag", language="english")
    blob = source_blob(result)
    assert_true(any(marker in blob for marker in ("land acquisition", "larr", "rehabilitation", "forest rights")), "legal source marker missing")


def test_weather_follow_up_language_and_evidence(client: TestClient) -> None:
    first = post_query(
        client,
        {
            "question": "Kal baarish hogi kya, spraying karu?",
            "conversation_id": "quality-weather",
        },
    )
    assert_common_quality(first, route="workflow", language="hinglish")
    assert_true(first.get("answer_kind") == "clarification", "weather without location should clarify")
    assert_true(first.get("missing_fields") == ["location"], "weather clarification should request location")

    with mocked_live_weather(fake_weather_request):
        second = post_query(
            client,
            {
                "question": "jaipur",
                "conversation_id": "quality-weather",
                "workflow_context": first["workflow_context"],
            },
        )
    assert_common_quality(second, route="dynamic_router", language="hinglish")
    assert_true(second.get("evidence_verified"), "weather evidence should verify before synthesis/fallback")
    assert_true(second.get("live_status") == "success", "mocked weather should return success")
    assert_true("weather" in source_blob(second) or "mausam" in source_blob(second), "weather source marker missing")


def test_english_follow_up_switches_language(client: TestClient) -> None:
    first = post_query(
        client,
        {
            "question": "Will it rain tomorrow for spraying?",
            "conversation_id": "quality-language",
        },
    )
    assert_common_quality(first, route="workflow", language="english")

    with mocked_live_weather(fake_weather_request):
        second = post_query(
            client,
            {
                "question": "what about delhi",
                "conversation_id": "quality-language",
                "workflow_context": first["workflow_context"],
            },
        )
    assert_common_quality(second, route="dynamic_router", language="english")
    assert_true(second.get("evidence_verified"), "English weather follow-up should verify evidence")


def main() -> int:
    client = TestClient(app)
    tests = [
        test_static_english_pmfby,
        test_static_hinglish_pmkisan,
        test_static_legal_rights,
        test_weather_follow_up_language_and_evidence,
        test_english_follow_up_switches_language,
    ]

    for test in tests:
        test(client)
        print(f"✓ {test.__name__}")

    print("\nOK: answer quality gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
