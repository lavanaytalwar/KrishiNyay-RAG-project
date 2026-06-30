"""
Validate Phase 8 workflow behavior.

This gate checks intent classification, slot filling, clarification prompts,
workflow follow-up state, and safe routing. It avoids external network and
model dependencies by monkeypatching live API calls and RAG generation where
those are not the subject of the test.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any

import live_data
import app as app_module
from fastapi.testclient import TestClient


class FakeStore:
    def stats(self) -> dict[str, Any]:
        return {
            "total_chunks": 1748,
            "collection": "krishinyay_chunks",
            "embedding_backend": "MiniLM",
            "embedding_dim": 384,
            "retrieval_mode": "hybrid_vector_lexical",
            "lexical_chunks": 1748,
        }


class FakeChain:
    llm_provider = "ollama:llama3.1:8b"
    store = FakeStore()

    def ask(
        self,
        question: str,
        category: str | None = None,
        state: str | None = None,
        answer_language: str | None = None,
    ) -> dict[str, Any]:
        return {
            "question": question,
            "normalized_question": question.lower(),
            "answer": f"Grounded {answer_language or 'unknown'} RAG answer for: {question}",
            "sources": [
                {
                    "display": "Mock official source",
                    "url": "https://example.gov.in",
                    "similarity": 0.9,
                    "vector_score": 0.9,
                    "lexical_score": 0.5,
                    "hybrid_score": 1.1,
                    "retrieval_method": "hybrid",
                    "category": category or "scheme",
                    "state": state or "central",
                    "source": "mock_official",
                    "text": "Mock official source text for workflow validation.",
                }
            ],
            "context": "mock context",
            "n_chunks": 1,
            "mode": "rag",
            "route": "rag",
            "llm_provider": self.llm_provider,
            "generation_status": "generated",
            "generation_error": None,
            "answer_language": answer_language or "unknown",
        }

    def synthesize_from_evidence(
        self,
        *,
        question: str,
        evidence: str,
        sources: list[dict],
        answer_language: str | None = None,
    ) -> dict[str, Any]:
        return {
            "answer": evidence,
            "sources": sources,
            "llm_provider": self.llm_provider,
            "generation_status": "generated_from_verified_evidence",
            "generation_error": None,
            "answer_language": answer_language or "unknown",
        }


@contextmanager
def mocked_chain():
    original = app_module.get_chain
    try:
        app_module.get_chain = lambda n_results: FakeChain()
        yield
    finally:
        app_module.get_chain = original


@contextmanager
def mocked_request_json(handler):
    original = live_data._request_json
    try:
        live_data._request_json = handler
        yield
    finally:
        live_data._request_json = original


@contextmanager
def temporary_env(**values: str):
    original = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            if value == "":
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def post_query(client: TestClient, payload: dict[str, Any]) -> dict[str, Any]:
    response = client.post("/query", json=payload)
    assert_true(response.status_code == 200, f"expected 200, got {response.status_code}: {response.text}")
    return response.json()


def fake_weather_request(url, params, timeout=12):
    assert_true("latitude" in params, "weather request should include latitude")
    return {
        "current": {
            "temperature_2m": 31.0,
            "precipitation": 0.0,
            "wind_speed_10m": 7.5,
        },
        "daily": {
            "time": ["2026-06-29", "2026-06-30", "2026-07-01"],
            "precipitation_probability_max": [30, 70, 20],
            "precipitation_sum": [0.2, 6.0, 0.0],
        },
    }


def test_weather_missing_location_and_follow_up(client: TestClient) -> None:
    first = post_query(client, {"question": "Kal baarish hogi kya, spraying karu?"})
    assert_true(first["route"] == "workflow", "weather without location should stay in workflow")
    assert_true(first["intent"] == "weather", "intent should be weather")
    assert_true(first["answer_kind"] == "clarification", "weather missing location should clarify")
    assert_true(first["answer_language"] == "hinglish", "Hinglish weather question should set Hinglish answer language")
    assert_true(first["missing_fields"] == ["location"], "location should be the missing field")
    assert_true(first["sources"] == [], "clarification should not return RAG sources")

    with mocked_request_json(fake_weather_request):
        second = post_query(
            client,
            {
                "question": "jaipur",
                "workflow_context": first["workflow_context"],
            },
        )
    assert_true(second["route"] == "dynamic_router", "weather follow-up should use dynamic route")
    assert_true(second["intent"] == "weather", "weather follow-up should keep weather intent")
    assert_true(second["workflow_state"] == "resumed", "weather follow-up should resume workflow")
    assert_true(second["live_status"] == "success", "weather follow-up should fetch mocked live data")
    assert_true(second["evidence_verified"], "weather evidence should verify before synthesis")
    assert_true(second["generation_status"] == "generated_from_verified_evidence", "weather should run evidence synthesis")
    assert_true(second["answer_language"] == "hinglish", "ambiguous follow-up should inherit Hinglish")
    assert_true("Jaipur ka live forecast" in second["answer"], "Hinglish weather follow-up should answer in Hinglish")
    assert_true("Spraying" in second["answer"], "Hinglish weather answer should keep farmer spraying advice")
    assert_true(second["live_data"]["location"] == "Jaipur", "follow-up should fill Jaipur as location")

    with mocked_request_json(fake_weather_request):
        third = post_query(
            client,
            {
                "question": "what about delhi",
                "workflow_context": second["workflow_context"],
            },
        )
    assert_true(third["route"] == "dynamic_router", "weather comparison follow-up should stay dynamic")
    assert_true(third["intent"] == "weather", "weather comparison should keep weather intent")
    assert_true(third["workflow_state"] == "resumed", "weather comparison should resume active context")
    assert_true(third["live_status"] == "success", "weather comparison should fetch mocked live data")
    assert_true(third["evidence_verified"], "weather comparison evidence should verify")
    assert_true(third["answer_language"] == "english", "English follow-up should switch answer language to English")
    assert_true(third["live_data"]["location"] == "Delhi", "comparison follow-up should switch location to Delhi")
    assert_true(third["answer"].startswith("Live forecast for Delhi"), "English follow-up should answer in English")


def test_weather_language_switch_from_english_to_hinglish(client: TestClient) -> None:
    first = post_query(client, {"question": "Will it rain tomorrow, should I spray?"})
    assert_true(first["route"] == "workflow", "English weather without location should clarify")
    assert_true(first["answer_language"] == "english", "English weather question should set English answer language")
    assert_true(first["answer"].startswith("I need your"), "English clarification should be in English")

    with mocked_request_json(fake_weather_request):
        second = post_query(
            client,
            {
                "question": "delhi kya",
                "workflow_context": first["workflow_context"],
            },
        )
    assert_true(second["route"] == "dynamic_router", "Hinglish follow-up should stay on weather route")
    assert_true(second["answer_language"] == "hinglish", "Hinglish follow-up should switch answer language")
    assert_true("Delhi ka live forecast" in second["answer"], "Hinglish follow-up should answer in Hinglish")


def test_mandi_missing_commodity_and_follow_up(client: TestClient) -> None:
    first = post_query(client, {"question": "Aaj mandi bhav kya hai?"})
    assert_true(first["route"] == "workflow", "mandi without commodity should stay in workflow")
    assert_true(first["intent"] == "mandi_price", "intent should be mandi_price")
    assert_true(first["missing_fields"] == ["commodity"], "commodity should be missing")
    assert_true(first["answer_kind"] == "clarification", "mandi missing commodity should clarify")

    with temporary_env(DATA_GOV_IN_API_KEY="", AGMARKNET_API_KEY=""):
        second = post_query(
            client,
            {
                "question": "soybean",
                "workflow_context": first["workflow_context"],
            },
        )
    assert_true(second["route"] == "dynamic_router", "commodity follow-up should use dynamic route")
    assert_true(second["workflow_state"] == "resumed", "mandi follow-up should resume workflow")
    assert_true(second["live_status"] == "unavailable_missing_api_key", "missing API key should be explicit")
    assert_true(second["live_data"]["commodity"] == "Soyabean", "commodity should be filled from follow-up")


def test_vague_eligibility_clarifies(client: TestClient) -> None:
    result = post_query(client, {"question": "am I eligible?"})
    assert_true(result["route"] == "workflow", "vague eligibility should not run RAG")
    assert_true(result["intent"] == "eligibility", "intent should be eligibility")
    assert_true(result["missing_fields"] == ["scheme_or_state"], "scheme/state should be missing")
    assert_true(result["sources"] == [], "clarification should not return unrelated RAG sources")
    assert_true("latency_ms" in result, "response should expose latency metadata")
    assert_true(result["source_count"] == 0, "clarification source_count should be zero")


def test_vague_eligibility_follow_up_uses_rag(client: TestClient) -> None:
    first = post_query(
        client,
        {
            "question": "am I eligible?",
            "conversation_id": "eligibility-follow-up",
        },
    )
    second = post_query(
        client,
        {
            "question": "PM-KISAN",
            "conversation_id": "eligibility-follow-up",
            "workflow_context": first["workflow_context"],
        },
    )
    assert_true(second["route"] == "rag", "scheme follow-up should resume eligibility into RAG")
    assert_true(second["workflow_state"] == "resumed", "eligibility follow-up should mark resumed workflow")
    assert_true(second["tool_used"] == "rag", "eligibility follow-up should use RAG")
    assert_true(second["answer_language"] == "english", "English first turn should keep English for ambiguous scheme follow-up")
    assert_true(second["evidence_verified"], "resumed eligibility RAG should verify evidence")


def test_static_questions_use_rag_workflow(client: TestClient) -> None:
    questions = [
        ("PM-KISAN ke liye eligible kaun hai?", "hinglish"),
        ("How do I claim PMFBY crop insurance after flood damage?", "english"),
        ("KCC ke liye kya documents chahiye?", "hinglish"),
        ("Forest Rights Act mein Gram Sabha ka role kya hai?", "hinglish"),
    ]
    for question, expected_language in questions:
        result = post_query(client, {"question": question})
        assert_true(result["route"] == "rag", f"{question} should use RAG")
        assert_true(result["intent"] == "static_rag", f"{question} should be static_rag")
        assert_true(result["tool_used"] == "rag", f"{question} should mark rag tool")
        assert_true(result["answer_kind"] == "generated", f"{question} should be generated in mocked chain")
        assert_true(result["answer_language"] == expected_language, f"{question} should pass language policy to RAG")
        assert_true(result["evidence_verified"], f"{question} should verify retrieved evidence")
        assert_true(len(result["sources"]) == 1, f"{question} should return mocked source")


def test_dynamic_status_and_system_info(client: TestClient) -> None:
    status = post_query(client, {"question": "Mera PM-KISAN beneficiary status kya hai?"})
    assert_true(status["route"] == "dynamic_router", "PM-KISAN status should use dynamic route")
    assert_true(status["intent"] == "pmkisan_status", "status intent should be pmkisan_status")
    assert_true(status["tool_used"] == "pmkisan_portal", "status should use portal tool")
    assert_true(status["source_count"] >= 1, "status response should expose source count")
    assert_true("source_types" in status, "status response should expose source types")

    system = post_query(client, {"question": "which model is this"})
    assert_true(system["route"] == "system_info", "model question should use system info")
    assert_true(system["intent"] == "system_info", "system intent should be system_info")
    assert_true("MiniLM" in system["answer"], "system answer should report retrieval backend")
    assert_true(system["evidence_verified"], "system info should include evidence verifier metadata")


def main() -> int:
    client = TestClient(app_module.app)
    tests = [
        test_weather_missing_location_and_follow_up,
        test_weather_language_switch_from_english_to_hinglish,
        test_mandi_missing_commodity_and_follow_up,
        test_vague_eligibility_clarifies,
        test_vague_eligibility_follow_up_uses_rag,
        test_static_questions_use_rag_workflow,
        test_dynamic_status_and_system_info,
    ]
    with mocked_chain():
        for test in tests:
            test(client)
            print(f"✓ {test.__name__}")
    print()
    print("OK: Phase 8 workflow validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
