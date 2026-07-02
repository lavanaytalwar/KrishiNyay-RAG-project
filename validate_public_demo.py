#!/usr/bin/env python3
"""
Validate the public demo packaging path.

Default mode is local-safe: it validates the full packaged Chroma index,
public safety gates, static RAG, and mocked weather follow-up without requiring
a hosted LLM key.

Use --require-gemini before launch to require GEMINI_API_KEY and assert that the
public demo is fully ready with Gemini selected.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

EXPECTED_CHUNKS = 1748
EXPECTED_DIM = 384
ROOT = Path(__file__).resolve().parent


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def configure_env(require_gemini: bool) -> None:
    os.environ["DEMO_PUBLIC"] = "true"
    os.environ["ENABLE_LIVE_INGEST"] = "false"
    os.environ["CHROMA_PATH"] = "demo_chroma_db"
    runtime_path = Path(tempfile.gettempdir()) / "krishinyay_public_demo_chroma"
    if runtime_path.exists():
        shutil.rmtree(runtime_path)
    os.environ["CHROMA_RUNTIME_PATH"] = str(runtime_path)
    os.environ["CHUNKS_DIR"] = "demo_data/chunks"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    if require_gemini:
        assert_true(
            bool(os.environ.get("GEMINI_API_KEY", "").strip()),
            "GEMINI_API_KEY is required when running validate_public_demo.py --require-gemini",
        )
        os.environ["LLM_PROVIDER"] = "gemini"
        return

    if os.environ.get("GEMINI_API_KEY", "").strip():
        os.environ.setdefault("LLM_PROVIDER", "gemini")
    else:
        os.environ["LLM_PROVIDER"] = "template"


def fake_weather_request(url: str, params: dict[str, Any], timeout: int = 12) -> dict[str, Any]:
    assert_true("latitude" in params, "weather request should include latitude")
    return {
        "current": {
            "temperature_2m": 31.0,
            "precipitation": 0.0,
            "wind_speed_10m": 7.5,
        },
        "daily": {
            "time": ["2026-07-02", "2026-07-03", "2026-07-04"],
            "precipitation_probability_max": [25, 15, 20],
            "precipitation_sum": [0.1, 0.0, 0.0],
        },
    }


@contextmanager
def mocked_weather_request(replacement: Callable[..., dict[str, Any]]):
    import live_data

    original = live_data._request_json
    live_data._request_json = replacement
    try:
        yield
    finally:
        live_data._request_json = original


def post_query(client, payload: dict[str, Any]) -> dict[str, Any]:
    response = client.post("/query", json=payload)
    assert_true(response.status_code == 200, f"query failed: {response.status_code} {response.text}")
    return response.json()


def assert_no_secret_leak(payloads: list[dict[str, Any]]) -> None:
    secret_values = [
        os.environ.get("GEMINI_API_KEY", ""),
        os.environ.get("DATA_GOV_IN_API_KEY", ""),
        os.environ.get("AGMARKNET_API_KEY", ""),
        os.environ.get("LIVE_INGEST_TOKEN", ""),
    ]
    rendered = json.dumps(payloads, ensure_ascii=False)
    for value in secret_values:
        if value and len(value) >= 8:
            assert_true(value not in rendered, "API secret leaked in public JSON response")


def validate_artifacts() -> None:
    chroma_path = ROOT / "demo_chroma_db"
    chunks_dir = ROOT / "demo_data" / "chunks"
    assert_true(chroma_path.exists(), "demo_chroma_db is missing")
    assert_true((chroma_path / "chroma.sqlite3").exists(), "demo_chroma_db/chroma.sqlite3 is missing")
    assert_true((chunks_dir / "all_chunks.jsonl").exists(), "demo_data/chunks/all_chunks.jsonl is missing")
    assert_true((chunks_dir / "embed_meta.json").exists(), "demo_data/chunks/embed_meta.json is missing")


def validate_api(require_gemini: bool) -> None:
    from fastapi.testclient import TestClient

    import app

    app.get_chain.cache_clear()
    client = TestClient(app.app)

    health_response = client.get("/health")
    assert_true(health_response.status_code == 200, f"/health failed: {health_response.text}")
    health = health_response.json()
    readiness = health.get("readiness") or {}
    assert_true(health["demo_public"] is True, "/health should report demo_public=true")
    assert_true(readiness["demo_public"] is True, "readiness should report demo_public=true")
    assert_true(health["total_chunks"] == EXPECTED_CHUNKS, "public Chroma count should be 1,748")
    assert_true(health["lexical_chunks"] == EXPECTED_CHUNKS, "public lexical chunk count should be 1,748")
    assert_true(health["embedding_dim"] == EXPECTED_DIM, "public embedding dimension should be 384")
    assert_true(health["chroma_path"] == "demo_chroma_db", "public Chroma path should be demo_chroma_db")
    assert_true(
        "krishinyay_public_demo_chroma" in health["chroma_runtime_path"],
        "public Chroma runtime path should use writable temp storage",
    )
    assert_true(health["chunks_dir"] == "demo_data/chunks", "public chunks path should be demo_data/chunks")
    assert_true(readiness["live_ingest_enabled"] is False, "live ingest must be disabled in public mode")

    if require_gemini:
        assert_true(health["llm_provider"] == "gemini", "Gemini should be selected for public launch")
        assert_true(health["public_demo_ready"] is True, "public_demo_ready should be true with Gemini configured")

    config_response = client.get("/demo-config")
    assert_true(config_response.status_code == 200, f"/demo-config failed: {config_response.text}")
    config = config_response.json()
    assert_true(config["public_demo"] is True, "/demo-config should report public_demo=true")
    assert_true(config["demo_hosting"] == "hugging_face_spaces", "demo hosting should be Hugging Face Spaces")
    assert_true(len(config.get("recommended_demo_flow") or []) >= 4, "recommended demo flow is missing")

    ingest_response = client.post(
        "/ingest",
        json={
            "title": "Public demo blocked ingest",
            "text": "This official-looking document should not be accepted in public demo mode. " * 2,
        },
    )
    assert_true(ingest_response.status_code == 403, "/ingest must return 403 in public mode")
    assert_true("public demo mode" in ingest_response.text.lower(), "/ingest should explain public demo lock")

    static = post_query(client, {"question": "PM-KISAN mein kitne paise milte hain?"})
    assert_true(static["route"] == "rag", "PM-KISAN sample should use static RAG")
    assert_true(static["sources"], "PM-KISAN sample should return sources")

    first = post_query(client, {"question": "Kal baarish hogi kya, spraying karu?"})
    assert_true(first["answer_kind"] == "clarification", "weather without location should ask for clarification")
    assert_true(first["workflow_context"], "weather clarification should return workflow context")

    with mocked_weather_request(fake_weather_request):
        second = post_query(
            client,
            {
                "question": "Jaipur",
                "workflow_context": first["workflow_context"],
            },
        )
    assert_true(second["route"] == "dynamic_router", "weather follow-up should use dynamic router")
    assert_true(second["live_status"] == "success", "weather follow-up should use mocked live data")
    assert_true(second["evidence_verified"] is True, "weather evidence should verify")

    assert_no_secret_leak([health, config, static, first, second])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-gemini",
        action="store_true",
        help="Require GEMINI_API_KEY and assert the public demo is fully ready with Gemini.",
    )
    args = parser.parse_args()

    try:
        configure_env(args.require_gemini)
        validate_artifacts()
        validate_api(args.require_gemini)
    except Exception as exc:
        print(f"❌ Public demo validation failed: {exc}")
        return 1

    mode = "Gemini launch" if args.require_gemini else "local dry-run"
    print(f"✅ Public demo validation passed ({mode})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
