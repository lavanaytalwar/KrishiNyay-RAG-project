"""
Validate Phase 7 live-data routing without relying on external network calls.

The tests monkeypatch the live API helper so the gate is deterministic. Live
API availability itself is intentionally runtime-dependent and reported through
`live_status` in `/query` responses.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import live_data
from app import route_dynamic_query


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


@contextmanager
def mocked_request_json(handler):
    original = live_data._request_json
    try:
        live_data._request_json = handler
        yield
    finally:
        live_data._request_json = original


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_mandi_missing_key_falls_back_safely() -> None:
    with temporary_env(DATA_GOV_IN_API_KEY="", AGMARKNET_API_KEY=""):
        result = live_data.get_mandi_price_snapshot("Aaj soybean ka mandi bhav kya hai?")
    assert_true(result["route"] == "dynamic_router", "mandi result should stay on dynamic route")
    assert_true(result["live_status"] == "unavailable_missing_api_key", "missing API key should be explicit")
    assert_true("not configured" in result["answer"], "answer should explain API setup")
    assert_true(result["live_data"]["commodity"] == "Soyabean", "commodity parser should detect soybean")


def test_mandi_mock_success() -> None:
    def fake_request(url, params, timeout=12):
        assert_true(params["filters[commodity]"] == "Soyabean", "commodity filter should be applied")
        return {
            "records": [
                {
                    "state": "Maharashtra",
                    "district": "Nashik",
                    "market": "Lasalgaon",
                    "commodity": "Soyabean",
                    "arrival_date": "27/06/2026",
                    "min_price": "3900",
                    "max_price": "4300",
                    "modal_price": "4100",
                }
            ]
        }

    with temporary_env(DATA_GOV_IN_API_KEY="test-key"):
        with mocked_request_json(fake_request):
            result = live_data.get_mandi_price_snapshot(
                "Aaj soybean ka mandi bhav kya hai?",
                state="Maharashtra",
            )

    assert_true(result["live_status"] == "success", "mock mandi fetch should succeed")
    assert_true("₹4,100/quintal" in result["answer"], "answer should include modal price")
    assert_true(result["live_data"]["market"] == "Lasalgaon", "market metadata should be returned")


def test_weather_mock_success() -> None:
    def fake_request(url, params, timeout=12):
        assert_true("latitude" in params, "forecast call should include latitude")
        return {
            "current": {
                "temperature_2m": 28.5,
                "precipitation": 0.0,
                "wind_speed_10m": 9.2,
            },
            "daily": {
                "time": ["2026-06-27", "2026-06-28", "2026-06-29"],
                "precipitation_probability_max": [20, 65, 40],
                "precipitation_sum": [0.1, 8.4, 2.0],
            },
        }

    with mocked_request_json(fake_request):
        result = live_data.get_weather_forecast("Kal Pune mein baarish hogi kya, spraying karu?")

    assert_true(result["live_status"] == "success", "mock weather fetch should succeed")
    assert_true("Pune" in result["answer"], "answer should include resolved location")
    assert_true("Pune ka live forecast" in result["answer"], "Hinglish question should get Hinglish answer")
    assert_true(
        "Avoid spraying" in result["answer"] or "spraying avoid" in result["answer"].lower(),
        "rainy forecast should warn against spraying",
    )
    assert_true(result["live_data"]["location"] == "Pune", "location metadata should be returned")


def test_router_returns_live_metadata() -> None:
    with temporary_env(DATA_GOV_IN_API_KEY="", AGMARKNET_API_KEY=""):
        mandi = route_dynamic_query("Aaj soybean ka mandi bhav kya hai?")
    weather = route_dynamic_query("Kal baarish hogi kya, spraying karu?")

    assert_true(mandi is not None, "mandi question should route dynamically")
    assert_true("live_status" in mandi, "mandi route should include live_status")
    assert_true(weather is not None, "weather question should route dynamically")
    assert_true(weather["live_status"] == "needs_more_input", "weather without location should ask for location")


def main() -> int:
    tests = [
        test_mandi_missing_key_falls_back_safely,
        test_mandi_mock_success,
        test_weather_mock_success,
        test_router_returns_live_metadata,
    ]
    for test in tests:
        test()
        print(f"✓ {test.__name__}")
    print()
    print("OK: Phase 7 live-data validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
