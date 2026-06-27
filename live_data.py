"""
Phase 7 live data helpers for dynamic farmer queries.

Mandi prices:
  - Uses the Data.gov.in Agmarknet commodity-price resource when an API key is
    configured through DATA_GOV_IN_API_KEY or AGMARKNET_API_KEY.
  - Falls back to official eNAM/Agmarknet portal guidance without inventing
    prices when a key or usable record is unavailable.

Weather:
  - Uses Open-Meteo forecast data for low-friction local demo forecasts.
  - Keeps IMD/Mausam as the official farmer-facing verification source.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import Any, Optional

import requests


AGMARKNET_RESOURCE_ID = os.environ.get(
    "AGMARKNET_RESOURCE_ID",
    "9ef84268-d588-465a-a308-a864a43d0070",
)
AGMARKNET_API_BASE = os.environ.get(
    "AGMARKNET_API_BASE",
    "https://api.data.gov.in/resource",
)
ENAM_LIVE_PRICE_URL = "https://enam.gov.in/web/dashboard/live_price"
AGMARKNET_PORTAL_URL = "https://agmarknet.gov.in/"
IMD_URL = "https://mausam.imd.gov.in/"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"


COMMODITY_ALIASES = {
    "soybean": "Soyabean",
    "soyabean": "Soyabean",
    "soya": "Soyabean",
    "wheat": "Wheat",
    "gehu": "Wheat",
    "gehun": "Wheat",
    "rice": "Rice",
    "dhan": "Paddy",
    "paddy": "Paddy",
    "cotton": "Cotton",
    "kapas": "Cotton",
    "onion": "Onion",
    "pyaz": "Onion",
    "potato": "Potato",
    "aloo": "Potato",
    "tomato": "Tomato",
    "maize": "Maize",
    "makka": "Maize",
    "mustard": "Mustard",
    "sarson": "Mustard",
    "gram": "Gram",
    "chana": "Gram",
}

STATE_ALIASES = {
    "andhra pradesh": "Andhra Pradesh",
    "bihar": "Bihar",
    "gujarat": "Gujarat",
    "haryana": "Haryana",
    "karnataka": "Karnataka",
    "madhya pradesh": "Madhya Pradesh",
    "maharashtra": "Maharashtra",
    "odisha": "Odisha",
    "punjab": "Punjab",
    "rajasthan": "Rajasthan",
    "telangana": "Telangana",
    "uttar pradesh": "Uttar Pradesh",
    "west bengal": "West Bengal",
}

LOCATION_COORDS = {
    "ahmedabad": (23.0225, 72.5714, "Ahmedabad"),
    "bhopal": (23.2599, 77.4126, "Bhopal"),
    "delhi": (28.6139, 77.2090, "Delhi"),
    "hyderabad": (17.3850, 78.4867, "Hyderabad"),
    "indore": (22.7196, 75.8577, "Indore"),
    "jaipur": (26.9124, 75.7873, "Jaipur"),
    "kolkata": (22.5726, 88.3639, "Kolkata"),
    "lucknow": (26.8467, 80.9462, "Lucknow"),
    "ludhiana": (30.9010, 75.8573, "Ludhiana"),
    "mumbai": (19.0760, 72.8777, "Mumbai"),
    "nagpur": (21.1458, 79.0882, "Nagpur"),
    "nashik": (19.9975, 73.7898, "Nashik"),
    "patna": (25.5941, 85.1376, "Patna"),
    "pune": (18.5204, 73.8567, "Pune"),
}

STATE_CAPITAL_COORDS = {
    "bihar": LOCATION_COORDS["patna"],
    "gujarat": LOCATION_COORDS["ahmedabad"],
    "madhya pradesh": LOCATION_COORDS["bhopal"],
    "maharashtra": LOCATION_COORDS["mumbai"],
    "punjab": LOCATION_COORDS["ludhiana"],
    "rajasthan": LOCATION_COORDS["jaipur"],
    "telangana": LOCATION_COORDS["hyderabad"],
    "uttar pradesh": LOCATION_COORDS["lucknow"],
    "west bengal": LOCATION_COORDS["kolkata"],
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _request_json(url: str, params: dict[str, Any], timeout: int = 12) -> dict[str, Any]:
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _find_alias(text: str, aliases: dict[str, str]) -> Optional[str]:
    lowered = text.lower()
    for marker, value in aliases.items():
        if re.search(rf"\b{re.escape(marker)}\b", lowered):
            return value
    return None


def parse_commodity(question: str, explicit: Optional[str] = None) -> Optional[str]:
    if explicit:
        return explicit.strip()
    return _find_alias(question, COMMODITY_ALIASES)


def parse_state(question: str, explicit: Optional[str] = None) -> Optional[str]:
    if explicit:
        return explicit.strip()
    return _find_alias(question, STATE_ALIASES)


def _normalise_record(record: dict[str, Any]) -> dict[str, Any]:
    return {str(key).lower(): value for key, value in record.items()}


def _format_price(value: Any) -> str:
    try:
        return f"₹{float(value):,.0f}"
    except (TypeError, ValueError):
        return str(value or "not reported")


def _base_dynamic_response(
    *,
    reason: str,
    answer: str,
    sources: list[dict[str, Any]],
    live_status: str,
    provider: str,
    live_data: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    return {
        "mode": "dynamic_router",
        "route": "dynamic_router",
        "route_reason": reason,
        "answer": answer,
        "sources": sources,
        "live_status": live_status,
        "data_provider": provider,
        "fetched_at": _now_iso(),
        "live_data": live_data or {},
    }


def _mandi_sources(text: str = "Current mandi prices should be checked on official live market data sources.") -> list[dict[str, Any]]:
    return [
        {
            "display": "Data.gov.in Agmarknet Commodity Prices",
            "url": f"{AGMARKNET_API_BASE}/{AGMARKNET_RESOURCE_ID}",
            "similarity": None,
            "category": "market_prices",
            "state": "india",
            "source": "live_portal",
            "doc_type": "Live API",
            "text": text,
        },
        {
            "display": "eNAM Live Price Dashboard",
            "url": ENAM_LIVE_PRICE_URL,
            "similarity": None,
            "category": "market_prices",
            "state": "india",
            "source": "official_portal",
            "doc_type": "Live Portal",
            "text": "Use the official live portal to verify prices before selling.",
        },
    ]


def get_mandi_price_snapshot(
    question: str,
    *,
    commodity: Optional[str] = None,
    state: Optional[str] = None,
    district: Optional[str] = None,
    market: Optional[str] = None,
) -> dict[str, Any]:
    commodity_name = parse_commodity(question, commodity)
    state_name = parse_state(question, state)
    api_key = os.environ.get("DATA_GOV_IN_API_KEY") or os.environ.get("AGMARKNET_API_KEY")

    if not commodity_name:
        return _base_dynamic_response(
            reason="mandi_price_live_data",
            answer=(
                "Mandi prices are live. Tell me the commodity name and, ideally, "
                "state/district/market so I can look up the current price. You can also "
                f"check the official eNAM dashboard: {ENAM_LIVE_PRICE_URL}"
            ),
            sources=_mandi_sources(),
            live_status="needs_more_input",
            provider="data.gov.in_agmarknet",
        )

    if not api_key:
        return _base_dynamic_response(
            reason="mandi_price_live_data",
            answer=(
                f"I detected the commodity as {commodity_name}, but live mandi API access "
                "is not configured on this machine. Set DATA_GOV_IN_API_KEY or "
                f"AGMARKNET_API_KEY, or verify current prices on eNAM: {ENAM_LIVE_PRICE_URL}"
            ),
            sources=_mandi_sources("Agmarknet API key is required for live API price lookup."),
            live_status="unavailable_missing_api_key",
            provider="data.gov.in_agmarknet",
            live_data={"commodity": commodity_name, "state": state_name or ""},
        )

    params: dict[str, Any] = {
        "api-key": api_key,
        "format": "json",
        "limit": int(os.environ.get("AGMARKNET_LIMIT", "10")),
        "filters[commodity]": commodity_name,
    }
    if state_name:
        params["filters[state]"] = state_name
    if district:
        params["filters[district]"] = district
    if market:
        params["filters[market]"] = market

    try:
        payload = _request_json(f"{AGMARKNET_API_BASE}/{AGMARKNET_RESOURCE_ID}", params)
    except Exception as exc:
        return _base_dynamic_response(
            reason="mandi_price_live_data",
            answer=(
                f"I could not fetch live {commodity_name} mandi prices right now "
                f"({str(exc)[:120]}). Please verify on eNAM before making a sale decision: "
                f"{ENAM_LIVE_PRICE_URL}"
            ),
            sources=_mandi_sources("Live Agmarknet request failed; fallback portal is provided."),
            live_status="unavailable_api_error",
            provider="data.gov.in_agmarknet",
            live_data={"commodity": commodity_name, "state": state_name or ""},
        )

    records = [_normalise_record(record) for record in payload.get("records", [])]
    if not records:
        state_part = f" in {state_name}" if state_name else ""
        return _base_dynamic_response(
            reason="mandi_price_live_data",
            answer=(
                f"I could not find a fresh Agmarknet record for {commodity_name}{state_part}. "
                f"Please verify on eNAM: {ENAM_LIVE_PRICE_URL}"
            ),
            sources=_mandi_sources("Agmarknet returned no matching records."),
            live_status="unavailable_no_records",
            provider="data.gov.in_agmarknet",
            live_data={"commodity": commodity_name, "state": state_name or ""},
        )

    top = records[0]
    market_name = top.get("market") or "reported market"
    district_name = top.get("district") or ""
    state_reported = top.get("state") or state_name or ""
    date = top.get("arrival_date") or top.get("date") or "latest available date"
    modal_price = _format_price(top.get("modal_price"))
    min_price = _format_price(top.get("min_price"))
    max_price = _format_price(top.get("max_price"))
    answer = (
        f"Latest available {commodity_name} price from Agmarknet: {market_name}"
        f"{', ' + district_name if district_name else ''}"
        f"{', ' + state_reported if state_reported else ''} reported modal price "
        f"{modal_price}/quintal on {date}. Range: {min_price}–{max_price}/quintal. "
        "Verify on the mandi/eNAM portal before selling because prices change during the day."
    )
    return _base_dynamic_response(
        reason="mandi_price_live_data",
        answer=answer,
        sources=_mandi_sources("Live mandi price summary fetched from the Agmarknet API."),
        live_status="success",
        provider="data.gov.in_agmarknet",
        live_data={
            "commodity": commodity_name,
            "state": state_reported,
            "district": district_name,
            "market": market_name,
            "arrival_date": date,
            "modal_price": top.get("modal_price"),
            "min_price": top.get("min_price"),
            "max_price": top.get("max_price"),
            "records_returned": len(records),
        },
    )


def _location_from_text(question: str, explicit_location: Optional[str], explicit_state: Optional[str]) -> Optional[tuple[float, float, str]]:
    if explicit_location:
        key = explicit_location.strip().lower()
        if key in LOCATION_COORDS:
            return LOCATION_COORDS[key]
        return _geocode_location(explicit_location)

    lowered = question.lower()
    for marker, coords in LOCATION_COORDS.items():
        if re.search(rf"\b{re.escape(marker)}\b", lowered):
            return coords

    state_name = parse_state(question, explicit_state)
    if state_name:
        return STATE_CAPITAL_COORDS.get(state_name.lower())
    return None


def _geocode_location(location: str) -> Optional[tuple[float, float, str]]:
    try:
        payload = _request_json(
            OPEN_METEO_GEOCODE_URL,
            {"name": location, "count": 1, "language": "en", "format": "json"},
            timeout=8,
        )
    except Exception:
        return None
    results = payload.get("results") or []
    if not results:
        return None
    top = results[0]
    return float(top["latitude"]), float(top["longitude"]), str(top.get("name") or location)


def _weather_sources(text: str = "Current weather should be checked from live forecast data and official advisories.") -> list[dict[str, Any]]:
    return [
        {
            "display": "Open-Meteo Forecast API",
            "url": OPEN_METEO_FORECAST_URL,
            "similarity": None,
            "category": "weather",
            "state": "global",
            "source": "live_api",
            "doc_type": "Live API",
            "text": text,
        },
        {
            "display": "India Meteorological Department",
            "url": IMD_URL,
            "similarity": None,
            "category": "weather",
            "state": "india",
            "source": "official_portal",
            "doc_type": "Live Portal",
            "text": "Use IMD or local agri advisory for official weather verification.",
        },
    ]


def get_weather_forecast(
    question: str,
    *,
    location: Optional[str] = None,
    state: Optional[str] = None,
) -> dict[str, Any]:
    resolved = _location_from_text(question, location, state)
    if not resolved:
        return _base_dynamic_response(
            reason="weather_live_data",
            answer=(
                "Weather and spraying advice depends on your exact location. Tell me your "
                "village/city or district, and verify official advisories on IMD/Mausam "
                f"before spraying: {IMD_URL}"
            ),
            sources=_weather_sources("A location is required before fetching a local forecast."),
            live_status="needs_more_input",
            provider="open_meteo_with_imd_verification",
        )

    latitude, longitude, place = resolved
    try:
        payload = _request_json(
            OPEN_METEO_FORECAST_URL,
            {
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,precipitation,wind_speed_10m",
                "daily": "precipitation_probability_max,precipitation_sum",
                "forecast_days": 3,
                "timezone": "auto",
            },
        )
    except Exception as exc:
        return _base_dynamic_response(
            reason="weather_live_data",
            answer=(
                f"I could not fetch the live forecast for {place} right now "
                f"({str(exc)[:120]}). Please check IMD/Mausam or your local agriculture "
                f"advisory before spraying: {IMD_URL}"
            ),
            sources=_weather_sources("Live weather request failed; official portal fallback is provided."),
            live_status="unavailable_api_error",
            provider="open_meteo_with_imd_verification",
            live_data={"location": place, "latitude": latitude, "longitude": longitude},
        )

    current = payload.get("current") or {}
    daily = payload.get("daily") or {}
    dates = daily.get("time") or []
    rain_probability = daily.get("precipitation_probability_max") or []
    rain_sum = daily.get("precipitation_sum") or []
    today_probability = rain_probability[0] if rain_probability else None
    tomorrow_probability = rain_probability[1] if len(rain_probability) > 1 else None
    today_rain = rain_sum[0] if rain_sum else None
    tomorrow_rain = rain_sum[1] if len(rain_sum) > 1 else None
    temp = current.get("temperature_2m")
    wind = current.get("wind_speed_10m")
    current_rain = current.get("precipitation")

    spray_caution = (
        "Avoid spraying until the rain risk drops and wind is calm."
        if (tomorrow_probability or 0) >= 50 or (today_probability or 0) >= 50
        else "Spraying may be possible, but confirm locally before applying chemicals."
    )
    answer = (
        f"Live forecast for {place}: current temperature {temp if temp is not None else 'not reported'}°C, "
        f"current rain {current_rain if current_rain is not None else 'not reported'} mm, "
        f"wind {wind if wind is not None else 'not reported'} km/h. "
        f"Rain chance: today {today_probability if today_probability is not None else 'not reported'}%, "
        f"tomorrow {tomorrow_probability if tomorrow_probability is not None else 'not reported'}%. "
        f"Expected rain: today {today_rain if today_rain is not None else 'not reported'} mm, "
        f"tomorrow {tomorrow_rain if tomorrow_rain is not None else 'not reported'} mm. "
        f"{spray_caution} Verify with IMD/local advisory before spraying."
    )
    return _base_dynamic_response(
        reason="weather_live_data",
        answer=answer,
        sources=_weather_sources("Live local forecast fetched from Open-Meteo; IMD is provided for official verification."),
        live_status="success",
        provider="open_meteo_with_imd_verification",
        live_data={
            "location": place,
            "latitude": latitude,
            "longitude": longitude,
            "current_temperature_c": temp,
            "current_precipitation_mm": current_rain,
            "current_wind_kmh": wind,
            "daily_dates": dates[:3],
            "rain_probability_max_percent": rain_probability[:3],
            "rain_sum_mm": rain_sum[:3],
        },
    )


def live_config_status() -> dict[str, Any]:
    return {
        "mandi_provider": "data.gov.in_agmarknet",
        "mandi_api_configured": bool(os.environ.get("DATA_GOV_IN_API_KEY") or os.environ.get("AGMARKNET_API_KEY")),
        "weather_provider": "open_meteo_with_imd_verification",
        "weather_api_configured": True,
    }
