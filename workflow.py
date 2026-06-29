"""
Phase 8 workflow planner for conversational query handling.

This module intentionally stays lightweight and deterministic for the first
workflow baseline. It coordinates the existing router, live-data tools,
MiniLM/hybrid retrieval, and Ollama generation without replacing them.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from live_data import LOCATION_COORDS, STATE_CAPITAL_COORDS, parse_commodity, parse_state
from query_utils import canonical_for_routing


WEATHER_TERMS = (
    "weather",
    "rain",
    "baarish",
    "barish",
    "paus",
    "mausam",
    "temperature",
    "forecast",
    "spray",
    "spraying",
)
WEATHER_TIME_TERMS = ("today", "tomorrow", "kal", "aaj", "parso", "forecast")
MANDI_TERMS = ("mandi", "bhav", "price", "rate", "market")
PMKISAN_TERMS = ("pm-kisan", "pm kisan", "pmkisan", "kisan samman")
PMKISAN_DYNAMIC_TERMS = (
    "status",
    "beneficiary",
    "registration",
    "rejected",
    "rejection",
    "installment",
    "instalment",
    "payment",
    "kist",
    "credited",
    "list",
)
SYSTEM_INFO_PATTERNS = (
    r"\b(which|what)\s+(model|llm)\s+(is this|are you using|are you)\b",
    r"\b(model|llm)\s+(name|provider)\b",
    r"\b(which|what)\s+ai\s+model\b",
    r"\b(is this|are you using)\s+(ollama|llama|claude|gemini|openrouter)\b",
    r"\bwhich\s+provider\b",
)
AGRICULTURE_MODEL_CONTEXT = (
    "yield",
    "pmfby",
    "fasal",
    "bima",
    "crop",
    "scheme",
    "insurance",
    "estimation",
)
SCHEME_TERMS = (
    "pm-kisan",
    "pm kisan",
    "pmkisan",
    "kisan samman",
    "pmfby",
    "fasal bima",
    "crop insurance",
    "kcc",
    "kisan credit card",
    "fra",
    "forest rights",
    "land acquisition",
    "larr",
    "namo shetkari",
    "krishak bandhu",
)
ELIGIBILITY_TERMS = (
    "eligible",
    "eligibility",
    "patra",
    "benefit",
    "labh",
    "apply",
    "apply kar",
)
FOLLOW_UP_BLOCKED_TERMS = WEATHER_TERMS + MANDI_TERMS + PMKISAN_TERMS + SCHEME_TERMS


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(re.search(rf"\b{re.escape(term)}\b", text) for term in terms)


def _is_system_info(text: str) -> bool:
    if not any(re.search(pattern, text, flags=re.I) for pattern in SYSTEM_INFO_PATTERNS):
        return False
    return not _contains_any(text, AGRICULTURE_MODEL_CONTEXT)


def _has_scheme_marker(text: str) -> bool:
    return _contains_any(text, SCHEME_TERMS) or bool(parse_state(text))


def _is_vague_eligibility(text: str) -> bool:
    return _contains_any(text, ELIGIBILITY_TERMS) and not _has_scheme_marker(text)


def _is_short_follow_up(question: str) -> bool:
    normalized = question.strip()
    if not normalized or "?" in normalized:
        return False
    if len(normalized.split()) > 5:
        return False
    return not _contains_any(canonical_for_routing(normalized), FOLLOW_UP_BLOCKED_TERMS)


def _extract_location(question: str, explicit: Optional[str], state: Optional[str]) -> Optional[str]:
    if explicit:
        return explicit.strip()
    text = canonical_for_routing(question)
    for marker, (_, _, label) in LOCATION_COORDS.items():
        if re.search(rf"\b{re.escape(marker)}\b", text):
            return label
    state_name = parse_state(text, state)
    if state_name and state_name.lower() in STATE_CAPITAL_COORDS:
        return state_name
    return None


def _extract_slots(
    question: str,
    *,
    location: Optional[str] = None,
    commodity: Optional[str] = None,
    state: Optional[str] = None,
    district: Optional[str] = None,
    market: Optional[str] = None,
) -> dict[str, str]:
    slots: dict[str, str] = {}
    state_name = parse_state(question, state)
    commodity_name = parse_commodity(question, commodity)
    location_name = _extract_location(question, location, state_name)

    if location_name:
        slots["location"] = location_name
    if commodity_name:
        slots["commodity"] = commodity_name
    if state_name:
        slots["state"] = state_name
    if district:
        slots["district"] = district.strip()
    if market:
        slots["market"] = market.strip()
    return slots


def _classify_intent(question: str, slots: dict[str, str]) -> str:
    text = canonical_for_routing(question)
    if _is_system_info(text):
        return "system_info"
    if _contains_any(text, PMKISAN_TERMS) and _contains_any(text, PMKISAN_DYNAMIC_TERMS):
        return "pmkisan_status"
    if _contains_any(text, WEATHER_TERMS) and (
        _contains_any(text, WEATHER_TIME_TERMS) or "location" in slots
    ):
        return "weather"
    if _contains_any(text, MANDI_TERMS) or ("commodity" in slots and _contains_any(text, ("aaj", "today"))):
        return "mandi_price"
    if _is_vague_eligibility(text):
        return "eligibility"
    return "static_rag"


def _pending_context(
    *,
    intent: str,
    question: str,
    missing_fields: list[str],
    slots: dict[str, str],
) -> dict[str, Any]:
    return {
        "pending": {
            "intent": intent,
            "question": question,
            "missing_fields": missing_fields,
            "filled_slots": slots,
        },
        "last_intent": intent,
        "filled_slots": slots,
    }


def _clear_context(intent: str, slots: dict[str, str]) -> dict[str, Any]:
    return {
        "pending": None,
        "last_intent": intent,
        "filled_slots": slots,
    }


def _decision(
    *,
    action: str,
    intent: str,
    question: str,
    slots: dict[str, str],
    missing_fields: Optional[list[str]] = None,
    answer: str = "",
    route: str = "rag",
    route_reason: str = "workflow_decision",
    tool_used: Optional[str] = None,
    answer_kind: str = "generated",
    workflow_state: str = "complete",
    workflow_context: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    return {
        "action": action,
        "intent": intent,
        "question": question,
        "filled_slots": slots,
        "missing_fields": missing_fields or [],
        "answer": answer,
        "route": route,
        "route_reason": route_reason,
        "tool_used": tool_used,
        "answer_kind": answer_kind,
        "workflow_state": workflow_state,
        "workflow_context": workflow_context if workflow_context is not None else _clear_context(intent, slots),
    }


def _resume_pending(
    question: str,
    workflow_context: Optional[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    pending = (workflow_context or {}).get("pending")
    if not pending or not _is_short_follow_up(question):
        return None

    intent = pending.get("intent")
    original_question = pending.get("question") or question
    missing_fields = list(pending.get("missing_fields") or [])
    slots = dict(pending.get("filled_slots") or {})
    value = question.strip()

    if intent == "weather" and "location" in missing_fields:
        slots["location"] = value
        return _decision(
            action="dynamic",
            intent="weather",
            question=original_question,
            slots=slots,
            route="dynamic_router",
            route_reason="weather_live_data",
            tool_used="weather",
            answer_kind="router_direct",
            workflow_state="resumed",
        )

    if intent == "mandi_price" and "commodity" in missing_fields:
        commodity_name = parse_commodity(value) or value
        slots["commodity"] = commodity_name
        return _decision(
            action="dynamic",
            intent="mandi_price",
            question=f"{original_question} {value} {commodity_name}",
            slots=slots,
            route="dynamic_router",
            route_reason="mandi_price_live_data",
            tool_used="mandi",
            answer_kind="router_direct",
            workflow_state="resumed",
        )

    if intent == "eligibility" and "scheme_or_state" in missing_fields:
        combined_question = f"{value} eligibility criteria"
        combined_slots = _extract_slots(combined_question)
        if _has_scheme_marker(canonical_for_routing(combined_question)):
            return _decision(
                action="rag",
                intent="static_rag",
                question=combined_question,
                slots=combined_slots,
                route="rag",
                route_reason="workflow_resumed_static_rag",
                tool_used="rag",
                answer_kind="generated",
                workflow_state="resumed",
            )

    return None


def plan_query_workflow(
    question: str,
    *,
    workflow_context: Optional[dict[str, Any]] = None,
    state: Optional[str] = None,
    location: Optional[str] = None,
    commodity: Optional[str] = None,
    district: Optional[str] = None,
    market: Optional[str] = None,
) -> dict[str, Any]:
    resumed = _resume_pending(question, workflow_context)
    if resumed:
        return resumed

    slots = _extract_slots(
        question,
        location=location,
        commodity=commodity,
        state=state,
        district=district,
        market=market,
    )
    intent = _classify_intent(question, slots)

    if intent == "system_info":
        return _decision(
            action="system_info",
            intent=intent,
            question=question,
            slots=slots,
            route="system_info",
            route_reason="system_model_info",
            tool_used="system_info",
            answer_kind="router_direct",
        )

    if intent == "weather":
        if "location" not in slots and "state" not in slots:
            missing = ["location"]
            return _decision(
                action="clarification",
                intent=intent,
                question=question,
                slots=slots,
                missing_fields=missing,
                answer=(
                    "Mujhe aapka village/city ya district chahiye. Location milte hi "
                    "main live weather forecast check karke bataunga ki spraying avoid "
                    "karni chahiye ya safe hai."
                ),
                route="workflow",
                route_reason="weather_missing_location",
                tool_used="weather",
                answer_kind="clarification",
                workflow_state="awaiting_input",
                workflow_context=_pending_context(
                    intent=intent,
                    question=question,
                    missing_fields=missing,
                    slots=slots,
                ),
            )
        return _decision(
            action="dynamic",
            intent=intent,
            question=question,
            slots=slots,
            route="dynamic_router",
            route_reason="weather_live_data",
            tool_used="weather",
            answer_kind="router_direct",
        )

    if intent == "mandi_price":
        if "commodity" not in slots:
            missing = ["commodity"]
            return _decision(
                action="clarification",
                intent=intent,
                question=question,
                slots=slots,
                missing_fields=missing,
                answer=(
                    "Mandi bhav live hota hai. Kaunsi commodity ka bhav chahiye? "
                    "Agar state/district/market bhi batayenge to lookup zyada useful hoga."
                ),
                route="workflow",
                route_reason="mandi_missing_commodity",
                tool_used="mandi",
                answer_kind="clarification",
                workflow_state="awaiting_input",
                workflow_context=_pending_context(
                    intent=intent,
                    question=question,
                    missing_fields=missing,
                    slots=slots,
                ),
            )
        return _decision(
            action="dynamic",
            intent=intent,
            question=question,
            slots=slots,
            route="dynamic_router",
            route_reason="mandi_price_live_data",
            tool_used="mandi",
            answer_kind="router_direct",
        )

    if intent == "pmkisan_status":
        return _decision(
            action="dynamic",
            intent=intent,
            question=question,
            slots=slots,
            route="dynamic_router",
            route_reason="pmkisan_live_status",
            tool_used="pmkisan_portal",
            answer_kind="router_direct",
        )

    if intent == "eligibility":
        missing = ["scheme_or_state"]
        return _decision(
            action="clarification",
            intent=intent,
            question=question,
            slots=slots,
            missing_fields=missing,
            answer=(
                "Eligibility kis scheme ke liye check karni hai? Scheme aur state bataiye, "
                "jaise PM-KISAN, PMFBY, KCC, Maharashtra Namo Shetkari, ya koi aur rajya yojana."
            ),
            route="workflow",
            route_reason="eligibility_missing_scheme_or_state",
            answer_kind="clarification",
            workflow_state="awaiting_input",
            workflow_context=_pending_context(
                intent=intent,
                question=question,
                missing_fields=missing,
                slots=slots,
            ),
        )

    return _decision(
        action="rag",
        intent="static_rag",
        question=question,
        slots=slots,
        route="rag",
        route_reason="workflow_static_rag",
        tool_used="rag",
        answer_kind="generated",
    )
