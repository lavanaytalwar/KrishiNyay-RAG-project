import hashlib
import json
import logging
import os
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from live_data import (
    get_mandi_price_snapshot,
    get_weather_forecast,
    live_config_status,
)
from language_policy import detect_answer_language, is_hinglish_language, normalise_answer_language
from ocr_utils import check_ocr_dependencies
from query_utils import canonical_for_routing
from rag_chain import RAGChain
from workflow import plan_query_workflow


ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("krishinyay.app")

app = FastAPI(
    title="KrishiNyay AI",
    description="Source-grounded RAG assistant for Indian agriculture schemes.",
    version="0.4.0",
)


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def demo_public_enabled() -> bool:
    return env_flag("DEMO_PUBLIC")


def configured_path_label(name: str, default: str) -> str:
    return os.environ.get(name, default).strip() or default


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=1000)
    conversation_id: Optional[str] = None
    workflow_context: Optional[dict[str, Any]] = None
    category: Optional[str] = None
    state: Optional[str] = None
    location: Optional[str] = None
    commodity: Optional[str] = None
    district: Optional[str] = None
    market: Optional[str] = None
    n_results: int = Field(default=4, ge=1, le=8)


class IngestRequest(BaseModel):
    title: str = Field(..., min_length=2, max_length=140)
    text: str = Field(..., min_length=40)
    url: str = ""
    category: str = "scheme"
    state: str = "central"
    doc_type: str = "User Document"


class IngestResponse(BaseModel):
    status: str
    doc_id: str
    chunks_added: int


PHASE_STATUS = [
    {
        "phase": "Phase 0",
        "title": "Repo hygiene and MiniLM baseline",
        "status": "completed",
        "summary": "Clean baseline with MiniLM-first retrieval, slim requirements, and ignored generated artifacts.",
    },
    {
        "phase": "Phase 1",
        "title": "Manual official PDF ingestion",
        "status": "completed",
        "summary": "Manifest-based ingestion for curated government PDFs with validation and rebuild workflow.",
    },
    {
        "phase": "Phase 2",
        "title": "Retrieval and routing stability",
        "status": "completed",
        "summary": "Query normalization, dynamic routing for live data, richer health metadata, and smoke validation.",
    },
    {
        "phase": "Phase 3",
        "title": "FastAPI and frontend demo polish",
        "status": "completed",
        "summary": "Demo-ready UI states, route transparency, optional motion assets, and deployment guidance.",
    },
    {
        "phase": "Phase 4",
        "title": "Farmer-facing evaluation set",
        "status": "completed",
        "summary": "100 realistic farmer questions with route, source-type, topic, and language validation.",
    },
    {
        "phase": "Phase 5",
        "title": "Retrieval quality upgrades",
        "status": "completed",
        "summary": "Hybrid MiniLM plus lexical retrieval, score fusion, and richer top-k validation metrics.",
    },
    {
        "phase": "Phase 6",
        "title": "OCR for scanned PDFs",
        "status": "completed",
        "summary": "Tesseract OCR path for scanned pages in manual official PDF ingestion, with image-PDF validation.",
    },
    {
        "phase": "Phase 7",
        "title": "Live mandi and weather APIs",
        "status": "completed",
        "summary": "Dynamic router fetches weather forecasts and optionally Agmarknet mandi prices, with safe portal fallback.",
    },
    {
        "phase": "Phase 8",
        "title": "Guarded workflow graph",
        "status": "completed",
        "summary": "Turn-level language policy, intent/slot planning, allowlisted tool routing, evidence verification, and LLM synthesis.",
    },
    {
        "phase": "Phase 9",
        "title": "Quality hardening",
        "status": "baseline complete",
        "summary": "Expanded 250-case eval coverage plus stricter source, state, scheme, legal, and dynamic-route guardrails.",
    },
    {
        "phase": "Phase 10",
        "title": "Answer and workflow reliability",
        "status": "baseline complete",
        "summary": "Answer-quality validation checks language, evidence metadata, source grounding, and follow-up behavior.",
    },
    {
        "phase": "Phase 11",
        "title": "Field UX polish",
        "status": "baseline complete",
        "summary": "Frontend exposes guarded workflow trace, field-message preview, and mobile-first chat refinements.",
    },
    {
        "phase": "Phase 12",
        "title": "Production readiness",
        "status": "baseline complete",
        "summary": "Structured query logs, readiness metadata, setup checks, and one-command regression suite.",
    },
]

REMAINING_WORK = [
    "Install Indic Tesseract language packs before strict Hindi/Marathi/Punjabi OCR validation.",
    "Add real scanned official Indic PDFs to the OCR fixture set when available.",
    "Evaluate reranking only if expanded retrieval misses show a measurable need.",
    "Add voice or WhatsApp channels only after text workflow quality remains stable.",
]

RECENT_CAPABILITIES = [
    {
        "phase": "Phase 4",
        "title": "Farmer eval set",
        "metric": "250 questions",
        "summary": "Hindi, Hinglish, English, and regional-romanized farmer questions validate routes, topics, language coverage, and source types.",
    },
    {
        "phase": "Phase 5",
        "title": "Hybrid retrieval",
        "metric": "93% top-1 / 99% top-3",
        "summary": "MiniLM remains primary while lexical boosts improve source selection without measured hybrid regressions.",
    },
    {
        "phase": "Phase 6",
        "title": "OCR ingestion",
        "metric": "Tesseract path",
        "summary": "Scanned official PDF pages can be rendered and OCR'd before entering the trusted manual ingestion pipeline.",
    },
    {
        "phase": "Generation",
        "title": "Local Ollama answers",
        "metric": "llama3.1:8b",
        "summary": "Retrieved chunks can now be synthesized locally, with explicit fallback metadata if local generation fails.",
    },
    {
        "phase": "Phase 7",
        "title": "Live data routing",
        "metric": "Weather live / mandi key-gated",
        "summary": "Weather uses live forecasts; mandi prices use Agmarknet when configured and otherwise return safe official-portal fallback.",
    },
    {
        "phase": "Phase 8",
        "title": "Guarded workflow",
        "metric": "Language → route → verify → synthesize",
        "summary": "The workflow sets answer language per turn, routes to allowlisted tools, verifies evidence, then synthesizes the final answer.",
    },
]

VALIDATION_GATES = [
    {
        "name": "Farmer eval dataset",
        "command": "validate_farmer_eval.py",
        "result": "250 items valid",
    },
    {
        "name": "Farmer eval spot-check",
        "command": "validate_farmer_eval.py --spot-check",
        "result": "21 router/retriever checks passed",
    },
    {
        "name": "Hybrid retrieval gate",
        "command": "validate_phase5_retrieval.py",
        "result": "93% static top-1, 99% static top-3, 0 hybrid regressions",
    },
    {
        "name": "OCR pipeline",
        "command": "validate_ocr_pipeline.py",
        "result": "Image-PDF OCR path validated",
    },
    {
        "name": "Local generation",
        "command": "validate_generation.py --provider ollama",
        "result": "3/3 Ollama generation cases passed",
    },
    {
        "name": "Live data",
        "command": "validate_phase7_live.py",
        "result": "Mandi/weather metadata and safe fallbacks passed",
    },
    {
        "name": "Workflow gate",
        "command": "validate_phase8_workflows.py",
        "result": "Intent, slot filling, clarification, and follow-up routing passed",
    },
    {
        "name": "Answer quality",
        "command": "validate_answer_quality.py",
        "result": "Language, evidence, source, and follow-up quality checks passed",
    },
    {
        "name": "Setup readiness",
        "command": "validate_setup_readiness.py",
        "result": "Required local dependencies checked with clear optional setup warnings",
    },
    {
        "name": "Public demo readiness",
        "command": "validate_public_demo.py",
        "result": "Packaged Chroma index, public ingest lock, static RAG, and workflow follow-up checked",
    },
    {
        "name": "Full regression suite",
        "command": "run_regression_suite.py",
        "result": "Runs all local regression gates in one command",
    },
    {
        "name": "Corpus smoke",
        "command": "validate_corpus.py",
        "result": "15/15 checks passed",
    },
]

API_KEY_GUIDE = {
    "title": "Mandi API key setup",
    "provider": "Data.gov.in / Open Government Data Platform India",
    "steps": [
        "Create or sign in to data.gov.in using Login/Register.",
        "Open the user dashboard/profile and copy the API key.",
        "Paste it into .env as DATA_GOV_IN_API_KEY or AGMARKNET_API_KEY.",
        "Restart the FastAPI backend and confirm /health shows mandi_api_configured=true.",
    ],
    "env": "DATA_GOV_IN_API_KEY=your_key_here",
    "note": "Weather does not need a key. PM-KISAN status remains a portal route because it is farmer-specific.",
}

PUBLIC_DEMO_FLOW = [
    "Open Home and show the farmer-first prompt buttons.",
    "Ask a PM-KISAN amount question and show official-source RAG evidence.",
    "Ask a land acquisition rights question and show legal-source routing.",
    "Ask 'Kal baarish hogi kya, spraying karu?' and provide Jaipur when asked.",
    "Open System to show MiniLM, Chroma, hybrid retrieval, Gemini, and validation readiness.",
]

DEMO_QUESTIONS = [
    {
        "label": "PM-KISAN amount",
        "question": "PM-KISAN mein kitne paise milte hain aur kab?",
        "kind": "rag",
    },
    {
        "label": "PMFBY claim",
        "question": "How do I claim PMFBY crop insurance after flood damage?",
        "kind": "rag",
    },
    {
        "label": "Live status",
        "question": "Mera PM-KISAN beneficiary status kya hai?",
        "kind": "dynamic_router",
    },
    {
        "label": "Mandi price",
        "question": "Aaj soybean ka mandi bhav kya hai?",
        "kind": "dynamic_router",
    },
    {
        "label": "Weather advisory",
        "question": "Kal baarish hogi kya, spraying karu?",
        "kind": "dynamic_router",
    },
    {
        "label": "Legal rights",
        "question": "Land acquisition mein farmer ke rights kya hain?",
        "kind": "rag",
    },
]

MOTION_SLOTS = [
    {
        "slot": "hero",
        "label": "Farmer AI assistant motion",
        "style": "css_farmer_ai",
    },
    {
        "slot": "rag-flow",
        "label": "RAG pipeline motion",
        "style": "css_pipeline",
    },
    {
        "slot": "router",
        "label": "Dynamic router motion",
        "style": "css_router",
    },
    {
        "slot": "retrieval",
        "label": "Retrieval motion",
        "style": "css_sticky_retrieval",
    },
    {
        "slot": "sources",
        "label": "Sources and citations motion",
        "style": "css_source_stack",
    },
]


DYNAMIC_PATTERNS = {
    "pmkisan_status": [
        r"\b(status|beneficiary status|registration status|naam|name|list|rejected|rejection|check)\b",
        r"\b(pm[\s-]?kisan|kisan)\b",
    ],
    "installment": [
        r"\b(kist|installment|instalment|payment status|kab aayegi|kab aaegi|credited)\b",
        r"\b(pm[\s-]?kisan)\b",
    ],
    "mandi_price": [
        r"\b(price|bhav|mandi|rate|aaj ka bhav|market)\b",
        r"\b(gehu|wheat|rice|dhan|paddy|cotton|soybean|soyabean|onion|potato|commodity)\b",
    ],
    "weather": [
        r"\b(weather|rain|baarish|barish|paus|mausam|temperature|forecast|spray|spraying)\b",
        r"\b(today|tomorrow|kal|aaj|parso|forecast|spray|spraying|temperature|paus)\b",
    ],
}


SYSTEM_INFO_PATTERNS = [
    r"\b(which|what)\s+(model|llm)\s+(is this|are you using|are you)\b",
    r"\b(model|llm)\s+(name|provider)\b",
    r"\b(which|what)\s+ai\s+model\b",
    r"\b(is this|are you using)\s+(ollama|llama|claude|gemini|openrouter)\b",
    r"\bwhich\s+provider\b",
]

AGRICULTURE_MODEL_CONTEXT = [
    "yield",
    "pmfby",
    "fasal",
    "bima",
    "crop",
    "scheme",
    "insurance",
    "estimation",
]


def _matches_all(question: str, patterns: list[str]) -> bool:
    return all(re.search(pattern, question, flags=re.I) for pattern in patterns)


def route_system_query(question: str) -> Optional[dict]:
    normalized = canonical_for_routing(question)
    if not any(re.search(pattern, normalized, flags=re.I) for pattern in SYSTEM_INFO_PATTERNS):
        return None
    if any(term in normalized for term in AGRICULTURE_MODEL_CONTEXT):
        return None

    chain = get_chain(4)
    stats = chain.store.stats()
    live_status = live_config_status()
    retrieval_mode = stats.get("retrieval_mode", "vector_only")
    embedding_backend = stats.get("embedding_backend", "unknown")
    total_chunks = stats.get("total_chunks", "unknown")
    mandi_status = "configured" if live_status.get("mandi_api_configured") else "not configured"

    return {
        "mode": "system_info",
        "route": "system_info",
        "route_reason": "system_model_info",
        "answer": (
            "This is KrishiNyay AI. Retrieval uses "
            f"{embedding_backend} embeddings with {retrieval_mode} over {total_chunks} indexed chunks. "
            f"Answer generation is currently configured as {chain.llm_provider}. "
            "Local demos can use Ollama with llama3.1:8b; public hosted demos should use Gemini. "
            f"Weather uses {live_status.get('weather_provider', 'the configured weather provider')}; "
            f"mandi prices use {live_status.get('mandi_provider', 'the configured mandi provider')} "
            f"when the API key is {mandi_status}."
        ),
        "sources": [
            {
                "display": "KrishiNyay runtime configuration",
                "url": "",
                "similarity": None,
                "category": "system",
                "state": "local",
                "source": "runtime_metadata",
                "doc_type": "System Metadata",
                "text": "Runtime metadata from the local FastAPI configuration, retrieval store, and live-data settings.",
            }
        ],
        "llm_provider": chain.llm_provider,
    }


def route_dynamic_query(
    question: str,
    *,
    state: Optional[str] = None,
    location: Optional[str] = None,
    commodity: Optional[str] = None,
    district: Optional[str] = None,
    market: Optional[str] = None,
    answer_language: Optional[str] = None,
) -> Optional[dict]:
    normalized = canonical_for_routing(question)
    resolved_answer_language = normalise_answer_language(answer_language) or detect_answer_language(question)
    farmer_hindi = is_hinglish_language(resolved_answer_language)

    if re.search(r"\b(pmfby|fasal bima|crop insurance)\b", normalized, flags=re.I):
        return None

    if _matches_all(normalized, DYNAMIC_PATTERNS["pmkisan_status"]):
        answer = (
            "Yeh live, farmer-specific information hai, isliye main static documents se "
            "status guess nahi karunga. Aadhaar, registration number, ya mobile details ke "
            "saath official PM-KISAN Beneficiary Status page check karein: "
            "https://pmkisan.gov.in/BeneficiaryStatus_New.aspx"
            if farmer_hindi
            else (
                "This is live, farmer-specific information, so I should not answer it "
                "from the static knowledge base. Please check the official PM-KISAN "
                "Beneficiary Status page with your Aadhaar, registration number, or "
                "mobile details: https://pmkisan.gov.in/BeneficiaryStatus_New.aspx"
            )
        )
        return {
            "mode": "dynamic_router",
            "route": "dynamic_router",
            "route_reason": "pmkisan_live_status",
            "answer": answer,
            "answer_language": resolved_answer_language,
            "sources": [
                {
                    "display": "PM-KISAN Beneficiary Status",
                    "url": "https://pmkisan.gov.in/BeneficiaryStatus_New.aspx",
                    "similarity": None,
                    "category": "live_status",
                    "state": "central",
                    "source": "official_portal",
                    "doc_type": "Live Portal",
                    "text": "Live beneficiary status must be checked on the official PM-KISAN portal.",
                }
            ],
        }

    if _matches_all(normalized, DYNAMIC_PATTERNS["installment"]):
        if re.search(r"\b(amount|kitna|kitne|paisa|paise|milta|milte|6000|rs|rupees|rule|hisson)\b", normalized, flags=re.I):
            return None
        answer = (
            "Instalment dates aur payment status time ke saath change hote hain, isliye "
            "main purane documents se guess nahi karunga. Current payment status ke liye "
            "official PM-KISAN portal use karein aur Aadhaar/registration details ready rakhein."
            if farmer_hindi
            else (
                "Instalment dates and payment status change over time, so I should not "
                "guess from old documents. Use the official PM-KISAN portal for current "
                "payment status and keep your Aadhaar or registration details ready."
            )
        )
        return {
            "mode": "dynamic_router",
            "route": "dynamic_router",
            "route_reason": "pmkisan_installment_live_status",
            "answer": answer,
            "answer_language": resolved_answer_language,
            "sources": [
                {
                    "display": "PM-KISAN Official Portal",
                    "url": "https://pmkisan.gov.in/",
                    "similarity": None,
                    "category": "live_status",
                    "state": "central",
                    "source": "official_portal",
                    "doc_type": "Live Portal",
                    "text": "Payment status and instalment timing are dynamic government records.",
                }
            ],
        }

    if _matches_all(normalized, DYNAMIC_PATTERNS["mandi_price"]):
        return get_mandi_price_snapshot(
            question,
            commodity=commodity,
            state=state,
            district=district,
            market=market,
            answer_language=resolved_answer_language,
        )

    if _matches_all(normalized, DYNAMIC_PATTERNS["weather"]):
        return get_weather_forecast(
            question,
            location=location,
            state=state,
            answer_language=resolved_answer_language,
        )

    return None


@lru_cache(maxsize=1)
def get_chain(n_results: int) -> RAGChain:
    return RAGChain(n_results=n_results)


def infer_doc_type(source: dict) -> str:
    if source.get("doc_type") == "System Metadata":
        return "System Metadata"
    text = " ".join(
        str(source.get(key, "")) for key in ["display", "source", "url", "category", "doc_type"]
    ).lower()
    if "faq" in text:
        return "FAQ"
    if "guideline" in text or "pdf" in text:
        return "Guidelines"
    if "live" in text or "portal" in text:
        return "Live Portal"
    if "state" in text or source.get("state") not in ("", "central", None):
        return "State Scheme"
    return source.get("doc_type") or "Scheme"


def enrich_sources(sources: list[dict]) -> list[dict]:
    enriched = []
    for source in sources:
        item = dict(source)
        item["doc_type"] = infer_doc_type(item)
        if item.get("text"):
            item["text"] = item["text"][:900]
        enriched.append(item)
    return enriched


def split_live_document(text: str, chunk_size: int = 650, overlap: int = 80) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= chunk_size:
        return [cleaned]

    chunks = []
    start = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        boundary = cleaned.rfind(". ", start, end)
        if boundary > start + 180:
            end = boundary + 1
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(cleaned):
            break
        start = max(0, end - overlap)
    return chunks


def setup_readiness(chain: RAGChain, live_status: dict[str, Any]) -> dict[str, Any]:
    ocr = check_ocr_dependencies()
    languages = set(ocr.get("tesseract_languages") or [])
    stats = chain.store.stats()
    public_mode = demo_public_enabled()
    configured_provider = os.environ.get("LLM_PROVIDER", "auto").strip().lower() or "auto"
    required_paths = {
        "eval_dataset": (ROOT / "eval" / "farmer_questions.jsonl").exists(),
        "web_ui": (WEB_DIR / "index.html").exists(),
        "frontend_js": (WEB_DIR / "app.js").exists(),
        "frontend_css": (WEB_DIR / "styles.css").exists(),
        "sample_data": (ROOT / "sample_data").exists(),
    }
    public_paths = {
        "demo_chroma": (ROOT / configured_path_label("CHROMA_PATH", "chroma_db")).exists(),
        "demo_chunks": (ROOT / configured_path_label("CHUNKS_DIR", "data/chunks") / "all_chunks.jsonl").exists(),
    }
    public_demo_ready = (
        public_mode
        and stats.get("total_chunks") == 1748
        and stats.get("lexical_chunks") == 1748
        and stats.get("embedding_dim") == 384
        and chain.llm_provider == "gemini"
        and not live_ingest_enabled()
        and all(public_paths.values())
    )
    return {
        "phase": "Phase 12",
        "demo_public": public_mode,
        "public_demo_ready": public_demo_ready,
        "configured_llm_provider": configured_provider,
        "ollama_generation_ready": chain.llm_provider.startswith("ollama:"),
        "gemini_generation_ready": chain.llm_provider == "gemini",
        "llm_provider": chain.llm_provider,
        "chroma_path": stats.get("chroma_path"),
        "chroma_runtime_path": stats.get("chroma_runtime_path"),
        "chunks_dir": stats.get("chunks_dir"),
        "ocr_ready": bool(ocr.get("available")),
        "ocr_engine": ocr.get("engine") or "",
        "ocr_languages": sorted(languages),
        "indic_ocr_ready": {"hin", "mar", "pan"}.issubset(languages),
        "mandi_api_configured": bool(live_status.get("mandi_api_configured")),
        "weather_api_configured": bool(live_status.get("weather_api_configured")),
        "live_ingest_enabled": live_ingest_enabled(),
        "live_ingest_requires_token": bool(os.environ.get("LIVE_INGEST_TOKEN", "").strip()),
        "required_paths": required_paths,
        "required_paths_ready": all(required_paths.values()),
        "public_paths": public_paths,
        "public_paths_ready": all(public_paths.values()),
        "notes": [
            "Fine-tuning is intentionally not part of the active roadmap.",
            "Install tesseract-lang before strict Indic OCR validation.",
            "Set DATA_GOV_IN_API_KEY or AGMARKNET_API_KEY for live mandi prices.",
            "Set LLM_PROVIDER=gemini and GEMINI_API_KEY for the hosted public demo.",
            "Set ENABLE_LIVE_INGEST=true only for trusted demo/admin environments.",
        ],
    }


def live_ingest_enabled() -> bool:
    if demo_public_enabled():
        return False
    return env_flag("ENABLE_LIVE_INGEST")


def verify_live_ingest_access(token: Optional[str]) -> None:
    if demo_public_enabled():
        raise HTTPException(
            status_code=403,
            detail="Live ingest is disabled in public demo mode.",
        )
    if not live_ingest_enabled():
        raise HTTPException(
            status_code=403,
            detail="Live ingest is disabled. Set ENABLE_LIVE_INGEST=true only in a trusted demo/admin environment.",
        )
    expected_token = os.environ.get("LIVE_INGEST_TOKEN", "").strip()
    if expected_token and token != expected_token:
        raise HTTPException(status_code=403, detail="Invalid or missing live ingest token.")


def source_type_summary(sources: list[dict[str, Any]]) -> list[str]:
    values = []
    for source in sources:
        value = source.get("doc_type") or source.get("category") or source.get("source") or "source"
        if value not in values:
            values.append(str(value))
    return values


def finalize_query_response(response: dict[str, Any], started_at: float) -> dict[str, Any]:
    sources = response.get("sources") or []
    response["latency_ms"] = round((time.perf_counter() - started_at) * 1000, 1)
    response["source_count"] = len(sources)
    response["source_types"] = source_type_summary(sources)

    log.info(
        "query_event %s",
        json.dumps(
            {
                "route": response.get("route"),
                "intent": response.get("intent"),
                "answer_language": response.get("answer_language"),
                "answer_kind": response.get("answer_kind"),
                "tool_used": response.get("tool_used"),
                "llm_provider": response.get("llm_provider"),
                "generation_status": response.get("generation_status"),
                "evidence_verified": response.get("evidence_verified"),
                "source_count": response.get("source_count"),
                "source_types": response.get("source_types"),
                "latency_ms": response.get("latency_ms"),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
    )
    return response


@app.get("/health")
def health():
    try:
        chain = get_chain(4)
        stats = chain.store.stats()
        live_status = live_config_status()
        readiness = setup_readiness(chain, live_status)
        return {
            "status": "ok",
            "app": "KrishiNyay AI",
            "total_chunks": stats["total_chunks"],
            "collection": stats["collection"],
            "embedding_backend": stats["embedding_backend"],
            "embedding_dim": stats.get("embedding_dim"),
            "llm_provider": chain.llm_provider,
            "retrieval_mode": stats.get("retrieval_mode", "vector_only"),
            "lexical_chunks": stats.get("lexical_chunks", 0),
            "chroma_path": stats.get("chroma_path"),
            "chroma_runtime_path": stats.get("chroma_runtime_path"),
            "chunks_dir": stats.get("chunks_dir"),
            "dynamic_router": "enabled",
            "live_data": live_status,
            "phase": "Phase 12",
            "workflow": "enabled",
            "quality_hardening": "enabled",
            "fine_tuning": "not_required",
            "readiness": readiness,
            "demo_public": demo_public_enabled(),
            "public_demo_ready": readiness.get("public_demo_ready", False),
            "demo_ready": readiness.get("public_demo_ready", True) if demo_public_enabled() else True,
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/demo-config")
def demo_config():
    return {
        "app": "KrishiNyay AI",
        "phase_status": PHASE_STATUS,
        "recent_capabilities": RECENT_CAPABILITIES,
        "validation_gates": VALIDATION_GATES,
        "api_key_guide": API_KEY_GUIDE,
        "remaining_work": REMAINING_WORK,
        "demo_questions": DEMO_QUESTIONS,
        "public_demo": demo_public_enabled(),
        "demo_hosting": "hugging_face_spaces",
        "recommended_demo_flow": PUBLIC_DEMO_FLOW,
        "public_safety_notice": (
            "Public demo mode is read-only. It uses official/public sources, "
            "keeps live ingest disabled, and should be verified against official portals."
        ),
        "motion_slots": MOTION_SLOTS,
        "media_note": (
            "The homepage uses lightweight CSS/HTML motion UI. Reference MP4s "
            "stay out of the app bundle unless deliberately converted into assets later."
        ),
    }


def workflow_metadata(
    decision: dict[str, Any],
    payload: QueryRequest,
    *,
    answer_kind: Optional[str] = None,
    tool_used: Optional[str] = None,
) -> dict[str, Any]:
    metadata = {
        "conversation_id": payload.conversation_id,
        "intent": decision["intent"],
        "workflow_state": decision["workflow_state"],
        "missing_fields": decision["missing_fields"],
        "filled_slots": decision["filled_slots"],
        "tool_used": tool_used if tool_used is not None else decision["tool_used"],
        "answer_kind": answer_kind if answer_kind is not None else decision["answer_kind"],
        "answer_language": decision["answer_language"],
        "workflow_context": decision["workflow_context"],
    }
    return metadata


def verify_workflow_evidence(
    *,
    decision: dict[str, Any],
    route: str,
    sources: list[dict],
    live_status: Optional[str] = None,
) -> dict[str, Any]:
    expected_category_by_intent = {
        "weather": "weather",
        "mandi_price": "market_prices",
        "pmkisan_status": "live_status",
    }
    expected_category = expected_category_by_intent.get(decision["intent"])
    route_match = route == decision["route"]
    has_sources = bool(sources)
    category_match = True
    if expected_category:
        category_match = any(source.get("category") == expected_category for source in sources)

    passed = route_match and has_sources and category_match
    if decision["action"] == "rag":
        passed = route == "rag" and has_sources

    status = "passed" if passed else "failed"
    if live_status == "needs_more_input":
        status = "needs_more_input"

    return {
        "status": status,
        "route_match": route_match,
        "has_sources": has_sources,
        "expected_category": expected_category,
        "category_match": category_match,
        "live_status": live_status,
    }


@app.post("/query")
def query(payload: QueryRequest):
    started_at = time.perf_counter()
    decision = plan_query_workflow(
        payload.question,
        workflow_context=payload.workflow_context,
        state=payload.state,
        location=payload.location,
        commodity=payload.commodity,
        district=payload.district,
        market=payload.market,
    )

    if decision["action"] == "clarification":
        verifier = {
            "status": "needs_more_input",
            "route_match": True,
            "has_sources": False,
            "expected_category": None,
            "category_match": True,
            "live_status": None,
        }
        return finalize_query_response({
            "question": payload.question,
            "answer": decision["answer"],
            "sources": [],
            "mode": "workflow",
            "route": decision["route"],
            "route_reason": decision["route_reason"],
            "n_chunks": 0,
            "llm_provider": "workflow",
            "evidence_verified": False,
            "evidence_verifier": verifier,
            **workflow_metadata(decision, payload),
        }, started_at)

    if decision["action"] == "system_info":
        system_info = route_system_query(decision["question"])
        if system_info:
            verifier = {
                "status": "passed",
                "route_match": True,
                "has_sources": True,
                "expected_category": "system",
                "category_match": True,
                "live_status": None,
            }
            return finalize_query_response({
                "question": payload.question,
                "answer": system_info["answer"],
                "sources": enrich_sources(system_info["sources"]),
                "mode": system_info["mode"],
                "route": system_info["route"],
                "route_reason": system_info["route_reason"],
                "n_chunks": 0,
                "llm_provider": system_info["llm_provider"],
                "evidence_verified": True,
                "evidence_verifier": verifier,
                **workflow_metadata(decision, payload),
            }, started_at)

    if decision["action"] == "dynamic":
        slots = decision["filled_slots"]
        dynamic = route_dynamic_query(
            decision["question"],
            state=slots.get("state") or payload.state,
            location=slots.get("location") or payload.location,
            commodity=slots.get("commodity") or payload.commodity,
            district=slots.get("district") or payload.district,
            market=slots.get("market") or payload.market,
            answer_language=decision["answer_language"],
        )
        if dynamic:
            sources = enrich_sources(dynamic["sources"])
            verifier = verify_workflow_evidence(
                decision=decision,
                route=dynamic["route"],
                sources=sources,
                live_status=dynamic.get("live_status"),
            )
            synthesis = {
                "answer": dynamic["answer"],
                "llm_provider": "router",
                "generation_status": "router_direct",
                "generation_error": None,
            }
            if verifier["status"] == "passed":
                chain = get_chain(payload.n_results)
                synthesis = chain.synthesize_from_evidence(
                    question=payload.question,
                    evidence=dynamic["answer"],
                    sources=sources,
                    answer_language=decision["answer_language"],
                )
            generation_status = synthesis.get("generation_status", "")
            response = {
                "question": payload.question,
                "answer": synthesis["answer"],
                "sources": sources,
                "mode": dynamic["mode"],
                "route": dynamic["route"],
                "route_reason": dynamic["route_reason"],
                "n_chunks": 0,
                "llm_provider": synthesis.get("llm_provider", "router"),
                "generation_status": generation_status,
                "generation_error": synthesis.get("generation_error"),
                "evidence_verified": verifier["status"] == "passed",
                "evidence_verifier": verifier,
                **workflow_metadata(
                    decision,
                    payload,
                    answer_kind="generated" if generation_status.startswith("generated") else "router_direct",
                ),
            }
            for key in ["live_status", "data_provider", "fetched_at", "live_data"]:
                if key in dynamic:
                    response[key] = dynamic[key]
            return finalize_query_response(response, started_at)

        return finalize_query_response({
            "question": payload.question,
            "answer": "I understood this as a live-data question, but I could not route it safely. Please add the missing commodity, location, or scheme details.",
            "sources": [],
            "mode": "workflow",
            "route": "workflow",
            "route_reason": "dynamic_route_unavailable",
            "n_chunks": 0,
            "llm_provider": "workflow",
            "evidence_verified": False,
            "evidence_verifier": {
                "status": "failed",
                "route_match": False,
                "has_sources": False,
                "expected_category": None,
                "category_match": False,
                "live_status": None,
            },
            **workflow_metadata(decision, payload, answer_kind="clarification"),
        }, started_at)

    chain = get_chain(payload.n_results)
    response = chain.ask(
        decision["question"],
        category=payload.category,
        state=decision["filled_slots"].get("state") or payload.state,
        answer_language=decision["answer_language"],
    )
    response["sources"] = enrich_sources(response.get("sources", []))
    verifier = verify_workflow_evidence(
        decision=decision,
        route=response.get("route", ""),
        sources=response["sources"],
    )
    generation_status = response.get("generation_status", "")
    response["evidence_verified"] = verifier["status"] == "passed"
    response["evidence_verifier"] = verifier
    response.update(
        workflow_metadata(
            decision,
            payload,
            answer_kind="generated" if generation_status.startswith("generated") else "template_fallback",
        )
    )
    return finalize_query_response(response, started_at)


@app.post("/ingest", response_model=IngestResponse)
def ingest(payload: IngestRequest, x_ingest_token: Optional[str] = Header(default=None)):
    verify_live_ingest_access(x_ingest_token)
    try:
        chain = get_chain(4)
        chunks = split_live_document(payload.text)
        digest = hashlib.sha1(f"{payload.title}:{payload.text}".encode("utf-8")).hexdigest()[:10]
        doc_id = f"live_{digest}"
        ids = [f"{doc_id}_c{i}" for i in range(len(chunks))]
        embeddings = [chain.store.embed_fn(chunk) for chunk in chunks]
        metadatas = [
            {
                "doc_id": doc_id,
                "source": "live_ingest",
                "display": payload.title,
                "url": payload.url,
                "category": payload.category,
                "state": payload.state,
                "priority": "live",
                "language": "mixed",
                "chunk_idx": str(i),
                "doc_type": payload.doc_type,
            }
            for i in range(len(chunks))
        ]
        chain.store.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )
        live_records = []
        for chunk_id, chunk, metadata in zip(ids, chunks, metadatas):
            searchable = " ".join(
                str(metadata.get(key, ""))
                for key in ["display", "source", "category", "state", "language", "doc_type"]
            )
            searchable = f"{chunk} {searchable}"
            live_records.append({
                "id": chunk_id,
                "text": chunk,
                "source": metadata["source"],
                "display": metadata["display"],
                "url": metadata["url"],
                "category": metadata["category"],
                "state": metadata["state"],
                "priority": metadata["priority"],
                "tokens": chain.store._tokenize(searchable),
            })
        if hasattr(chain.store, "chunk_records"):
            live_ids = set(ids)
            chain.store.chunk_records = [
                record for record in chain.store.chunk_records if record.get("id") not in live_ids
            ]
            chain.store.chunk_records.extend(live_records)
        return IngestResponse(status="indexed", doc_id=doc_id, chunks_added=len(chunks))
    except Exception as exc:
        log.exception("Live ingest failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


app.mount("/assets", StaticFiles(directory=str(WEB_DIR)), name="assets")


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/{page_name}")
def page(page_name: str):
    if page_name in {"chat", "knowledge", "ingest", "system"}:
        return FileResponse(WEB_DIR / "index.html")
    raise HTTPException(status_code=404, detail="Not found")
