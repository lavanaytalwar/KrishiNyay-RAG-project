import hashlib
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from live_data import (
    get_mandi_price_snapshot,
    get_weather_forecast,
    live_config_status,
)
from query_utils import canonical_for_routing
from rag_chain import RAGChain


ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("krishinyay.app")

app = FastAPI(
    title="KrishiNyay AI",
    description="Source-grounded RAG assistant for Indian agriculture schemes.",
    version="0.4.0",
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=1000)
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
]

REMAINING_WORK = [
    "Add Indic OCR language packs and validate on real Hindi/Marathi scanned official PDFs.",
    "Add stronger reranking and category/state/source-type filtering beyond the hybrid baseline.",
    "Add LangGraph workflows, voice/WhatsApp channels, and optional fine-tuning after enough validated data exists.",
]

RECENT_CAPABILITIES = [
    {
        "phase": "Phase 4",
        "title": "Farmer eval set",
        "metric": "100 questions",
        "summary": "Hindi, Hinglish, English, and regional-romanized farmer questions validate routes, topics, language coverage, and source types.",
    },
    {
        "phase": "Phase 5",
        "title": "Hybrid retrieval",
        "metric": "98% top-1 / 100% top-3",
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
]

VALIDATION_GATES = [
    {
        "name": "Farmer eval dataset",
        "command": "validate_farmer_eval.py",
        "result": "100 items valid",
    },
    {
        "name": "Farmer eval spot-check",
        "command": "validate_farmer_eval.py --spot-check",
        "result": "15 router/retriever checks passed",
    },
    {
        "name": "Hybrid retrieval gate",
        "command": "validate_phase5_retrieval.py",
        "result": "98% static top-1, 100% static top-3, 0 regressions",
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
        r"\b(gehu|wheat|rice|dhan|paddy|cotton|soybean|onion|potato|commodity)\b",
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
            "On this local setup, that should be Ollama with llama3.1:8b when Ollama is running. "
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
) -> Optional[dict]:
    normalized = canonical_for_routing(question)

    if re.search(r"\b(pmfby|fasal bima|crop insurance)\b", normalized, flags=re.I):
        return None

    if _matches_all(normalized, DYNAMIC_PATTERNS["pmkisan_status"]):
        return {
            "mode": "dynamic_router",
            "route": "dynamic_router",
            "route_reason": "pmkisan_live_status",
            "answer": (
                "This is live, farmer-specific information, so I should not answer it "
                "from the static knowledge base. Please check the official PM-KISAN "
                "Beneficiary Status page with your Aadhaar, registration number, or "
                "mobile details: https://pmkisan.gov.in/BeneficiaryStatus_New.aspx"
            ),
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
        return {
            "mode": "dynamic_router",
            "route": "dynamic_router",
            "route_reason": "pmkisan_installment_live_status",
            "answer": (
                "Instalment dates and payment status change over time, so I should not "
                "guess from old documents. Use the official PM-KISAN portal for current "
                "payment status and keep your Aadhaar or registration details ready."
            ),
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
        )

    if _matches_all(normalized, DYNAMIC_PATTERNS["weather"]):
        return get_weather_forecast(question, location=location, state=state)

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


@app.get("/health")
def health():
    try:
        chain = get_chain(4)
        stats = chain.store.stats()
        live_status = live_config_status()
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
            "dynamic_router": "enabled",
            "live_data": live_status,
            "phase": "Phase 7",
            "demo_ready": True,
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
        "motion_slots": MOTION_SLOTS,
        "media_note": (
            "The homepage uses lightweight CSS/HTML motion UI. Reference MP4s "
            "stay out of the app bundle unless deliberately converted into assets later."
        ),
    }


@app.post("/query")
def query(payload: QueryRequest):
    system_info = route_system_query(payload.question)
    if system_info:
        return {
            "question": payload.question,
            "answer": system_info["answer"],
            "sources": enrich_sources(system_info["sources"]),
            "mode": system_info["mode"],
            "route": system_info["route"],
            "route_reason": system_info["route_reason"],
            "n_chunks": 0,
            "llm_provider": system_info["llm_provider"],
        }

    dynamic = route_dynamic_query(
        payload.question,
        state=payload.state,
        location=payload.location,
        commodity=payload.commodity,
        district=payload.district,
        market=payload.market,
    )
    if dynamic:
        response = {
            "question": payload.question,
            "answer": dynamic["answer"],
            "sources": enrich_sources(dynamic["sources"]),
            "mode": dynamic["mode"],
            "route": dynamic["route"],
            "route_reason": dynamic["route_reason"],
            "n_chunks": 0,
            "llm_provider": "router",
        }
        for key in ["live_status", "data_provider", "fetched_at", "live_data"]:
            if key in dynamic:
                response[key] = dynamic[key]
        return response

    chain = get_chain(payload.n_results)
    response = chain.ask(
        payload.question,
        category=payload.category,
        state=payload.state,
    )
    response["sources"] = enrich_sources(response.get("sources", []))
    return response


@app.post("/ingest", response_model=IngestResponse)
def ingest(payload: IngestRequest):
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
