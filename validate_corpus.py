"""
KrishiNyay — validate_corpus.py
Run 12 Hindi + English test queries and print top-2 retrieved chunks.
This is your Phase 2 proof-of-work — run this after chunk_and_embed.py.

Run: python validate_corpus.py
Run with LLM answers: python validate_corpus.py --with-llm
"""

import sys, logging, argparse
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.WARNING)  # quiet for clean output
log = logging.getLogger("krishinyay.validate")

from query_utils import normalize_query


ROUTE_RAG = "rag"
ROUTE_DYNAMIC = "dynamic_router"
ROUTE_FUTURE_LIVE = "future_live_api"


# ── Test queries covering retrieval and live-data routing ──────────────────
TEST_QUERIES = [
    # Central schemes — Hindi
    {
        "question": "PM-KISAN ke liye kaun eligible hai?",
        "category": "income_support",
        "route": ROUTE_RAG,
        "expectation": "Should retrieve PM-KISAN eligibility info",
    },
    {
        "question": "PM-KISAN mein kitne paise milte hain aur kab?",
        "category": "income_support",
        "route": ROUTE_RAG,
        "expectation": "Should retrieve ₹6000/year in 3 installments",
    },

    # Central schemes — English
    {
        "question": "What documents do I need to apply for PM-KISAN?",
        "category": "income_support",
        "route": ROUTE_RAG,
        "expectation": "Should retrieve Aadhaar, land records, bank details",
    },
    {
        "question": "How do I claim PMFBY crop insurance after flood damage?",
        "category": "crop_insurance",
        "route": ROUTE_RAG,
        "expectation": "Should retrieve 72-hour intimation, claim process",
    },

    # State schemes
    {
        "question": "Maharashtra mein farmers ko kya extra scheme milti hai?",
        "category": None,
        "route": ROUTE_RAG,
        "expectation": "Should retrieve Namo Shetkari ₹6000 extra",
    },

    # Crop insurance Hindi
    {
        "question": "Fasal bima ka premium kitna hota hai kharif crops ke liye?",
        "category": "crop_insurance",
        "route": ROUTE_RAG,
        "expectation": "Should retrieve 2% of sum insured",
    },

    # Legal
    {
        "question": "Tribal farmers ke liye forest rights kya hain?",
        "category": None,
        "route": ROUTE_RAG,
        "expectation": "Should retrieve FRA 2006 content",
    },
    {
        "question": "Zameen khareedne par kisan ke kya adhikar hain?",
        "category": None,
        "route": ROUTE_RAG,
        "expectation": "Should retrieve land acquisition rights",
    },

    # Financial
    {
        "question": "Kisan Credit Card ke liye kaise apply karein?",
        "category": "credit",
        "route": ROUTE_RAG,
        "expectation": "Should retrieve KCC application process",
    },

    # Agri science
    {
        "question": "Cotton crop mein pest attack ke liye kya karein?",
        "category": None,
        "route": ROUTE_RAG,
        "expectation": "Should retrieve crop management content",
    },

    # Mixed Hindi-English (Hinglish)
    {
        "question": "PM-KISAN ka helpline number kya hai?",
        "category": "income_support",
        "route": ROUTE_RAG,
        "expectation": "Should retrieve 155261 / 1800115526",
    },
    {
        "question": "PMFBY claim karne ke baad kitne din mein paise milte hain?",
        "category": "crop_insurance",
        "route": ROUTE_RAG,
        "expectation": "Should retrieve claim settlement timeline",
    },

    # Dynamic/live data router cases
    {
        "question": "Mera PM-KISAN beneficiary status kya hai?",
        "category": None,
        "route": ROUTE_DYNAMIC,
        "expectation": "Should route to official PM-KISAN live status portal",
    },
    {
        "question": "Aaj soybean ka mandi bhav kya hai?",
        "category": None,
        "route": ROUTE_DYNAMIC,
        "expectation": "Should route to live mandi/eNAM source",
    },
    {
        "question": "Kal baarish hogi kya, spraying karu?",
        "category": None,
        "route": ROUTE_DYNAMIC,
        "expectation": "Should route to live weather/advisory source",
    },
]


def _dynamic_route(question: str) -> Optional[dict]:
    try:
        from app import route_dynamic_query
        return route_dynamic_query(question)
    except Exception:
        return None


def run_validation(with_llm: bool = False):
    print()
    print("═" * 70)
    print("  KRISHINYAY — CORPUS VALIDATION")
    print(f"  {len(TEST_QUERIES)} queries  ·  RAG top-2 + dynamic routes")
    print("═" * 70)

    # Load vector store
    try:
        from vector_store import VectorStore
        vs = VectorStore()
        stats = vs.stats()
        print(f"\n  Vector store: {stats['total_chunks']} chunks indexed")
        print(f"  Embeddings  : {stats['embedding_backend']} ({stats.get('embedding_dim', 'unknown')} dim)")
    except Exception as e:
        print(f"\n  ✗ Could not load VectorStore: {e}")
        print("    Run: python chunk_and_embed.py first")
        return

    # Optionally load RAG chain for full LLM answers
    chain = None
    if with_llm:
        try:
            from rag_chain import RAGChain
            chain = RAGChain()
            print("  LLM: enabled")
        except Exception as e:
            print(f"  LLM: not available ({e})")

    # Run queries
    passed   = 0
    total    = len(TEST_QUERIES)
    all_sims = []

    for i, case in enumerate(TEST_QUERIES, 1):
        query = case["question"]
        category = case.get("category")
        expected_route = case.get("route", ROUTE_RAG)
        expectation = case["expectation"]

        print(f"\n{'─' * 70}")
        print(f"  [{i:02d}] {query}")
        print(f"       Route   : {expected_route}")
        print(f"       Expected: {expectation}")

        dynamic = _dynamic_route(query)
        if expected_route == ROUTE_DYNAMIC:
            ok = bool(dynamic and dynamic.get("route") == ROUTE_DYNAMIC)
            if ok:
                passed += 1
                print(f"  ✓ Dynamic route: {dynamic.get('route_reason', 'dynamic')}")
                print(f"       {dynamic['answer'][:160].strip()}...")
            else:
                print("  ✗  Expected dynamic route but query was not routed")
            continue

        if dynamic:
            print(f"  ✗  Unexpected dynamic route: {dynamic.get('route_reason', 'dynamic')}")
            continue

        normalized_query = normalize_query(query)
        results = vs.query(normalized_query, n=2, category=category)

        if not results:
            print("  ✗  No results returned")
            continue

        best_sim = results[0]["similarity"]
        all_sims.append(best_sim)

        # A result is "passing" if similarity > 0.1 (anything relevant)
        ok = best_sim > 0.05
        if ok:
            passed += 1

        status = "✓" if ok else "✗"
        for j, r in enumerate(results, 1):
            print(f"  {status if j == 1 else ' '} [{j}] {r['display']:35s} "
                  f"sim={r['similarity']:.3f}")
            print(f"       {r['text'][:110].strip()}...")

        if with_llm and chain:
            try:
                resp = chain.ask(query, category=category)
                print(f"\n  💬 Answer: {resp['answer'][:200].strip()}...")
            except Exception as e:
                print(f"\n  💬 LLM error: {e}")

    # Summary
    avg_sim = sum(all_sims) / len(all_sims) if all_sims else 0
    print(f"\n{'═' * 70}")
    print(f"  RESULTS: {passed}/{total} queries returned relevant chunks")
    print(f"  Avg similarity score : {avg_sim:.3f}")

    if passed == total:
        print("  ✅  All queries passing — RAG is working correctly")
    elif passed >= total * 0.7:
        print("  ⚠   Most queries passing — check the failed ones above")
        print("      Tip: add more documents to the knowledge base for those topics")
    else:
        print("  ✗   Low pass rate — check chunk_and_embed.py ran correctly")
        print("      Tip: run python chunk_and_embed.py --force to re-embed")

    print()
    print("  Next step: python evaluate_rag.py  (RAGAS metrics)")
    print("  Or jump to Phase 3: fine-tuning")
    print()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--with-llm", action="store_true",
                   help="Also generate LLM answers for each query")
    args = p.parse_args()
    run_validation(with_llm=args.with_llm)
