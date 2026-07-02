"""
KrishiNyay — validate_corpus.py
Run Hindi + English test queries and print top retrieved chunks plus routing metrics.

Run: python validate_corpus.py
Run with LLM answers: python validate_corpus.py --with-llm
"""

import sys, logging, argparse
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("krishinyay.validate")

from query_utils import normalize_query


ROUTE_RAG = "rag"
ROUTE_DYNAMIC = "dynamic_router"
ROUTE_FUTURE_LIVE = "future_live_api"


TEST_QUERIES = [
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
    {
        "question": "Maharashtra mein farmers ko kya extra scheme milti hai?",
        "category": None,
        "route": ROUTE_RAG,
        "expectation": "Should retrieve Namo Shetkari ₹6000 extra",
    },
    {
        "question": "Fasal bima ka premium kitna hota hai kharif crops ke liye?",
        "category": "crop_insurance",
        "route": ROUTE_RAG,
        "expectation": "Should retrieve 2% of sum insured",
    },
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
    {
        "question": "Kisan Credit Card ke liye kaise apply karein?",
        "category": "credit",
        "route": ROUTE_RAG,
        "expectation": "Should retrieve KCC application process",
    },
    {
        "question": "Cotton crop mein pest attack ke liye kya karein?",
        "category": None,
        "route": ROUTE_RAG,
        "expectation": "Should retrieve crop management content",
    },
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


def _is_relevant(result: dict) -> bool:
    similarity = float(result.get("similarity") or 0.0)
    hybrid_score = float(result.get("hybrid_score") or 0.0)
    return similarity > 0.05 or hybrid_score > 0.20


def _format_scores(result: dict) -> str:
    return (
        f"sim={float(result.get('similarity') or 0):.3f} "
        f"hybrid={float(result.get('hybrid_score') or 0):.3f} "
        f"lex={float(result.get('lexical_score') or 0):.3f} "
        f"method={result.get('retrieval_method', 'vector')}"
    )


def run_validation(with_llm: bool = False) -> int:
    print()
    print("═" * 70)
    print("  KRISHINYAY — CORPUS VALIDATION")
    print(f"  {len(TEST_QUERIES)} queries  ·  RAG top-3 + dynamic routes")
    print("═" * 70)

    try:
        from vector_store import VectorStore
        vs = VectorStore()
        stats = vs.stats()
        print()
        print(f"  Vector store: {stats['total_chunks']} chunks indexed")
        print(f"  Embeddings  : {stats['embedding_backend']} ({stats.get('embedding_dim', 'unknown')} dim)")
        print(f"  Retrieval   : {stats.get('retrieval_mode', 'vector_only')}")
        if stats.get("lexical_chunks"):
            print(f"  Lexical idx : {stats['lexical_chunks']} chunks")
    except Exception as e:
        print()
        print(f"  ✗ Could not load VectorStore: {e}")
        print("    Run: python chunk_and_embed.py first")
        return 1

    chain = None
    llm_errors = 0
    if with_llm:
        try:
            from rag_chain import RAGChain
            chain = RAGChain()
            print("  LLM: enabled")
        except Exception as e:
            print(f"  LLM: not available ({e})")
            llm_errors += 1

    passed = 0
    total = len(TEST_QUERIES)
    route_correct = 0
    static_total = 0
    top1_hits = 0
    top3_hits = 0
    all_sims = []
    all_hybrid_scores = []

    for i, case in enumerate(TEST_QUERIES, 1):
        query = case["question"]
        category = case.get("category")
        expected_route = case.get("route", ROUTE_RAG)
        expectation = case["expectation"]

        print()
        print("─" * 70)
        print(f"  [{i:02d}] {query}")
        print(f"       Route   : {expected_route}")
        print(f"       Expected: {expectation}")

        dynamic = _dynamic_route(query)
        if expected_route == ROUTE_DYNAMIC:
            ok = bool(dynamic and dynamic.get("route") == ROUTE_DYNAMIC)
            if ok:
                passed += 1
                route_correct += 1
                print(f"  ✓ Dynamic route: {dynamic.get('route_reason', 'dynamic')}")
                print(f"       {dynamic['answer'][:160].strip()}...")
            else:
                print("  ✗  Expected dynamic route but query was not routed")
            continue

        static_total += 1
        if dynamic:
            print(f"  ✗  Unexpected dynamic route: {dynamic.get('route_reason', 'dynamic')}")
            continue
        route_correct += 1

        normalized_query = normalize_query(query)
        results = vs.query(normalized_query, n=3, category=category)

        if not results:
            print("  ✗  No results returned")
            continue

        first = results[0]
        first_relevant = _is_relevant(first)
        any_relevant = any(_is_relevant(result) for result in results)
        if first_relevant:
            top1_hits += 1
        if any_relevant:
            top3_hits += 1
            passed += 1

        all_sims.append(float(first.get("similarity") or 0.0))
        all_hybrid_scores.append(float(first.get("hybrid_score") or 0.0))

        status = "✓" if any_relevant else "✗"
        for j, result in enumerate(results, 1):
            prefix = status if j == 1 else " "
            print(f"  {prefix} [{j}] {result['display']:35s} {_format_scores(result)}")
            print(f"       {result['text'][:110].strip()}...")

        if with_llm and chain:
            try:
                resp = chain.ask(query, category=category)
                print()
                print(f"  💬 Answer: {resp['answer'][:200].strip()}...")
            except Exception as e:
                print()
                print(f"  💬 LLM error: {e}")
                llm_errors += 1

    avg_sim = sum(all_sims) / len(all_sims) if all_sims else 0
    avg_hybrid = sum(all_hybrid_scores) / len(all_hybrid_scores) if all_hybrid_scores else 0
    route_accuracy = route_correct / total if total else 0
    top1_rate = top1_hits / static_total if static_total else 0
    top3_rate = top3_hits / static_total if static_total else 0

    print()
    print("═" * 70)
    print(f"  RESULTS: {passed}/{total} checks passed")
    print(f"  Route accuracy     : {route_correct}/{total} ({route_accuracy:.0%})")
    print(f"  Static top-1 hit   : {top1_hits}/{static_total} ({top1_rate:.0%})")
    print(f"  Static top-3 hit   : {top3_hits}/{static_total} ({top3_rate:.0%})")
    print(f"  Avg similarity     : {avg_sim:.3f}")
    print(f"  Avg hybrid score   : {avg_hybrid:.3f}")

    if passed == total and route_correct == total:
        print("  ✅  All routing and retrieval checks passing")
    elif passed >= total * 0.7:
        print("  ⚠   Most checks passing — inspect misses above")
    else:
        print("  ✗   Low pass rate — check chunk_and_embed.py and retrieval configuration")

    print()
    print("  Full eval gate: validate_phase5_retrieval.py covers the 250-item farmer set")
    print()
    if passed != total or route_correct != total:
        return 1
    if with_llm and llm_errors:
        return 1
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--with-llm", action="store_true", help="Also generate LLM answers for each query")
    args = p.parse_args()
    raise SystemExit(run_validation(with_llm=args.with_llm))
