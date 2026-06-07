"""
KrishiNyay — validate_corpus.py
Run 12 Hindi + English test queries and print top-2 retrieved chunks.
This is your Phase 2 proof-of-work — run this after chunk_and_embed.py.

Run: python validate_corpus.py
Run with LLM answers: python validate_corpus.py --with-llm
"""

import sys, logging, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.WARNING)  # quiet for clean output
log = logging.getLogger("krishinyay.validate")

# ── 12 test queries covering all 3 pillars ─────────────────────────────────
TEST_QUERIES = [
    # Central schemes — Hindi
    ("PM-KISAN ke liye kaun eligible hai?",
     "income_support", "Should retrieve PM-KISAN eligibility info"),
    ("PM-KISAN mein kitne paise milte hain aur kab?",
     "income_support", "Should retrieve ₹6000/year in 3 installments"),

    # Central schemes — English
    ("What documents do I need to apply for PM-KISAN?",
     "income_support", "Should retrieve Aadhaar, land records, bank details"),
    ("How do I claim PMFBY crop insurance after flood damage?",
     "crop_insurance", "Should retrieve 72-hour intimation, claim process"),

    # State schemes
    ("Maharashtra mein farmers ko kya extra scheme milti hai?",
     None,             "Should retrieve Namo Shetkari ₹6000 extra"),

    # Crop insurance Hindi
    ("Fasal bima ka premium kitna hota hai kharif crops ke liye?",
     "crop_insurance", "Should retrieve 2% of sum insured"),

    # Legal
    ("Tribal farmers ke liye forest rights kya hain?",
     None,             "Should retrieve FRA 2006 content"),
    ("Zameen khareedne par kisan ke kya adhikar hain?",
     None,             "Should retrieve land acquisition rights"),

    # Financial
    ("Kisan Credit Card ke liye kaise apply karein?",
     "credit",         "Should retrieve KCC application process"),

    # Agri science
    ("Cotton crop mein pest attack ke liye kya karein?",
     None,             "Should retrieve crop management content"),

    # Mixed Hindi-English (Hinglish)
    ("PM-KISAN ka helpline number kya hai?",
     "income_support", "Should retrieve 155261 / 1800115526"),
    ("PMFBY claim karne ke baad kitne din mein paise milte hain?",
     "crop_insurance", "Should retrieve claim settlement timeline"),
]


def run_validation(with_llm: bool = False):
    print()
    print("═" * 70)
    print("  KRISHINYAY — CORPUS VALIDATION")
    print(f"  {len(TEST_QUERIES)} queries  ·  top-2 results each")
    print("═" * 70)

    # Load vector store
    try:
        from vector_store import VectorStore
        vs = VectorStore()
        stats = vs.stats()
        print(f"\n  Vector store: {stats['total_chunks']} chunks indexed")
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

    for i, (query, category, expectation) in enumerate(TEST_QUERIES, 1):
        print(f"\n{'─' * 70}")
        print(f"  [{i:02d}] {query}")
        print(f"       Expected: {expectation}")

        results = vs.query(query, n=2, category=category)

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
