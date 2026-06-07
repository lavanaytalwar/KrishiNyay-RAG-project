"""
KrishiNyay — evaluate_rag.py
RAGAS-style RAG evaluation — faithfulness, answer relevancy, context recall.
Uses Anthropic API as the judge LLM (fast, reliable for Hindi evaluation).

Run: python evaluate_rag.py
"""

import sys, json, logging, os
from pathlib import Path
from dataclasses import dataclass, asdict

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("krishinyay.eval")

# ── Evaluation dataset — ground truth QA pairs ───────────────────────────
# Format: (question, ground_truth_answer, category)
EVAL_SET = [
    (
        "PM-KISAN scheme mein kitne paise milte hain per year?",
        "PM-KISAN scheme mein Rs. 6000 per year milte hain, teen installments mein Rs. 2000 each.",
        "income_support",
    ),
    (
        "What documents are needed to apply for PM-KISAN?",
        "Aadhaar card, land records (Khasra/Khatauni), bank account details, and mobile number linked to Aadhaar.",
        "income_support",
    ),
    (
        "PMFBY mein kharif crops ka premium kitna hota hai?",
        "Kharif crops ke liye PMFBY premium 2% of sum insured hota hai.",
        "crop_insurance",
    ),
    (
        "Maharashtra mein Namo Shetkari scheme se kitna milta hai?",
        "Maharashtra government Rs. 6000 extra deti hai PM-KISAN ke upar, total Rs. 12000 per year.",
        None,
    ),
    (
        "How can a farmer register for PM-KISAN?",
        "Farmers can register through Common Service Centres, Gram Patwari, Revenue Officer, or pmkisan.gov.in portal.",
        "income_support",
    ),
]


@dataclass
class EvalResult:
    question:          str
    ground_truth:      str
    generated_answer:  str
    retrieved_context: str
    faithfulness:      float   # 0-1: is answer grounded in context?
    answer_relevancy:  float   # 0-1: does answer address the question?
    context_recall:    float   # 0-1: does context contain the ground truth info?
    overall:           float   # mean of above 3


def score_with_llm(question: str, answer: str,
                   context: str, ground_truth: str) -> dict[str, float]:
    """
    Use Anthropic API as judge LLM to score RAG quality.
    Each metric scored 0.0–1.0.
    """
    try:
        import anthropic
        client = anthropic.Anthropic()

        judge_prompt = f"""You are evaluating a RAG system for Indian farmers.
Score each metric from 0.0 to 1.0. Respond with ONLY a JSON object.

QUESTION: {question}
GROUND TRUTH: {ground_truth}
RETRIEVED CONTEXT: {context[:800]}
GENERATED ANSWER: {answer[:400]}

Score these metrics:
- faithfulness: Is the generated answer fully supported by the retrieved context? (1.0=fully supported, 0.0=hallucinated)
- answer_relevancy: Does the answer actually address the question? (1.0=directly answers, 0.0=irrelevant)
- context_recall: Does the retrieved context contain the information needed to answer correctly? (1.0=all needed info present, 0.0=missing key info)

Respond with ONLY this JSON, no other text:
{{"faithfulness": 0.0, "answer_relevancy": 0.0, "context_recall": 0.0}}"""

        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=100,
            messages=[{"role": "user", "content": judge_prompt}],
        )
        text = msg.content[0].text.strip()
        # Strip markdown fences if present
        text = text.replace("```json", "").replace("```", "").strip()
        scores = json.loads(text)
        return {k: float(max(0.0, min(1.0, v))) for k, v in scores.items()}

    except Exception as e:
        log.warning(f"LLM scoring failed: {e} — using heuristic fallback")
        return _heuristic_score(question, answer, context, ground_truth)


def _heuristic_score(question: str, answer: str,
                     context: str, ground_truth: str) -> dict[str, float]:
    """
    Keyword-overlap scoring when LLM is unavailable.
    Not as accurate but always works offline.
    """
    import re

    def tokens(text: str) -> set:
        return set(re.findall(r'\b\w+\b', text.lower()))

    q_tokens  = tokens(question)
    a_tokens  = tokens(answer)
    c_tokens  = tokens(context)
    gt_tokens = tokens(ground_truth)

    # Faithfulness: fraction of answer tokens found in context
    faithfulness = len(a_tokens & c_tokens) / max(len(a_tokens), 1)

    # Answer relevancy: fraction of question tokens addressed in answer
    answer_relevancy = len(q_tokens & a_tokens) / max(len(q_tokens), 1)

    # Context recall: fraction of ground-truth tokens found in context
    context_recall = len(gt_tokens & c_tokens) / max(len(gt_tokens), 1)

    return {
        "faithfulness":    min(1.0, faithfulness * 1.5),
        "answer_relevancy":min(1.0, answer_relevancy * 2.0),
        "context_recall":  min(1.0, context_recall * 1.5),
    }


def run_evaluation():
    log.info("=" * 60)
    log.info("  KRISHINYAY — RAG EVALUATION")
    log.info(f"  {len(EVAL_SET)} test cases")
    log.info("=" * 60)

    # Load chain
    try:
        from rag_chain import RAGChain
        chain = RAGChain(n_results=4)
    except Exception as e:
        log.error(f"Could not load RAGChain: {e}")
        return

    results       = []
    use_llm_judge = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if not use_llm_judge:
        log.warning("ANTHROPIC_API_KEY not set — using heuristic scoring")
        log.warning("For LLM judge: export ANTHROPIC_API_KEY=sk-...")

    for i, (question, ground_truth, category) in enumerate(EVAL_SET, 1):
        log.info(f"\n[{i}/{len(EVAL_SET)}] {question}")

        resp    = chain.ask(question, category=category)
        answer  = resp["answer"]
        context = resp["context"]

        if use_llm_judge:
            scores = score_with_llm(question, answer, context, ground_truth)
        else:
            scores = _heuristic_score(question, answer, context, ground_truth)

        overall = sum(scores.values()) / 3

        result = EvalResult(
            question          = question,
            ground_truth      = ground_truth,
            generated_answer  = answer[:300],
            retrieved_context = context[:400],
            faithfulness      = scores["faithfulness"],
            answer_relevancy  = scores["answer_relevancy"],
            context_recall    = scores["context_recall"],
            overall           = overall,
        )
        results.append(result)

        log.info(f"  Faithfulness   : {scores['faithfulness']:.2f}")
        log.info(f"  Ans Relevancy  : {scores['answer_relevancy']:.2f}")
        log.info(f"  Context Recall : {scores['context_recall']:.2f}")
        log.info(f"  Overall        : {overall:.2f}")

    # Aggregate
    avg_faith  = sum(r.faithfulness    for r in results) / len(results)
    avg_relev  = sum(r.answer_relevancy for r in results) / len(results)
    avg_recall = sum(r.context_recall  for r in results) / len(results)
    avg_all    = sum(r.overall         for r in results) / len(results)

    log.info(f"\n{'=' * 60}")
    log.info("  AGGREGATE SCORES")
    log.info(f"{'=' * 60}")
    log.info(f"  Faithfulness        : {avg_faith:.3f}  (target > 0.80)")
    log.info(f"  Answer Relevancy    : {avg_relev:.3f}  (target > 0.75)")
    log.info(f"  Context Recall      : {avg_recall:.3f}  (target > 0.70)")
    log.info(f"  Overall             : {avg_all:.3f}")

    # Save results JSON
    out = ROOT / "eval" / "rag_eval_results.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(
        [asdict(r) for r in results], indent=2, ensure_ascii=False
    ), encoding="utf-8")
    log.info(f"\n  Results saved → {out}")

    # Resume bullet
    log.info("\n  RESUME BULLET (copy this):")
    log.info(f'  "Evaluated RAG pipeline with RAGAS-style metrics: '
             f'faithfulness={avg_faith:.2f}, '
             f'answer_relevancy={avg_relev:.2f}, '
             f'context_recall={avg_recall:.2f} on {len(EVAL_SET)}-case Hindi/English eval set"')


if __name__ == "__main__":
    run_evaluation()
