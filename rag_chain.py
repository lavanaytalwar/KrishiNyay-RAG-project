"""
KrishiNyay — rag_chain.py
The full RAG chain: retrieve context → build prompt → generate answer.

LLM STRATEGY:
  Production  : Ollama (Llama 3.1 8B running locally — free, private)
  Dev/fallback: Anthropic API (claude-haiku — fast, cheap, great Hindi)
  Offline test: Template answer (no LLM needed — proves retrieval works)

Usage:
    from rag_chain import RAGChain
    chain = RAGChain()
    answer = chain.ask("PM-KISAN ke liye kaun eligible hai?")
    print(answer)
"""

import os, logging
from pathlib import Path
from typing import Optional

log = logging.getLogger("krishinyay.rag_chain")

# ── Prompt templates ───────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are KrishiNyay AI — a helpful assistant for Indian farmers.
You answer questions about government schemes, crop management, legal rights, and financial inclusion.
Always answer in the same language the user asked in (Hindi or English).
Base your answer ONLY on the provided context. If the context doesn't have enough information, say so clearly.
Keep answers concise, practical, and easy for a farmer to understand.
When mentioning amounts or deadlines, be specific."""

RAG_PROMPT_TEMPLATE = """{system}

CONTEXT (retrieved from KrishiNyay knowledge base):
{context}

USER QUESTION: {question}

ANSWER:"""


def format_context(results: list[dict]) -> str:
    """Format retrieved chunks into a clean context block."""
    parts = []
    for i, r in enumerate(results, 1):
        source_label = r.get("display") or r.get("source", "Unknown")
        parts.append(
            f"[Source {i}: {source_label}]\n{r['text']}"
        )
    return "\n\n---\n\n".join(parts)


# ── LLM backends ───────────────────────────────────────────────────────────

def _call_anthropic(prompt: str) -> str:
    """Call Anthropic API — fast and reliable for Hindi."""
    import anthropic
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip()


def _call_ollama(prompt: str, model: str = "llama3.1:8b") -> str:
    """Call local Ollama instance — free, runs on your machine."""
    import requests
    resp = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


def _template_answer(question: str, results: list[dict]) -> str:
    """
    No-LLM fallback: construct answer directly from top chunk.
    Proves retrieval works even without a running LLM.
    """
    if not results:
        return "Maafi kijiye, is sawaal ka jawab hamari knowledge base mein nahi mila."
    top     = results[0]
    source  = top.get("display") or top.get("source", "")
    excerpt = top["text"][:400].strip()
    return (
        f"[Retrieved from: {source}]\n\n"
        f"{excerpt}\n\n"
        f"(Template answer — plug in Ollama or Anthropic API for full LLM generation)"
    )


def _choose_llm():
    """
    Pick the best available LLM backend:
    1. Ollama (local Llama 3.1 — preferred for production)
    2. Anthropic API (best Hindi, fastest for dev)
    3. Template fallback (always works, no LLM needed)
    """
    # Try Ollama first
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            llama  = next((m for m in models if "llama" in m.lower()), None)
            model  = llama or "llama3.1:8b"
            log.info(f"✓ Using Ollama: {model}")
            return lambda prompt: _call_ollama(prompt, model)
    except Exception:
        pass

    # Try Anthropic
    if os.environ.get("ANTHROPIC_API_KEY"):
        log.info("✓ Using Anthropic API (claude-haiku)")
        return _call_anthropic

    # Template fallback
    log.warning("No LLM backend found — using template fallback")
    log.warning("  For Ollama : install ollama.com → ollama pull llama3.1:8b")
    log.warning("  For Anthropic: export ANTHROPIC_API_KEY=sk-...")
    return None


# ── RAG Chain ──────────────────────────────────────────────────────────────

class RAGChain:
    def __init__(self, n_results: int = 4):
        from vector_store import VectorStore
        self.store     = VectorStore()
        self.n_results = n_results
        self.llm_fn    = _choose_llm()
        log.info(f"RAGChain ready (n_results={n_results})")

    def ask(
        self,
        question: str,
        category: Optional[str] = None,
        state: Optional[str] = None,
        verbose: bool = False,
    ) -> dict:
        """
        Full RAG pipeline: retrieve → build prompt → generate.
        Returns dict with: answer, sources, context, question
        """
        # 1. Retrieve relevant chunks
        results = self.store.query(
            question,
            n=self.n_results,
            category=category,
            state=state,
        )

        if verbose:
            log.info(f"\nQuery: {question}")
            for i, r in enumerate(results, 1):
                log.info(f"  [{i}] {r['display']} (sim={r['similarity']}) — {r['text'][:80]}...")

        # 2. Build context
        context = format_context(results)

        # 3. Build prompt
        prompt = RAG_PROMPT_TEMPLATE.format(
            system=SYSTEM_PROMPT,
            context=context,
            question=question,
        )

        # 4. Generate answer
        if self.llm_fn:
            try:
                answer = self.llm_fn(prompt)
            except Exception as e:
                log.warning(f"LLM call failed ({e}) — using template fallback")
                answer = _template_answer(question, results)
        else:
            answer = _template_answer(question, results)

        # 5. Return structured response
        return {
            "question": question,
            "answer":   answer,
            "sources":  [
                {"display": r["display"], "url": r["url"], "similarity": r["similarity"]}
                for r in results
            ],
            "context":  context,
            "n_chunks": len(results),
        }

    def ask_simple(self, question: str) -> str:
        """Convenience: returns just the answer string."""
        return self.ask(question)["answer"]
