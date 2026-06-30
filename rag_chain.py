"""
KrishiNyay — rag_chain.py
The full RAG chain: retrieve context → build prompt → generate answer.

LLM STRATEGY:
  Local demo  : Ollama (Llama 3.1 8B running locally — free, private)
  Hosted demo : Gemini or OpenRouter via API key
  Dev/fallback: Anthropic API (claude-haiku — fast, cheap, great Hindi)
  Offline test: Template answer (no LLM needed — proves retrieval works)

Usage:
    from rag_chain import RAGChain
    chain = RAGChain()
    answer = chain.ask("PM-KISAN ke liye kaun eligible hai?")
    print(answer)
"""

import os, logging, re, time
from pathlib import Path
from typing import Optional

from language_policy import detect_answer_language, prompt_language_instruction, requires_english_answer
from query_utils import normalize_query

log = logging.getLogger("krishinyay.rag_chain")

# ── Prompt templates ───────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are KrishiNyay AI — a helpful assistant for Indian farmers.
You answer questions about government schemes, crop management, legal rights, and financial inclusion.
Always follow REQUIRED ANSWER LANGUAGE.
If REQUIRED ANSWER LANGUAGE says English, answer in English only. Do not use Hindi or Devanagari except official scheme names.
If REQUIRED ANSWER LANGUAGE says Hindi/Hinglish, answer in simple Roman Hindi/Hinglish, not formal English.
Base your answer ONLY on the provided context. If the context doesn't have enough information, say so clearly.
Keep answers concise, practical, and easy for a farmer to understand.
When mentioning amounts or deadlines, be specific.
Do not invent eligibility, prices, weather, status, or application outcomes.
Prefer 2-5 short bullets or short paragraphs.
If the question needs live farmer-specific data, say it must be checked on the official/live source."""

RAG_PROMPT_TEMPLATE = """{system}

CONTEXT (retrieved from KrishiNyay knowledge base):
{context}

USER QUESTION: {question}
REQUIRED ANSWER LANGUAGE: {answer_language}

ANSWER:"""

EVIDENCE_SYNTHESIS_PROMPT_TEMPLATE = """{system}

VERIFIED EVIDENCE:
{evidence}

USER QUESTION: {question}
REQUIRED ANSWER LANGUAGE: {answer_language}

Write the final farmer-facing answer using only the verified evidence.
Preserve official URLs exactly when they appear in the evidence.
Do not add new facts beyond the verified evidence.

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


def _answer_looks_english(answer: str) -> bool:
    devanagari_chars = sum(1 for char in answer if "\u0900" <= char <= "\u097F")
    return devanagari_chars <= max(4, len(answer) // 20)


def _sanitize_llm_error(exc: Exception) -> str:
    """Return a short, user-safe generation error message."""
    message = str(exc).strip()
    if not message:
        return "The local model did not return a usable response."
    if len(message) > 220:
        message = message[:217].rstrip() + "..."
    return message


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


def _call_gemini(prompt: str) -> str:
    """Call Gemini REST API without adding a Google SDK dependency."""
    import requests

    api_key = os.environ["GEMINI_API_KEY"]
    model = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    resp = requests.post(
        url,
        json={"contents": [{"parts": [{"text": prompt}]}]},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()


def _call_openrouter(prompt: str) -> str:
    """Call OpenRouter's OpenAI-compatible chat API."""
    import requests

    model = os.environ.get("OPENROUTER_MODEL", "google/gemini-2.0-flash-exp:free")
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.environ.get("APP_URL", "http://localhost:8000"),
            "X-Title": "KrishiNyay AI",
        },
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 700,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _call_ollama(prompt: str, model: str = "llama3.1:8b") -> str:
    """Call local Ollama instance — free, runs on your machine."""
    import requests
    timeout = int(os.environ.get("OLLAMA_TIMEOUT", "180"))
    num_predict = int(os.environ.get("OLLAMA_NUM_PREDICT", "450"))
    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "keep_alive": "10m",
                "options": {
                    "temperature": 0.1,
                    "num_predict": num_predict,
                },
            },
            timeout=timeout,
        )
    except requests.Timeout as exc:
        raise RuntimeError(
            f"Ollama timed out after {timeout}s. The model may still be loading; retry the request."
        ) from exc
    except requests.ConnectionError as exc:
        raise RuntimeError(
            "Ollama is not reachable at http://localhost:11434. Start it with: brew services start ollama"
        ) from exc
    except requests.RequestException as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc

    if resp.status_code >= 400:
        try:
            detail = resp.json().get("error", resp.text)
        except ValueError:
            detail = resp.text
        raise RuntimeError(f"Ollama HTTP {resp.status_code}: {detail[:180]}")

    try:
        payload = resp.json()
    except ValueError as exc:
        raise RuntimeError("Ollama returned invalid JSON.") from exc

    if payload.get("error"):
        raise RuntimeError(f"Ollama error: {payload['error']}")

    answer = str(payload.get("response", "")).strip()
    if not answer:
        raise RuntimeError("Ollama returned an empty response.")
    return answer


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
    1. Ollama (local Llama 3.1, preferred for private generation)
    2. Gemini API (hosted demo)
    3. OpenRouter API (hosted model router)
    4. Anthropic API (dev fallback)
    5. Template fallback (always works, no LLM needed)
    """
    configured_ollama = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            if configured_ollama in models:
                log.info(f"✓ Using Ollama: {configured_ollama}")
                return (
                    f"ollama:{configured_ollama}",
                    lambda prompt: _call_ollama(prompt, configured_ollama),
                )
            log.warning(
                "Ollama is running, but required model '%s' is not installed. "
                "Run: ollama pull %s",
                configured_ollama,
                configured_ollama,
            )
    except Exception:
        pass

    if os.environ.get("GEMINI_API_KEY"):
        log.info("Using Gemini API")
        return "gemini", _call_gemini

    if os.environ.get("OPENROUTER_API_KEY"):
        log.info("Using OpenRouter API")
        return "openrouter", _call_openrouter

    # Try Anthropic
    if os.environ.get("ANTHROPIC_API_KEY"):
        log.info("✓ Using Anthropic API (claude-haiku)")
        return "anthropic", _call_anthropic

    # Template fallback
    log.warning("No LLM backend found — using template fallback")
    log.warning("  For Gemini : export GEMINI_API_KEY=...")
    log.warning("  For OpenRouter: export OPENROUTER_API_KEY=...")
    log.warning("  For Ollama : install ollama.com → ollama pull %s", configured_ollama)
    log.warning("  For Anthropic: export ANTHROPIC_API_KEY=sk-...")
    return "template", None


# ── RAG Chain ──────────────────────────────────────────────────────────────

class RAGChain:
    def __init__(self, n_results: int = 4):
        from vector_store import VectorStore
        self.store     = VectorStore()
        self.n_results = n_results
        self.llm_provider, self.llm_fn = _choose_llm()
        log.info(f"RAGChain ready (n_results={n_results})")

    def ask(
        self,
        question: str,
        category: Optional[str] = None,
        state: Optional[str] = None,
        answer_language: Optional[str] = None,
        verbose: bool = False,
    ) -> dict:
        """
        Full RAG pipeline: retrieve → build prompt → generate.
        Returns dict with: answer, sources, context, question
        """
        normalized_question = normalize_query(question)

        # 1. Retrieve relevant chunks
        results = self.store.query(
            normalized_question,
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
        resolved_answer_language = detect_answer_language(question, fallback=answer_language)
        answer_language_instruction = prompt_language_instruction(resolved_answer_language)
        prompt = RAG_PROMPT_TEMPLATE.format(
            system=SYSTEM_PROMPT,
            context=context,
            question=question,
            answer_language=answer_language_instruction,
        )

        # 4. Generate answer
        generation_status = "template_fallback"
        generation_error = None
        if self.llm_fn:
            try:
                answer = self.llm_fn(prompt)
                generation_status = "generated"
                if requires_english_answer(resolved_answer_language) and not _answer_looks_english(answer):
                    retry_prompt = (
                        f"{prompt}\n\n"
                        "IMPORTANT: Answer in English only. Do not use Hindi or Devanagari. "
                        "Give the final answer now in concise farmer-facing English."
                    )
                    answer = self.llm_fn(retry_prompt)
                    generation_status = "generated_after_language_retry"
            except Exception as exc:
                if self.llm_provider.startswith("ollama:"):
                    try:
                        time.sleep(float(os.environ.get("OLLAMA_RETRY_DELAY", "1.5")))
                        answer = self.llm_fn(prompt)
                        generation_status = "generated_after_retry"
                    except Exception as retry_exc:
                        generation_error = _sanitize_llm_error(retry_exc)
                        log.warning("LLM call failed after retry (%s) — using template fallback", generation_error)
                        answer = _template_answer(question, results)
                else:
                    generation_error = _sanitize_llm_error(exc)
                    log.warning("LLM call failed (%s) — using template fallback", generation_error)
                    answer = _template_answer(question, results)
        else:
            generation_error = "No LLM backend configured."
            answer = _template_answer(question, results)

        if generation_status == "template_fallback" and not generation_error and self.llm_fn:
            generation_error = "The configured model did not generate an answer."

        # 5. Return structured response
        return {
            "question": question,
            "normalized_question": normalized_question,
            "answer_language": resolved_answer_language,
            "answer":   answer,
            "sources":  [
                {
                    "display": r["display"],
                    "url": r["url"],
                    "similarity": r["similarity"],
                    "vector_score": r.get("vector_score", r.get("similarity", 0.0)),
                    "lexical_score": r.get("lexical_score", 0.0),
                    "hybrid_score": r.get("hybrid_score", r.get("similarity", 0.0)),
                    "retrieval_method": r.get("retrieval_method", "vector"),
                    "category": r.get("category", ""),
                    "state": r.get("state", ""),
                    "source": r.get("source", ""),
                    "text": r.get("text", ""),
                }
                for r in results
            ],
            "context":  context,
            "n_chunks": len(results),
            "mode":     "rag",
            "route":    "rag",
            "llm_provider": self.llm_provider,
            "generation_status": generation_status,
            "generation_error": generation_error,
        }

    def synthesize_from_evidence(
        self,
        *,
        question: str,
        evidence: str,
        sources: list[dict],
        answer_language: Optional[str] = None,
    ) -> dict:
        """Generate a final answer from already-verified tool evidence."""
        resolved_answer_language = detect_answer_language(question, fallback=answer_language)
        answer_language_instruction = prompt_language_instruction(resolved_answer_language)
        prompt = EVIDENCE_SYNTHESIS_PROMPT_TEMPLATE.format(
            system=SYSTEM_PROMPT,
            evidence=evidence,
            question=question,
            answer_language=answer_language_instruction,
        )

        generation_status = "tool_answer_fallback"
        generation_error = None
        answer = evidence

        if self.llm_fn:
            try:
                answer = self.llm_fn(prompt)
                generation_status = "generated_from_verified_evidence"
                if requires_english_answer(resolved_answer_language) and not _answer_looks_english(answer):
                    retry_prompt = (
                        f"{prompt}\n\n"
                        "IMPORTANT: Answer in English only. Do not use Hindi or Devanagari. "
                        "Give the final answer now in concise farmer-facing English."
                    )
                    answer = self.llm_fn(retry_prompt)
                    generation_status = "generated_from_verified_evidence_after_language_retry"
            except Exception as exc:
                generation_error = _sanitize_llm_error(exc)
                log.warning("Evidence synthesis failed (%s) — using verified tool answer", generation_error)

        return {
            "answer": answer,
            "sources": sources,
            "llm_provider": self.llm_provider,
            "generation_status": generation_status,
            "generation_error": generation_error,
            "answer_language": resolved_answer_language,
        }

    def ask_simple(self, question: str) -> str:
        """Convenience: returns just the answer string."""
        return self.ask(question)["answer"]
