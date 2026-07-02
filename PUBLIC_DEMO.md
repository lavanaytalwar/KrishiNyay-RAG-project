# KrishiNyay AI Public Demo

This guide is for showing KrishiNyay AI as an unlisted public demo.

## Demo Stack

- Hosting: Hugging Face Spaces Docker
- Backend: FastAPI
- Frontend: static HTML/CSS/JS served by FastAPI
- Retrieval: full packaged Chroma index
- Embeddings: MiniLM, 384 dimensions
- Hybrid search: MiniLM vector retrieval plus lexical scoring
- Public generation: Gemini
- Local backup generation: Ollama `llama3.1:8b`
- Public ingest: disabled

## Required Public Demo Environment

```env
DEMO_PUBLIC=true
ENABLE_LIVE_INGEST=false
CHROMA_PATH=demo_chroma_db
CHUNKS_DIR=demo_data/chunks
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_hugging_face_space_secret
GEMINI_MODEL=gemini-1.5-flash
```

Optional live mandi prices:

```env
DATA_GOV_IN_API_KEY=your_data_gov_key
AGMARKNET_API_KEY=your_data_gov_key
```

Weather does not require a key. It uses Open-Meteo with IMD/Mausam shown for official verification.

## Hugging Face Spaces Setup

Use a Docker Space.

Recommended Space settings:

```yaml
sdk: docker
app_port: 7860
```

Set secrets in the Hugging Face Space settings, not in Git:

- `GEMINI_API_KEY`
- optional `DATA_GOV_IN_API_KEY`
- optional `AGMARKNET_API_KEY`

The included Dockerfile starts:

```bash
uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}
```

The public demo relies on these packaged artifacts:

```text
demo_chroma_db/
demo_data/chunks/all_chunks.jsonl
demo_data/chunks/embed_meta.json
```

Do not delete them from the public demo branch.

At runtime, public mode copies `demo_chroma_db/` to writable temp storage before opening Chroma. This avoids dirtying checked-in index files and prevents SQLite read-only errors on hosts where the app bundle is not writable.

## Validation

Local dry-run without a Gemini key:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/krishinyay_pycache krishinyay-env/bin/python validate_public_demo.py
```

Launch validation with Gemini:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/krishinyay_pycache LLM_PROVIDER=gemini GEMINI_API_KEY=... krishinyay-env/bin/python validate_public_demo.py --require-gemini
```

Full local regression:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/krishinyay_pycache krishinyay-env/bin/python run_regression_suite.py
node --check web/app.js
git diff --check
```

## 5-Minute Farmer Demo Script

1. Open the Home page.
2. Explain: “KrishiNyay is a farmer-facing RAG assistant using official/public sources.”
3. Click or ask:
   ```text
   PM-KISAN mein kitna paisa milta hai?
   ```
4. Show the answer, retrieved source cards, and route trace.
5. Ask:
   ```text
   Land acquisition mein farmer ke rights kya hain?
   ```
6. Show legal-rights routing and citations.
7. Ask:
   ```text
   Kal baarish hogi kya, spraying karu?
   ```
8. The app should ask for location.
9. Reply:
   ```text
   Jaipur
   ```
10. Show live weather routing, spraying advice, and source/tool metadata.

## 3-Minute Technical Demo Script

1. Open the System page.
2. Show:
   - full Chroma index count;
   - MiniLM embedding backend and dimension;
   - lexical chunk count;
   - Gemini provider status;
   - public ingest disabled;
   - validation/readiness status.
3. Open a chat answer’s route trace.
4. Explain:
   ```text
   MiniLM retrieves evidence. Lexical scoring boosts exact scheme/source matches.
   The workflow chooses static RAG or a live tool. Gemini writes the final answer
   only from verified retrieved/live evidence.
   ```

## Expected Fallbacks

- Missing Gemini key: `/health` should show the demo is not public-launch ready.
- Missing mandi key: mandi route should show safe official-portal guidance.
- Weather failure: the app should advise checking IMD/Mausam or local advisory.
- Public ingest attempt: `/ingest` should return `403`.

## Public Safety Rules

- Do not expose API keys in frontend or API responses.
- Do not enable live ingest on the public Space.
- Do not ask users to enter private beneficiary IDs in the public demo.
- Treat answers as guidance and always link users back to official portals.
