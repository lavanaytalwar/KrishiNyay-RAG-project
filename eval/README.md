# Farmer-Facing Evaluation Set

Phase 4 adds a farmer-question evaluation dataset for KrishiNyay AI. Phase 9 extends it to 250 total evaluation cases with generated hardening prompts for retrieval, dynamic routing, state matching, legal-rights routing, OCR-adjacent questions, and follow-up realism. This dataset is used for evaluation only; it is not an authoritative grounding corpus and should not be indexed into ChromaDB.

## Files

- `farmer_questions.jsonl` - 100 Phase 4 realistic farmer-facing questions, one JSON object per line.
- `phase9_hardening_cases.py` - 150 generated Phase 9 hardening cases using the same schema.
- `../validate_farmer_eval.py` - structural validator for schema, duplicate questions, language minimums, topic minimums, route/source-type values, and minimum 250 total items.

## JSONL Schema

Each item contains:

- `question`: farmer-style query in Hindi/Hinglish, English, or regional-flavored romanized language.
- `language`: one of `hinglish`, `english`, or `regional_romanized`.
- `topic`: one of the Phase 4 topic buckets.
- `expected_route`: `rag` for static source-grounded retrieval, or `dynamic_router` for live-changing data.
- `expected_source_type`: expected evidence/source class.
- `reference_answer`: concise target behavior or grounded answer.
- `source_basis`: short provenance label from existing indexed sources or dynamic router behavior.

## Policy

The dataset is manually drafted from existing repo sources such as PM-KISAN FAQs, PMFBY guidelines, KCC/Vikaspedia material, FRA/LARR documents, state scheme pages, Soil Health/PMKSY/AIF sources, and current dynamic-router behavior. Forum, YouTube, and social content are not copied here; they may only inspire future query realism after review.

## Validation

Run from the repository root:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/krishi_pycache krishinyay-env/bin/python validate_farmer_eval.py
PYTHONPYCACHEPREFIX=/private/tmp/krishi_pycache krishinyay-env/bin/python validate_phase5_retrieval.py
```
