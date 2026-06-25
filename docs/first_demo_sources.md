# First Demo Source Shortlist

This tracker defines the source set for the first KrishiNyay demo. The goal is a small, defensible corpus with official grounding documents plus farmer-facing language sources for evaluation and query realism.

## Source Tiers

### Tier 1: Demo Grounding Corpus

Use these as the first RAG sources. They should answer the demo questions directly and should be prioritized for manual PDF ingestion.

| Source | Coverage | Intended use | Priority | Notes |
| --- | --- | --- | --- | --- |
| PM-KISAN | Income support, eligibility, exclusions, eKYC, status checks | RAG grounding | P0 | Use official operational guidelines and FAQs from `pmkisan.gov.in`. |
| PMFBY | Crop insurance enrollment, claims, loss reporting, premium, exclusions | RAG grounding | P0 | Use official PMFBY portal guidelines/FAQs. Download manually if the portal is JS-heavy. |
| Kisan Credit Card | Farm credit eligibility, loan purpose, application documents | RAG grounding | P0 | Prefer RBI/NABARD/official bank circulars plus PM-KISAN KCC form. |
| Forest Rights Act | Forest land rights, Gram Sabha process, titles, community rights | RAG grounding | P0 | Use Ministry of Tribal Affairs FRA booklet, FAQs, and selected executive directions. |
| Land Acquisition / LARR | Compensation, rehabilitation, consent, affected-family rights | RAG grounding | P1 | Use India Code or Ministry of Law PDFs. Keep scope to farmer-facing rights. |
| PMKSY | Irrigation support, micro-irrigation, per-drop-more-crop | RAG grounding | P1 | Use PMKSY guidelines and official FAQ. |
| Soil Health Card | Soil testing, card download, nutrient recommendations | RAG grounding | P1 | Use official Soil Health portal docs where downloadable. |
| e-NAM | Mandi trading, farmer registration, price discovery, manuals | RAG grounding | P1 | Useful for demo if price/mandi questions are included. |
| Punjab state schemes | PMFBY/PM-KISAN state notices, crop residue management, farm mechanization, DSR/seed subsidies | RAG grounding | P1 | Use Agriculture Department of Punjab pages and downloadable notices. |
| Haryana state schemes | State agriculture schemes, soil cards, payment status, farmer portal | RAG grounding | P1 | Use Haryana agriculture department scheme pages/PDFs. |
| Maharashtra state schemes | State agriculture schemes, Marathi coverage, farmer advisories | RAG grounding | P1 | Use Maharashtra agriculture department scheme pages/PDFs. |

### Tier 2: Farmer-Language and Evaluation Sources

Use these to build realistic eval questions, not as authoritative policy grounding unless licensing and provenance are clear.

| Source | Coverage | Intended use | Priority | Notes |
| --- | --- | --- | --- | --- |
| Vikaspedia | Farmer-facing explainers in Indian languages | Eval questions + supplemental grounding | P0 | Good bridge between official policy language and farmer-readable explanations. Verify page provenance. |
| CABI Plantwise / PlantwisePlus Knowledge Bank | Plant health factsheets and extension content | Eval questions + crop advisory supplement | P1 | Good for pest/disease language; confirm reuse terms before committing content. |
| Access Agriculture | Multilingual agriculture training videos | Eval questions + video-derived query patterns | P1 | The site exposes categories and many Indian languages. Treat transcripts/video text as licensing-sensitive. |
| Digital Green | FarmerChat, participatory video, farmer advisory context | Research/reference + possible partnership target | P2 | Strong thematic fit, but direct datasets/transcripts may need permission. |
| aAQUA farmer Q&A | Real farmer questions in Indian languages | Eval questions if accessible/licensed | P2 | Valuable but must check site availability and content license before ingestion. |
| Kisan Call Centre data | Real farmer helpline query logs | Research reference only unless public access is found | P2 | High-value dataset, but raw logs are not assumed public. Papers can guide schema and eval design. |
| AgriGov dataset | Structured multilingual government scheme dataset | Candidate eval/augmentation source | P1 | Newly published research source. Use only if dataset is actually released or permission is granted. |

## Ingestion Rules

- Grounding corpus entries must have clear provenance: official URL, downloaded date, source name, category, state, language, and priority.
- No raw web/forum/comment content should be mixed into the main RAG corpus unless its license and quality are acceptable.
- Farmer-facing Q&A, forum posts, video comments, and transcripts should first go into an eval-question workflow.
- Scanned PDFs should be deferred until OCR is added; Phase 1 should prefer text-extractable PDFs.
- Generated artifacts stay ignored: `data/manual_pdfs/`, `data/processed/`, `data/chunks/`, `chroma_db/`, and `logs/`.

## Phase 1 Minimum Demo Corpus

For the first runnable demo, target 8-12 curated PDFs/pages:

1. PM-KISAN operational guidelines.
2. PM-KISAN revised FAQ.
3. PMFBY operational guidelines or farmer FAQ.
4. KCC official form or circular.
5. FRA Act/Rules/Guidelines booklet.
6. Land Acquisition/LARR Act or farmer-facing explainer.
7. PMKSY guidelines or FAQ.
8. e-NAM operational guidelines or farmer FAQ.
9. One Punjab agriculture scheme/notice.
10. One Haryana agriculture scheme/notice.
11. One Maharashtra agriculture scheme/notice.
12. One Vikaspedia farmer-facing explainer for language realism.

## Manual Upload Checklist

For each file added to `data/manual_pdfs/staging/`, update `data/manual_pdfs/manifest.json` with:

- `name`: stable slug, e.g. `pmkisan_operational_guidelines`.
- `display`: human-readable title.
- `file`: path to the local PDF.
- `url`: official source URL, when available.
- `source_note`: required if no URL is available.
- `category`: one of `scheme`, `insurance`, `credit`, `legal_rights`, `irrigation`, `soil_health`, `market`, `state_scheme`, `crop_advisory`.
- `state`: `central`, `punjab`, `haryana`, `maharashtra`, or another explicit state.
- `language`: `en`, `hi`, `mr`, `pa`, or another explicit language code.
- `priority`: `0` for core demo, `1` for useful demo expansion, `2` for later.

## Acquisition Script

Run the first-demo acquisition script to download public sources into ignored local folders and generate the manual PDF manifest:

```bash
python acquire_first_demo_sources.py
python ingest_manual_pdfs.py --manifest data/manual_pdfs/manifest.json
```

Current acquisition status is tracked in `docs/first_demo_acquisition_status.md`.

## Known Access Risks

- PMFBY and some state portals may require manual browser download because JavaScript-heavy pages do not expose PDFs cleanly to scripts.
- Kisan Call Centre raw data should not be assumed accessible; use it as a research reference unless a public/authorized dataset is obtained.
- Digital Green, Access Agriculture, CABI, and aAQUA may require license review before committing copied content.
- AgriGov is promising, but the repo should only ingest it after confirming the dataset download location and usage terms.
