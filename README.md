# 🛡️ RAGuard — RAG Evaluation & Governance Lab

**A RAG system that enforces document access control inside the vector query, with an evaluation harness that measures both answer quality and leakage.**

Most RAG tutorials stop at "retrieve, stuff, generate." RAGuard answers the two questions production actually asks:

1. **Is this RAG system retrieving and generating well?** — measured with classic IR metrics *and* LLM-judged RAGAS metrics, not vibes.
2. **Can it leak a document a user isn't authorized to read?** Access is enforced in the query itself, and a leak test sweeps every question for every user; the leak count must be exactly **0**.

## Results

Small synthetic corpus: 4 documents of a fictional company (public, confidential, restricted), 18 golden questions including 2 deliberately unanswerable ones, 3 users.

| Category | Metric | Run 1 (2026-07-02, Qdrant server) | Run 2 (2026-11-08, after refactor; Qdrant server and embedded mode agree) |
|---|---|---|---|
| Governance | **Leaks** (18 questions × 3 users × top 5 = 270 chunk checks) | **0** | **0** (and 0 access mismatches) |
| Retrieval | Recall@5 / Hit Rate@5 | 1.000 / 1.000 | 1.000 / 1.000 |
| Retrieval | MRR / Precision@5 | 0.938 / 0.391 | 0.906 / 0.323 |
| Latency | Retrieval p50 / p95 | 11.2 ms / 14.5 ms | 6.3 ms / 7.7 ms (server) |
| Generation (RAGAS) | Faithfulness | 0.870 | not rerun |
| Generation (RAGAS) | Context Precision / Context Recall | 0.796 / 1.000 | not rerun |
| Generation (RAGAS) | Answer Relevancy | 0.665 † | not rerun |

The governance result reproduced exactly. Precision and MRR moved between runs. Run 2 gives identical numbers on the Qdrant server and in embedded mode, so the index mode is not the cause; the July figures came from an earlier state of the code or library versions that was not recorded. Precision@5 is low by construction here, because 5 chunks are returned from a 4-document corpus.

† A known artifact of terse-but-correct answers, confirmed stable across three different judge models (see [Judge A/B](#the-llm-judge-ab-experiment)).

## The core idea: relevant ≠ permitted

The governance gate is a **pre-filter inside the vector query** — not a post-retrieval filter, and not a prompt instruction the model can ignore:

```
classification_rank <= user_clearance      # enforced as a Qdrant payload filter
```

```
                         ┌──────────────────────────────────────────┐
 Query + user clearance ►│ 1. Embed query (nomic-embed-text, 768-d) │
                         └──────────────────┬───────────────────────┘
                                            ▼
                         ┌──────────────────────────────────────────┐
                         │ 2. Qdrant search WITH payload filter:    │ ◄── GOVERNANCE GATE
                         │    classification_rank ≤ user_clearance │     (pre-filter)
                         │    Unauthorized docs never enter the     │
                         │    candidate set                         │
                         └──────────────────┬───────────────────────┘
                                            ▼
                         ┌──────────────────────────────────────────┐
                         │ 3. Generate from permitted context only  │
                         │    (llama3.1:8b, temperature 0)          │
                         └──────────────────────────────────────────┘
```

Because unauthorized documents never enter the candidate set, the model cannot quote what it never receives, so protection does not depend on the model behaving. The guarantee is only as strong as the classification labels and the filter itself, which is why both are tested. Two invariants follow:

1. **Pre-filter, never post-filter.** Filtering happens via `query_filter` against an indexed integer payload field. Retrieve-broadly-then-drop hurts recall and leaks the moment the filter is skipped.
2. **Fail-closed.** A document missing from the manifest — or carrying an unrecognized label — defaults to the *most* restricted class, never the least.

Clearance model: `public=0 · confidential=1 · restricted=2`; a user's clearance must be ≥ a document's rank. Demo users: alice (0), bob (1), carol (2).

## What's in the box

| File | Role |
|---|---|
| `config.py` | Single source of truth: models, dims, chunking, top-k, clearance ranks, users, Qdrant location |
| `governance.py` | Fail-closed classification, clearance validation and the payload pre-filter, with no model dependencies |
| `ingest.py` | Chunk → embed → upsert with governance payload (fail-closed classification) |
| `rag.py` | Authorized retrieval + grounded generation; refuses when no permitted context exists |
| `evaluate_retrieval.py` | Precision@k, Recall@k, MRR, Hit Rate, latency percentiles (model-independent) |
| `evaluate_ragas.py` | RAGAS faithfulness / relevancy / context precision + recall (LLM-as-judge) |
| `test_governance.py` | End-to-end leak test: sweeps every question × every user; exits non-zero on any leak or access mismatch |
| `tests/test_offline.py` | Unit tests that need no Ollama or Qdrant server: fail-closed labels, clearance validation, and a randomized check that the filter never returns a chunk above the user's clearance |
| `api.py` + `web/` | FastAPI + React observability UI: per-stage timings, retrieved vs. **withheld** chunks, live leak status |
| `docs/build-guide.md` | The original step-by-step build guide the project started from |
| `documents/`, `manifest.csv`, `golden_set.json` | Synthetic corpus (fictional company), classification ground truth, eval ground truth — including deliberately unanswerable questions to catch hallucination |

## Stack

**Qdrant** (vectors, Docker) · **Ollama** (all models, local daemon or remote GPU server via SSH tunnel — the app is endpoint-agnostic) · **RAGAS 0.2.x** (LLM-judged quality) · **LangChain** (splitters/adapters) · **FastAPI + React/Vite** (observability UI) · Python 3.10

Models: `nomic-embed-text` (768-d embeddings) · `llama3.1:8b` (generation) · `qwen2.5:32b` (RAGAS judge).

## Quick start

```bash
# 1. Vector DB (or skip Docker: export RAGUARD_QDRANT=./local_qdrant for embedded mode)
docker run -d --name qdrant-lab -p 6333:6333 -p 6334:6334 \
  -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant

# 2. Models (any Ollama endpoint; set the URL in config.py)
ollama pull nomic-embed-text && ollama pull llama3.1:8b && ollama pull qwen2.5:32b

# 3. Python env (3.10; requirements are version-pinned deliberately)
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 4. The pipeline, in order
python ingest.py                 # chunk → embed → index with governance payload
python rag.py                    # smoke test
python evaluate_retrieval.py     # IR metrics + latency
python evaluate_ragas.py         # LLM-judged quality (slow)
python test_governance.py        # leak test: must print 0 and exit 0
python -m pytest -q              # offline governance tests

# 5. Observability UI (optional)
./run_app.sh                     # FastAPI :8090 + Vite dev server
```

The decisive demo: ask *"What is the Project Helios budget?"* (a restricted fact) as alice (clearance 0). The system retrieves nothing restricted and answers "I don't have that information" — then flip to carol (clearance 2) in the UI and watch the same question answered with sources, with the sidebar showing exactly which chunks the gate withheld.

## Known side channel: the "withheld" view

The demo UI can show which chunks the filter withheld from a user. Even without their text, that view reveals that a restricted document exists, its file name, its class and how relevant it is to the question, and an unauthorized user can learn a lot from that. It is therefore **off by default**: the API returns no withheld metadata unless `RAGUARD_SHOW_WITHHELD=1` is set for teaching. The leak test covers retrieved content, not metadata channels like this one, and it checks one top-k retrieval path, not every possible inference from scores, timing or refusals.

## The LLM-judge A/B experiment

RAGAS scores depend on the judge, so the judge itself was A/B tested (`ragas_results_*.csv`):

| Judge | Outcome |
|---|---|
| `llama3.1:8b` | JSON-format failures → NaN rows; under-scored faithfulness |
| `llama3.1:70b` | Needed aggressive throttling (`max_workers=4`, 600s timeout) on a single GPU; unthrottled runs produced timeout-NaNs that silently corrupted metric means |
| `qwen2.5:32b` ✅ | 72 judge jobs in 5m25s, zero NaNs/timeouts, scores consistent with the 70B baseline at half the VRAM |

Two transferable lessons: **always check the sample count behind a mean** (a "0.33 context precision" turned out to be 3 surviving samples out of 18), and **judge capability ≥ generator capability**, but bigger isn't automatically better under real serving constraints.

## Evaluation discipline

- **Fix recall first, then precision** — if the right context isn't retrieved, no prompt tuning helps.
- Retrieval and generation quality are measured at max clearance (so retrieval is fully exercised); governance is tested separately by sweeping all users.
- Local judges are directional: watch *relative* movement between config changes, not absolute scores.

## Roadmap

Markdown-header-aware chunking → hybrid search (BM25 + dense, RRF) → cross-encoder reranking (targets Precision@5) → embedder A/B (`mxbai-embed-large`) → scalar quantization (measure the recall cost) → ABAC (per-department/project tags instead of a single clearance rank).
