# 🛡️ RAGuard — RAG Evaluation & Governance Lab

**A RAG system that provably cannot leak documents to users who aren't cleared to see them — with the evaluation harness to prove it.**

Most RAG tutorials stop at "retrieve, stuff, generate." RAGuard answers the two questions production actually asks:

1. **Is this RAG system retrieving and generating well?** — measured with classic IR metrics *and* LLM-judged RAGAS metrics, not vibes.
2. **Can it leak a document a user isn't authorized to read?** — enforced structurally, tested exhaustively, leak rate must be exactly **0**.

## Verified results

| Category | Metric | Result |
|---|---|---|
| Governance | **Leak rate** (18 questions × 3 clearance levels, 270 chunk checks) | **0 leaks** |
| Retrieval | Recall@5 / Hit Rate@5 | 1.000 / 1.000 |
| Retrieval | MRR / Precision@5 | 0.938 / 0.391 |
| Latency | Retrieval p50 / p95 | 11.2 ms / 14.5 ms |
| Generation (RAGAS) | Faithfulness | 0.870 |
| Generation (RAGAS) | Context Precision / Context Recall | 0.796 / 1.000 |
| Generation (RAGAS) | Answer Relevancy | 0.665 † |

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

Because unauthorized documents never enter the candidate set, leakage is **structurally impossible** rather than dependent on the model behaving. Two invariants follow:

1. **Pre-filter, never post-filter.** Filtering happens via `query_filter` against an indexed integer payload field. Retrieve-broadly-then-drop hurts recall and leaks the moment the filter is skipped.
2. **Fail-closed.** A document missing from the manifest — or carrying an unrecognized label — defaults to the *most* restricted class, never the least.

Clearance model: `public=0 · confidential=1 · restricted=2`; a user's clearance must be ≥ a document's rank. Demo users: alice (0), bob (1), carol (2).

## What's in the box

| File | Role |
|---|---|
| `config.py` | Single source of truth: models, dims, chunking, top-k, clearance ranks, users |
| `ingest.py` | Chunk → embed → upsert with governance payload (fail-closed classification) |
| `rag.py` | Authorized retrieval + grounded generation; refuses when no permitted context exists |
| `evaluate_retrieval.py` | Precision@k, Recall@k, MRR, Hit Rate, latency percentiles (model-independent) |
| `evaluate_ragas.py` | RAGAS faithfulness / relevancy / context precision + recall (LLM-as-judge) |
| `test_governance.py` | The leak test: sweeps every question × every user; any leak is a failure |
| `api.py` + `web/` | FastAPI + React observability UI: per-stage timings, retrieved vs. **withheld** chunks, live leak status |
| `documents/`, `manifest.csv`, `golden_set.json` | Synthetic corpus (fictional company), classification ground truth, eval ground truth — including deliberately unanswerable questions to catch hallucination |

## Stack

**Qdrant** (vectors, Docker) · **Ollama** (all models, local daemon or remote GPU server via SSH tunnel — the app is endpoint-agnostic) · **RAGAS 0.2.x** (LLM-judged quality) · **LangChain** (splitters/adapters) · **FastAPI + React/Vite** (observability UI) · Python 3.10

Models: `nomic-embed-text` (768-d embeddings) · `llama3.1:8b` (generation) · `qwen2.5:32b` (RAGAS judge).

## Quick start

```bash
# 1. Vector DB
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
python test_governance.py        # leak test — must print 0

# 5. Observability UI (optional)
./run_app.sh                     # FastAPI :8090 + Vite dev server
```

The decisive demo: ask *"What is the Project Helios budget?"* (a restricted fact) as alice (clearance 0). The system retrieves nothing restricted and answers "I don't have that information" — then flip to carol (clearance 2) in the UI and watch the same question answered with sources, with the sidebar showing exactly which chunks the gate withheld.

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
