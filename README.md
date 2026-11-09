# 🛡️ RAGuard — RAG Evaluation & Governance Lab

**A RAG system that enforces document access control inside the vector query, with an evaluation harness that measures both answer quality and leakage.**

Most RAG tutorials stop at "retrieve, stuff, generate." RAGuard answers the two questions production actually asks:

1. **Is this RAG system retrieving and generating well?** — measured with classic IR metrics *and* LLM-judged RAGAS metrics, not vibes.
2. **Can it leak a document a user isn't authorized to read?** Access is enforced in the query itself, and a leak test sweeps every question for every user; the leak count must be exactly **0**. Scope: the demo trusts the clearance the client sends (authentication is out of scope), and the tests cover retrieved content on the top-k path, not inference from refusals, timing or scores.

## Results

Small synthetic corpus: 4 documents of a fictional company (public, confidential, restricted), 18 golden questions including 2 deliberately unanswerable ones, 3 users.

| Category | Metric | Run 1 (2026-07-02, Qdrant server) | Run 2 (2026-11-08, after refactor; Qdrant server and embedded mode agree) |
|---|---|---|---|
| Governance | **Leaks** (18 questions × 3 users × top 5 = 270 chunk checks; 180 of them by the two restricted users, since carol cannot leak by definition) | **0** | **0**, 0 stored labels disagreeing with `manifest.csv`, 0 of 33 cleared lookups failed |
| Retrieval (16 answerable questions) | Recall@5 / Hit Rate@5 | 1.000 / 1.000 | 1.000 / 1.000 |
| Retrieval | MRR / Doc precision@5 | 0.938 / 0.391 | 0.906 / 0.323 |
| Latency | Retrieval p50 / p95 | 11.2 ms / 14.5 ms | 6.4 ms / 9.5 ms (server) |
| Generation (RAGAS) | Faithfulness (all 18 / 16 answerable) | 0.870 / 0.979 ‡ | not rerun |
| Generation (RAGAS) | Context Precision / Context Recall | 0.796 / 1.000 | not rerun |
| Generation (RAGAS) | Answer Relevancy | 0.665 † | not rerun |

Run 2 records: `results/leak_sweep_*.json` and `results/retrieval_*.json` (per-check and per-question rows, commit, Qdrant location, models).

The governance result reproduced exactly. MRR moved by one question dropping from rank 1 to rank 2. Corpus, golden set, chunking and the filter condition did not change between runs (the refactor in `rag.py` and `ingest.py` moved the same filter and labels into `governance.py`; see `git diff f560307 dc6e9cf`), and run 2 gives identical numbers on the Qdrant server and in embedded mode, so the shift came from the environment (library versions, model weights or hardware), which run 1 did not record. Scale caveat: the index has 12 chunks, so random top-5 retrieval would already score hit rate 0.84 and MRR 0.48; MRR is the informative number. Doc precision@5 counts unique documents in the top 5 and is low by construction with one relevant document per question.

‡ The two unanswerable questions were both refused ("I don't have that information"); the Qwen2.5-32B judge scores a correct refusal as faithfulness 0, the Llama-3.1-70B judge as 1. On the 16 answerable questions the two judges agree closely (0.979 vs 0.958).

† A known artifact of terse-but-correct answers, confirmed stable across three different judge models (see [Judge A/B](#the-llm-judge-ab-experiment)).

### Environment of the 2026-11-08 runs

| Component | Version |
|---|---|
| Python packages | `requirements.lock` (qdrant-client 1.18.0, ragas 0.2.14, langchain-ollama 0.2.3, fastapi 0.138.0) |
| Qdrant server | 1.18.2 (image digest `sha256:75eab8c4ba42`) |
| Ollama | 0.14.1, `nomic-embed-text` digest `0a109f422b47` (local Mac) |
| Python | 3.10.8 |

Run 1 (July) used models served from GVSU's H100 cluster over an SSH tunnel; its library and model versions were not recorded. The RAGAS rows come from run 1 only.

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
| `evaluate_retrieval.py` | Doc precision@k, Recall@k, MRR, Hit Rate, latency percentiles; writes a run record to `results/` |
| `evaluate_ragas.py` | RAGAS faithfulness / relevancy / context precision + recall (LLM-as-judge) |
| `test_governance.py` | Live leak sweep: every question × every user. Expected levels come from `manifest.csv`, not the stored payload, so a mislabeled chunk is caught. Writes a run record to `results/`; exits non-zero on any leak, label disagreement or access mismatch |
| `tests/test_offline.py` | Unit tests that need no Ollama or Qdrant server: fail-closed labels, clearance validation, and a randomized check of the filter on qdrant-client's embedded engine |
| `tests/test_api.py` | API tests with fakes: every search is filtered, no unfiltered probe and no withheld metadata when the teaching flag is off, out-of-range clearances rejected |
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
  -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant:v1.18.2

# 2. Models (any Ollama endpoint; set the URL in config.py)
ollama pull nomic-embed-text && ollama pull llama3.1:8b && ollama pull qwen2.5:32b

# 3. Python env (3.10). requirements.lock = exact versions of the 2026-11-08 runs
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock

# 4. The pipeline, in order
python ingest.py                 # chunk → embed → index with governance payload
python rag.py                    # smoke test
python evaluate_retrieval.py     # IR metrics + latency
python evaluate_ragas.py         # LLM-judged quality (slow)
python test_governance.py        # leak test: must print 0 and exit 0
python -m pytest -q              # offline governance + API tests (no models needed)

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
