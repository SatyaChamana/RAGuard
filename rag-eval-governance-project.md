# RAG Eval + Governance Lab — Project Setup

A self-contained, fully local RAG system for **practicing evaluation and governance** on your own data. Everything runs on your machine: **Qdrant** for vectors, **Ollama** for embeddings *and* generation (open-weight), **RAGAS** for LLM-judged quality, plus retrieval metrics and a per-user authorization gate.

The goal isn't "build a chatbot." It's to answer one question rigorously: **is this RAG system actually working well — and can it leak data it shouldn't?**

---

## What you'll be able to measure

| Category | Metrics | What it tells you |
|---|---|---|
| **Generation quality** (RAGAS) | Faithfulness, Response Relevancy | Is the answer grounded in retrieved context, and does it actually address the question (hallucination signal) |
| **Retrieval quality** (RAGAS, needs references) | Context Precision, Context Recall | Did you retrieve the right context, and was the right context complete |
| **Retrieval quality** (classic, needs labels) | Precision@k, Recall@k, MRR, Hit Rate | Pure retriever health, model-independent |
| **Performance** | p50 / p95 retrieval latency, end-to-end latency | Production viability |
| **Governance** | Leak rate, authorization precision/recall | Does a restricted doc *ever* reach the LLM for an unauthorized user |

The governance piece is the one most "RAG tutorials" skip — and it's exactly the Adobe "relevant ≠ permitted" pattern: **authorization happens before the model ever sees the data, so leakage is structurally impossible, not instruction-dependent.**

---

## Architecture

```
                          ┌─────────────────────────────────────────┐
  Query + User(role) ───► │  1. Embed query (Ollama: nomic-embed)    │
                          └─────────────────────┬───────────────────┘
                                                ▼
                          ┌─────────────────────────────────────────┐
                          │  2. Qdrant search WITH payload filter:   │
                          │     classification_rank <= user_clearance│  ◄── GOVERNANCE GATE
                          │     (unauthorized docs never enter the   │      (pre-filter)
                          │      candidate set)                      │
                          └─────────────────────┬───────────────────┘
                                                ▼
                          ┌─────────────────────────────────────────┐
                          │  3. (optional) rerank top-N              │
                          └─────────────────────┬───────────────────┘
                                                ▼
                          ┌─────────────────────────────────────────┐
                          │  4. Assemble prompt from PERMITTED ctx   │
                          │     → Ollama generate (llama3.1 / qwen)  │
                          └─────────────────────┬───────────────────┘
                                                ▼
                          Answer  +  retrieved_contexts  +  latencies
                                                │
                                                ▼
                          ┌─────────────────────────────────────────┐
                          │  EVAL: RAGAS + retrieval metrics +       │
                          │        governance leak test              │
                          └─────────────────────────────────────────┘
```

---

## Prerequisites

- **Python 3.10+**
- **Docker** (for Qdrant)
- **Ollama** installed and running — https://ollama.com

Pull the models (one embedding model, one generator; the generator doubles as the RAGAS judge):

```bash
ollama pull nomic-embed-text      # 768-dim embeddings, fast, local
ollama pull llama3.1:8b           # generator + RAGAS judge
# Optional, stronger judge if you have the RAM:
# ollama pull qwen2.5:14b
```

Start Qdrant locally:

```bash
docker run -d --name qdrant-lab \
  -p 6333:6333 -p 6334:6334 \
  -v "$(pwd)/qdrant_storage:/qdrant/storage" \
  qdrant/qdrant
```

> ⚠️ **Honest caveat on the local judge.** RAGAS metrics like Faithfulness use an *LLM as judge*. A small local model (8B) gives noisier scores than a frontier judge. That's fine for practice — and a real lesson — but treat absolute scores as directional, watch *relative* movement between config changes, and consider `qwen2.5:14b` as judge if you can run it. This "quantify the cost on your own setup, don't assume" discipline is itself the point.

---

## Project structure

```
rag-eval-governance/
├── requirements.txt
├── config.py
├── documents/                 # ← your data goes here (.md / .txt)
│   ├── public_handbook.md
│   ├── confidential_finance.md
│   └── restricted_board.md
├── manifest.csv               # ← governance labels for each doc
├── ingest.py                  # chunk → embed → upsert (with governance payload)
├── rag.py                     # authorized retrieval + generation pipeline
├── golden_set.json            # eval questions + ground-truth + access labels
├── evaluate_ragas.py          # RAGAS quality metrics
├── evaluate_retrieval.py      # precision@k / recall@k / MRR / latency
└── test_governance.py         # multi-user leak test
```

---

## 1. `requirements.txt`

```txt
qdrant-client>=1.12
langchain-ollama>=0.2
langchain-text-splitters>=0.3
ragas>=0.2.8
datasets>=3.0
pandas>=2.2
numpy>=1.26
tqdm>=4.66
```

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## 2. `config.py`

```python
# Central config so every script agrees on names/dims/models.
QDRANT_URL = "http://localhost:6333"
COLLECTION = "rag_lab"

EMBED_MODEL = "nomic-embed-text"     # 768-dim
EMBED_DIM = 768
GEN_MODEL = "llama3.1:8b"            # generation
JUDGE_MODEL = "llama3.1:8b"          # RAGAS judge (bump to qwen2.5:14b if you can)

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TOP_K = 5

# Clearance model: a user's clearance_rank must be >= a doc's classification_rank.
CLASSIFICATION_RANK = {
    "public": 0,
    "confidential": 1,
    "restricted": 2,
}
# Users you'll test with (mirrors Adobe's Alice/Bob/Carol demo).
USERS = {
    "alice": {"clearance": 0},   # public only
    "bob":   {"clearance": 1},   # public + confidential
    "carol": {"clearance": 2},   # everything
}
```

---

## 3. Your data + governance labels

Drop your documents into `documents/` (`.md` or `.txt`). Then label each one in `manifest.csv` — **this is the governance ground truth**:

```csv
filename,classification
public_handbook.md,public
confidential_finance.md,confidential
restricted_board.md,restricted
```

Tip for a meaningful test: put a *distinctive fact* in the restricted doc (e.g., a fake "Project Helios budget is $4.2M") that appears **nowhere else**. Later you'll ask about it as Alice and confirm the system can't surface it.

---

## 4. `ingest.py` — chunk, embed, upsert with governance payload

```python
import csv, uuid, pathlib
from qdrant_client import QdrantClient, models
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from tqdm import tqdm
import config as C

client = QdrantClient(url=C.QDRANT_URL)
embedder = OllamaEmbeddings(model=C.EMBED_MODEL)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=C.CHUNK_SIZE, chunk_overlap=C.CHUNK_OVERLAP
)

def load_manifest():
    with open("manifest.csv") as f:
        return {r["filename"]: r["classification"] for r in csv.DictReader(f)}

def main():
    client.recreate_collection(
        collection_name=C.COLLECTION,
        vectors_config=models.VectorParams(size=C.EMBED_DIM, distance=models.Distance.COSINE),
    )
    # Index the governance field so payload filtering is fast.
    client.create_payload_index(
        collection_name=C.COLLECTION,
        field_name="classification_rank",
        field_schema=models.PayloadSchemaType.INTEGER,
    )

    manifest = load_manifest()
    points = []
    for path in pathlib.Path("documents").glob("*"):
        if path.suffix not in (".md", ".txt"):
            continue
        classification = manifest.get(path.name, "restricted")  # fail-closed default
        rank = C.CLASSIFICATION_RANK[classification]
        chunks = splitter.split_text(path.read_text(encoding="utf-8"))
        vectors = embedder.embed_documents(chunks)
        for chunk, vec in zip(chunks, vectors):
            points.append(models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={
                    "text": chunk,
                    "source": path.name,
                    "classification": classification,
                    "classification_rank": rank,
                },
            ))
    client.upsert(collection_name=C.COLLECTION, points=points)
    print(f"Indexed {len(points)} chunks from {len(manifest)} documents.")

if __name__ == "__main__":
    main()
```

> **Fail-closed** is deliberate: an unlabeled doc defaults to the *most* restricted class, not the least. That's the safe default any governance system should have.

---

## 5. `rag.py` — authorized retrieval + generation

```python
import time
from qdrant_client import QdrantClient, models
from langchain_ollama import OllamaEmbeddings, ChatOllama
import config as C

client = QdrantClient(url=C.QDRANT_URL)
embedder = OllamaEmbeddings(model=C.EMBED_MODEL)
llm = ChatOllama(model=C.GEN_MODEL, temperature=0)

def authorized_search(query: str, user_clearance: int, top_k: int = C.TOP_K):
    """Retrieval WITH the governance gate baked in.
    Docs above the user's clearance never enter the candidate set."""
    qvec = embedder.embed_query(query)
    t0 = time.perf_counter()
    hits = client.query_points(
        collection_name=C.COLLECTION,
        query=qvec,
        limit=top_k,
        query_filter=models.Filter(
            must=[models.FieldCondition(
                key="classification_rank",
                range=models.Range(lte=user_clearance),  # doc_rank <= user_clearance
            )]
        ),
        with_payload=True,
    ).points
    latency_ms = (time.perf_counter() - t0) * 1000
    return hits, latency_ms

PROMPT = """Answer the question using ONLY the context below. \
If the context doesn't contain the answer, say "I don't have that information."

Context:
{context}

Question: {question}
Answer:"""

def answer(query: str, user_clearance: int):
    hits, ret_ms = authorized_search(query, user_clearance)
    contexts = [h.payload["text"] for h in hits]
    sources = [h.payload["source"] for h in hits]
    t0 = time.perf_counter()
    resp = llm.invoke(PROMPT.format(context="\n\n".join(contexts), question=query))
    gen_ms = (time.perf_counter() - t0) * 1000
    return {
        "question": query,
        "answer": resp.content,
        "retrieved_contexts": contexts,
        "sources": sources,
        "retrieval_ms": ret_ms,
        "generation_ms": gen_ms,
    }

if __name__ == "__main__":
    out = answer("What does the handbook say about remote work?", user_clearance=2)
    print(out["answer"], "\n\nsources:", out["sources"])
```

**Pre-filter vs post-filter (worth understanding, this is a real interview question):** filtering inside the Qdrant query (above) means unauthorized docs are *never candidates* — cleanest for governance. The alternative — retrieve broadly, then drop unauthorized docs afterward — can hurt recall (your top-k fills with docs you'll discard) and risks a leak if the filter is ever skipped. Pre-filtering with an indexed payload field avoids both. Qdrant's filterable HNSW is built for exactly this.

---

## 6. `golden_set.json` — your eval set

This is the labor that separates real evaluation from vibes. Each entry: a question, a ground-truth answer (unlocks Context Recall + Precision), the relevant source docs (unlocks retrieval metrics), and the **minimum clearance** the question requires (for the governance test).

```json
[
  {
    "question": "What is the company's remote work policy?",
    "ground_truth": "Employees may work remotely up to three days per week with manager approval.",
    "relevant_sources": ["public_handbook.md"],
    "min_clearance": 0
  },
  {
    "question": "What was Q3 operating margin?",
    "ground_truth": "Q3 operating margin was 18.4%.",
    "relevant_sources": ["confidential_finance.md"],
    "min_clearance": 1
  },
  {
    "question": "What is the Project Helios budget?",
    "ground_truth": "The Project Helios budget is $4.2M.",
    "relevant_sources": ["restricted_board.md"],
    "min_clearance": 2
  }
]
```

Aim for **15–30 questions** spread across classifications for stable numbers. Use a few "unanswerable" questions too — a good system says "I don't have that information" instead of inventing one.

---

## 7. `evaluate_ragas.py` — generation + retrieval quality

```python
import json
from ragas import evaluate, EvaluationDataset
from ragas.metrics import (
    Faithfulness, ResponseRelevancy,
    LLMContextPrecisionWithReference, LLMContextRecall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import ChatOllama, OllamaEmbeddings
from rag import answer
import config as C

judge = LangchainLLMWrapper(ChatOllama(model=C.JUDGE_MODEL, temperature=0))
judge_emb = LangchainEmbeddingsWrapper(OllamaEmbeddings(model=C.EMBED_MODEL))

def build_samples():
    golden = json.load(open("golden_set.json"))
    rows = []
    for item in golden:
        # Run each question at MAX clearance so retrieval is fully tested
        # (governance is tested separately in test_governance.py).
        out = answer(item["question"], user_clearance=2)
        rows.append({
            "user_input": item["question"],
            "retrieved_contexts": out["retrieved_contexts"],
            "response": out["answer"],
            "reference": item["ground_truth"],
        })
    return EvaluationDataset.from_list(rows)

def main():
    dataset = build_samples()
    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),                          # answer grounded in context?
            ResponseRelevancy(),                     # answer addresses the question?
            LLMContextPrecisionWithReference(),      # retrieved context relevant?
            LLMContextRecall(),                      # did we retrieve all needed context?
        ],
        llm=judge,
        embeddings=judge_emb,
    )
    print(result)
    result.to_pandas().to_csv("ragas_results.csv", index=False)
    print("\nPer-question detail written to ragas_results.csv")

if __name__ == "__main__":
    main()
```

**Reading it:** Faithfulness and Context Recall near 1.0 with low Response Relevancy usually means retrieval is fine but the prompt/model is rambling. Low Faithfulness with high Context Precision means the model is *hallucinating despite* good context — a generation problem, not a retrieval one. Diagnosing *which axis* is broken is the entire skill.

---

## 8. `evaluate_retrieval.py` — model-independent retriever health + latency

```python
import json, numpy as np
from rag import authorized_search
import config as C

def main():
    golden = json.load(open("golden_set.json"))
    precisions, recalls, rr, hits_at_k, latencies = [], [], [], [], []
    for item in golden:
        hits, ms = authorized_search(item["question"], user_clearance=2, top_k=C.TOP_K)
        latencies.append(ms)
        retrieved = [h.payload["source"] for h in hits]
        relevant = set(item["relevant_sources"])

        tp = sum(1 for s in retrieved if s in relevant)
        precisions.append(tp / max(len(retrieved), 1))
        recalls.append(tp / max(len(relevant), 1))
        hits_at_k.append(1.0 if tp > 0 else 0.0)
        rank = next((i + 1 for i, s in enumerate(retrieved) if s in relevant), None)
        rr.append(1.0 / rank if rank else 0.0)

    print(f"Precision@{C.TOP_K}: {np.mean(precisions):.3f}")
    print(f"Recall@{C.TOP_K}:    {np.mean(recalls):.3f}")
    print(f"Hit Rate@{C.TOP_K}:  {np.mean(hits_at_k):.3f}")
    print(f"MRR:               {np.mean(rr):.3f}")
    print(f"Latency p50:       {np.percentile(latencies, 50):.1f} ms")
    print(f"Latency p95:       {np.percentile(latencies, 95):.1f} ms")

if __name__ == "__main__":
    main()
```

---

## 9. `test_governance.py` — the leak test (the most important script)

```python
import json
import config as C
from rag import authorized_search

def main():
    golden = json.load(open("golden_set.json"))
    leaks, checks = [], 0
    for item in golden:
        needed = item["min_clearance"]
        for username, u in C.USERS.items():
            clearance = u["clearance"]
            hits, _ = authorized_search(item["question"], user_clearance=clearance)
            # Any retrieved chunk above this user's clearance = a leak.
            for h in hits:
                checks += 1
                if h.payload["classification_rank"] > clearance:
                    leaks.append((username, item["question"], h.payload["source"]))

    print(f"Authorization checks run: {checks}")
    print(f"LEAKS DETECTED: {len(leaks)}")
    for user, q, src in leaks:
        print(f"  ❌ {user} retrieved '{src}' via: {q!r}")
    if not leaks:
        print("  ✅ No leaks. Authorization is structurally enforced.")

    # Sanity check the inverse: authorized users CAN reach what they need.
    print("\nAccess sanity check:")
    for item in golden:
        for username, u in C.USERS.items():
            should_access = u["clearance"] >= item["min_clearance"]
            hits, _ = authorized_search(item["question"], user_clearance=u["clearance"])
            got = any(h.payload["source"] in item["relevant_sources"] for h in hits)
            status = "ok" if got == should_access else "MISMATCH"
            if status == "MISMATCH":
                print(f"  ⚠️  {username} / {item['question'][:40]!r}: "
                      f"expected_access={should_access}, got={got}")
```

**The decisive demo:** ask *"What is the Project Helios budget?"* as `alice` (clearance 0). A correct system retrieves nothing restricted and the model answers "I don't have that information." Leak rate must be **exactly 0** — anything else is a finding, not a rounding error.

---

## How to run the whole thing

```bash
python ingest.py                 # build the vector DB from your data
python rag.py                    # smoke test one query
python evaluate_retrieval.py     # retriever health + latency
python evaluate_ragas.py         # generation + retrieval quality (slow; LLM judge)
python test_governance.py        # leak test — must be 0
```

---

## Is the model working well? — the verdict rubric

| Signal | Healthy | Investigate |
|---|---|---|
| Faithfulness | > 0.85 | < 0.7 → model hallucinating; tighten prompt, check context |
| Response Relevancy | > 0.8 | low → rambling / off-topic answers |
| Context Precision | > 0.7 | low → retriever pulling noise; add reranking |
| Context Recall | > 0.8 | low → right doc not retrieved; fix chunking / hybrid search |
| Hit Rate@k | > 0.9 | low → embedding or chunking problem |
| p95 latency | within your budget | spikes → quantization / index tuning |
| **Leak rate** | **exactly 0** | **anything > 0 is a hard fail** |

The mental model (straight from the eval discipline): **fix recall first, then precision.** If the right context isn't being retrieved, no amount of prompt tuning helps. Once recall is solid, reranking is your highest-leverage lever for precision.

---

## Extensions (once the baseline runs)

1. **Hybrid search** — add BM25/sparse vectors alongside dense, fuse with RRF. Watch Context Precision move. This is the standard production retrieval baseline.
2. **Reranking** — over-fetch ~20 candidates, re-score with a cross-encoder, keep top 5. Usually the single biggest quality jump.
3. **Quantization** — enable scalar quantization in Qdrant and re-run the eval set. *Measure* the recall cost on your data instead of assuming it's free.
4. **A/B two generators** — run the eval set with `llama3.1:8b` vs `qwen2.5:7b`, compare faithfulness/latency, pick with evidence rather than vibes.
5. **Attribute-based access (ABAC)** — extend governance beyond a single clearance rank to per-department / per-project tags.

---

### Why this is also tomorrow's best talking point
You now have a hands-on artifact that hits every theme of the Deloitte event: **evaluation** (RAGAS, retrieval metrics), **governance** (authorization before the model, structurally leak-proof), **cost/privacy** (fully local open-weight via Ollama), and the **diagnostic discipline** of knowing *which* axis is broken. If a hiring manager asks "how do you know a RAG system is production-ready?" — this project is your answer, built, not described.
