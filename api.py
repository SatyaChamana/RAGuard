"""FastAPI bridge between the React frontend (web/) and the RAG pipeline.

It reuses rag.py's live components (the same Qdrant client, embedder, LLM, the
governance pre-filter, and the prompt) and returns a rich payload the UI can
visualise: the answer, the retrieved chunks (with scores + classifications),
what was *withheld* by the user's clearance, the leak status, and per-stage
timings.

Run:  uvicorn api:app --reload --port 8000   (from the project root, venv active)
"""
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import config as C
from rag import client, embedder, llm, PROMPT
from governance import clearance_filter

app = FastAPI(title="RAGuard API", version="1.0")

# The React dev server runs on a different origin (5173); allow it.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

RANK_TO_NAME = {v: k for k, v in C.CLASSIFICATION_RANK.items()}
MAX_RANK = max(C.CLASSIFICATION_RANK.values())
# Map a clearance level back to the demo user that has it (alice/bob/carol).
CLEARANCE_USER = {u["clearance"]: name for name, u in C.USERS.items()}
# How many candidates to inspect (unfiltered) when computing what was withheld.
PROBE_K = 15


class AskRequest(BaseModel):
    question: str
    # Reject out-of-range clearances at the boundary (422) rather than letting
    # e.g. clearance=99 silently behave like full access.
    clearance: int = Field(ge=0, le=MAX_RANK)


def _search(qvec, clearance, top_k):
    """Vector search with the governance pre-filter baked in (None = no filter)."""
    qfilter = None if clearance is None else clearance_filter(clearance)
    t0 = time.perf_counter()
    hits = client.query_points(
        collection_name=C.COLLECTION,
        query=qvec,
        limit=top_k,
        query_filter=qfilter,
        with_payload=True,
    ).points
    ms = (time.perf_counter() - t0) * 1000
    return hits, ms


def _chunk(h, include_text=True):
    out = {
        "source": h.payload["source"],
        "classification": h.payload["classification"],
        "rank": h.payload["classification_rank"],
        "score": round(float(h.score), 4),
    }
    if include_text:
        out["text"] = h.payload["text"]
    return out


@app.get("/api/config")
def get_config():
    """Everything the UI needs to render the clearance selector + legend."""
    users = sorted(C.USERS.items(), key=lambda kv: kv[1]["clearance"])
    classifications = sorted(C.CLASSIFICATION_RANK.items(), key=lambda kv: kv[1])
    return {
        "users": [
            {
                "name": name,
                "clearance": u["clearance"],
                "label": RANK_TO_NAME[u["clearance"]],
                "accessible": [k for k, v in C.CLASSIFICATION_RANK.items() if v <= u["clearance"]],
            }
            for name, u in users
        ],
        "classifications": [{"name": k, "rank": v} for k, v in classifications],
        "gen_model": C.GEN_MODEL,
        "embed_model": C.EMBED_MODEL,
        "embed_dim": C.EMBED_DIM,
        "top_k": C.TOP_K,
        "collection": C.COLLECTION,
    }


@app.post("/api/ask")
def ask(req: AskRequest):
    total0 = time.perf_counter()

    # 1. Embed the query (same model as ingestion -> same 768-d space).
    t0 = time.perf_counter()
    qvec = embedder.embed_query(req.question)
    embed_ms = (time.perf_counter() - t0) * 1000

    # 2. Authorized search: governance pre-filter at the user's clearance.
    user_hits, retrieval_ms = _search(qvec, req.clearance, C.TOP_K)

    # 3. Optional teaching probe: what did the filter WITHHOLD? Off by default,
    #    because names, classes and scores of restricted chunks are a side channel.
    withheld = []
    if C.SHOW_WITHHELD_METADATA:
        full_hits, _ = _search(qvec, MAX_RANK, PROBE_K)
        user_ids = {h.id for h in user_hits}
        withheld = [
            h for h in full_hits
            if h.id not in user_ids and h.payload["classification_rank"] > req.clearance
        ]

    # 4. Generate the answer from ONLY the permitted context.
    contexts = [h.payload["text"] for h in user_hits]
    t0 = time.perf_counter()
    if contexts:
        answer = llm.invoke(PROMPT.format(
            context="\n\n".join(contexts), question=req.question)).content
    else:
        answer = "I don't have that information."
    gen_ms = (time.perf_counter() - t0) * 1000

    # Leak check: any permitted chunk above the user's clearance? (Must be 0.)
    leaks = [h for h in user_hits if h.payload["classification_rank"] > req.clearance]
    total_ms = (time.perf_counter() - total0) * 1000

    return {
        "question": req.question,
        "clearance": req.clearance,
        "user": CLEARANCE_USER.get(req.clearance),
        "answer": answer,
        "retrieved": [_chunk(h) for h in user_hits],
        # NB: withheld chunks expose metadata only — never their text (that
        # would itself be a leak in the demo UI).
        "withheld": [_chunk(h, include_text=False) for h in withheld],
        "governance": {
            "accessible": [k for k, v in C.CLASSIFICATION_RANK.items() if v <= req.clearance],
            "withheld_shown": C.SHOW_WITHHELD_METADATA,
            "withheld_classes": sorted({h.payload["classification"] for h in withheld}),
            "withheld_count": len(withheld) if C.SHOW_WITHHELD_METADATA else None,
            "leak": len(leaks) > 0,
            "leak_count": len(leaks),
        },
        "timings": {
            "embed_ms": round(embed_ms, 1),
            "retrieval_ms": round(retrieval_ms, 1),
            "generation_ms": round(gen_ms, 1),
            "total_ms": round(total_ms, 1),
        },
        "pipeline": [
            {"step": "Embed query", "detail": f"{C.EMBED_MODEL} → {C.EMBED_DIM}-d vector", "ms": round(embed_ms, 1)},
            {"step": "Governance pre-filter", "detail": f"classification_rank ≤ {req.clearance}", "ms": 0.0},
            {"step": "Vector search", "detail": f"top-{C.TOP_K} · Qdrant '{C.COLLECTION}'", "ms": round(retrieval_ms, 1)},
            {"step": "Generate answer", "detail": C.GEN_MODEL, "ms": round(gen_ms, 1)},
        ],
    }
