import json, subprocess, time
from pathlib import Path

import numpy as np
from rag import authorized_search
import config as C

def main():
    golden = json.load(open("golden_set.json"))
    precisions, recalls, rr, hits_at_k, latencies, per_q = [], [], [], [], [], []
    for item in golden:
        # Skip "unanswerable" questions (no relevant sources) for retrieval scoring.
        if not item["relevant_sources"]:
            continue
        hits, ms = authorized_search(item["question"], user_clearance=2, top_k=C.TOP_K)
        latencies.append(ms)
        # Labels are at DOCUMENT granularity, but retrieval returns CHUNKS and
        # several chunks can share a source. Dedupe to unique docs (order-preserving)
        # before scoring, otherwise true positives can exceed the relevant count
        # and recall blows past 1.0.
        retrieved = []
        for h in hits:
            src = h.payload["source"]
            if src not in retrieved:
                retrieved.append(src)
        relevant = set(item["relevant_sources"])

        tp = sum(1 for s in retrieved if s in relevant)
        precisions.append(tp / max(len(retrieved), 1))
        recalls.append(tp / max(len(relevant), 1))
        hits_at_k.append(1.0 if tp > 0 else 0.0)
        rank = next((i + 1 for i, s in enumerate(retrieved) if s in relevant), None)
        rr.append(1.0 / rank if rank else 0.0)
        per_q.append({"question": item["question"], "retrieved_docs": retrieved, "first_relevant_rank": rank})

    # Precision over UNIQUE documents in the top-k (chunks deduped to documents), not chunk P@k.
    print(f"Doc precision@{C.TOP_K}: {np.mean(precisions):.3f}")
    print(f"Recall@{C.TOP_K}:    {np.mean(recalls):.3f}")
    print(f"Hit Rate@{C.TOP_K}:  {np.mean(hits_at_k):.3f}")
    print(f"MRR:               {np.mean(rr):.3f}")
    print(f"Latency p50:       {np.percentile(latencies, 50):.1f} ms")
    print(f"Latency p95:       {np.percentile(latencies, 95):.1f} ms")
    print(f"Questions scored:  {len(rr)}")

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    Path("results").mkdir(exist_ok=True)
    out = Path("results") / f"retrieval_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({
        "commit": commit, "qdrant": C.QDRANT_URL, "embed_model": C.EMBED_MODEL, "top_k": C.TOP_K,
        "n_questions": len(rr), "doc_precision": float(np.mean(precisions)), "recall": float(np.mean(recalls)),
        "hit_rate": float(np.mean(hits_at_k)), "mrr": float(np.mean(rr)),
        "latency_p50_ms": float(np.percentile(latencies, 50)), "latency_p95_ms": float(np.percentile(latencies, 95)),
        "per_question": per_q}, indent=1))
    print(f"Run record: {out}")

if __name__ == "__main__":
    main()
