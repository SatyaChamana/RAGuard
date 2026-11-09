"""Leak sweep and access check against the live index (needs Ollama embeddings and Qdrant).

Two independent checks:
  1. Leak sweep: for every golden question and every user, no retrieved chunk may be above the
     user's clearance. The expected level of each chunk is derived from manifest.csv via
     governance.classify (keyed by source file), NOT from the stored payload, so a chunk indexed
     with the wrong label is caught as a leak instead of agreeing with itself.
  2. Access check: cleared users must reach the golden set's relevant document (uses the golden
     set's min_clearance as ground truth, independent of the stored labels).

Writes a run record to results/leak_sweep_<timestamp>.json. Exits non-zero on any failure.
Run: python test_governance.py
"""
import csv, json, os, subprocess, sys, time
from pathlib import Path

import config as C
from governance import classify
from rag import authorized_search


def load_manifest(path="manifest.csv"):
    with open(path, newline="") as f:
        return {r["filename"]: r["classification"] for r in csv.DictReader(f)}


def main():
    golden = json.load(open("golden_set.json"))
    manifest = load_manifest()
    rows, leaks, label_mismatch, checks = [], [], [], 0
    for item in golden:
        for username, u in C.USERS.items():
            clearance = u["clearance"]
            hits, _ = authorized_search(item["question"], user_clearance=clearance)
            for h in hits:
                checks += 1
                src = h.payload["source"]
                expected = C.CLASSIFICATION_RANK[classify(src, manifest)]
                stored = h.payload["classification_rank"]
                leak = expected > clearance or stored > clearance
                if stored != expected:
                    label_mismatch.append((src, stored, expected))
                if leak:
                    leaks.append((username, item["question"], src))
                rows.append({"user": username, "clearance": clearance, "question": item["question"],
                             "source": src, "stored_rank": stored, "manifest_rank": expected, "leak": leak})

    print(f"Authorization checks run: {checks}")
    print(f"LEAKS DETECTED: {len(leaks)}")
    for user, q, src in leaks:
        print(f"  LEAK: {user} retrieved '{src}' via: {q!r}")
    print(f"Stored labels that disagree with manifest.csv: {len(label_mismatch)}")

    print("\nAccess check:")
    access, mismatches = [], 0
    for item in golden:
        # "Unanswerable" questions have no relevant source, so there is nothing to reach.
        if not item["relevant_sources"]:
            continue
        for username, u in C.USERS.items():
            should = u["clearance"] >= item["min_clearance"]
            hits, _ = authorized_search(item["question"], user_clearance=u["clearance"])
            got = any(h.payload["source"] in item["relevant_sources"] for h in hits)
            if got != should:
                mismatches += 1
                print(f"  MISMATCH {username} / {item['question'][:40]!r}: expected={should}, got={got}")
            access.append({"user": username, "question": item["question"], "should_access": should, "got": got})
    cleared = sum(a["should_access"] for a in access)
    print(f"Access mismatches: {mismatches} (cleared pairs: {cleared}, uncleared pairs: {len(access) - cleared})")

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    Path("results").mkdir(exist_ok=True)
    out = Path("results") / f"leak_sweep_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({
        "commit": commit, "qdrant": C.QDRANT_URL, "embed_model": C.EMBED_MODEL, "top_k": C.TOP_K,
        "checks": checks, "leaks": len(leaks), "label_mismatches": len(label_mismatch),
        "access_mismatches": mismatches, "cleared_pairs": cleared, "uncleared_pairs": len(access) - cleared,
        "chunk_checks": rows, "access_checks": access}, indent=1))
    print(f"Run record: {out}")
    return 1 if (leaks or mismatches or label_mismatch) else 0


if __name__ == "__main__":
    sys.exit(main())
