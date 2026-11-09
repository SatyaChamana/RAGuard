"""Governance tests that need no Ollama and no Qdrant server (embedded in-memory Qdrant).

Run: python -m pytest -q
"""
import random

import pytest
from qdrant_client import QdrantClient, models

import config as C
from governance import MAX_RANK, check_clearance, classify, clearance_filter

MANIFEST = {"a.md": "public", "b.md": "confidential", "c.md": "restricted", "d.md": "Top Secret"}


def test_known_labels_pass_through():
    assert classify("a.md", MANIFEST) == "public"
    assert classify("b.md", MANIFEST) == "confidential"


def test_missing_document_fails_closed():
    assert classify("not_in_manifest.md", MANIFEST) == "restricted"


def test_unknown_label_fails_closed():
    assert classify("d.md", MANIFEST) == "restricted"


@pytest.mark.parametrize("bad", [-1, MAX_RANK + 1, 99, "2", 1.0, None, True])
def test_out_of_range_clearance_rejected(bad):
    with pytest.raises(ValueError):
        check_clearance(bad)


def test_filter_never_returns_chunks_above_clearance():
    """Random vectors, every clearance, many queries: no hit may exceed the user's clearance."""
    rng = random.Random(0)
    dim = 16
    client = QdrantClient(":memory:")
    client.create_collection("t", vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE))
    points = [models.PointStruct(id=i, vector=[rng.uniform(-1, 1) for _ in range(dim)],
                                 payload={"classification_rank": i % (MAX_RANK + 1)}) for i in range(300)]
    client.upsert("t", points=points)
    for clearance in range(MAX_RANK + 1):
        for _ in range(50):
            q = [rng.uniform(-1, 1) for _ in range(dim)]
            hits = client.query_points("t", query=q, limit=C.TOP_K, query_filter=clearance_filter(clearance),
                                       with_payload=True).points
            assert len(hits) == C.TOP_K  # the filter does not starve results
            assert all(h.payload["classification_rank"] <= clearance for h in hits)


def test_withheld_metadata_off_by_default(monkeypatch):
    """Default must be off when the teaching flag is unset, whatever the caller's shell has."""
    import importlib
    monkeypatch.delenv("RAGUARD_SHOW_WITHHELD", raising=False)
    try:
        assert importlib.reload(C).SHOW_WITHHELD_METADATA is False
    finally:
        importlib.reload(C)
