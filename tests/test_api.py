"""API-level governance tests with fakes (no Ollama, no Qdrant server).

A fake `rag` module stands in for the live components, so these tests exercise api.py's own
search path: the filter is always applied, the unfiltered probe never runs when the teaching
flag is off, and no withheld metadata reaches the response.
"""
import sys, types
from types import SimpleNamespace

import pytest

CHUNKS = [  # (id, source, classification, rank)
    (1, "public_handbook.md", "public", 0),
    (2, "confidential_q3_review.md", "confidential", 1),
    (3, "restricted_board_brief.md", "restricted", 2),
]


class FakeClient:
    def __init__(self):
        self.calls = []

    def query_points(self, collection_name, query, limit, query_filter, with_payload):
        lte = query_filter.must[0].range.lte if query_filter is not None else None
        self.calls.append(lte)
        pts = [SimpleNamespace(id=i, score=0.9 - 0.1 * i,
                               payload={"source": s, "classification": c, "classification_rank": r, "text": f"text {i}"})
               for i, s, c, r in CHUNKS if lte is None or r <= lte]
        return SimpleNamespace(points=pts[:limit])


@pytest.fixture()
def api(monkeypatch):
    fake = types.ModuleType("rag")
    fake.client = FakeClient()
    fake.embedder = SimpleNamespace(embed_query=lambda q: [0.0] * 4)
    fake.llm = SimpleNamespace(invoke=lambda prompt: SimpleNamespace(content="stub answer"))
    fake.PROMPT = "{context}\n{question}"
    monkeypatch.setitem(sys.modules, "rag", fake)
    sys.modules.pop("api", None)
    import api as api_module
    yield api_module, fake.client
    sys.modules.pop("api", None)


def _ask(api_module, clearance):
    from fastapi.testclient import TestClient
    return TestClient(api_module.app).post("/api/ask", json={"question": "q", "clearance": clearance})


@pytest.mark.parametrize("clearance", [0, 1, 2])
def test_flag_off_returns_no_withheld_metadata_and_never_probes(api, monkeypatch, clearance):
    api_module, client = api
    monkeypatch.setattr(api_module.C, "SHOW_WITHHELD_METADATA", False)
    body = _ask(api_module, clearance).json()
    assert body["withheld"] == []
    assert body["governance"]["withheld_count"] is None
    assert body["governance"]["withheld_classes"] == []
    assert client.calls == [clearance]  # one filtered search, no unfiltered probe
    assert all(c["rank"] <= clearance for c in body["retrieved"])
    assert body["governance"]["leak_count"] == 0


def test_every_search_is_filtered_even_with_teaching_flag(api, monkeypatch):
    api_module, client = api
    monkeypatch.setattr(api_module.C, "SHOW_WITHHELD_METADATA", True)
    _ask(api_module, 0)
    assert None not in client.calls  # the probe also goes through the filter (at max rank)


def test_client_cannot_send_out_of_range_clearance(api):
    api_module, _ = api
    assert _ask(api_module, 99).status_code == 422
