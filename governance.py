"""Governance rules in one place, with no model or network dependencies.

- classify(): fail-closed document classification from the manifest.
- clearance_filter(): the Qdrant payload pre-filter applied inside every search.
- make_client(): Qdrant server or embedded local mode, from config.
"""
import config as C

MOST_RESTRICTED = max(C.CLASSIFICATION_RANK, key=C.CLASSIFICATION_RANK.get)
MAX_RANK = C.CLASSIFICATION_RANK[MOST_RESTRICTED]


def classify(filename: str, manifest: dict) -> str:
    """Label for a document. Missing or unknown labels fail closed to the most restricted class."""
    label = manifest.get(filename, MOST_RESTRICTED)
    return label if label in C.CLASSIFICATION_RANK else MOST_RESTRICTED


def check_clearance(clearance: int) -> int:
    """Reject out-of-range clearances instead of letting them act as full access."""
    if not isinstance(clearance, int) or isinstance(clearance, bool) or not 0 <= clearance <= MAX_RANK:
        raise ValueError(f"clearance must be an int in [0, {MAX_RANK}], got {clearance!r}")
    return clearance


def clearance_filter(clearance: int):
    """Pre-filter: only chunks with classification_rank <= clearance can become candidates."""
    from qdrant_client import models
    return models.Filter(must=[models.FieldCondition(
        key="classification_rank", range=models.Range(lte=check_clearance(clearance)))])


def make_client():
    from qdrant_client import QdrantClient
    if C.QDRANT_URL.startswith(("http://", "https://")):
        return QdrantClient(url=C.QDRANT_URL)
    return QdrantClient(path=C.QDRANT_URL)
