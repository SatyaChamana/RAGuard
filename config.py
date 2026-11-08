# Central config so every script agrees on names/dims/models.
import os

# Qdrant location: a server URL (Docker, default) or a local directory for
# qdrant-client's embedded mode, e.g. RAGUARD_QDRANT=./local_qdrant (no Docker).
QDRANT_URL = os.environ.get("RAGUARD_QDRANT", "http://localhost:6333")
COLLECTION = "rag_lab"

# --- Ollama endpoints --------------------------------------------------------
# Works with a local Ollama daemon or an SSH tunnel to a remote GPU server —
# the app is endpoint-agnostic. Each model role (embed / generate / judge) has
# its own base_url so they can be split across machines independently.
OLLAMA_URL = "http://localhost:11434"

EMBED_BASE_URL = OLLAMA_URL
GEN_BASE_URL   = OLLAMA_URL
JUDGE_BASE_URL = OLLAMA_URL

# Embedder A/B candidate: mxbai-embed-large (set EMBED_DIM=1024 + re-ingest).
EMBED_MODEL = "nomic-embed-text"     # 768-dim (served by EMBED_BASE_URL)
EMBED_DIM = 768
GEN_MODEL = "llama3.1:8b"            # generation baseline — keep on 8B for clean A/Bs
JUDGE_MODEL = "qwen2.5:32b"          # RAGAS judge — validating vs llama3.1:70b; rm 70b once scores are clean

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TOP_K = 5

# Clearance model: a user's clearance_rank must be >= a doc's classification_rank.
CLASSIFICATION_RANK = {
    "public": 0,
    "confidential": 1,
    "restricted": 2,
}
# Demo users (the common Alice/Bob/Carol convention).
USERS = {
    "alice": {"clearance": 0},   # public only
    "bob":   {"clearance": 1},   # public + confidential
    "carol": {"clearance": 2},   # everything
}

# Demo UI only. When True, /api/ask also returns metadata (file name,
# classification, similarity score) of the top chunks the user was NOT cleared
# to see. That is a deliberate side channel for teaching, so it is OFF by
# default: existence, names and relevance of restricted documents are
# themselves sensitive.
SHOW_WITHHELD_METADATA = os.environ.get("RAGUARD_SHOW_WITHHELD", "0") == "1"
