#!/usr/bin/env bash
# Launch the RAGuard web app: FastAPI backend (:8090) + React/Vite frontend.
# Prereqs: Qdrant container up, Ollama tunnel up, `ingest.py` already run,
# the .venv created, and `web/` deps installed (npm install).
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo "▶ Starting FastAPI on http://localhost:8090 …"
source .venv/bin/activate
uvicorn api:app --port 8090 &
API_PID=$!

cleanup() { echo; echo "Stopping…"; kill $API_PID 2>/dev/null || true; }
trap cleanup EXIT INT TERM

echo "▶ Starting Vite dev server (auto-picks 5173/5174) …"
cd web
npm run dev

# When `npm run dev` exits, the trap stops the API too.
