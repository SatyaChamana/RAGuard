# RAGuard Web UI

React/Vite observability frontend for RAGuard. Talks to the FastAPI backend (`api.py`, port 8090) through a dev-server proxy (`/api` → `localhost:8090`).

Shows, per question: the answer, per-stage pipeline timings, the governance gate status (live leak check), the retrieved chunks with scores and classifications, and the chunks **withheld** by the user's clearance (metadata only — withheld text never reaches the client).

```bash
npm install
npm run dev        # or use ../run_app.sh to launch backend + frontend together
```
