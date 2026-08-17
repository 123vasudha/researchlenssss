# Stacks — Research Paper Recommendation System

Semantic search over research papers: type a title, abstract snippet, keyword phrase,
or DOI, and get back the ten most similar papers, ranked by cosine similarity over
sentence embeddings.

```
User → React UI → FastAPI → Semantic Scholar / arXiv → Embeddings → FAISS → Ranked Results
```

## Contents

- [Features](#features)
- [Architecture](#architecture)
- [Project layout](#project-layout)
- [Quick start (Docker)](#quick-start-docker)
- [Local development](#local-development)
- [Building the paper index](#building-the-paper-index)
- [API reference](#api-reference)
- [Evaluation](#evaluation)
- [Testing](#testing)
- [Configuration](#configuration)
- [Future improvements](#future-improvements)

## Features

- 🔍 Search by title, abstract, keywords, or DOI
- 🧠 Semantic similarity via `sentence-transformers` (`all-MiniLM-L6-v2`) + FAISS cosine search
- 🎛️ Filters by year range, author, and field of study
- 📄 Paper detail page with abstract, authors, venue, citation count, and source link
- 🕸️ Related-papers graph visualization (ego-graph of top neighbors)
- ⭐ Favorites and 🕓 search history, tied to a JWT-authenticated account
- 🐳 Docker Compose for one-command local deployment
- ✅ GitHub Actions CI (backend tests + lint, frontend build)

## Architecture

**Backend** — FastAPI (async), modular by concern:

```
backend/app/
├── main.py            # app factory, router wiring, CORS, startup
├── config.py           # pydantic-settings, reads .env
├── database.py          # SQLAlchemy engine/session
├── models/              # SQLAlchemy ORM models (User, Paper, Favorite, SearchHistory)
├── schemas/              # Pydantic request/response models
├── api/                  # route handlers: auth, search, papers, favorites, history
├── core/                 # security (JWT/bcrypt), auth dependencies
├── services/              # embeddings, FAISS wrapper, Semantic Scholar & arXiv clients
└── ml/                     # offline index-building and evaluation scripts
```

**Frontend** — React + TypeScript + Tailwind, organized as pages/components/api/hooks.

**Data flow for a search:**
1. User submits a query in the React UI.
2. FastAPI embeds the query text with `sentence-transformers`.
3. The embedding is compared against the FAISS index (`IndexFlatIP` over L2-normalized
   vectors, i.e. cosine similarity) to retrieve the closest paper IDs.
4. Full metadata for those IDs is hydrated from Postgres, filtered (year/author/field),
   and returned ranked by similarity score.
5. If the user is logged in, the query is recorded to their search history.

Paper metadata itself is fetched from the **Semantic Scholar Graph API** (primary) and
the **arXiv API** (fallback/supplement for recent preprints), then cached in Postgres and
embedded into FAISS by the offline `app.ml.build_index` pipeline — search requests never
call these external APIs directly, so search stays fast and doesn't depend on third-party
uptime.

## Project layout

```
research-paper-recommender/
├── backend/            # FastAPI service (see above)
├── frontend/           # React + TS + Tailwind app
├── docker-compose.yml
├── .env.example
└── .github/workflows/ci.yml
```

## Quick start (Docker)

Requires Docker and Docker Compose.

```bash
cp .env.example .env
# edit .env and set a real SECRET_KEY before doing anything beyond local testing

docker compose up --build
```

This starts Postgres, the FastAPI backend (`http://localhost:8000`, Swagger docs at
`http://localhost:8000/docs`), and the frontend (`http://localhost:5173`, proxied to the
backend through nginx).

The database starts empty — you still need to build the FAISS index (next section)
before search will return anything.

```bash
docker compose exec backend python -m app.ml.build_index --input data/sample_papers.json
```

That loads a small synthetic demo dataset (`backend/data/sample_papers.json`) so you can
try the UI immediately. To pull real papers from Semantic Scholar instead:

```bash
docker compose exec backend python scripts/fetch_sample_data.py
```

## Local development

Running the backend and frontend directly (without Docker) is faster for iterating.

### Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp ../.env.example .env
# For local dev without Postgres installed, you can point DATABASE_URL at SQLite instead:
#   DATABASE_URL=sqlite:///./dev.db

uvicorn app.main:app --reload --port 8000
```

Swagger UI: http://localhost:8000/docs

### Frontend

```bash
cd frontend
cp .env.example .env
npm install
npm run dev
```

Visit http://localhost:5173. In dev mode, Vite proxies `/api` requests to
`http://localhost:8000` (configurable via `VITE_API_PROXY_TARGET`).

## Building the paper index

The FAISS index and Postgres `papers` table are populated by an offline pipeline, not by
the live API, so indexing can be re-run any time without affecting request latency.

```bash
# From a local JSON file of paper dicts (see backend/data/sample_papers.json for the shape)
python -m app.ml.build_index --input data/sample_papers.json

# Or fetch fresh results from Semantic Scholar for a topic
python -m app.ml.build_index --query "graph neural networks" --limit 200

# Or seed a broader multi-topic starter dataset
python scripts/fetch_sample_data.py
```

Re-running indexing with new papers appends to the existing FAISS index — it doesn't
rebuild from scratch, so you can incrementally grow the corpus.

## API reference

Full interactive documentation (Swagger/OpenAPI) is served at `/docs` once the backend is
running. Key endpoints:

| Method | Path                              | Auth  | Description                         |
|--------|-----------------------------------|-------|--------------------------------------|
| POST   | `/api/v1/auth/register`           | No    | Create an account                   |
| POST   | `/api/v1/auth/login`              | No    | Get a JWT access token              |
| GET    | `/api/v1/auth/me`                 | Yes   | Current user profile                |
| POST   | `/api/v1/search`                  | Optional | Semantic search, top-k + filters |
| GET    | `/api/v1/papers/{id}`             | No    | Paper detail                        |
| GET    | `/api/v1/papers/{id}/related-graph` | No  | Ego-graph of similar papers          |
| GET    | `/api/v1/favorites`               | Yes   | List saved papers                   |
| POST   | `/api/v1/favorites`               | Yes   | Save a paper                        |
| DELETE | `/api/v1/favorites/{id}`          | Yes   | Remove a saved paper                |
| GET    | `/api/v1/history`                 | Yes   | List past searches                  |

## Evaluation

`app.ml.evaluate` computes **Precision@k**, **Recall@k**, and **Mean Reciprocal Rank
(MRR)** against a labeled relevance set — a JSON file mapping a query paper to the set of
papers considered ground-truth-relevant recommendations for it (e.g. papers that cite it,
or that a human annotator judged as topically related):

```bash
python -m app.ml.evaluate --ground-truth data/eval_set.json --k 10
```

`backend/data/eval_set.json` contains a tiny illustrative example built on the synthetic
sample dataset — replace it with real labels (e.g. derived from citation graphs) for a
meaningful benchmark.

## Testing

```bash
cd backend
pytest -v
```

Tests use an isolated SQLite database and cover authentication, the FAISS wrapper, and
health checks. Extend `backend/tests/` alongside new endpoints.

## Configuration

All settings are environment variables, documented with defaults in `.env.example` and
loaded via `app/config.py` (`pydantic-settings`). Notable ones:

| Variable | Purpose |
|---|---|
| `SECRET_KEY` | JWT signing secret — **set a real random value outside of local dev** |
| `DATABASE_URL` | SQLAlchemy connection string (Postgres in Docker, SQLite works for local dev) |
| `EMBEDDING_MODEL_NAME` | Any `sentence-transformers` model name |
| `FAISS_INDEX_PATH` | Where the index (and its sidecar id-map JSON) is persisted |
| `CORS_ORIGINS` | Comma-separated list of allowed frontend origins |

## Future improvements

- Swap the flat FAISS index for an approximate index (`IndexIVFFlat` / HNSW) once the
  corpus grows past a few hundred thousand papers, trading a little recall for much
  faster search.
- Incremental/streaming indexing (currently a batch script) via a background worker
  queue, so newly published papers appear searchably within minutes.
- Personalized ranking that blends semantic similarity with a user's favorites/history.
- Author disambiguation (the current author filter is a simple substring match).
- Citation-graph-aware recommendations (papers that co-cite or are co-cited), blended
  with the embedding similarity signal.
- Alembic migrations in place of `Base.metadata.create_all` for schema evolution.
- Rate limiting and caching on the external Semantic Scholar/arXiv clients.
