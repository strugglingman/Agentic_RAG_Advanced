# Agentic RAG Advanced

This repository is an advanced agentic RAG system.

Code-verified stack:
- Frontend: Next.js App Router + NextAuth
- Backend: FastAPI + Dishka DI
- Data/Auth: Prisma + PostgreSQL
- Retrieval: Qdrant (dense + sparse hybrid)
- Orchestration: ReAct agent + LangGraph plan-execute
- Streaming: SSE (`text`, `hitl`, `context`)
- Integrations: Slack endpoints, Prometheus metrics

## Current Architecture

### Runtime Services
- `frontend` serves UI and proxy API routes under `frontend/app/api/*`
- `backend` serves FastAPI routes from `backend/src/fastapi_app.py`
- `postgres` stores users, conversations, messages, file registry, query logs
- `qdrant` stores vectorized chunks (dense + sparse)
- `redis` is optional for cache/job/state persistence

### Backend API Surface
Mounted in `backend/src/fastapi_app.py`:
- `/chat` (`/chat/agent`, `/chat`, `/chat/resume`)
- `/conversations`
- `/upload`
- `/files`
- `/org-structure`
- `/ingest` (`/ingest`, `/ingest/cancel`, `/ingest/active`)
- `/slack` (`/slack/events`, `/slack/interactive`)
- `/metrics`

### Frontend Request Pattern
Browser calls Next.js API routes, then proxy to FastAPI:
- `frontend/app/api/chat/route.ts` -> `POST {FASTAPI_URL}/chat/agent`
- `frontend/app/api/chat/resume/route.ts` -> `POST {FASTAPI_URL}/chat/resume`
- `frontend/app/api/conversations/*` -> backend conversation endpoints
- Upload/ingest/files/org also go through Next.js proxy routes

Proxy routes mint a short-lived service JWT from NextAuth session claims (`email`, `dept`) using `frontend/lib/service-auth.ts`.

### Chat + Streaming Contract
Backend chat streaming (`/chat/agent`, `/chat/resume`) returns SSE events:
- `event: text` -> incremental answer chunk
- `event: hitl` -> human-in-the-loop interrupt payload
- `event: context` -> retrieved context array

Frontend parsing is in:
- `frontend/lib/sse-parse.ts`
- `frontend/components/ChatUI.tsx`

`ChatUI` sends only the latest user message plus `conversation_id`, filters, and attachments. Full history is fetched server-side.

## Repository Layout
```text
backend/     FastAPI app, application/domain layers, retrieval, ingestion, LangGraph
frontend/    Next.js App Router UI + API proxy + NextAuth
docs/        Project docs (some files may lag implementation)
eval/        Evaluation assets/scripts
nginx/       Reverse-proxy config for Docker deployment
reports/     Reports and analysis artifacts
```

## Quick Start (Docker Compose)

1. Prepare env files:
```bash
cp .env.docker.example .env.docker
cp backend/.env.example backend/.env
```

2. Fill required secrets in those files at minimum:
- `NEXTAUTH_SECRET`
- `SERVICE_AUTH_SECRET`
- `SERVICE_AUTH_ISSUER`
- `SERVICE_AUTH_AUDIENCE`
- `OPENAI_API_KEY` (in `backend/.env`)
- DB credentials (`POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`)

3. Create external observability network once:
```bash
docker network create observability-network
```

4. Start:
```bash
docker compose --env-file .env.docker up --build
```

Default endpoints:
- App via nginx: `http://localhost`
- Frontend direct: `http://localhost:3000`
- Backend direct: `http://localhost:5001`
- Backend docs: `http://localhost:5001/docs`

## Quick Start (Local Development)

### 1) Start infra
Use Docker for infra only:
```bash
docker compose --env-file .env.docker up -d postgres redis qdrant
```

### 2) Backend
```bash
cd backend
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env
prisma generate
prisma migrate deploy
python run_fastapi.py
```

### 3) Frontend
```bash
cd frontend
npm ci
npx prisma generate
# create/update .env.local (see required keys below)
npm run dev
```

## Required Environment Variables (Minimum)

### Backend (`backend/.env`)
- `OPENAI_API_KEY`
- `DATABASE_URL` (PostgreSQL)
- `QDRANT_URL`
- `SERVICE_AUTH_SECRET`
- `SERVICE_AUTH_ISSUER`
- `SERVICE_AUTH_AUDIENCE`

### Frontend (`frontend/.env.local`)
- `FASTAPI_URL` (example: `http://127.0.0.1:5001`)
- `DATABASE_URL` (same Postgres, Prisma JS client)
- `NEXTAUTH_URL` (example: `http://localhost:3000`)
- `NEXTAUTH_SECRET`
- `SERVICE_AUTH_SECRET`
- `SERVICE_AUTH_ISSUER`
- `SERVICE_AUTH_AUDIENCE`

Important: service-auth values must match between frontend and backend.

## Retrieval and Agent Orchestration

### Active retrieval path
- `backend/src/services/vector_db_qdrant.py`
- `backend/src/services/retrieval_qdrant.py`

Behavior:
- Qdrant dense retrieval or hybrid dense+sparse retrieval
- Server-side RRF fusion for hybrid path
- Optional reranking via local CrossEncoder or Cohere
- Optional decomposition/contextual retrieval flags via config

### Agent routing
- `backend/src/services/query_supervisor.py`
  - Simple query -> `AgentService` (ReAct)
  - Complex query -> LangGraph workflow
- `backend/src/services/langgraph_builder.py`
  - Nodes: plan, retrieve, reflect, refine, generate, verify
  - Tools: web_search, download_file, create_documents, send_email, code_execution
  - HITL interruption before `tool_send_email` when checkpointing is enabled

## Data Model

Primary backend Prisma models (`backend/prisma/schema.prisma`):
- `User`
- `Conversation`
- `Message`
- `QueryLog`
- `FileRegistry`

Conversation/message persistence is server-side PostgreSQL. Frontend localStorage paths are mostly disabled and are not source of truth.

## Notes on Legacy/Transition Code
- Legacy retrieval modules and Chroma naming still exist in parts of the codebase.
- Active runtime path is FastAPI + Qdrant + SSE typed events as described above.

For implementation truth, prefer code under:
- `backend/src/fastapi_app.py`
- `backend/src/presentation/api/*`
- `backend/src/services/*`
- `frontend/app/api/*`
- `frontend/components/ChatUI.tsx`
