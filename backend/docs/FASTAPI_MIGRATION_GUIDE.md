# FastAPI Migration Guide (Status + Current Conventions)

Updated: 2026-03-13

This migration is functionally complete.  
This file now documents current backend conventions and remaining cleanup items.

## 1. Migration Status

Completed:
- FastAPI app factory and router mounting are active (`backend/src/fastapi_app.py`)
- Flask routes have FastAPI equivalents under `backend/src/presentation/api/*`
- DI is handled by Dishka container (`backend/src/setup/ioc/container.py`)
- Request auth uses backend JWT dependency (`presentation/dependencies/auth.py`)
- SSE endpoints are active for chat and ingest flows
- CQRS command/query handlers are active in application layer

Not in scope anymore:
- Maintaining Flask as primary runtime
- Planning-level migration tasks that are already implemented

## 2. Current Backend Layout

### Core entrypoints
- Runtime app: `backend/src/fastapi_app.py`
- Local run script: `backend/run_fastapi.py`

### Active routers
- `chat.py`
- `conversations.py`
- `files.py`
- `ingest.py`
- `org.py`
- `metrics.py`
- Slack adapter routes (`src/adapters/slack/slack_routes.py`)

### Layered structure in use
- `src/domain`: entities/value objects/ports
- `src/application`: commands/queries/handlers/services
- `src/infrastructure`: persistence/cache/storage/jobs
- `src/presentation`: FastAPI routes + auth dependency

## 3. Endpoint Behavior Notes

### Chat
- `POST /chat/agent`: SSE stream with `text`, `hitl`, `context`
- `POST /chat`: non-streaming JSON response
- `POST /chat/resume`: resume HITL workflow (SSE)

### Ingestion
- `POST /ingest`: SSE progress stream (`progress`, `complete`, `cancelled`, `error`)
- `POST /ingest/cancel`
- `GET /ingest/active`

### Conversations and files
- Conversation CRUD and history are API-backed, persisted in PostgreSQL
- File upload/list/download/delete are API-backed with access checks

## 4. Auth Convention

- Backend expects service JWT with `email` and `dept` claims.
- JWT validation checks:
  - signature (`SERVICE_AUTH_SECRET`)
  - issuer (`SERVICE_AUTH_ISSUER`)
  - audience (`SERVICE_AUTH_AUDIENCE`)
  - required registered claims (`exp`, `iat`, `aud`, `iss`)

## 5. Retrieval and Agent Convention

Active retrieval path:
- `src/services/vector_db_qdrant.py`
- `src/services/retrieval_qdrant.py`

Agent orchestration:
- `QuerySupervisor` routes simple vs complex
- simple -> `AgentService`
- complex -> LangGraph (`langgraph_builder.py`, `langgraph_nodes.py`)

HITL:
- interrupt before `tool_send_email` when checkpointing is enabled
- resume handled via `thread_id` and `/chat/resume`

## 6. What To Update Going Forward

When backend behavior changes, update these docs first:
- `README.md` (root runtime overview)
- `AGENTIC_RAG_IMPLEMENTATION_GUIDE.md` (cross-system implementation map)
- `backend/docs/FASTAPI_MIGRATION_GUIDE.md` (this file)

## 7. Remaining Cleanup (Non-blocking)

- Remove historical Flask/Chroma naming in comments, env examples, and field names where safe.
- Keep legacy modules only if still needed for transition/testing.
- Archive planning-era migration/task documents to avoid architecture confusion.
