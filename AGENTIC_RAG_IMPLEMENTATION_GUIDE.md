# Agentic RAG Implementation Guide (Current State)

Updated: 2026-03-13

This document reflects the current codebase implementation. It is not a future roadmap.

## 1. System Snapshot

- Frontend: Next.js App Router (`frontend/app`, `frontend/components`)
- Backend: FastAPI (`backend/src/fastapi_app.py`)
- Auth/session: NextAuth + Prisma (frontend) + service JWT to backend
- Data persistence: PostgreSQL via Prisma (backend + frontend)
- Retrieval: Qdrant dense+sparse hybrid (`vector_db_qdrant.py`, `retrieval_qdrant.py`)
- Agent orchestration: QuerySupervisor + ReAct + LangGraph
- Streaming: SSE typed events (`text`, `hitl`, `context`)
- Integrations: Slack routes, Prometheus metrics endpoint

## 2. Request Path (Web Chat)

1. Browser sends request to `POST /api/chat` (Next.js route).
2. Next.js route mints short-lived service JWT from NextAuth session (`email`, `dept`).
3. Next.js proxies to backend `POST /chat/agent`.
4. Backend validates service JWT (`backend/src/presentation/dependencies/auth.py`).
5. `SendMessageHandler` persists user message, resolves conversation, builds context.
6. `QuerySupervisor` routes execution:
   - simple -> `AgentService` (ReAct)
   - complex -> LangGraph workflow
7. Backend streams SSE events back to frontend.
8. `SendMessageHandler` persists assistant message.

## 3. Backend Runtime Structure

Main entrypoints:
- App factory: `backend/src/fastapi_app.py`
- Dev run: `backend/run_fastapi.py`
- API routers: `backend/src/presentation/api/*`

Mounted routers:
- `/chat`
- `/conversations`
- `/upload`
- `/files`
- `/org-structure`
- `/ingest`
- `/slack`
- `/metrics`

Infrastructure patterns in use:
- Dishka DI container (`backend/src/setup/ioc/container.py`)
- CQRS-style command/query handlers (`backend/src/application`)
- Prisma repositories (`backend/src/infrastructure/persistence`)
- Optional Redis-backed cache/state (`backend/src/infrastructure/cache`, `agent_state.py`)

## 4. Agent Execution Model

### 4.1 QuerySupervisor
File: `backend/src/services/query_supervisor.py`

- Uses LLM JSON classification (`simple` / `complex`)
- Routes to:
  - `AgentService` for fast ReAct loop
  - LangGraph for multi-step workflows
- Supports HITL resume flow with `thread_id`
- Supports checkpointer via `AsyncPostgresSaver` when enabled

### 4.2 LangGraph
Key files:
- `langgraph_builder.py`
- `langgraph_nodes.py`
- `langgraph_routing.py`
- `langgraph_state.py`

Nodes implemented:
- `plan`, `retrieve`, `reflect`, `refine`, `generate`, `verify`
- tool nodes: `web_search`, `download_file`, `create_documents`, `send_email`, `code_execution`
- `direct_answer`, `error`

HITL behavior:
- Graph can interrupt before `tool_send_email` when checkpointing is enabled.
- Resume path: backend `POST /chat/resume`.

## 5. Retrieval Stack

Active path:
- `backend/src/services/vector_db_qdrant.py`
- `backend/src/services/retrieval_qdrant.py`
- `backend/src/services/ingestion.py`

Capabilities:
- Dense retrieval (OpenAI embeddings or local sentence-transformers)
- Hybrid retrieval (dense + sparse BM25 vectors in Qdrant)
- Server-side RRF fusion for hybrid mode
- Optional reranking (local CrossEncoder or Cohere)
- Optional decomposition/contextual retrieval via settings flags

Legacy code:
- `backend/src/services/retrieval.py` still exists but is not the active integration path.
- Some field names still contain historical `chroma` naming.

## 6. Persistence Model

Backend schema: `backend/prisma/schema.prisma`

Primary models:
- `User`
- `Conversation`
- `Message`
- `QueryLog`
- `FileRegistry`

Notes:
- Conversations/messages are persisted in PostgreSQL.
- `source_channel_id` supports bot channels (for example Slack channel mapping).
- Frontend localStorage is mostly disabled; backend DB is source of truth.

## 7. Frontend Integration Facts

Relevant files:
- Chat UI: `frontend/components/ChatUI.tsx`
- SSE parser: `frontend/lib/sse-parse.ts`
- Chat proxy: `frontend/app/api/chat/route.ts`
- Resume proxy: `frontend/app/api/chat/resume/route.ts`
- Conversation proxies: `frontend/app/api/conversations/*`

Behavior:
- Chat request sends latest user message with `conversation_id`, filters, attachments.
- UI parses `text`, `hitl`, `context` SSE events.
- HITL confirm/cancel calls `/api/chat/resume`.

## 8. Operational Switches

Main settings file: `backend/src/config/settings.py`

Major switch groups include:
- Agent and checkpoint behavior
- Query routing and semantic router
- Retrieval/reranker/decomposition/contextual retrieval
- Web search and browser automation
- Code execution and Slack integration
- Upload limits, auth, and rate limits

## 9. What This Guide Is For

Use this file as:
- implementation map for current architecture
- source-of-truth index for core execution paths

Do not use this file as:
- a historical migration diary
- a speculative future plan
