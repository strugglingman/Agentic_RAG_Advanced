# AGENTS.md

## Purpose

This repository is an advanced Agentic RAG system, not a simple demo chatbot.

Work from the real codebase, not from outdated assumptions in README or old notes.

## Evaluation Baseline (Load on Startup)

- Always load `docs/evaluations/EVALUATION_LOG.md` at the start of a new session.
- The log is maintained in reverse chronological order (latest evaluation first).
- Every new formal evaluation must include an ISO-8601 timestamp and be inserted at the top.

## Project Shape

Current implementation is centered around:

- **Frontend:** Next.js App Router
- **Backend:** FastAPI
- **Auth / app data:** Prisma + PostgreSQL
- **Retrieval:** Qdrant hybrid retrieval
- **Agent orchestration:** LangGraph
- **Streaming:** SSE
- **Extra surfaces:** Slack / channel integrations

## General Working Rules

- Inspect code first before proposing changes
- Prefer the smallest safe change first
- Do not hallucinate architecture or runtime behavior
- If something is unclear, verify from source files, logs, or commands
- Summarize findings briefly before large changes
- After changes, verify with the smallest relevant test or command

## Important Warning

Do **not** rely only on:

- README
- `.md` guides
- old config comments
- legacy naming

Some repository descriptions may lag behind the actual implementation.

## High-Value Areas to Inspect

When debugging or extending the system, inspect these areas first:

### Frontend
- `frontend/app`
- `frontend/components`
- `frontend/app/api/chat`
- protected routes, chat UI, SSE handling, upload flow, interrupt/resume flow

### Backend
- `backend/src/fastapi_app.py`
- `backend/src/presentation/api`
- `backend/src/services`
- `backend/src/config/settings.py`

### Agent / RAG flow
- query routing / supervisor
- LangGraph workflow
- retrieval pipeline
- Qdrant vector DB integration
- reranker / hybrid retrieval / decomposition / contextual retrieval switches

### Data / persistence
- `backend/prisma/schema.prisma`
- conversation/message/file registry models
- user / dept / channel related fields

## Preferred Debugging Order

For runtime issues, follow this order:

1. define the failing symptom exactly
2. identify frontend, backend, retrieval, or infra layer
3. inspect the request path end-to-end
4. check settings / env-dependent branches
5. inspect logs and streamed responses
6. make the smallest safe fix
7. verify behavior again

## Change Rules

- Do not make broad refactors unless explicitly requested
- Do not silently rename public API behavior
- Do not change schema, auth, or retrieval strategy casually
- Call out impact radius before changing shared infrastructure code

## Retrieval / RAG Changes

When changing retrieval logic, explicitly state:

- what changed
- why it changed
- expected effect on quality / latency / cost
- what should be tested afterward

### Retrieval Note
- [2026-03-26T15:07:28+01:00] Known issue to fix later: Qdrant hybrid retrieval without reranker is the main score-semantics risk. In this mode, server-side RRF returns only fused hybrid scores, so the workflow loses separate dense vs sparse evidence that existed in the old Chroma path. Example symptom: an exact-policy query like "employee vacation carry-over policy" may be wrongly rejected or wrongly accepted because evaluation/refinement sees only fused `hybrid` scores, not underlying lexical and semantic signals.

## Definition of Done

A task is not done until you provide:

- what changed
- where it changed
- why it changed
- how it was verified
- any remaining risks or follow-up items
