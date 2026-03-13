# Frontend Conversation Guide (Current State)

Updated: 2026-03-13

This guide describes the current Next.js App Router implementation.

## 1. Current Architecture

Frontend chat does not call backend FastAPI directly from the browser.  
Browser calls Next.js API routes, then Next.js proxies to FastAPI with a service JWT.

Core files:
- Chat UI: `frontend/components/ChatUI.tsx`
- Chat context/store: `frontend/components/chat-context.tsx`
- SSE parsing: `frontend/lib/sse-parse.ts`
- Auth config: `frontend/lib/auth.ts`
- Service token minting: `frontend/lib/service-auth.ts`
- Route protection: `frontend/middleware.ts`

## 2. API Proxy Map

### Chat and resume
- `POST /api/chat` -> backend `POST /chat/agent`
- `POST /api/chat/resume` -> backend `POST /chat/resume`

### Conversations
- `GET /api/conversations` -> backend `GET /conversations`
- `GET /api/conversations/[id]` -> backend `GET /conversations/{id}`
- `PATCH /api/conversations/[id]` -> backend `PATCH /conversations/{id}`
- `DELETE /api/conversations/[id]` -> backend `DELETE /conversations/{id}`

### Files and ingestion
- `POST /api/upload` -> backend `POST /upload`
- `GET /api/files` -> backend `GET /files`
- `GET /api/files/[fileId]` -> backend `GET /files/{file_id}`
- `POST /api/files/delete` -> backend `POST /files/delete`
- `POST /api/ingest` -> backend `POST /ingest` (SSE)
- `POST /api/ingest/cancel` -> backend `POST /ingest/cancel`
- `GET /api/ingest/active` -> backend `GET /ingest/active`

## 3. Auth and Token Flow

1. User signs in via NextAuth credentials provider.
2. Session includes `email`, `dept`.
3. Each protected API route calls `mintServiceToken(...)`.
4. Next.js sends backend request with:
   - `Authorization: Bearer <service_jwt>`
   - `X-Correlation-ID`
5. Backend validates JWT claims and signature.

Important:
- `SERVICE_AUTH_SECRET`, `SERVICE_AUTH_ISSUER`, `SERVICE_AUTH_AUDIENCE` must match frontend and backend.

## 4. Conversation State and Persistence

Current state model:
- Source of truth is backend PostgreSQL conversations/messages.
- `chat-context.tsx` fetches conversations from `/api/conversations` on mount.
- LocalStorage persistence code exists but is mostly commented out.

Chat send behavior in `ChatUI.tsx`:
- Sends only latest user message in `messages`.
- Includes `conversation_id` if selected.
- Includes optional filters and attachments.
- For new chat (`selectedConversation == null`), refreshes conversation list after first send.

## 5. SSE Contract

Frontend expects typed SSE events:
- `event: text` -> append assistant text
- `event: hitl` -> parse HITL payload
- `event: context` -> parse context array

Parser:
- `consumeSSEStream(...)` in `frontend/lib/sse-parse.ts`
- Buffers chunks and splits by SSE event boundary (`\n\n`)

Resume flow:
- If HITL interrupt exists, UI sends `thread_id` + `confirmed` to `/api/chat/resume`.

## 6. Protected Routes

Middleware protects:
- pages: `/chat/*`, `/upload/*`
- API: `/api/chat/*`, `/api/upload/*`, `/api/ingest/*`, `/api/files/*`, `/api/conversations/*`

Not protected by middleware:
- `/api/org-structure` (used by signup flow)

## 7. Current Folder Paths (Important)

Use current paths:
- `frontend/app/...`
- `frontend/components/...`
- `frontend/lib/...`

Do not use old `frontend/src/...` paths in new docs or implementation notes.

## 8. Debug Checklist

1. Verify NextAuth session exists.
2. Verify service JWT claims include `email` and `dept`.
3. Check `FASTAPI_URL` value in frontend env.
4. Inspect Network panel for SSE event stream shape.
5. Confirm backend returns `text`, `hitl`, `context` events.
