"""
Chat API Router - FastAPI endpoints for chat/messaging.

Guidelines:
- Uses FastAPI router (not Flask Blueprint)
- Receives handlers via Dependency Injection (Dishka)
- Thin layer: only handles HTTP concerns (request/response)
- Delegates business logic to Application layer handlers
- Handles streaming responses for chat

Flow:
  HTTP Request → Router → Command → Handler → QuerySupervisor → LLM
                                 ↓
  HTTP Response (Streaming) ← Router ← SendMessageResult ←

Maps from: src/routes/chat.py
"""

import time
from typing import Optional, Any
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.responses import StreamingResponse
from dishka.integrations.fastapi import FromDishka, inject
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.application.commands.chat.send_message import (
    SendMessageCommand,
    SendMessageHandler,
    SendMessageResult,
)
from src.domain.value_objects.conversation_id import ConversationId
from src.presentation.dependencies.auth import AuthUser, get_current_user
from src.utils.stream_utils import stream_text_smart, sse_event
from src.observability.metrics import observe_request_latency
from logging import getLogger

logger = getLogger(__name__)

# Get limiter from app state (will be set in fastapi_app.py)
limiter = Limiter(key_func=get_remote_address)


# ==================== REQUEST/RESPONSE MODELS ====================


class ChatMessage(BaseModel):
    """Single message in chat history."""

    role: str
    content: str


class ChatMessageRequest(BaseModel):
    """
    Request body for sending a chat message.

    Matches frontend:
    {
        "messages": [{"role": "user", "content": "..."}],
        "conversation_id": "uuid" or null,
        "source_channel_id": "slack:C0123ABC" or null (for bot adapters),
        "filters": [{"exts": [...]}, {"tags": [...]}],
        "attachments": [{"type": "image", "filename": "...", "mime_type": "...", "data": "base64..."}]
    }
    """

    messages: list[ChatMessage]
    conversation_id: Optional[str] = None
    source_channel_id: Optional[str] = None  # For Slack/Teams: "slack:C0123ABC"
    filters: Optional[list[dict[str, Any]]] = None
    attachments: Optional[list[dict[str, Any]]] = None
    conversation_history: Optional[list[dict[str, str]]] = None  # Pre-fetched history from bot adapters


class ChatMessageResponse(BaseModel):
    """Response for non-streaming chat and bot adapters (Slack, etc.)."""

    message: str
    conversation_id: str
    contexts: list[dict[str, Any]]
    hitl: Optional[dict[str, Any]] = None


class HITLInterruptResponse(BaseModel):
    """Response when workflow is interrupted for human confirmation."""

    status: str  # "awaiting_confirmation"
    action: str  # "send_email", etc.
    thread_id: str  # Thread ID for resumption
    details: dict[str, Any]  # Action-specific details
    previous_steps: list[dict[str, Any]]  # Completed step results
    partial_answer: str  # Answer from completed steps
    conversation_id: str


class ResumeWorkflowRequest(BaseModel):
    """Request to resume an interrupted workflow."""

    thread_id: str
    confirmed: bool  # True to proceed, False to cancel
    conversation_id: Optional[str] = None


# ==================== ROUTER ====================

router = APIRouter(prefix="/chat", tags=["chat"])


# ==================== ENDPOINTS ====================


@router.post("/agent")
@limiter.limit("30/minute;1000/day")
@inject
async def chat_agent(
    request: Request,
    body: ChatMessageRequest,
    handler: FromDishka[SendMessageHandler],
    current_user: AuthUser = Depends(get_current_user),
):
    """
    Agentic chat endpoint with streaming response.

    Rate limited: 30/minute, 1000/day (matches Flask)

    Returns a streaming SSE response with event types:
    - "text": answer text chunks
    - "hitl": HITL interrupt data (if workflow paused)
    - "context": retrieved context array

    Maps from: routes/chat.py chat_agent()
    """
    start_time = time.time()
    status_code = 200
    try:
        # Extract latest user message
        msgs = body.messages
        latest_user_msg = None
        if msgs and msgs[-1].role == "user":
            latest_user_msg = msgs[-1]

        if not latest_user_msg:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No user message found",
            )
        if not latest_user_msg.content.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty user message",
            )

        # Build payload dict for filters (matches Flask request_data)
        filters = (
            {
                "filters": body.filters,
            }
            if body.filters
            else None
        )

        # Build command from body
        command = SendMessageCommand(
            conversation_id=ConversationId(body.conversation_id or ""),
            user_email=current_user.email,
            dept_id=current_user.dept,
            content=latest_user_msg.content.strip(),
            attachments=body.attachments,
            filters=filters,
            source_channel_id=body.source_channel_id,
            conversation_history=body.conversation_history,
        )

        # Execute handler
        result: SendMessageResult = await handler.execute(command)

        # Check for HITL interrupt
        if result.requires_confirmation:
            hitl = result.hitl_interrupt
            hitl_response = {
                "status": "awaiting_confirmation",
                "action": hitl.action,
                "thread_id": hitl.thread_id,
                "details": hitl.details,
                "previous_steps": hitl.previous_steps,
                "partial_answer": result.answer,
                "conversation_id": result.conversation_id.value,
            }

            # Stream partial answer + HITL + context as SSE events
            def generate_hitl():
                for chunk in stream_text_smart(result.answer):
                    yield sse_event("text", chunk)
                yield sse_event("hitl", hitl_response)
                yield sse_event("context", result.contexts)

            return StreamingResponse(
                generate_hitl(),
                media_type="text/event-stream",
                headers={
                    "X-Conversation-Id": result.conversation_id.value,
                    "X-HITL-Required": "true",
                    "Cache-Control": "no-cache",
                },
            )

        # Normal response - stream the answer as SSE events
        def generate():
            for chunk in stream_text_smart(result.answer):
                yield sse_event("text", chunk)
            yield sse_event("context", result.contexts)

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "X-Conversation-Id": result.conversation_id.value,
                "Cache-Control": "no-cache",
            },
        )

    except ValueError as e:
        status_code = 400
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except PermissionError as e:
        status_code = 403
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except Exception as e:
        status_code = 500
        logger.exception("Chat error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )
    finally:
        duration = time.time() - start_time
        observe_request_latency("POST", "/api/chat/agent", status_code, duration)


@router.post("", response_model=ChatMessageResponse)
@limiter.limit("30/minute;1000/day")
@inject
async def chat_simple(
    request: Request,
    body: ChatMessageRequest,
    handler: FromDishka[SendMessageHandler],
    current_user: AuthUser = Depends(get_current_user),
):
    """
    Simple chat endpoint (non-streaming).

    Rate limited: 30/minute, 1000/day (matches Flask)

    Returns full response as JSON. Use /chat/agent for streaming.

    Maps from: routes/chat.py chat()
    """
    try:
        # Extract latest user message
        msgs = body.messages
        latest_user_msg = None
        if msgs and msgs[-1].role == "user":
            latest_user_msg = msgs[-1]

        if not latest_user_msg:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No user message found",
            )
        if not latest_user_msg.content.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty user message",
            )

        # Build payload dict for filters
        filters = (
            {
                "filters": body.filters,
            }
            if body.filters
            else None
        )

        command = SendMessageCommand(
            conversation_id=ConversationId(body.conversation_id or ""),
            user_email=current_user.email,
            dept_id=current_user.dept,
            content=latest_user_msg.content.strip(),
            attachments=body.attachments,
            filters=filters,
            source_channel_id=body.source_channel_id,
            conversation_history=body.conversation_history,
        )

        result: SendMessageResult = await handler.execute(command)

        hitl_data = None
        if result.requires_confirmation:
            hitl = result.hitl_interrupt
            hitl_data = {
                "status": "awaiting_confirmation",
                "action": hitl.action,
                "thread_id": hitl.thread_id,
                "details": hitl.details,
                "previous_steps": hitl.previous_steps,
            }

        return ChatMessageResponse(
            message=result.answer,
            conversation_id=result.conversation_id.value,
            contexts=result.contexts,
            hitl=hitl_data,
        )

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except Exception as e:
        logger.exception("Chat error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.post("/resume")
@limiter.limit("30/minute;1000/day")
@inject
async def resume_workflow(
    request: Request,
    body: ResumeWorkflowRequest,
    handler: FromDishka[SendMessageHandler],
    current_user: AuthUser = Depends(get_current_user),
):
    """
    Resume an interrupted workflow after human confirmation.

    Called when user confirms or cancels a pending action (e.g., send_email).

    Request body:
    {
        "thread_id": "uuid",
        "confirmed": true/false,
        "conversation_id": "uuid" (optional)
    }

    Returns streaming response with final answer.
    """
    try:
        # Build context for resume (reuse handler's dependencies)
        agent_context = {
            "vector_db": handler.vector_db,
            "dept_id": str(current_user.dept),
            "user_id": str(current_user.email),
            "conversation_id": body.conversation_id or "",
            "conversation_history": [],  # Resume doesn't need history - state is in checkpoint
            "file_service": handler.file_service,
            "available_files": [],
            "attachment_file_ids": [],
        }

        # Resume the workflow
        query_result = await handler.query_supervisor.resume_workflow(
            thread_id=body.thread_id,
            context=agent_context,
            confirmed=body.confirmed,
        )

        # Check for another HITL interrupt (unlikely but possible)
        if query_result.hitl_interrupt:
            hitl = query_result.hitl_interrupt
            hitl_response = {
                "status": "awaiting_confirmation",
                "action": hitl.action,
                "thread_id": hitl.thread_id,
                "details": hitl.details,
                "previous_steps": hitl.previous_steps,
                "partial_answer": query_result.answer,
                "conversation_id": body.conversation_id or "",
            }

            def generate_hitl():
                for chunk in stream_text_smart(query_result.answer):
                    yield sse_event("text", chunk)
                yield sse_event("hitl", hitl_response)
                yield sse_event("context", query_result.contexts)

            return StreamingResponse(
                generate_hitl(),
                media_type="text/event-stream",
                headers={"X-HITL-Required": "true", "Cache-Control": "no-cache"},
            )

        # Normal completion
        def generate():
            for chunk in stream_text_smart(query_result.answer):
                yield sse_event("text", chunk)
            yield sse_event("context", query_result.contexts)

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception("[HITL] Resume workflow error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )
