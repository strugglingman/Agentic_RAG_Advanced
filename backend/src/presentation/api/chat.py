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

import json
from typing import Optional, Any
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from dishka.integrations.fastapi import FromDishka, inject
from pydantic import BaseModel

from src.application.commands.chat.send_message import (
    SendMessageCommand,
    SendMessageHandler,
    SendMessageResult,
)
from src.domain.value_objects.conversation_id import ConversationId
from src.presentation.dependencies.auth import AuthUser, get_current_user
from src.utils.stream_utils import stream_text_smart
from logging import getLogger

logger = getLogger(__name__)


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
        "filters": [{"exts": [...]}, {"tags": [...]}],
        "attachments": [{"type": "image", "filename": "...", "mime_type": "...", "data": "base64..."}]
    }
    """

    messages: list[ChatMessage]
    conversation_id: Optional[str] = None
    filters: Optional[list[dict[str, Any]]] = None
    attachments: Optional[list[dict[str, Any]]] = None


class ChatMessageResponse(BaseModel):
    """Response for non-streaming chat (if needed)."""

    message: str
    conversation_id: str
    contexts: list[dict[str, Any]]


# ==================== ROUTER ====================

router = APIRouter(prefix="/chat", tags=["chat"])


# ==================== ENDPOINTS ====================


@router.post("/agent")
@inject
async def chat_agent(
    request: ChatMessageRequest,
    handler: FromDishka[SendMessageHandler],
    current_user: AuthUser = Depends(get_current_user),
):
    """
    Agentic chat endpoint with streaming response.

    Returns a streaming response that:
    1. Streams the answer text in chunks (mimics LLM streaming)
    2. Appends context at the end as __CONTEXT__:{json}

    Maps from: routes/chat.py chat_agent()
    """
    try:
        # Extract latest user message
        msgs = request.messages
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
                "filters": request.filters,
            }
            if request.filters
            else None
        )

        # Build command from request
        command = SendMessageCommand(
            conversation_id=ConversationId(request.conversation_id or ""),
            user_email=current_user.email,
            dept_id=current_user.dept,
            content=latest_user_msg.content.strip(),
            attachments=request.attachments,
            filters=filters,
        )

        # Execute handler
        logger.debug(
            f"Before executing SendMessageCommand for user {current_user.email.value}"
        )
        result: SendMessageResult = await handler.execute(command)

        # Stream the response
        def generate():
            # Stream answer in chunks
            for chunk in stream_text_smart(result.answer):
                yield chunk

            # Append context at the end
            yield f"\n__CONTEXT__:{json.dumps(result.contexts)}"

        return StreamingResponse(
            generate(),
            media_type="text/plain",
            headers={
                "X-Conversation-Id": result.conversation_id.value,
            },
        )

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat error: {str(e)}",
        )


@router.post("", response_model=ChatMessageResponse)
@inject
async def chat_simple(
    request: ChatMessageRequest,
    handler: FromDishka[SendMessageHandler],
    current_user: AuthUser = Depends(get_current_user),
):
    """
    Simple chat endpoint (non-streaming).

    Returns full response as JSON. Use /chat/agent for streaming.

    Maps from: routes/chat.py chat()
    """
    try:
        # Extract latest user message
        msgs = request.messages
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
                "filters": request.filters,
            }
            if request.filters
            else None
        )

        command = SendMessageCommand(
            conversation_id=ConversationId(request.conversation_id or ""),
            user_email=current_user.email,
            dept_id=current_user.dept,
            content=latest_user_msg.content.strip(),
            attachments=request.attachments,
            filters=filters,
        )

        result: SendMessageResult = await handler.execute(command)

        return ChatMessageResponse(
            message=result.answer,
            conversation_id=result.conversation_id.value,
            contexts=result.contexts,
        )

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat error: {str(e)}",
        )
