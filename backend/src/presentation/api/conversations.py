"""
Conversations API Router - FastAPI endpoints for conversation management.

Guidelines:
- Uses FastAPI router (not Flask Blueprint)
- Receives handlers via Dependency Injection (Dishka)
- Thin layer: only handles HTTP concerns (request/response)
- Delegates business logic to Application layer handlers
- Uses DTOs for request/response schemas

Steps to implement create_conversation endpoint:
1. Define request/response models (Pydantic)
2. Create router with FastAPI
3. Inject handler via Depends (Dishka)
4. Extract data from request
5. Create Command and call handler.execute()
6. Return response DTO

Flow:
  HTTP Request → Router → Command → Handler → Repository → Database
                                 ↓
  HTTP Response ← Router ← Result ←
"""

from logging import getLogger
from fastapi import APIRouter, Depends, HTTPException, status
from dishka.integrations.fastapi import FromDishka, inject
from pydantic import BaseModel
from src.application.commands.conversations import (
    CreateConversationCommand,
    CreateConversationHandler,
    DeleteConversationCommand,
    DeleteConversationHandler,
    UpdateTitleCommand,
    UpdateTitleHandler,
)
from src.application.queries.conversations import (
    ListConversationsQuery,
    ListConversationsHandler,
)
from src.application.queries.chat import (
    GetChatHistoryQuery,
    GetChatHistoryHandler,
)
from src.application.dto.chat import MessageDTO
from src.domain.exceptions import EntityNotFoundError, AccessDeniedError
from src.domain.value_objects.conversation_id import ConversationId
from src.presentation.dependencies.auth import AuthUser, get_current_user
from src.config.settings import Config

logger = getLogger(__name__)


# ==================== REQUEST/RESPONSE MODELS ====================


class CreateConversationRequest(BaseModel):
    """Request body for creating a conversation."""

    title: str = "New Conversation"


class CreateConversationResponse(BaseModel):
    """Response for created conversation."""

    id: str
    title: str


class DeleteConversationResponse(BaseModel):
    """Response for deleted conversation."""

    success: bool


class UpdateConversationTitleRequest(BaseModel):
    title: str


class UpdateConversationTitleResponse(BaseModel):
    """
    Response for update conversation title.

    Matches Flask format:
    {
        "id": "uuid",
        "title": "Updated title",
        "updated_at": "2025-01-27T12:45:00Z"
    }
    """

    id: str
    title: str
    updated_at: str


class ConversationListItemResponse(BaseModel):
    """
    Single conversation item in list.

    Matches Flask format:
    {
        "id": "uuid",
        "title": "Conversation title",
        "updated_at": "2025-01-27T12:00:00Z",
        "preview": "Last message preview..."
    }
    """

    id: str
    title: str
    updated_at: str
    preview: str = ""


class ListConversationsResponse(BaseModel):
    conversations: list[ConversationListItemResponse]


class GetConversationResponse(BaseModel):
    """
    Response for get conversation (with messages).

    Matches Flask format:
    {
        "id": "uuid",
        "title": "Conversation title",
        "created_at": "2025-01-27T12:00:00Z",
        "updated_at": "2025-01-27T12:30:00Z",
        "messages": [...]
    }
    """

    id: str
    title: str
    created_at: str
    updated_at: str
    messages: list[MessageDTO]


# ==================== ROUTER ====================

router = APIRouter(prefix="/conversations", tags=["conversations"])


# ==================== ENDPOINTS ====================


@router.post(
    "",
    response_model=CreateConversationResponse,
    status_code=status.HTTP_201_CREATED,
)
@inject
async def create_conversation(
    request: CreateConversationRequest,
    handler: FromDishka[CreateConversationHandler],
    current_user: AuthUser = Depends(get_current_user),
):
    """Create a new conversation."""
    command = CreateConversationCommand(
        user_email=current_user.email,
        title=request.title,
    )
    conversation_id = await handler.execute(command)

    return CreateConversationResponse(
        id=conversation_id.value,
        title=request.title,
    )


@router.delete(
    "/{conversation_id}",
    response_model=DeleteConversationResponse,
    status_code=status.HTTP_200_OK,
)
@inject
async def delete_conversation(
    conversation_id: str,
    handler: FromDishka[DeleteConversationHandler],
    current_user: AuthUser = Depends(get_current_user),
):
    """Delete conversation by ID."""
    try:
        command = DeleteConversationCommand(
            conversation_id=ConversationId(conversation_id),
            user_email=current_user.email,
        )

        success = await handler.execute(command)
        return DeleteConversationResponse(success=success)
    except EntityNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
    except AccessDeniedError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e)) from e


@router.patch(
    "/{conversation_id}",
    response_model=UpdateConversationTitleResponse,
    status_code=status.HTTP_200_OK,
)
@inject
async def update_conversation(
    conversation_id: str,
    request: UpdateConversationTitleRequest,
    handler: FromDishka[UpdateTitleHandler],
    current_user: AuthUser = Depends(get_current_user),
):
    """
    Update conversation (currently only title).

    Matches Flask: PATCH /conversations/{id}
    Request: {"title": "New title"}
    Response: {"id": "uuid", "title": "New title", "updated_at": "2025-01-27T12:45:00Z"}
    """
    try:
        command = UpdateTitleCommand(
            conversation_id=ConversationId(conversation_id),
            user_email=current_user.email,
            new_title=request.title,
        )
        updated_conversation = await handler.execute(command)

        return UpdateConversationTitleResponse(
            id=updated_conversation.id.value,
            title=updated_conversation.title,
            updated_at=updated_conversation.updated_at.isoformat(),
        )
    except EntityNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
    except AccessDeniedError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e)) from e


@router.get(
    "",
    response_model=ListConversationsResponse,
    status_code=status.HTTP_200_OK,
)
@inject
async def list_conversations(
    handler: FromDishka[ListConversationsHandler],
    current_user: AuthUser = Depends(get_current_user),
):
    """
    List all conversations for current user.

    Matches Flask format:
    {
        "conversations": [
            {"id": "uuid", "title": "...", "updated_at": "ISO", "preview": "..."},
            ...
        ]
    }
    """
    query = ListConversationsQuery(
        user_email=current_user.email, limit=Config.CONVERSATION_USER_LIMIT
    )

    conversations = await handler.execute(query)
    conversation_items = [
        ConversationListItemResponse(
            id=conv.id.value,
            title=conv.title,
            updated_at=conv.updated_at.isoformat() if conv.updated_at else "",
            preview=conv.last_message or "",
        )
        for conv in conversations
    ]

    return ListConversationsResponse(conversations=conversation_items)


@router.get(
    "/{conversation_id}",
    response_model=GetConversationResponse,
    status_code=status.HTTP_200_OK,
)
@inject
async def get_conversation(
    conversation_id: str,
    handler: FromDishka[GetChatHistoryHandler],
    current_user: AuthUser = Depends(get_current_user),
    limit: int = Config.CONVERSATION_MESSAGE_LIMIT,
):
    """
    Get conversation with messages.

    Matches Flask: GET /conversations/{id}
    Response:
    {
        "id": "uuid",
        "title": "Conversation title",
        "created_at": "2025-01-27T12:00:00Z",
        "updated_at": "2025-01-27T12:30:00Z",
        "messages": [
            {"id": "uuid", "role": "user", "content": "...", "created_at": "..."},
            ...
        ]
    }
    """
    try:
        query = GetChatHistoryQuery(
            conversation_id=ConversationId(conversation_id),
            user_email=current_user.email,
            limit=limit,
        )

        result = await handler.execute(query)

        message_dtos = [
            MessageDTO(
                id=msg.id.value,
                conversation_id=msg.conversation_id.value,
                role=msg.role,
                content=msg.content,
                created_at=msg.created_at,
                tokens_used=msg.tokens_used,
                latency_ms=msg.latency_ms,
            )
            for msg in result.messages
        ]

        return GetConversationResponse(
            id=result.conversation.id.value,
            title=result.conversation.title,
            created_at=result.conversation.created_at.isoformat(),
            updated_at=result.conversation.updated_at.isoformat(),
            messages=message_dtos,
        )

    except EntityNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
    except AccessDeniedError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e)) from e
