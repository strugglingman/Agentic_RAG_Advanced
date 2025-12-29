"""
GetChatHistory Query - Get conversation with messages.

This is for the frontend chat window to load conversation + messages
when clicking on a conversation title.

Maps from: src/routes/conversations.py get_conversation()
"""

from dataclasses import dataclass

from src.application.common.interfaces import Query, QueryHandler
from src.domain.entities.conversation import Conversation
from src.domain.entities.message import Message
from src.domain.exceptions import AccessDeniedError, EntityNotFoundError
from src.domain.ports.repositories import ConversationRepository, MessageRepository
from src.domain.value_objects.conversation_id import ConversationId
from src.domain.value_objects.user_email import UserEmail
from src.config.settings import Config


@dataclass
class GetChatHistoryResult:
    """Result containing conversation metadata and messages."""

    conversation: Conversation
    messages: list[Message]


@dataclass(frozen=True)
class GetChatHistoryQuery(Query[GetChatHistoryResult]):
    """
    Query to get chat history for a conversation.

    Used by frontend to load conversation + messages when opening a conversation.
    """

    conversation_id: ConversationId
    user_email: UserEmail
    limit: int = Config.CONVERSATION_MESSAGE_LIMIT


class GetChatHistoryHandler(QueryHandler[GetChatHistoryResult]):
    """
    Handler for GetChatHistoryQuery.

    Returns conversation metadata + messages for frontend display.
    """

    def __init__(
        self,
        conv_repo: ConversationRepository,
        msg_repo: MessageRepository,
    ):
        self._conv_repo = conv_repo
        self._msg_repo = msg_repo

    async def execute(self, query: GetChatHistoryQuery) -> GetChatHistoryResult:
        """
        Get conversation with chat history.

        Steps:
        1. Verify conversation exists
        2. Verify user owns conversation
        3. Load messages from DB

        Returns:
            GetChatHistoryResult with conversation and messages

        Raises:
            EntityNotFoundError: If conversation doesn't exist
            AccessDeniedError: If user doesn't own the conversation
        """
        # 1. Get conversation
        conversation = await self._conv_repo.get_by_id(query.conversation_id)
        if not conversation:
            raise EntityNotFoundError(
                f"Conversation {query.conversation_id.value} not found"
            )

        # 2. Verify ownership
        if conversation.user_email.value != query.user_email.value:
            raise AccessDeniedError("You don't have access to this conversation")

        # 3. Load messages from DB (latest limit messages, from oldest to latest)
        messages = await self._msg_repo.get_by_conversation(
            query.conversation_id, limit=query.limit
        )

        return GetChatHistoryResult(
            conversation=conversation,
            messages=messages,
        )
