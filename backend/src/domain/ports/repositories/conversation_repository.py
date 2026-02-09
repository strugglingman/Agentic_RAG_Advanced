"""
Conversation Repository Port - Interface for conversation persistence.
Implementation: src/infrastructure/persistence/prisma_conversation_repository.py
"""

from abc import ABC, abstractmethod
from typing import Optional
from src.domain.entities.conversation import Conversation
from src.domain.value_objects.conversation_id import ConversationId
from src.domain.value_objects.user_email import UserEmail


class ConversationRepository(ABC):
    @abstractmethod
    async def get_by_id(
        self, conversation_id: ConversationId
    ) -> Optional[Conversation]: ...

    @abstractmethod
    async def get_by_user(
        self, user_email: UserEmail, limit: int
    ) -> list[Conversation]: ...

    @abstractmethod
    async def get_by_source(
        self, user_email: UserEmail, source_channel_id: str
    ) -> Optional[Conversation]:
        """
        Get conversation by source channel (Slack/Teams) and user.

        Args:
            user_email: User's email
            source_channel_id: Source identifier (e.g., "slack:C0123ABC")

        Returns:
            Conversation if found, None otherwise
        """
        ...

    @abstractmethod
    async def find_conversation(
        self,
        user_email: UserEmail,
        conversation_id: Optional[ConversationId] = None,
        source_channel_id: Optional[str] = None,
    ) -> Optional[Conversation]:
        """
        Unified conversation lookup with priority:
        1. source_channel_id (if provided) - for Slack/Teams
        2. conversation_id (if provided) - for web UI

        Args:
            user_email: User's email (for source lookup and ownership verification)
            conversation_id: Optional conversation UUID
            source_channel_id: Optional source (e.g., "slack:C0123ABC")

        Returns:
            Conversation if found, None otherwise
        """
        ...

    @abstractmethod
    async def save(self, conversation: Conversation) -> None: ...

    @abstractmethod
    async def delete(self, conversation_id: ConversationId) -> bool: ...
