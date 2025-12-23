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
    async def save(self, conversation: Conversation) -> None: ...

    @abstractmethod
    async def delete(self, conversation_id: ConversationId) -> bool: ...
