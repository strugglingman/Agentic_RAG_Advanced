"""
Message Repository Port - Interface for message persistence.
Implementation: src/infrastructure/persistence/prisma_message_repository.py
"""

from abc import ABC, abstractmethod
from typing import Optional

from src.domain.entities.message import Message
from src.domain.value_objects.message_id import MessageId
from src.domain.value_objects.conversation_id import ConversationId


class MessageRepository(ABC):
    @abstractmethod
    async def get_by_id(self, message_id: MessageId) -> Optional[Message]: ...

    @abstractmethod
    async def get_by_conversation(
        self, conversation_id: ConversationId, limit: int = 20
    ) -> list[Message]: ...

    @abstractmethod
    async def save(self, message: Message) -> None: ...

    @abstractmethod
    async def delete(self, message_id: MessageId) -> bool: ...
