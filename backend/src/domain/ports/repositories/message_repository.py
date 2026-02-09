"""
Message Repository Port - Interface for message persistence.
Implementation: src/infrastructure/persistence/prisma_message_repository.py
"""

from abc import ABC, abstractmethod
from typing import Optional, Any, Dict

from src.domain.entities.message import Message
from src.domain.value_objects.message_id import MessageId
from src.domain.value_objects.conversation_id import ConversationId


class MessageRepository(ABC):
    @abstractmethod
    async def get_by_id(self, message_id: MessageId) -> Optional[Message]: ...

    @abstractmethod
    async def get_by_conversation(
        self, conversation_id: ConversationId, limit: int = 200
    ) -> list[Message]: ...

    @abstractmethod
    async def get_smart_history(
        self,
        query: str,
        conversation_id: ConversationId,
        context: Optional[Dict[str, Any]] = None
    ) -> list[Message]:
        """
        Smart conversation history retrieval based on query intent.

        Analyzes the query and retrieves appropriate conversation history using:
        - LLM-based intent detection (initial implementation)
        - Later: RAG with vector search for semantic queries
        - Later: Timestamp filtering for temporal queries
        - Later: Quantitative queries (all, half, first N)

        Args:
            query: User's query to analyze for intent
            conversation_id: Conversation to retrieve history from
            context: Optional context (openai_client, vector_db, etc.)

        Returns:
            List of Message entities in chronological order (oldest first)
        """
        ...

    @abstractmethod
    async def save(self, message: Message) -> None: ...

    @abstractmethod
    async def delete(self, message_id: MessageId) -> bool: ...
