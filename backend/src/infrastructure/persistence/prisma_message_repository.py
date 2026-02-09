"""
Prisma Message Repository Implementation.

Guidelines:
- Implements MessageRepository port from domain layer
- Uses Prisma client for database operations
- Maps between Prisma models and domain entities
- All methods are async

Prisma Message Model (from schema.prisma):
    model Message {
        id              String       @id @default(uuid())
        conversation_id String
        role            String
        content         String
        tokens_used     Int?
        latency_ms      Int?
        created_at      DateTime     @default(now())
        conversation    Conversation @relation(...)
    }

Domain Message Entity:
    @dataclass
    class Message:
        id: MessageId
        conversation_id: ConversationId
        role: str
        content: str
        created_at: datetime
        tokens_used: Optional[int] = None
        latency_ms: Optional[int] = None

Mapping:
- Prisma: id (str) ←→ Domain: id (MessageId)
- Prisma: conversation_id (str) ←→ Domain: conversation_id (ConversationId)
- Other fields map directly

Steps to implement each method:

1. _to_entity(record: PrismaMessage) -> Message
   - Convert Prisma record to domain entity
   - Wrap id with MessageId(record.id)
   - Wrap conversation_id with ConversationId(record.conversation_id)
   - Pass other fields directly

2. get_by_id(message_id: MessageId) -> Optional[Message]
   - Query: self._prisma.message.find_unique(where={"id": message_id.value})
   - If found, return self._to_entity(record)
   - If not found, return None

3. get_by_conversation(conversation_id: ConversationId, limit: int) -> list[Message]
   - Query: self._prisma.message.find_many(
       where={"conversation_id": conversation_id.value},
       order={"created_at": "asc"},  # Oldest first for chat history
       take=limit
     )
   - Return [self._to_entity(r) for r in records]
   - NOTE: "asc" order = oldest first (chronological for display)

4. save(message: Message) -> None
   - Use upsert to handle both create and update
   - Extract values: message.id.value, message.conversation_id.value, etc.
   - self._prisma.message.upsert(
       where={"id": message.id.value},
       data={
           "create": {...all fields...},
           "update": {...fields that can change...}
       }
     )

5. delete(message_id: MessageId) -> bool
   - Try: self._prisma.message.delete(where={"id": message_id.value})
   - Return True if successful
   - Catch exception, return False
"""

import logging
from typing import Optional, Any, Dict
from prisma import Prisma
from prisma.models import Message as PrismaMessage
from src.domain.entities.message import Message
from src.domain.ports.repositories.message_repository import MessageRepository
from src.domain.value_objects.message_id import MessageId
from src.domain.value_objects.conversation_id import ConversationId
from src.utils.history_utils import determine_message_count

logger = logging.getLogger(__name__)


class PrismaMessageRepository(MessageRepository):
    """
    Prisma implementation of MessageRepository.

    Handles persistence of Message entities to PostgreSQL via Prisma.
    """

    _prisma: Prisma

    def __init__(self, prisma: Prisma):
        """
        Initialize repository with Prisma client.

        Args:
            prisma: Connected Prisma client (injected by DI container)
        """
        self._prisma = prisma

    def _to_entity(self, record: PrismaMessage) -> Message:
        """
        Map Prisma record to domain entity.

        Args:
            record: Prisma Message model instance

        Returns:
            Domain Message entity with value objects
        """
        return Message(
            id=MessageId(record.id),
            conversation_id=ConversationId(record.conversation_id),
            role=record.role,
            content=record.content,
            created_at=record.created_at,
            tokens_used=record.tokens_used,
            latency_ms=record.latency_ms,
        )

    async def get_by_id(self, message_id: MessageId) -> Optional[Message]:
        """
        Get message by ID.

        Args:
            message_id: MessageId value object

        Returns:
            Message entity if found, None otherwise
        """
        record = await self._prisma.message.find_unique(where={"id": message_id.value})
        return self._to_entity(record) if record else None

    async def get_by_conversation(
        self, conversation_id: ConversationId, limit: int = 200
    ) -> list[Message]:
        """
        Get messages for a conversation, ordered chronologically (oldest first).

        Args:
            conversation_id: ConversationId value object
            limit: Maximum number of messages to return

        Returns:
            List of Message entities in chronological order (oldest first)

        Note:
            Uses "asc" order so messages display in correct chat order.
            For "most recent N messages", query with "desc" then reverse,
            or use offset-based pagination.
        """
        records = await self._prisma.message.find_many(
            where={"conversation_id": conversation_id.value},
            order={"created_at": "desc"},
            take=limit,
        )
        records.reverse()  # Now oldest first
        return [self._to_entity(record) for record in records]

    async def save(self, message: Message) -> None:
        """
        Save (create or update) a message.

        Args:
            message: Message entity to persist

        Note:
            Uses upsert to handle both new messages and updates.
            In practice, messages are rarely updated after creation.
        """
        await self._prisma.message.upsert(
            where={"id": message.id.value},
            data={
                "create": {
                    "id": message.id.value,
                    "conversation_id": message.conversation_id.value,
                    "role": message.role,
                    "content": message.content,
                    "created_at": message.created_at,
                    "tokens_used": message.tokens_used,
                    "latency_ms": message.latency_ms,
                },
                "update": {
                    "content": message.content,
                    "tokens_used": message.tokens_used,
                    "latency_ms": message.latency_ms,
                },
            },
        )

    async def get_smart_history(
        self,
        query: str,
        conversation_id: ConversationId,
        context: Optional[Dict[str, Any]] = None,
    ) -> list[Message]:
        """
        Smart conversation history retrieval based on query intent.

        Phase 1 Implementation (LLM-based):
        - Uses LLM to determine how many messages are needed
        - Simple and effective for most queries
        - No vector DB infrastructure required yet

        Future phases:
        - Phase 2: Add RAG with vector search for semantic queries
        - Phase 3: Add timestamp filtering for temporal queries
        - Phase 4: Add quantitative query support (all, half, first N)

        Args:
            query: User's query to analyze for intent
            conversation_id: Conversation to retrieve history from
            context: Optional context with openai_client

        Returns:
            List of Message entities in chronological order (oldest first)
        """
        # Extract OpenAI client from context (optional)
        openai_client = context.get("openai_client") if context else None

        # Determine how many messages to fetch (shared utility used by both Web UI and Slack)
        messages_needed = determine_message_count(query, openai_client)

        # Fetch messages using standard method
        logger.info(
            f"[get_smart_history] Fetching {messages_needed} messages "
            f"for query: {query[:50]}..."
        )
        return await self.get_by_conversation(conversation_id, limit=messages_needed)

    async def delete(self, message_id: MessageId) -> bool:
        """
        Delete message by ID.

        Args:
            message_id: MessageId value object

        Returns:
            True if deleted, False if not found or error
        """
        try:
            await self._prisma.message.delete(where={"id": message_id.value})
            return True
        except Exception:
            return False
