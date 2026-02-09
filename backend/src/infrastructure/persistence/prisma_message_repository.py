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

import json
import logging
from typing import Optional, Any, Dict
from prisma import Prisma
from prisma.models import Message as PrismaMessage
from src.domain.entities.message import Message
from src.domain.ports.repositories.message_repository import MessageRepository
from src.domain.value_objects.message_id import MessageId
from src.domain.value_objects.conversation_id import ConversationId
from src.config.settings import Config

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
        openai_client = None
        if context:
            openai_client = context.get("openai_client")

        # Determine how many messages to fetch
        if openai_client:
            # LLM-based determination (smart)
            messages_needed = await self._determine_message_count_llm(
                query, openai_client
            )
        else:
            # Fallback: Use default limit
            logger.info(
                "[get_smart_history] No OpenAI client provided, "
                f"using default limit {Config.REDIS_CACHE_LIMIT}"
            )
            messages_needed = Config.REDIS_CACHE_LIMIT

        # Fetch messages using standard method
        logger.info(
            f"[get_smart_history] Fetching {messages_needed} messages "
            f"for query: {query[:50]}..."
        )
        return await self.get_by_conversation(conversation_id, limit=messages_needed)

    async def _determine_message_count_llm(self, query: str, openai_client: Any) -> int:
        """
        Use LLM to determine how many messages are needed for the query.

        Phase 1: Simple LLM-based count determination.

        Args:
            query: User's query
            openai_client: OpenAI client instance

        Returns:
            Number of messages needed (0-200)
        """
        try:
            from src.services.llm_client import chat_completion

            prompt = f"""Analyze this user query and determine how many previous conversation messages are needed to answer it accurately.

Query: "{query}"

Consider:
- Does it reference previous conversation? (e.g., "what did we discuss?")
- Is it a follow-up? (e.g., "what about...", "and also...")
- Does it need context? (e.g., "based on what you said...")
- Is it asking for a summary? (e.g., "summarize our chat")
- Or is it standalone? (e.g., "what is Python?")

Respond with ONLY a JSON object:
{{
  "messages_needed": <number between 0 and 200>,
  "reasoning": "<brief explanation>"
}}

Examples:
- "What is Python?" → {{"messages_needed": 0, "reasoning": "Standalone general knowledge"}}
- "What about Java?" → {{"messages_needed": 3, "reasoning": "Follow-up to recent topic"}}
- "What did we discuss about databases?" → {{"messages_needed": 20, "reasoning": "Search recent history for topic"}}
- "Summarize our conversation" → {{"messages_needed": 200, "reasoning": "Full conversation summary"}}
"""

            response = chat_completion(
                client=openai_client,
                model=Config.OPENAI_SIMPLE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=100,
                temperature=0,
            )

            result = json.loads(response.choices[0].message.content)
            messages_needed = min(result["messages_needed"], 200)  # Cap at 200

            logger.info(
                f"[LLM Intent] Query needs {messages_needed} messages - "
                f"Reason: {result['reasoning']}"
            )

            return messages_needed

        except Exception as e:
            logger.warning(
                f"[LLM Intent] Failed to determine message count: {e}, "
                f"falling back to {Config.REDIS_CACHE_LIMIT}"
            )
            return Config.REDIS_CACHE_LIMIT

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
