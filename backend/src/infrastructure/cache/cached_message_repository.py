"""
Cached Message Repository - Decorator pattern for Redis caching.

Guidelines:
- Implements MessageRepository interface (same as PrismaMessageRepository)
- Wraps underlying repository with Redis caching layer
- Transparent to callers - they don't know caching exists
- All methods are async (pure async, no event loop issues)

Architecture:
    CachedMessageRepository (decorator)
        ↓ wraps
    PrismaMessageRepository (concrete implementation)
        ↓ implements
    MessageRepository (abstract interface)

Cache Strategy:
- Read-Through: Check cache first, fallback to DB, populate cache
- Write-Through: Write to DB first, then update/invalidate cache
- TTL-based expiration: No explicit invalidation needed

Redis Data Structure (LIST):
- Key pattern: "conv:{conversation_id}:msgs"
- Type: LIST (not STRING)
- Each element: JSON string for ONE message
- Order: Position 0 = newest, Position N = oldest
- TTL: Config.REDIS_CACHE_TTL (default 3600 seconds = 1 hour)
- Limit: Config.REDIS_CACHE_LIMIT (default 15 messages)

Redis Commands Used:
- LPUSH: Add new message to front (save)
- RPUSH: Populate cache from DB (bulk insert)
- LRANGE: Read messages from cache
- LTRIM: Keep only most recent N messages
- EXPIRE: Set TTL

Error Handling:
- Cache failures should NOT fail the operation
- Log warnings and fallback to DB
- All Redis operations wrapped in try/except

Serialization:
- Message entity → dict → JSON string (for Redis storage)
- JSON string → dict → Message entity (for retrieval)

Example Usage:
    # In DI container:
    base_repo = PrismaMessageRepository(prisma)
    cached_repo = CachedMessageRepository(base_repo, redis_client)

    # Caller uses it like normal repository:
    messages = await cached_repo.get_by_conversation(conv_id, limit=15)

Steps to implement:

1. __init__(self, repo: MessageRepository, redis: Redis)
   - Store underlying repository
   - Store Redis client
   - Define cache key prefix and TTL from Config

2. _cache_key(conversation_id: ConversationId) -> str
   - Return formatted key: f"conv:{conversation_id.value}:msgs"

3. _serialize_messages(messages: list[Message]) -> str
   - Convert Message entities to list of dicts
   - Handle value objects: message.id.value, message.conversation_id.value
   - Handle datetime: use .isoformat() or default=str in json.dumps
   - Return JSON string

4. _deserialize_messages(json_str: str) -> list[Message]
   - Parse JSON string to list of dicts
   - Reconstruct Message entities with value objects
   - Handle datetime parsing (fromisoformat)
   - Return list of Message entities

5. async get_by_id(message_id: MessageId) -> Optional[Message]
   - Delegate directly to underlying repo (single message not worth caching)
   - return await self._repo.get_by_id(message_id)

6. async get_by_conversation(conversation_id: ConversationId, limit: int) -> list[Message]
   - Try cache first:
     - key = self._cache_key(conversation_id)
     - cached = await self._redis.get(key)
     - If cached: return self._deserialize_messages(cached)
   - Cache miss - fetch from DB:
     - messages = await self._repo.get_by_conversation(conversation_id, limit)
   - Populate cache (best effort):
     - json_str = self._serialize_messages(messages)
     - await self._redis.setex(key, Config.REDIS_CACHE_TTL, json_str)
   - Return messages
   - Wrap Redis operations in try/except, log warnings

7. async save(message: Message) -> None
   - Write to DB first (source of truth):
     - await self._repo.save(message)
   - Invalidate cache (let next read repopulate):
     - key = self._cache_key(message.conversation_id)
     - await self._redis.delete(key)
   - Or: Could update cache if exists (more complex)
   - Wrap Redis operations in try/except

8. async delete(message_id: MessageId) -> bool
   - Delegate to underlying repo
   - Note: Would need conversation_id to invalidate cache
   - For simplicity, just delegate (TTL will handle staleness)
   - return await self._repo.delete(message_id)
"""

import json
import logging
from typing import Optional
from datetime import datetime
from redis.asyncio import Redis
from src.domain.entities.message import Message
from src.domain.ports.repositories.message_repository import MessageRepository
from src.domain.value_objects.message_id import MessageId
from src.domain.value_objects.conversation_id import ConversationId
from src.config.settings import Config

logger = logging.getLogger(__name__)


class CachedMessageRepository(MessageRepository):
    """
    Decorator: adds Redis caching to MessageRepository.

    Wraps an underlying MessageRepository implementation with a Redis cache layer.
    Implements the same interface, so callers don't know caching exists.
    """

    def __init__(self, repo: MessageRepository, redis: Redis):
        """
        Initialize cached repository.

        Args:
            repo: Underlying MessageRepository implementation (e.g., PrismaMessageRepository)
            redis: Async Redis client for caching

        TODO: Implement
        - Store self._repo = repo
        - Store self._redis = redis
        """
        self._repo = repo
        self._redis = redis

    def _cache_key(self, conversation_id: ConversationId) -> str:
        """
        Generate Redis cache key for conversation messages.

        Args:
            conversation_id: ConversationId value object

        Returns:
            Cache key string: "conv:{id}:msgs"

        TODO: Implement
        - Return f"conv:{conversation_id.value}:msgs"
        """
        return f"conv:{conversation_id.value}:msgs"

    def _serialize_message(self, message: Message) -> str:
        """
        Serialize a single Message entity to JSON string.

        Args:
            message: Message domain entity

        Returns:
            JSON string for one message

        Note:
            Redis LIST stores each message as separate JSON string.
            This is different from storing one big JSON array.
        """
        msg_dict = {
            "id": message.id.value,
            "conversation_id": message.conversation_id.value,
            "role": message.role,
            "content": message.content,
            "created_at": message.created_at.isoformat(),
            "tokens_used": message.tokens_used,
            "latency_ms": message.latency_ms,
        }
        return json.dumps(msg_dict)

    def _deserialize_message(self, json_str: str) -> Message:
        """
        Deserialize a single JSON string to Message entity.

        Args:
            json_str: JSON string for one message

        Returns:
            Message domain entity
        """
        d = json.loads(json_str)
        return Message(
            id=MessageId(d["id"]),
            conversation_id=ConversationId(d["conversation_id"]),
            role=d["role"],
            content=d["content"],
            created_at=datetime.fromisoformat(d["created_at"]),
            tokens_used=d.get("tokens_used"),
            latency_ms=d.get("latency_ms"),
        )

    async def get_by_id(self, message_id: MessageId) -> Optional[Message]:
        """
        Get message by ID.

        Note: Single message lookup - not worth caching, delegate directly.

        TODO: Implement
        - return await self._repo.get_by_id(message_id)
        """
        return await self._repo.get_by_id(message_id)

    async def get_by_conversation(
        self, conversation_id: ConversationId, limit: int = 200
    ) -> list[Message]:
        """
        Get messages for conversation with Redis caching.

        Redis LIST Structure:
        - Position 0 = newest message (LPUSH adds to front)
        - Position N = oldest message
        - lrange returns [newest, ..., oldest]
        - We reverse to get chronological order [oldest, ..., newest]

        Flow:
        1. Try cache (fast path): lrange → reverse → return
        2. Cache miss → fetch from DB (slow path)
        3. Populate cache: DB returns oldest-first, reverse to newest-first, rpush

        Args:
            conversation_id: ConversationId value object
            limit: Maximum messages to return

        Returns:
            List of Message entities (oldest first, chronological order)
        """
        # If requesting more than cache limit, skip cache entirely
        # Cache only stores REDIS_CACHE_LIMIT messages (e.g., 15)
        # Large requests (e.g., limit=200 for frontend) must go to DB
        if limit > Config.REDIS_CACHE_LIMIT:
            logger.debug(
                f"Limit {limit} > cache limit {Config.REDIS_CACHE_LIMIT}, bypassing cache"
            )
            return await self._repo.get_by_conversation(conversation_id, limit)

        cache_key = self._cache_key(conversation_id)

        # 1. Try cache first (fast path) - only for small limits
        try:
            cached_json_list = await self._redis.lrange(cache_key, 0, limit - 1)
            if cached_json_list:
                logger.debug(f"Cache HIT for {cache_key}")
                # lrange returns [newest, ..., oldest], each item is a JSON string
                messages = [
                    self._deserialize_message(json_str) for json_str in cached_json_list
                ]
                # Reverse to get chronological order [oldest, ..., newest]
                messages.reverse()
                return messages
        except Exception as e:
            logger.warning(f"Redis cache read error for {cache_key}: {str(e)}")

        # 2. Cache miss - fetch from DB
        logger.debug(f"Cache MISS for {cache_key}")
        messages = await self._repo.get_by_conversation(conversation_id, limit)

        # 3. Populate cache (best effort)
        # DB returns [oldest, ..., newest] (chronological)
        # Redis LIST needs newest at position 0
        # So we reverse to [newest, ..., oldest] and use RPUSH to add in order
        try:
            if messages:
                # Clear existing cache first to avoid duplicates/stale data
                await self._redis.delete(cache_key)
                # Reverse to get [newest, ..., oldest] for Redis storage
                messages_reversed = list(reversed(messages))
                # Serialize each message individually
                json_strings = [
                    self._serialize_message(msg) for msg in messages_reversed
                ]
                # RPUSH adds to the end, preserving our newest-first order
                await self._redis.rpush(cache_key, *json_strings)
                await self._redis.ltrim(cache_key, 0, Config.REDIS_CACHE_LIMIT - 1)
                await self._redis.expire(cache_key, Config.REDIS_CACHE_TTL)
                logger.debug(f"Cache POPULATED for {cache_key}")
        except Exception as e:
            logger.warning(f"Redis cache write error for {cache_key}: {str(e)}")

        return messages

    async def save(self, message: Message) -> None:
        """
        Save message to DB and update cache (Write-Through pattern).

        Flow:
        1. Write to DB (must succeed)
        2. Add to cache front with LPUSH (newest at position 0)
        3. Trim to keep cache size limited

        Args:
            message: Message entity to save
        """
        # 1. Write to DB first (source of truth)
        await self._repo.save(message)

        # 2. Update cache (best effort)
        try:
            cache_key = self._cache_key(message.conversation_id)
            json_str = self._serialize_message(message)
            # LPUSH adds to front (position 0 = newest)
            await self._redis.lpush(cache_key, json_str)
            # LTRIM keeps only the most recent messages
            await self._redis.ltrim(cache_key, 0, Config.REDIS_CACHE_LIMIT - 1)
            await self._redis.expire(cache_key, Config.REDIS_CACHE_TTL)
            logger.debug(f"Cache UPDATED for {cache_key}")
        except Exception as e:
            logger.warning(f"Redis cache update error: {str(e)}")

    async def delete(self, message_id: MessageId) -> bool:
        """
        Delete message by ID.

        Note: Delegated directly - would need conversation_id to invalidate cache.
        TTL will handle eventual consistency.
        """
        return await self._repo.delete(message_id)
