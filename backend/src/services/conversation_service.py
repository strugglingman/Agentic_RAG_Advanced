"""
Conversation service for managing chat history with hybrid Redis + PostgreSQL storage.
"""

import json
from datetime import datetime, timezone
from typing import List, Dict, Optional
from prisma import Prisma
from src.config.redis_client import get_redis
from src.config.settings import Config


class ConversationService:
    """Manages conversation history with Redis cache + PostgreSQL persistence."""

    def __init__(self):
        """
        TODO: Initialize service
        - Create Prisma client instance
        - Get Redis client from get_redis()
        """
        self.prisma_client = Prisma()
        self.redis = get_redis().redis

    async def create_conversation(
        self, user_email: str, title: Optional[str] = None
    ) -> str:
        """
        TODO: Create new conversation
        - Connect to Prisma
        - Create conversation record in PostgreSQL
        - Return conversation_id
        """
        self.connect()
        conversation = await self.prisma_client.conversation.create(
            data={
                "user_email": user_email,
                "title": title or "New Conversation",
                "created_at": datetime.now(timezone.utc),
            }
        )

        return conversation.id

    async def save_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        tokens_used: Optional[int] = None,
        latency_ms: Optional[int] = None,
    ) -> Dict:
        """
        TODO: Save message to PostgreSQL and update Redis cache
        - Connect to Prisma
        - Create message in PostgreSQL
        - Invalidate Redis cache for this conversation
        - Return message data
        """
        self.connect()
        message = await self.prisma_client.message.create(
            data={
                "conversation_id": conversation_id,
                "role": role,
                "content": content,
                "tokens_used": tokens_used,
                "latency_ms": latency_ms,
                "created_at": datetime.now(timezone.utc),
            }
        )
        # Invalidate Redis cache
        cache_key = self._get_cache_key(conversation_id)
        await self.redis.delete(cache_key)

        return message.model_dump()

    async def get_conversation_history(
        self, conversation_id: str, limit: int = 20
    ) -> List[Dict]:
        """
        TODO: Get conversation history (Redis cache + PostgreSQL fallback)
        - Check Redis cache first (key: "conversation:{id}")
        - If hit: return cached messages
        - If miss: load from PostgreSQL → cache in Redis with TTL
        - Return list of messages
        """
        cache_key = self._get_cache_key(conversation_id)
        cached_data = await self.redis.get(cache_key)
        if cached_data:
            return self._deserialize_messages(cached_data)

        self.connect()
        query = {
            "where": {"conversation_id": conversation_id},
            "order": {"created_at": "asc"},
            "take": limit,
        }
        messages = await self.prisma_client.message.find_many(**query)
        await self.redis.set(cache_key, self._serialize_messages(messages), ex=3600)

        return [m.model_dump() for m in messages]

    async def delete_conversation(self, conversation_id: str) -> bool:
        """
        TODO: Delete conversation from both Redis and PostgreSQL
        - Delete from Redis cache
        - Delete from PostgreSQL (cascade deletes messages)
        - Return success status
        """
        try:
            cache_key = self._get_cache_key(conversation_id)
            await self.redis.delete(cache_key)

            self.connect()
            await self.prisma_client.conversation.delete(where={"id": conversation_id})

            return True
        except Exception as e:
            print(f"Error deleting conversation: {e}")
            return False

    def _serialize_messages(self, messages: List) -> str:
        """
        TODO: Serialize messages to JSON string for Redis
        - Convert Prisma models to dicts
        - Return JSON string
        """
        msg_dicts = [m.model_dump() for m in messages]
        try:
            return json.dumps(msg_dicts, default=str)
        except Exception as e:
            print(f"Error serializing messages: {e}")
            return "[]"

    def _deserialize_messages(self, data: str) -> List[Dict]:
        """
        TODO: Deserialize messages from Redis JSON
        - Parse JSON string
        - Return list of message dicts
        """
        return json.loads(data)

    def _get_cache_key(self, conversation_id: str) -> str:
        """
        TODO: Generate Redis cache key
        - Return formatted key: "conversation:{id}"
        """
        return f"conversation:{conversation_id}"

    def connect(self):
        if self.prisma_client and self.prisma_client.is_connected():
            return

        if not self.prisma_client:
            self.prisma_client = Prisma()

        self.prisma_client.connect()
