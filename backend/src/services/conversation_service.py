"""
Conversation service for managing chat history with hybrid Redis + PostgreSQL storage.

IMPLEMENTATION GUIDE - Redis LIST-based Caching
===============================================

KEY CONCEPTS:
-------------
1. PostgreSQL = Source of truth (all messages forever)
2. Redis LIST = Performance layer (last 15 messages, 1 hour TTL)
3. Write-Through: Save to DB first, then update cache
4. Read-Through: Check cache first, fallback to DB

REDIS LIST COMMANDS YOU'LL USE:
-------------------------------
- LPUSH: Add to front of list (newest first)
- LRANGE: Get range of items from list
- LTRIM: Trim list to keep only N items
- RPUSH: Add to back of list (for bulk populate)
- EXPIRE: Set TTL on key

EXAMPLE REDIS OPERATIONS:
-------------------------
# Add new message to front:
await self.redis.lpush("conversation:123:messages", message_json)

# Keep only last 15:
await self.redis.ltrim("conversation:123:messages", 0, 19)

# Get last 15 messages:
messages = await self.redis.lrange("conversation:123:messages", 0, 19)

# Set 1 hour expiry:
await self.redis.expire("conversation:123:messages", 3600)
"""

import json
from datetime import datetime, timezone
from typing import List, Dict, Optional
from prisma import Prisma
from src.config.settings import Config
import redis.asyncio as redis


class ConversationService:
    """Manages conversation history with Redis cache + PostgreSQL persistence."""

    # Configuration constants
    CACHE_TTL = Config.REDIS_CACHE_TTL
    CACHE_LIMIT = Config.REDIS_CACHE_LIMIT

    def __init__(self):
        """
        Initialize service. Both Prisma and Redis clients created lazily on first connect.

        Clients are NOT created here to avoid event loop issues with Flask.
        They will be created in connect() when first needed.
        """
        self.prisma_client = None  # Created lazily in connect()
        self.redis = None  # Created lazily in connect()

    # ==================== Core CRUD Operations ====================

    async def create_conversation(
        self, user_email: str, title: Optional[str] = None
    ) -> str:
        """
        Create new conversation in PostgreSQL.

        TODO: Implement this
        STEPS:
        1. Ensure connected to Prisma: self.connect()
        2. Create conversation in DB:
           - Use: await self.prisma_client.conversation.create(data={...})
           - Fields: user_email, title (default "New Conversation"), created_at
           - created_at: use datetime.now(timezone.utc)
        3. Return conversation.id

        HINT: Look at your schema.prisma for field names
        """
        await self.connect()
        conversation = await self.prisma_client.conversation.create(
            data={
                "user_email": user_email,
                "title": title or "New Conversation",
                "created_at": datetime.now(),
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
        Save message to PostgreSQL AND update Redis cache (Write-Through pattern).

        TODO: Implement this
        STEPS:
        1. Save to PostgreSQL (CRITICAL - must succeed):
           - Ensure connected: self.connect()
           - Create message: await self.prisma_client.message.create(data={...})
           - Fields: conversation_id, role, content, tokens_used, latency_ms, created_at

        2. Update Redis cache (BEST EFFORT - failures shouldn't fail request):
           - Wrap in try/except
           - Get cache key: cache_key = self._get_cache_key(conversation_id)
           - Convert message to JSON: json.dumps(message.model_dump(), default=str)
           - Add to front: await self.redis.lpush(cache_key, message_json)
           - Trim to limit: await self.redis.ltrim(cache_key, 0, self.CACHE_LIMIT - 1)
           - Set expiry: await self.redis.expire(cache_key, self.CACHE_TTL)
           - If error, print warning but continue

        3. Return message.model_dump()

        WHY THIS WORKS:
        - LPUSH adds newest message to position 0
        - LTRIM keeps positions 0-19 (last 15 messages)
        - Older messages (15+) are automatically removed
        - All operations are O(1)
        """
        await self.connect()
        message = await self.prisma_client.message.create(
            data={
                "conversation_id": conversation_id,
                "role": role,
                "content": content,
                "tokens_used": tokens_used,
                "latency_ms": latency_ms,
                "created_at": datetime.now(),
            }
        )

        # Update Redis cache
        try:
            cache_key = self._get_cache_key(conversation_id)
            message_dict = message.model_dump()
            message_json = json.dumps(message_dict, default=str)
            await self.redis.lpush(cache_key, message_json)
            await self.redis.ltrim(cache_key, 0, self.CACHE_LIMIT - 1)
            await self.redis.expire(cache_key, self.CACHE_TTL)
        except Exception as e:
            print(f"Warning: Failed to update Redis cache: {str(e)}")

        return message.model_dump()

    async def get_message_history(
        self, conversation_id: str, limit: int = 15
    ) -> List[Dict]:
        """
        Get conversation messages (Read-Through pattern).

        TODO: Implement this
        STEPS:
        1. Try Redis cache first (FAST PATH):
           - Wrap in try/except
           - Get cache key: cache_key = self._get_cache_key(conversation_id)
           - Get messages: await self.redis.lrange(cache_key, 0, limit - 1)
           - If found: return [json.loads(msg) for msg in cached_messages]
           - If error, log warning and continue to DB

        2. Load from PostgreSQL (SLOW PATH):
           - Ensure connected: self.connect()
           - Query: await self.prisma_client.message.find_many(
                where={"conversation_id": conversation_id},
                order={"created_at": "desc"},  # ← IMPORTANT: desc for newest first
                take=limit
             )

        3. Populate cache for next request (BEST EFFORT):
           - If messages exist, wrap in try/except:
           - Convert to JSON list: [json.dumps(m.model_dump(), default=str) for m in messages]
           - Add all to Redis: await self.redis.rpush(cache_key, *message_jsons)
             (Use RPUSH because messages already in reverse order from DB)
           - Set expiry: await self.redis.expire(cache_key, self.CACHE_TTL)
           - If error, log warning

        4. Return [m.model_dump() for m in messages]

        WHY ORDER MATTERS:
        - Cache stores newest first (position 0 = newest)
        - DB query must be "desc" to match cache order
        - When populating cache, use RPUSH because messages already reversed
        """
        # Try Redis cache first
        try:
            cache_key = self._get_cache_key(conversation_id)
            cached_messages = await self.redis.lrange(cache_key, 0, limit - 1)
            if cached_messages:
                return [json.loads(msg) for msg in cached_messages]
        except Exception as e:
            print(f"Warning: Failed to read from Redis cache: {str(e)}")

        # Load from DB if not found in cache
        await self.connect()
        query = {
            "where": {"conversation_id": conversation_id},
            "order": {"created_at": "desc"},
            "take": limit,
        }
        messages = await self.prisma_client.message.find_many(**query)

        # Populate cache for next time
        if messages:
            try:
                msg_dict = [msg.model_dump() for msg in messages]
                message_jsons = [json.dumps(m, default=str) for m in msg_dict]
                await self.redis.rpush(cache_key, *message_jsons)
                await self.redis.expire(cache_key, self.CACHE_TTL)
            except Exception as e:
                print(f"Warning: Failed to populate Redis cache: {str(e)}")

        return [m.model_dump() for m in messages]

    async def load_message_history_db(
        self, conversation_id: str, limit: int = Config.CONVERSATION_MESSAGE_LIMIT
    ) -> List[Dict]:
        await self.connect()
        query = {
            "where": {"conversation_id": conversation_id},
            "order": {"created_at": "asc"},
            "take": limit,
        }
        messages = await self.prisma_client.message.find_many(**query)

        return [m.model_dump() for m in messages]

    async def get_user_conversations_list(self, user_id: str, limit: int = 50) -> List[Dict]:
        """
        Get list of conversations for a user from PostgreSQL.
        """
        try:
            await self.connect()
            conversations_list = await self.prisma_client.conversation.find_many(
                where={"user_email": user_id},
                order={"updated_at": "desc"},
                take=limit,
                include={
                    "messages": {
                        "order_by": {"created_at": "desc"},
                        "take": 1,
                    }
                },
            )

            return [conv.model_dump() for conv in conversations_list]
        except Exception as e:
            print(f"Error fetching conversations list: {str(e)}")
            return []

    async def update_conversation_title(self, conversation_id: str, user_id: str, new_title: str) -> Optional[Dict]:
        try:
            await self.connect()
            conversation = await self.prisma_client.conversation.find_unique(
                where={"id": conversation_id}
            )
            if not conversation or conversation.user_email != user_id:
                print("Error: Unauthorized or conversation not found")
                return None

            updated_conversation = await self.prisma_client.conversation.update(
                where={"id": conversation_id},
                data={"title": new_title, "updated_at": datetime.now()}
            )

            return updated_conversation.model_dump()
        except Exception as e:
            print(f"Error updating conversation title: {str(e)}")
            return None

    async def delete_conversation(self, conversation_id: str, user_id: str) -> bool:
        """
        Delete conversation from both Redis and PostgreSQL.

        TODO: Implement this
        STEPS:
        1. Wrap entire function in try/except
        2. Delete from Redis cache:
           - Get cache key: cache_key = self._get_cache_key(conversation_id)
           - Delete: await self.redis.delete(cache_key)
        3. Delete from PostgreSQL:
           - Ensure connected: self.connect()
           - Delete: await self.prisma_client.conversation.delete(
                where={"id": conversation_id}
             )
        4. Return True if successful
        5. In except block: print error and return False

        NOTE: PostgreSQL cascade will auto-delete all messages
        """
        try:
            # Check if conversation belongs to user
            await self.connect()
            conversation = await self.prisma_client.conversation.find_unique(
                where={"id": conversation_id}
            )
            if not conversation or conversation.user_email != user_id:
                print("Error: Unauthorized or conversation not found")
                return False

            # Remove from Redis
            cache_key = self._get_cache_key(conversation_id)
            await self.redis.delete(cache_key)

            # Remove from PostgreSQL
            await self.prisma_client.conversation.delete(where={"id": conversation_id})

            return True
        except Exception as e:
            print(f"Error deleting conversation: {str(e)}")
            return False

    # ==================== Helper Methods ====================

    def _get_cache_key(self, conversation_id: str) -> str:
        """
        Generate Redis cache key for conversation messages.

        TODO: Implement this
        STEPS:
        1. Return formatted string: f"conversation:{conversation_id}:messages"

        EXAMPLE: "conversation:abc123:messages"
        """
        return f"conversation:{conversation_id}:messages"

    async def connect(self):
        """
        Ensure Prisma and Redis clients are connected.

        Always creates fresh connections to avoid event loop and state issues.
        Both Prisma and Redis connections are lightweight and designed to be created per-request.
        """
        # Disconnect old Prisma client if exists
        if self.prisma_client:
            try:
                await self.prisma_client.disconnect()
            except:
                pass

        # Close old Redis connection if exists
        if self.redis:
            try:
                await self.redis.close()
            except:
                pass

        # Always create fresh clients for each request
        self.prisma_client = Prisma()
        await self.prisma_client.connect()

        # Create fresh Redis connection
        self.redis = await redis.from_url(Config.REDIS_URL, decode_responses=True)


# ==================== TESTING GUIDE ====================

"""
HOW TO TEST YOUR IMPLEMENTATION:
=================================

1. Test Redis LIST structure:

   # In redis-cli:
   docker exec -it chatbot-redis-1 redis-cli

   TYPE conversation:abc123:messages
   # Should return: list

   LRANGE conversation:abc123:messages 0 -1
   # Should return: JSON strings of messages

   LLEN conversation:abc123:messages
   # Should return: number (max 15)

2. Test with Python:

   import asyncio
   from src.services.conversation_service_skeleton import ConversationService

   async def test():
       service = ConversationService()

       # Create conversation
       conv_id = await service.create_conversation(
           user_email="test@example.com",
           title="Test"
       )
       print(f"Created: {conv_id}")

       # Add messages
       for i in range(5):
           await service.save_message(
               conversation_id=conv_id,
               role="user" if i % 2 == 0 else "assistant",
               content=f"Message {i}"
           )

       # Get messages
       messages = await service.get_conversation_history(conv_id)
       print(f"Retrieved {len(messages)} messages")
       print(f"Newest: {messages[0]['content']}")  # Should be "Message 4"
       print(f"Oldest: {messages[-1]['content']}")  # Should be "Message 0"

   asyncio.run(test())

3. Expected output:
   Created: {some_uuid}
   Retrieved 5 messages
   Newest: Message 4
   Oldest: Message 0

4. Verify cache in Redis:
   LRANGE conversation:{conv_id}:messages 0 -1
   # Should show 5 messages, newest at position 0


COMMON MISTAKES TO AVOID:
==========================

1. ❌ Using self.redis.set() instead of lpush
   ✅ Use lpush for adding individual messages

2. ❌ Forgetting to call ltrim after lpush
   ✅ Always trim to maintain size limit

3. ❌ Using "asc" order when querying DB
   ✅ Use "desc" to match cache order (newest first)

4. ❌ Using lpush when populating cache from DB
   ✅ Use rpush because messages already in reverse order

5. ❌ Not wrapping cache operations in try/except
   ✅ Cache failures shouldn't fail the request


REDIS COMMANDS REFERENCE:
=========================

# Add to front (newest message)
LPUSH conversation:123:messages '{"role":"user","content":"hi"}'

# Keep only last 15 (positions 0-19)
LTRIM conversation:123:messages 0 19

# Get last 15 messages
LRANGE conversation:123:messages 0 19

# Get first 5 messages
LRANGE conversation:123:messages 0 4

# Check length
LLEN conversation:123:messages

# Set expiry (1 hour = 3600 seconds)
EXPIRE conversation:123:messages 3600

# Check TTL
TTL conversation:123:messages

# Delete
DEL conversation:123:messages


NEXT STEPS AFTER IMPLEMENTING:
===============================

Once this file is working:
1. Test thoroughly with the test script above
2. Verify in redis-cli that LIST structure is correct
3. Check that messages are in correct order (newest first)
4. Verify cache limit (max 15 messages)
5. Check TTL is set (should be 3600 seconds)

Then we'll move to Step 2: Replace SESSIONS in chat.py
"""
