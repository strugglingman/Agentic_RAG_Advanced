# Conversation History Implementation Guide

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Database Schema](#database-schema)
3. [Caching Strategy](#caching-strategy)
4. [Service Layer Design](#service-layer-design)
5. [Migration from SESSIONS](#migration-from-sessions)
6. [API Integration](#api-integration)
7. [Performance Optimization](#performance-optimization)
8. [Best Practices](#best-practices)
9. [Implementation Checklist](#implementation-checklist)

---

## Architecture Overview

### High-Level Design

```
┌─────────────────────────┐
│   FastAPI Endpoints     │
│   - POST /conversations │
│   - POST /messages      │
│   - GET /history        │
└───────────┬─────────────┘
            │
┌───────────▼──────────────────────┐
│   ConversationService            │
│   - Business Logic Layer         │
│   - Caching Management           │
│   - Persistence Orchestration    │
└───────────┬──────────────────────┘
            │
      ┌─────┴─────┐
      │           │
┌─────▼─────┐ ┌──▼──────────┐
│   Redis   │ │ PostgreSQL  │
│   Cache   │ │  (Prisma)   │
│  (Fast)   │ │ (Persistent)│
└───────────┘ └─────────────┘
```

### Data Flow Strategy

```
User Request (Send Message)
    ↓
┌───▼────────────────────────────────┐
│ 1. Save to PostgreSQL              │  ← Source of truth
│    - Guaranteed persistence        │
└───┬────────────────────────────────┘
    ↓
┌───▼────────────────────────────────┐
│ 2. Update Redis Cache              │  ← Performance layer
│    - LPUSH new message             │
│    - LTRIM to limit (keep last N)  │
│    - SET expiry (TTL)              │
└───┬────────────────────────────────┘
    ↓
┌───▼────────────────────────────────┐
│ 3. Return to User                  │
└────────────────────────────────────┘

User Request (Get Messages)
    ↓
┌───▼────────────────────────────────┐
│ 1. Check Redis Cache               │
│    - LRANGE 0 N                    │
└───┬────────────────────────────────┘
    │
    ├─ HIT  → Return immediately (fast path)
    │
    └─ MISS ↓
       ┌────▼─────────────────────────┐
       │ 2. Load from PostgreSQL      │
       │    - Query with LIMIT        │
       └────┬─────────────────────────┘
            ↓
       ┌────▼─────────────────────────┐
       │ 3. Populate Redis Cache      │
       │    - RPUSH messages          │
       │    - SET expiry              │
       └────┬─────────────────────────┘
            ↓
       ┌────▼─────────────────────────┐
       │ 4. Return to User            │
       └──────────────────────────────┘
```

### Design Principles

1. **PostgreSQL = Source of Truth**
   - All messages are persisted here
   - Survives server restarts
   - Enables search, analytics, auditing

2. **Redis = Performance Layer**
   - Caches recent messages (last 20-50)
   - Fast read access (~1-5ms)
   - Disposable (can rebuild from PostgreSQL)

3. **Write-Through Caching**
   - Write to DB first, then update cache
   - Ensures consistency

4. **Read-Through Caching**
   - Check cache first, fallback to DB
   - Auto-populate cache on miss

---

## Database Schema

### Prisma Schema

```prisma
// backend/prisma/schema.prisma

model Conversation {
  id         String    @id @default(cuid())
  user_email String
  title      String
  created_at DateTime  @default(now())
  updated_at DateTime  @updatedAt

  messages   Message[]

  @@index([user_email])
  @@index([created_at])
}

model Message {
  id              String   @id @default(cuid())
  conversation_id String
  role            String   // "user", "assistant", "system"
  content         String   @db.Text
  tokens_used     Int?
  latency_ms      Int?
  created_at      DateTime @default(now())

  conversation Conversation @relation(fields: [conversation_id], references: [id], onDelete: Cascade)

  @@index([conversation_id, created_at])
  @@index([created_at])
}
```

### Key Design Decisions

1. **`conversation_id` Foreign Key**
   - Enables cascade delete
   - Ensures referential integrity

2. **Indexes**
   - `conversation_id + created_at`: Fast message retrieval
   - `user_email`: Fast user conversation lookup
   - `created_at`: Time-based queries

3. **`@db.Text`**
   - Supports long messages (no VARCHAR limit)

4. **Cascade Delete**
   - Deleting conversation auto-deletes messages
   - Maintains data consistency

---

## Caching Strategy

### Redis Data Structures Comparison

#### Option A: LIST (Recommended for Chat)

```redis
Key:   "conversation:{conversation_id}:messages"
Type:  LIST
Value: JSON strings of messages (newest first)
TTL:   3600 seconds (1 hour)

Commands:
  LPUSH key message_json      # Add new message to front
  LRANGE key 0 19            # Get last 20 messages
  LTRIM key 0 19             # Keep only last 20
  EXPIRE key 3600            # Set 1-hour expiry
```

**Pros:**
- ✅ Natural order (newest first with LPUSH)
- ✅ Efficient append (O(1))
- ✅ Efficient trim (O(N) where N = items removed)
- ✅ Simple to implement

**Cons:**
- ❌ No time-range queries
- ❌ No score-based filtering

#### Option B: SORTED SET (For Advanced Queries)

```redis
Key:   "conversation:{conversation_id}:messages"
Type:  ZSET
Score: Timestamp (Unix epoch)
Value: JSON message

Commands:
  ZADD key timestamp message_json
  ZRANGE key 0 19 REV
  ZREMRANGEBYRANK key 0 -21  # Keep last 20
```

**Pros:**
- ✅ Time-range queries (ZRANGEBYSCORE)
- ✅ Automatic ordering by timestamp

**Cons:**
- ❌ More complex
- ❌ Slightly more memory

**Recommendation:** Use **LIST** for simplicity unless you need time-range queries.

### Cache Key Pattern

```python
def _get_cache_key(self, conversation_id: str) -> str:
    return f"conversation:{conversation_id}:messages"
```

**Benefits:**
- Namespacing prevents key collisions
- Easy to identify in Redis
- Supports pattern matching for debugging

### Cache Limits

```python
CACHE_TTL = 3600           # 1 hour
CACHE_MESSAGE_LIMIT = 20   # Last 20 messages
```

**Rationale:**
- 20 messages ≈ typical chat context window
- 1 hour TTL balances memory vs. stale data
- Can be adjusted based on usage patterns

### Cache Invalidation Strategy

```python
# On new message: Update cache (don't invalidate)
await redis.lpush(key, new_message)
await redis.ltrim(key, 0, LIMIT - 1)

# On message edit: Invalidate (simpler than updating)
await redis.delete(key)

# On conversation delete: Invalidate
await redis.delete(key)
```

---

## Service Layer Design

### ConversationService Architecture

```python
class ConversationService:
    """
    Manages conversation lifecycle with hybrid Redis + PostgreSQL storage.

    Responsibilities:
    1. Conversation CRUD operations
    2. Message persistence (PostgreSQL)
    3. Cache management (Redis)
    4. Read-through/write-through caching

    Design Patterns:
    - Repository Pattern: Abstracts data access
    - Cache-Aside Pattern: Cache checked before DB
    - Write-Through Pattern: Write to DB + cache atomically
    """

    # Configuration
    CACHE_TTL = 3600              # 1 hour cache lifetime
    CACHE_MESSAGE_LIMIT = 20      # Max messages in cache

    def __init__(self):
        self.prisma_client = Prisma()
        self.redis = get_redis().redis
```

### Core Operations

#### 1. Create Conversation

```python
async def create_conversation(
    self,
    user_email: str,
    title: Optional[str] = None
) -> str:
    """
    Create new conversation in PostgreSQL.

    Args:
        user_email: User identifier
        title: Optional conversation title

    Returns:
        conversation_id: Unique conversation identifier
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
```

#### 2. Save Message (Write-Through Pattern)

```python
async def save_message(
    self,
    conversation_id: str,
    role: str,
    content: str,
    tokens_used: Optional[int] = None,
    latency_ms: Optional[int] = None,
) -> Dict:
    """
    Save message to PostgreSQL and update Redis cache.

    Write-Through Pattern:
    1. Save to PostgreSQL (source of truth)
    2. Add to Redis cache (performance)
    3. Trim cache to limit
    4. Set cache expiry

    Args:
        conversation_id: Conversation identifier
        role: "user", "assistant", or "system"
        content: Message text
        tokens_used: Optional token count
        latency_ms: Optional response latency

    Returns:
        Message dict with all fields
    """
    self.connect()

    # 1. Save to PostgreSQL
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

    # 2. Update Redis cache
    cache_key = self._get_cache_key(conversation_id)
    message_json = json.dumps(message.model_dump(), default=str)

    # Add to front of list (newest first)
    await self.redis.lpush(cache_key, message_json)

    # 3. Trim to keep only last N messages
    await self.redis.ltrim(cache_key, 0, self.CACHE_MESSAGE_LIMIT - 1)

    # 4. Set expiry
    await self.redis.expire(cache_key, self.CACHE_TTL)

    return message.model_dump()
```

#### 3. Get Recent Messages (Read-Through Pattern)

```python
async def get_recent_messages(
    self,
    conversation_id: str,
    limit: int = 20
) -> List[Dict]:
    """
    Get recent messages with Redis cache fallback to PostgreSQL.

    Read-Through Pattern:
    1. Check Redis cache
    2. On hit: Return cached data (fast)
    3. On miss: Load from PostgreSQL
    4. Populate cache for next request

    Args:
        conversation_id: Conversation identifier
        limit: Max messages to return

    Returns:
        List of message dicts (newest first)
    """
    cache_key = self._get_cache_key(conversation_id)

    # 1. Try Redis cache first
    cached_messages = await self.redis.lrange(cache_key, 0, limit - 1)

    if cached_messages:
        # Cache hit - parse and return
        return [json.loads(msg) for msg in cached_messages]

    # 2. Cache miss - load from PostgreSQL
    self.connect()
    messages = await self.prisma_client.message.find_many(
        where={"conversation_id": conversation_id},
        order={"created_at": "desc"},  # Newest first
        take=limit
    )

    # 3. Populate cache for next request
    if messages:
        message_jsons = [
            json.dumps(m.model_dump(), default=str)
            for m in messages
        ]
        # RPUSH because messages are already in reverse order
        await self.redis.rpush(cache_key, *message_jsons)
        await self.redis.expire(cache_key, self.CACHE_TTL)

    return [m.model_dump() for m in messages]
```

#### 4. Get Full History (Paginated)

```python
async def get_full_history(
    self,
    conversation_id: str,
    page: int = 1,
    page_size: int = 50
) -> Dict:
    """
    Get full conversation history from PostgreSQL with pagination.

    Use cases:
    - Export conversation
    - Full history view
    - Search through old messages

    Args:
        conversation_id: Conversation identifier
        page: Page number (1-indexed)
        page_size: Items per page

    Returns:
        Dict with messages, pagination metadata
    """
    self.connect()

    skip = (page - 1) * page_size

    # Get messages for current page
    messages = await self.prisma_client.message.find_many(
        where={"conversation_id": conversation_id},
        order={"created_at": "asc"},  # Oldest first for history
        skip=skip,
        take=page_size
    )

    # Get total count for pagination
    total = await self.prisma_client.message.count(
        where={"conversation_id": conversation_id}
    )

    return {
        "messages": [m.model_dump() for m in messages],
        "pagination": {
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size,
            "has_next": skip + len(messages) < total,
            "has_prev": page > 1
        }
    }
```

#### 5. Delete Conversation

```python
async def delete_conversation(self, conversation_id: str) -> bool:
    """
    Delete conversation from both Redis and PostgreSQL.

    Order matters:
    1. Delete from cache (fast, can fail silently)
    2. Delete from DB (critical, with error handling)

    Args:
        conversation_id: Conversation identifier

    Returns:
        True if successful, False otherwise
    """
    try:
        # 1. Remove from cache
        cache_key = self._get_cache_key(conversation_id)
        await self.redis.delete(cache_key)

        # 2. Delete from PostgreSQL (cascade deletes messages)
        self.connect()
        await self.prisma_client.conversation.delete(
            where={"id": conversation_id}
        )

        return True
    except Exception as e:
        print(f"Error deleting conversation: {e}")
        return False
```

### Helper Methods

```python
def _get_cache_key(self, conversation_id: str) -> str:
    """Generate Redis cache key for conversation messages."""
    return f"conversation:{conversation_id}:messages"

def _serialize_messages(self, messages: List) -> str:
    """Serialize Prisma messages to JSON string for Redis."""
    msg_dicts = [m.model_dump() for m in messages]
    return json.dumps(msg_dicts, default=str)

def _deserialize_messages(self, data: str) -> List[Dict]:
    """Deserialize JSON string from Redis to message dicts."""
    return json.loads(data)

def connect(self):
    """Ensure Prisma client is connected."""
    if self.prisma_client and self.prisma_client.is_connected():
        return
    if not self.prisma_client:
        self.prisma_client = Prisma()
    self.prisma_client.connect()
```

---

## Migration from SESSIONS

### Current State Analysis

#### Old SESSIONS Pattern

```python
# Global in-memory dictionary
SESSIONS = {}

# Session creation
def create_session(session_id: str):
    SESSIONS[session_id] = {
        "messages": [],
        "context": {},
        "created_at": datetime.now()
    }

# Message append
def add_message(session_id: str, message: dict):
    if session_id not in SESSIONS:
        create_session(session_id)
    SESSIONS[session_id]["messages"].append(message)

# Message retrieval
def get_messages(session_id: str) -> List[dict]:
    return SESSIONS.get(session_id, {}).get("messages", [])

# Session cleanup
def clear_session(session_id: str):
    if session_id in SESSIONS:
        del SESSIONS[session_id]
```

**Problems:**
- ❌ Lost on server restart
- ❌ All messages in RAM (memory issues)
- ❌ Single server only (can't scale horizontally)
- ❌ No persistence or history
- ❌ No search or analytics
- ❌ No audit trail

### New Pattern with ConversationService

```python
# Service instance (can be dependency-injected)
conversation_service = ConversationService()

# Conversation creation
async def create_conversation(user_email: str):
    conversation_id = await conversation_service.create_conversation(
        user_email=user_email,
        title="New Chat"
    )
    return conversation_id

# Message append
async def add_message(conversation_id: str, role: str, content: str):
    message = await conversation_service.save_message(
        conversation_id=conversation_id,
        role=role,
        content=content
    )
    return message

# Message retrieval
async def get_messages(conversation_id: str) -> List[dict]:
    messages = await conversation_service.get_recent_messages(
        conversation_id=conversation_id,
        limit=20
    )
    return messages

# Conversation cleanup
async def delete_conversation(conversation_id: str):
    success = await conversation_service.delete_conversation(
        conversation_id=conversation_id
    )
    return success
```

**Benefits:**
- ✅ Persistent across restarts
- ✅ Only recent messages in cache
- ✅ Multi-server ready (Redis shared)
- ✅ Full history in PostgreSQL
- ✅ Searchable and queryable
- ✅ Complete audit trail

### Step-by-Step Migration

#### Step 1: Find All SESSIONS Usage

```bash
# Search for SESSIONS usage
grep -r "SESSIONS" backend/src/

# Common patterns to find:
# - SESSIONS[session_id]
# - session_id in SESSIONS
# - SESSIONS.get()
# - SESSIONS[session_id]["messages"]
```

#### Step 2: Replace Session Creation

```python
# OLD
if session_id not in SESSIONS:
    SESSIONS[session_id] = {"messages": [], "context": {}}

# NEW
# Option A: Create on first message
conversation_id = await conversation_service.create_conversation(
    user_email=current_user.email
)

# Option B: Store conversation_id in session/JWT
# Then reuse existing conversation_id
```

#### Step 3: Replace Message Append

```python
# OLD
SESSIONS[session_id]["messages"].append({
    "role": "user",
    "content": user_message
})
SESSIONS[session_id]["messages"].append({
    "role": "assistant",
    "content": ai_response
})

# NEW
# Save user message
await conversation_service.save_message(
    conversation_id=conversation_id,
    role="user",
    content=user_message
)

# Save AI response
await conversation_service.save_message(
    conversation_id=conversation_id,
    role="assistant",
    content=ai_response,
    tokens_used=token_count,
    latency_ms=response_time_ms
)
```

#### Step 4: Replace Message Retrieval

```python
# OLD
messages = SESSIONS.get(session_id, {}).get("messages", [])

# NEW
messages = await conversation_service.get_recent_messages(
    conversation_id=conversation_id,
    limit=20
)
```

#### Step 5: Replace Session Deletion

```python
# OLD
if session_id in SESSIONS:
    del SESSIONS[session_id]

# NEW
await conversation_service.delete_conversation(
    conversation_id=conversation_id
)
```

### Handling Session ID to Conversation ID Mapping

#### Option A: Use Conversation ID as Session ID

```python
# When creating conversation
conversation_id = await conversation_service.create_conversation(...)

# Store in session/cookie/JWT
response.set_cookie("session_id", conversation_id)

# Or return to frontend
return {"session_id": conversation_id}
```

#### Option B: Maintain Mapping

```python
# Redis mapping: session_id -> conversation_id
await redis.set(f"session:{session_id}:conversation", conversation_id)

# Lookup when needed
conversation_id = await redis.get(f"session:{session_id}:conversation")
```

#### Option C: Store in JWT/Session

```python
# In JWT payload
token_payload = {
    "user_email": user.email,
    "conversation_id": conversation_id
}

# Or in session data
session["conversation_id"] = conversation_id
```

---

## API Integration

### FastAPI Endpoints

#### 1. Create Conversation

```python
# routes/conversations.py
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from src.services.conversation_service import ConversationService
from src.auth import get_current_user

router = APIRouter(prefix="/api/conversations", tags=["conversations"])

class CreateConversationRequest(BaseModel):
    title: Optional[str] = None

class ConversationResponse(BaseModel):
    conversation_id: str
    title: str
    created_at: datetime

@router.post("", response_model=ConversationResponse)
async def create_conversation(
    request: CreateConversationRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Create a new conversation.

    Returns conversation_id to be used for subsequent messages.
    """
    service = ConversationService()

    conversation_id = await service.create_conversation(
        user_email=current_user.email,
        title=request.title or "New Conversation"
    )

    return {
        "conversation_id": conversation_id,
        "title": request.title or "New Conversation",
        "created_at": datetime.now(timezone.utc)
    }
```

#### 2. Send Message

```python
class SendMessageRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None  # Create new if not provided

class MessageResponse(BaseModel):
    user_message: Dict
    assistant_message: Dict
    conversation_id: str

@router.post("/messages", response_model=MessageResponse)
async def send_message(
    request: SendMessageRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Send a message and get AI response.

    If conversation_id not provided, creates new conversation.
    """
    service = ConversationService()

    # Create conversation if needed
    conversation_id = request.conversation_id
    if not conversation_id:
        conversation_id = await service.create_conversation(
            user_email=current_user.email
        )

    # Save user message
    user_msg = await service.save_message(
        conversation_id=conversation_id,
        role="user",
        content=request.message
    )

    # Get conversation context (last 20 messages)
    history = await service.get_recent_messages(
        conversation_id=conversation_id,
        limit=20
    )

    # Generate AI response (your existing logic)
    ai_response = await generate_ai_response(
        messages=history,
        user_query=request.message
    )

    # Save assistant message
    assistant_msg = await service.save_message(
        conversation_id=conversation_id,
        role="assistant",
        content=ai_response["content"],
        tokens_used=ai_response.get("tokens"),
        latency_ms=ai_response.get("latency_ms")
    )

    return {
        "user_message": user_msg,
        "assistant_message": assistant_msg,
        "conversation_id": conversation_id
    }
```

#### 3. Get Conversation History

```python
class GetHistoryRequest(BaseModel):
    limit: int = 20
    page: Optional[int] = None  # If provided, return full paginated history

@router.get("/{conversation_id}/messages")
async def get_conversation_messages(
    conversation_id: str,
    limit: int = 20,
    page: Optional[int] = None,
    current_user: User = Depends(get_current_user)
):
    """
    Get conversation messages.

    - Without page: Returns recent N messages (from cache, fast)
    - With page: Returns full paginated history (from DB)
    """
    service = ConversationService()

    if page:
        # Full history with pagination
        result = await service.get_full_history(
            conversation_id=conversation_id,
            page=page,
            page_size=50
        )
        return result
    else:
        # Recent messages (from cache)
        messages = await service.get_recent_messages(
            conversation_id=conversation_id,
            limit=limit
        )
        return {"messages": messages}
```

#### 4. List User Conversations

```python
@router.get("")
async def list_conversations(
    current_user: User = Depends(get_current_user),
    page: int = 1,
    page_size: int = 20
):
    """List all conversations for current user."""
    service = ConversationService()

    # This requires a new method in ConversationService
    conversations = await service.list_user_conversations(
        user_email=current_user.email,
        page=page,
        page_size=page_size
    )

    return conversations
```

#### 5. Delete Conversation

```python
@router.delete("/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_user)
):
    """Delete a conversation and all its messages."""
    service = ConversationService()

    # Verify ownership (add to service method)
    success = await service.delete_conversation(
        conversation_id=conversation_id
    )

    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return {"message": "Conversation deleted successfully"}
```

### Integration with Existing Chat Endpoint

```python
# Example: Update existing chat endpoint
@app.post("/api/chat")
async def chat(
    request: ChatRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Main chat endpoint (updated to use ConversationService).
    """
    service = ConversationService()

    # Get or create conversation
    conversation_id = request.conversation_id
    if not conversation_id:
        conversation_id = await service.create_conversation(
            user_email=current_user.email
        )

    # Save user message
    await service.save_message(
        conversation_id=conversation_id,
        role="user",
        content=request.message
    )

    # Get context from cache/DB
    history = await service.get_recent_messages(
        conversation_id=conversation_id,
        limit=20
    )

    # Your existing agent logic
    agent_response = await run_agent(
        query=request.message,
        history=history,
        tools=["web_search", "clarification"]
    )

    # Save assistant response
    await service.save_message(
        conversation_id=conversation_id,
        role="assistant",
        content=agent_response["content"],
        tokens_used=agent_response.get("tokens"),
        latency_ms=agent_response.get("latency")
    )

    return {
        "response": agent_response["content"],
        "conversation_id": conversation_id
    }
```

---

## Performance Optimization

### Cache Hit Rate Monitoring

```python
import logging

logger = logging.getLogger(__name__)

async def get_recent_messages(self, conversation_id: str, limit: int = 20):
    cache_key = self._get_cache_key(conversation_id)
    cached = await self.redis.lrange(cache_key, 0, limit - 1)

    if cached:
        logger.info(f"Cache HIT for conversation {conversation_id}")
        return [json.loads(msg) for msg in cached]

    logger.info(f"Cache MISS for conversation {conversation_id}")
    # Load from DB and populate cache...
```

**Target Metrics:**
- Cache hit rate: >80%
- Average latency (cache hit): <10ms
- Average latency (cache miss): <100ms

### Database Query Optimization

#### Use Indexes

```prisma
model Message {
  // Composite index for common query
  @@index([conversation_id, created_at])

  // Individual indexes
  @@index([conversation_id])
  @@index([created_at])
}
```

#### Limit Query Results

```python
# Always use LIMIT/TAKE
messages = await prisma.message.find_many(
    where={"conversation_id": conversation_id},
    take=20  # Never fetch all messages
)
```

#### Use SELECT to Reduce Data Transfer

```python
# If you only need certain fields
messages = await prisma.message.find_many(
    where={"conversation_id": conversation_id},
    select={
        "id": True,
        "role": True,
        "content": True,
        "created_at": True
        # Don't fetch tokens_used, latency_ms if not needed
    }
)
```

### Redis Optimization

#### Pipeline Multiple Commands

```python
# Instead of multiple awaits
await redis.lpush(key, msg)
await redis.ltrim(key, 0, 19)
await redis.expire(key, 3600)

# Use pipeline for atomic operations
pipe = redis.pipeline()
pipe.lpush(key, msg)
pipe.ltrim(key, 0, 19)
pipe.expire(key, 3600)
await pipe.execute()
```

#### Set Appropriate TTLs

```python
# Short TTL for active conversations (being accessed frequently)
ACTIVE_TTL = 3600  # 1 hour

# Longer TTL for recent but idle conversations
IDLE_TTL = 7200  # 2 hours

# Adaptive TTL based on access pattern
async def set_cache_with_adaptive_ttl(self, key: str, value: str):
    last_access = await self.redis.get(f"{key}:last_access")
    if last_access:
        # Recently accessed - shorter TTL (will be refreshed)
        ttl = 1800
    else:
        # First access - longer TTL
        ttl = 3600

    await self.redis.setex(key, ttl, value)
    await self.redis.set(f"{key}:last_access", time.time(), ex=ttl)
```

### Connection Pooling

```python
# Redis connection pool (already handled by redis.asyncio)
redis_client = redis.from_url(
    Config.REDIS_URL,
    decode_responses=True,
    max_connections=50  # Adjust based on load
)

# Prisma connection pool (configured in schema.prisma)
datasource db {
  provider = "postgresql"
  url      = env("DATABASE_URL")
  connection_limit = 20  # Adjust based on load
}
```

---

## Best Practices

### 1. Error Handling

```python
async def save_message(self, conversation_id: str, role: str, content: str):
    try:
        # Save to PostgreSQL (critical)
        message = await self.prisma_client.message.create(...)
    except Exception as e:
        logger.error(f"Failed to save message to DB: {e}")
        raise  # Re-raise DB errors (critical)

    try:
        # Update Redis cache (non-critical)
        await self.redis.lpush(...)
    except Exception as e:
        logger.warning(f"Failed to update cache: {e}")
        # Don't raise - cache failure shouldn't fail the request

    return message.model_dump()
```

### 2. Data Consistency

```python
# Always save to DB first (source of truth)
# Then update cache

# ✅ Correct order
message = await db.save(message)  # 1. DB first
await cache.update(message)       # 2. Cache second

# ❌ Wrong order
await cache.update(message)       # If this succeeds but DB fails,
message = await db.save(message)  # cache has stale data
```

### 3. Cache as Disposable

```python
# Never rely on cache for critical data
# Cache should always be rebuildable from DB

async def get_messages(self, conversation_id: str):
    # Try cache
    cached = await self.redis.get(key)
    if cached:
        return cached

    # Cache miss - rebuild from DB
    messages = await self.db.get_messages(conversation_id)
    await self.redis.set(key, messages)
    return messages
```

### 4. Graceful Degradation

```python
async def get_recent_messages(self, conversation_id: str):
    try:
        # Try cache first
        cached = await self.redis.lrange(...)
        if cached:
            return [json.loads(msg) for msg in cached]
    except Exception as e:
        logger.warning(f"Redis error, falling back to DB: {e}")
        # Fall through to DB query

    # Fallback to DB
    messages = await self.prisma_client.message.find_many(...)
    return [m.model_dump() for m in messages]
```

### 5. Logging and Monitoring

```python
import time
import logging

logger = logging.getLogger(__name__)

async def save_message(self, *args, **kwargs):
    start = time.time()

    try:
        result = await self._save_message_impl(*args, **kwargs)

        latency = (time.time() - start) * 1000
        logger.info(f"save_message completed in {latency:.2f}ms")

        return result
    except Exception as e:
        logger.error(f"save_message failed: {e}", exc_info=True)
        raise
```

### 6. Testing Strategy

```python
# Unit tests with mocked dependencies
@pytest.mark.asyncio
async def test_save_message():
    # Mock Prisma and Redis
    mock_prisma = MagicMock()
    mock_redis = MagicMock()

    service = ConversationService()
    service.prisma_client = mock_prisma
    service.redis = mock_redis

    # Test
    result = await service.save_message(...)

    # Assertions
    mock_prisma.message.create.assert_called_once()
    mock_redis.lpush.assert_called_once()

# Integration tests with real Redis + PostgreSQL
@pytest.mark.asyncio
async def test_save_and_retrieve_message_integration():
    service = ConversationService()

    # Create conversation
    conv_id = await service.create_conversation("test@example.com")

    # Save message
    msg = await service.save_message(conv_id, "user", "Hello")

    # Retrieve from cache
    messages = await service.get_recent_messages(conv_id)
    assert len(messages) == 1
    assert messages[0]["content"] == "Hello"
```

---

## Implementation Checklist

### Phase 1: Setup Infrastructure ✓

- [x] PostgreSQL running (docker-compose)
- [x] Redis running (docker-compose)
- [x] Prisma schema defined
- [x] Prisma client generated (`prisma generate`)
- [x] Database migrated (`prisma db push`)

### Phase 2: Core Service Implementation

- [ ] Update `ConversationService` with LIST-based caching
  - [ ] Implement `save_message` with LPUSH/LTRIM
  - [ ] Implement `get_recent_messages` with LRANGE
  - [ ] Implement `get_full_history` with pagination
  - [ ] Add error handling and logging

### Phase 3: Find and Replace SESSIONS

- [ ] Search codebase for `SESSIONS` usage
  ```bash
  grep -rn "SESSIONS" backend/src/
  ```
- [ ] Create mapping document:
  - [ ] List all files using SESSIONS
  - [ ] Document current usage patterns
  - [ ] Plan replacement strategy

### Phase 4: Migrate Endpoints

- [ ] Create new conversation endpoints
  - [ ] `POST /api/conversations` - Create
  - [ ] `POST /api/conversations/messages` - Send message
  - [ ] `GET /api/conversations/{id}/messages` - Get history
  - [ ] `DELETE /api/conversations/{id}` - Delete

- [ ] Update existing chat endpoint
  - [ ] Replace SESSIONS with ConversationService
  - [ ] Handle conversation_id in request/response
  - [ ] Update frontend integration

### Phase 5: Testing

- [ ] Unit tests
  - [ ] `test_create_conversation`
  - [ ] `test_save_message`
  - [ ] `test_get_recent_messages_cache_hit`
  - [ ] `test_get_recent_messages_cache_miss`
  - [ ] `test_delete_conversation`

- [ ] Integration tests
  - [ ] End-to-end message flow
  - [ ] Cache invalidation
  - [ ] Pagination

- [ ] Load testing
  - [ ] Concurrent message sending
  - [ ] Cache hit rate under load
  - [ ] Database query performance

### Phase 6: Monitoring and Optimization

- [ ] Add logging
  - [ ] Cache hit/miss rates
  - [ ] Query latencies
  - [ ] Error rates

- [ ] Add metrics
  - [ ] Prometheus/Grafana integration
  - [ ] Track message volume
  - [ ] Track cache performance

- [ ] Optimize based on metrics
  - [ ] Adjust cache TTL
  - [ ] Adjust cache size limit
  - [ ] Add database indexes if needed

### Phase 7: Deployment

- [ ] Update environment variables
  - [ ] `DATABASE_URL`
  - [ ] `REDIS_URL`

- [ ] Run database migrations
  ```bash
  prisma migrate deploy
  ```

- [ ] Deploy backend with new code

- [ ] Update frontend
  - [ ] Handle `conversation_id` in requests
  - [ ] Display conversation history

- [ ] Monitor production
  - [ ] Check error logs
  - [ ] Verify cache hit rates
  - [ ] Monitor database load

---

## Next Steps

1. **Update ConversationService** with LIST-based caching (detailed above)
2. **Find SESSIONS usage** in your codebase
3. **Create API endpoints** for conversations
4. **Test locally** with Redis + PostgreSQL
5. **Migrate incrementally** - replace one endpoint at a time
6. **Deploy and monitor**

### Recommended Implementation Order

```
Week 1:
├─ Update ConversationService with LIST caching
├─ Write unit tests
└─ Test with local Redis + PostgreSQL

Week 2:
├─ Find all SESSIONS usage
├─ Create new API endpoints
└─ Write integration tests

Week 3:
├─ Migrate chat endpoint
├─ Update frontend
└─ End-to-end testing

Week 4:
├─ Add monitoring/logging
├─ Load testing
└─ Deploy to production
```

---

## Questions and Support

**Common Issues:**

1. **Redis connection errors**
   - Check Redis is running: `docker ps`
   - Verify REDIS_URL in .env
   - Test connection: `redis-cli ping`

2. **Prisma connection errors**
   - Check PostgreSQL is running
   - Verify DATABASE_URL in .env
   - Run migrations: `prisma db push`

3. **Cache not working**
   - Check Redis commands are awaited
   - Verify async Redis client (`redis.asyncio`)
   - Check TTL isn't too short

4. **Performance issues**
   - Monitor cache hit rate (should be >80%)
   - Add database indexes
   - Use connection pooling

---

**End of Guide**

This comprehensive guide should provide everything needed to implement industrial-grade conversation history with Redis caching and PostgreSQL persistence.