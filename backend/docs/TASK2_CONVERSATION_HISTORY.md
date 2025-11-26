# Task 2: Conversation History (Redis + PostgreSQL)

**Goal**: Add persistent multi-turn conversation support with session management

**Time Estimate**: 3-4 hours
**Difficulty**: Medium-High
**Dependencies**: Redis, PostgreSQL, observability completed

---

## 📋 Overview

### Current State (Stateless)
```
User: "What is our vacation policy?"
Agent: [Searches docs] → "You get 15 days PTO"

User: "How do I request it?"
Agent: ❌ No context → Doesn't know "it" refers to vacation
```

### After Implementation (Stateful)
```
User: "What is our vacation policy?"
Agent: [Searches docs] → "You get 15 days PTO"
      ↓ [Saves to history]

User: "How do I request it?"
Agent: [Loads history] → Knows "it" = vacation
      → "Submit request in HR portal"
```

---

## 🎯 Architecture

### Two-Tier Storage

```
┌─────────────────────────────────────────────────────┐
│                    User Request                     │
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│              Session Management (Redis)             │
│  • Fast in-memory cache (< 5ms lookup)             │
│  • Stores: session_id → user_id, metadata          │
│  • TTL: 30 minutes (auto-expires inactive sessions) │
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│          Conversation History (PostgreSQL)          │
│  • Persistent storage (survives restarts)           │
│  • Stores: messages, contexts, metadata             │
│  • Never expires (user can review old chats)        │
└─────────────────────────────────────────────────────┘
```

### Why Both?

| Feature | Redis | PostgreSQL |
|---------|-------|------------|
| Speed | ⚡ 1-5ms | 🐢 10-50ms |
| Persistence | ❌ In-memory | ✅ Disk-based |
| Use Case | Active sessions | Long-term storage |
| TTL | 30 min auto-expire | Permanent |

---

## 📊 Database Schema

### PostgreSQL Tables

#### 1. `conversations` (session metadata)
```sql
CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) NOT NULL,
    dept_id VARCHAR(100),
    title VARCHAR(500),              -- Auto-generated from first message
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB                    -- Store: model, config, etc.
);
```

#### 2. `messages` (chat history)
```sql
CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL,        -- 'user' | 'assistant' | 'system'
    content TEXT NOT NULL,
    tokens_used INTEGER,
    latency_ms INTEGER,
    created_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB                    -- Store: tool_calls, contexts, reflection
);
```

#### 3. `message_contexts` (retrieved documents)
```sql
CREATE TABLE message_contexts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    message_id UUID REFERENCES messages(id) ON DELETE CASCADE,
    doc_id VARCHAR(255),
    content TEXT,
    score FLOAT,
    metadata JSONB
);
```

### Redis Keys

```
# Session tracking
session:{session_id} → { user_id, dept_id, created_at, last_active }
TTL: 30 minutes

# Recent messages cache (for fast retrieval)
history:{conversation_id}:recent → [last 10 messages]
TTL: 30 minutes

# User active sessions
user:{user_id}:sessions → SET of active session_ids
TTL: 30 minutes
```

---

## 🔄 Flow Diagram

### Message Processing Flow

```
1. User sends message
   ↓
2. Check Redis for session
   ├─ Found? → Load from Redis (5ms)
   └─ Not found? → Load from PostgreSQL (50ms) → Cache in Redis
   ↓
3. Retrieve conversation history
   ├─ Last 6 messages (configurable via MAX_HISTORY)
   └─ Format: [{"role": "user", "content": "..."}, ...]
   ↓
4. Agent processes with history context
   ├─ LLM receives full history
   └─ Agent can reference previous turns
   ↓
5. Save to both storages
   ├─ PostgreSQL: Persistent save (async, 50ms)
   └─ Redis: Update cache (sync, 5ms)
   ↓
6. Return response to user
```

---

## 🛠️ Implementation Plan

### Phase 1: Database Setup (30 min)

**Files to create**:
- `backend/src/db/postgres.py` - PostgreSQL connection pool
- `backend/src/db/redis_client.py` - Redis connection
- `backend/migrations/001_create_tables.sql` - Table creation

**Tasks**:
1. Add dependencies to `requirements.txt`:
   ```
   psycopg2-binary>=2.9.0
   redis>=5.0.0
   ```
2. Create PostgreSQL tables
3. Initialize Redis connection
4. Add connection configs to `.env` and `settings.py`

---

### Phase 2: Data Models (30 min)

**Files to create**:
- `backend/src/models/conversation.py` - Pydantic models

**Models needed**:
```python
class Message:
    id: UUID
    conversation_id: UUID
    role: str  # 'user' | 'assistant'
    content: str
    tokens_used: Optional[int]
    latency_ms: Optional[int]
    created_at: datetime
    metadata: dict

class Conversation:
    id: UUID
    user_id: str
    dept_id: Optional[str]
    title: Optional[str]
    created_at: datetime
    updated_at: datetime
    messages: List[Message]
```

---

### Phase 3: Storage Services (60 min)

**Files to create**:
- `backend/src/services/conversation_service.py` - Core logic

**Key functions**:
```python
# Session Management
create_session(user_id, dept_id) → session_id
get_session(session_id) → Session | None
extend_session(session_id) → None  # Reset TTL

# Message Operations
save_message(conversation_id, role, content, metadata) → Message
get_history(conversation_id, limit=6) → List[Message]

# Conversation Operations
create_conversation(user_id, dept_id) → Conversation
get_conversation(conversation_id) → Conversation
list_user_conversations(user_id, limit=20) → List[Conversation]
delete_conversation(conversation_id) → None
```

---

### Phase 4: Integration (60 min)

**Files to modify**:
1. `backend/src/services/agent_service.py`
   - Add history loading before agent execution
   - Save message after response

2. `backend/src/routes/chat.py`
   - Add `conversation_id` to request/response
   - Handle session management

3. `backend/src/routes/conversations.py` (NEW)
   - `GET /conversations` - List user conversations
   - `GET /conversations/{id}` - Get full conversation
   - `DELETE /conversations/{id}` - Delete conversation

---

### Phase 5: Testing (30 min)

**Files to create**:
- `backend/tests/test_conversation_service.py`

**Test scenarios**:
```python
# Test 1: Create and retrieve conversation
def test_conversation_lifecycle():
    conv = create_conversation(user_id="test_user")
    save_message(conv.id, "user", "Hello")
    save_message(conv.id, "assistant", "Hi there")
    history = get_history(conv.id)
    assert len(history) == 2

# Test 2: Session expiration
def test_session_expires():
    session = create_session("test_user")
    # Wait 31 minutes (mock)
    assert get_session(session) is None

# Test 3: History limit (MAX_HISTORY)
def test_history_limit():
    conv = create_conversation("test_user")
    # Save 10 messages
    for i in range(10):
        save_message(conv.id, "user", f"Message {i}")
    history = get_history(conv.id, limit=6)
    assert len(history) == 6  # Only last 6
```

---

## ⚙️ Configuration

### `.env` additions
```bash
# PostgreSQL
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=chatbot
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_DB=0

# Session Settings
SESSION_TTL_MINUTES=30
MAX_HISTORY_MESSAGES=6
```

### `settings.py` additions
```python
# Database Settings
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "chatbot")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

# Session Configuration
SESSION_TTL_MINUTES = int(os.getenv("SESSION_TTL_MINUTES", "30"))
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "6"))
```

---

## 🧪 Testing Strategy

### Local Setup

**1. Start PostgreSQL** (Docker):
```bash
docker run -d \
  --name chatbot-postgres \
  -e POSTGRES_DB=chatbot \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=test123 \
  -p 5432:5432 \
  postgres:15
```

**2. Start Redis** (Docker):
```bash
docker run -d \
  --name chatbot-redis \
  -p 6379:6379 \
  redis:7
```

**3. Run Migrations**:
```bash
psql -h localhost -U postgres -d chatbot -f backend/migrations/001_create_tables.sql
```

**4. Test Connections**:
```bash
python -c "from src.db.postgres import get_db; print('PostgreSQL OK')"
python -c "from src.db.redis_client import get_redis; print('Redis OK')"
```

---

## ✅ Success Criteria

### Must Have
- [ ] Sessions stored in Redis with 30-min TTL
- [ ] Messages persisted in PostgreSQL
- [ ] Multi-turn conversations work (agent remembers context)
- [ ] History limited to last 6 messages (MAX_HISTORY)
- [ ] API endpoint: `GET /conversations` returns user's chats
- [ ] API endpoint: `GET /conversations/{id}` returns full history
- [ ] LangSmith traces show conversation_id in metadata

### Should Have
- [ ] Auto-generate conversation titles from first message
- [ ] Store retrieved contexts per message
- [ ] Store tokens/latency per message
- [ ] Handle session expiration gracefully

### Nice to Have
- [ ] Search conversations by content
- [ ] Export conversation as JSON/CSV
- [ ] Conversation branching (multiple threads)

---

## 🐛 Troubleshooting

### Issue 1: "Connection refused" (PostgreSQL)
**Fix**:
```bash
# Check if PostgreSQL is running
docker ps | grep postgres

# Check logs
docker logs chatbot-postgres
```

### Issue 2: "Redis connection timeout"
**Fix**:
```bash
# Test Redis connection
redis-cli -h localhost -p 6379 ping
# Should return: PONG
```

### Issue 3: Agent doesn't remember previous messages
**Debug**:
```python
# Add logging to agent_service.py
print(f"[DEBUG] Loading history for conversation_id={conversation_id}")
history = get_history(conversation_id)
print(f"[DEBUG] Loaded {len(history)} messages")
```

---

## 📁 File Checklist

### Files to Create
```
backend/
├── src/
│   ├── db/
│   │   ├── __init__.py
│   │   ├── postgres.py              # PostgreSQL connection pool
│   │   └── redis_client.py          # Redis client
│   ├── models/
│   │   └── conversation.py          # Conversation/Message models
│   ├── services/
│   │   └── conversation_service.py  # Core conversation logic
│   └── routes/
│       └── conversations.py         # REST API endpoints
├── migrations/
│   └── 001_create_tables.sql        # Database schema
└── tests/
    └── test_conversation_service.py
```

### Files to Modify
```
backend/
├── .env                             # Add PostgreSQL/Redis config
├── requirements.txt                 # Add psycopg2, redis
├── src/
│   ├── config/
│   │   └── settings.py              # Add database settings
│   ├── services/
│   │   └── agent_service.py         # Integrate history loading
│   └── routes/
│       └── chat.py                  # Add conversation_id handling
```

---

## 💰 Cost Impact

| Component | Cost | Notes |
|-----------|------|-------|
| Redis | Free (local) | $0.10-$1/mo for cloud (Railway/Redis Cloud) |
| PostgreSQL | Free (local) | $5-$10/mo for cloud (Railway/Supabase) |
| Storage | ~1KB/message | 10,000 messages = 10MB |
| Tokens | No change | History uses existing token budget |

---

## 🚀 Next Steps After Implementation

1. **Conversation Analytics**
   - Track average messages per conversation
   - Identify most common queries
   - Measure response quality over time

2. **Advanced Features**
   - Conversation search (full-text)
   - Export/import conversations
   - Share conversations via link

3. **Performance Optimization**
   - Connection pooling (pgBouncer)
   - Redis cluster for scale
   - Archive old conversations to S3

---

## 📚 References

- [PostgreSQL Best Practices](https://wiki.postgresql.org/wiki/Don't_Do_This)
- [Redis Session Management](https://redis.io/docs/manual/programmability/lua-scripts/)
- [LangChain Memory](https://python.langchain.com/docs/modules/memory/)

---

## 📝 Summary

**What you'll build**:
- Persistent conversation history (PostgreSQL)
- Fast session management (Redis)
- Multi-turn context awareness
- REST API for conversation management

**Key benefits**:
- Agent remembers previous messages
- Users can review chat history
- Better context for complex queries
- Production-ready architecture

**Estimated time**: 3-4 hours (includes testing)