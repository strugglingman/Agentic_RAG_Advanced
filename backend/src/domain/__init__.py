"""
DOMAIN LAYER - The Heart of Your Application

This layer contains:
- Entities: Business objects with identity (Conversation, Message, User)
- Value Objects: Immutable types (UserId, ConversationId, DeptId)
- Ports: Interfaces/abstractions that infrastructure implements
- Services: Pure domain logic (no I/O)
- Exceptions: Domain-specific errors

RULES:
1. NO framework imports (no FastAPI, SQLAlchemy, Pydantic, etc.)
2. NO I/O operations (no database, no HTTP, no file system)
3. Only depends on Python stdlib
4. This is where business rules live
"""
