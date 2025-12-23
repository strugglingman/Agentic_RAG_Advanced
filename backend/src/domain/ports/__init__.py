"""
PORTS - Interfaces that infrastructure implements

A "port" is an abstract interface that defines WHAT the domain needs,
without specifying HOW it's done.

Think of it as a contract:
- Domain says: "I need to save conversations"
- Infrastructure implements: "I'll use PostgreSQL"

Subfolders:
- repositories/  → Data persistence interfaces
- (root files)   → Other external service interfaces

Files to create:
- repositories/conversation_repository.py
- repositories/message_repository.py
- repositories/user_repository.py
- repositories/file_repository.py
- vector_store.py   → ChromaDB interface
- cache.py          → Redis interface
- llm_client.py     → OpenAI interface
- web_search.py     → Search interface
"""
