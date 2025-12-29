"""
APPLICATION LAYER - Use Cases & Orchestration

This layer contains:
- commands/  → Write operations (CQRS)
- queries/   → Read operations (CQRS)
- services/  → Complex orchestration (agent, retrieval)
- dto/       → Data Transfer Objects
- common/    → Shared interfaces (Command, Query base classes)

Rules:
- Depends on Domain layer only
- No HTTP/framework code here
- Coordinates entities, repositories, external services
"""