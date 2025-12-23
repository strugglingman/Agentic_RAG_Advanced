# Flask to FastAPI Migration Guide
## Cutting-Edge Clean Architecture (2025-2026)

> **Note**: This guide implements **true Clean Architecture with DDD patterns**, not just a Flask-compatible port. Based on latest research from [PyCon India 2025](https://cfp.in.pycon.org/2025/talk/LHLX8U/), [fastapi-clean-example](https://github.com/ivan-borovets/fastapi-clean-example), and [fast-clean-architecture CLI](https://pypi.org/project/fast-clean-architecture/).

---

## Executive Summary

### What We're Building

| Aspect | Current (Flask) | Target (FastAPI Clean Architecture) |
|--------|-----------------|-------------------------------------|
| Architecture | Flat structure with routes/services | **4-Layer Clean Architecture with DDD** |
| DI Pattern | Flask `g` + manual injection | **Dishka DI Container** (framework-agnostic) |
| Data Flow | Mixed CRUD | **CQRS (Command/Query Separation)** |
| Validation | Flask request + manual | **Pydantic v2 with domain validation** |
| Async | `asyncio.run()` workarounds | **Native async throughout** |
| Testing | Coupled to Flask | **Framework-agnostic, highly testable** |

### Why Clean Architecture?

Based on [Clean Architecture principles](https://www.glukhov.org/post/2025/11/python-design-patterns-for-clean-architecture/):

> "Business logic should not depend on infrastructure. Dependencies point inward toward core business rules."

This ensures:
- **Testability**: Test business logic without databases/HTTP
- **Flexibility**: Swap databases, frameworks, or external services easily
- **Maintainability**: Clear boundaries between concerns
- **Scalability**: Independent scaling of read/write operations (CQRS)

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Project Structure](#2-project-structure)
3. [Layer Responsibilities](#3-layer-responsibilities)
4. [Dependency Injection with Dishka](#4-dependency-injection-with-dishka)
5. [CQRS Implementation](#5-cqrs-implementation)
6. [Migration Mapping](#6-migration-mapping)
7. [Implementation Details](#7-implementation-details)
8. [Technology Stack](#8-technology-stack)
9. [Migration Phases](#9-migration-phases)
10. [References](#10-references)

---

## 1. Architecture Overview

### 1.1 The Four Layers

Based on [fastapi-clean-example](https://github.com/ivan-borovets/fastapi-clean-example):

```
┌─────────────────────────────────────────────────────────────────────┐
│                      PRESENTATION LAYER                              │
│  (FastAPI Controllers, HTTP validation, routing, OpenAPI)           │
│  Dependencies: Application Layer                                     │
├─────────────────────────────────────────────────────────────────────┤
│                      APPLICATION LAYER                               │
│  (Use Cases, Commands, Queries, DTOs, Application Services)         │
│  Dependencies: Domain Layer                                          │
├─────────────────────────────────────────────────────────────────────┤
│                        DOMAIN LAYER                                  │
│  (Entities, Value Objects, Domain Services, Ports/Interfaces)       │
│  Dependencies: NONE (Pure Python, no framework code)                │
├─────────────────────────────────────────────────────────────────────┤
│                     INFRASTRUCTURE LAYER                             │
│  (Repositories, External APIs, Database, Redis, ChromaDB)           │
│  Dependencies: Domain Layer (implements Ports)                       │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 Dependency Rule

**Critical**: Dependencies ONLY point inward. Domain layer has ZERO external dependencies.

```
Presentation → Application → Domain ← Infrastructure
                              ↑______________|
                         (implements ports)
```

### 1.3 CQRS Pattern

Based on [PyCon Greece 2025 presentation](https://dev.to/markoulis/how-i-learned-to-stop-worrying-and-love-raw-events-event-sourcing-cqrs-with-fastapi-and-celery-477e):

```
                    ┌─────────────────┐
                    │  HTTP Request   │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
     ┌────────────────┐           ┌────────────────┐
     │    COMMAND     │           │     QUERY      │
     │   (Write Op)   │           │   (Read Op)    │
     └────────┬───────┘           └────────┬───────┘
              │                             │
              ▼                             ▼
     ┌────────────────┐           ┌────────────────┐
     │ Command Handler│           │ Query Handler  │
     │  (Use Case)    │           │ (Read Model)   │
     └────────┬───────┘           └────────┬───────┘
              │                             │
              ▼                             ▼
     ┌────────────────┐           ┌────────────────┐
     │   Repository   │           │  Query Service │
     │  (Full Entity) │           │ (Optimized DTO)│
     └────────────────┘           └────────────────┘
```

---

## 2. Project Structure

### 2.1 Clean Architecture Structure

Based on [fast-clean-architecture CLI](https://pypi.org/project/fast-clean-architecture/) and [Architecture Patterns with Python](https://www.cosmicpython.com/book/chapter_02_repository.html):

```
backend/
├── src/
│   ├── __init__.py
│   │
│   ├── domain/                           # 🔵 DOMAIN LAYER (Pure Python)
│   │   ├── __init__.py
│   │   │
│   │   ├── entities/                     # Business objects with identity
│   │   │   ├── __init__.py
│   │   │   ├── user.py                   # User entity
│   │   │   ├── conversation.py           # Conversation entity
│   │   │   ├── message.py                # Message entity
│   │   │   └── file_registry.py          # FileRegistry entity
│   │   │
│   │   ├── value_objects/                # Immutable domain types
│   │   │   ├── __init__.py
│   │   │   ├── user_id.py                # Email-based ID
│   │   │   ├── dept_id.py                # Department ID
│   │   │   ├── conversation_id.py        # UUID wrapper
│   │   │   └── file_path.py              # Validated file path
│   │   │
│   │   ├── services/                     # Domain-level operations
│   │   │   ├── __init__.py
│   │   │   ├── retrieval_evaluator.py    # Self-reflection logic
│   │   │   └── query_refiner.py          # Query optimization logic
│   │   │
│   │   ├── ports/                        # Interfaces (abstractions)
│   │   │   ├── __init__.py
│   │   │   ├── repositories/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── conversation_repository.py
│   │   │   │   ├── message_repository.py
│   │   │   │   ├── file_repository.py
│   │   │   │   └── user_repository.py
│   │   │   ├── vector_store.py           # ChromaDB interface
│   │   │   ├── cache.py                  # Redis interface
│   │   │   ├── llm_client.py             # OpenAI interface
│   │   │   └── web_search.py             # Search interface
│   │   │
│   │   └── exceptions/                   # Domain exceptions
│   │       ├── __init__.py
│   │       ├── entity_not_found.py
│   │       ├── access_denied.py
│   │       └── validation_error.py
│   │
│   ├── application/                      # 🟢 APPLICATION LAYER
│   │   ├── __init__.py
│   │   │
│   │   ├── commands/                     # Write operations (CQRS)
│   │   │   ├── __init__.py
│   │   │   ├── chat/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── send_message.py       # SendMessageCommand + Handler
│   │   │   │   └── create_conversation.py
│   │   │   ├── files/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── upload_file.py
│   │   │   │   └── ingest_document.py
│   │   │   └── conversations/
│   │   │       ├── __init__.py
│   │   │       ├── delete_conversation.py
│   │   │       └── update_title.py
│   │   │
│   │   ├── queries/                      # Read operations (CQRS)
│   │   │   ├── __init__.py
│   │   │   ├── chat/
│   │   │   │   ├── __init__.py
│   │   │   │   └── get_chat_history.py
│   │   │   ├── conversations/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── list_conversations.py
│   │   │   │   └── get_conversation.py
│   │   │   └── files/
│   │   │       ├── __init__.py
│   │   │       ├── list_files.py
│   │   │       └── get_file.py
│   │   │
│   │   ├── services/                     # Application services
│   │   │   ├── __init__.py
│   │   │   ├── agent_service.py          # ReAct agent orchestration
│   │   │   ├── query_supervisor.py       # Routes queries
│   │   │   ├── retrieval_service.py      # RAG retrieval
│   │   │   └── langgraph_service.py      # LangGraph orchestration
│   │   │
│   │   ├── dto/                          # Data Transfer Objects
│   │   │   ├── __init__.py
│   │   │   ├── chat.py                   # ChatRequest, ChatResponse
│   │   │   ├── conversation.py           # ConversationDTO
│   │   │   ├── file.py                   # FileDTO
│   │   │   └── context.py                # RAGContext
│   │   │
│   │   └── common/                       # Shared application code
│   │       ├── __init__.py
│   │       ├── interfaces.py             # Command/Query base classes
│   │       └── unit_of_work.py           # UoW pattern
│   │
│   ├── infrastructure/                   # 🟠 INFRASTRUCTURE LAYER
│   │   ├── __init__.py
│   │   │
│   │   ├── persistence/                  # Database implementations
│   │   │   ├── __init__.py
│   │   │   ├── database.py               # Async SQLAlchemy setup
│   │   │   ├── models/                   # SQLAlchemy ORM models
│   │   │   │   ├── __init__.py
│   │   │   │   ├── user.py
│   │   │   │   ├── conversation.py
│   │   │   │   ├── message.py
│   │   │   │   └── file_registry.py
│   │   │   ├── repositories/             # Repository implementations
│   │   │   │   ├── __init__.py
│   │   │   │   ├── sqlalchemy_conversation_repo.py
│   │   │   │   ├── sqlalchemy_message_repo.py
│   │   │   │   ├── sqlalchemy_file_repo.py
│   │   │   │   └── sqlalchemy_user_repo.py
│   │   │   └── unit_of_work.py           # SQLAlchemy UoW
│   │   │
│   │   ├── cache/                        # Redis implementation
│   │   │   ├── __init__.py
│   │   │   └── redis_cache.py
│   │   │
│   │   ├── vector_store/                 # ChromaDB implementation
│   │   │   ├── __init__.py
│   │   │   └── chromadb_store.py
│   │   │
│   │   ├── llm/                          # LLM client implementations
│   │   │   ├── __init__.py
│   │   │   ├── openai_client.py
│   │   │   └── langchain_client.py
│   │   │
│   │   ├── external/                     # External service adapters
│   │   │   ├── __init__.py
│   │   │   ├── web_search/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── duckduckgo.py
│   │   │   │   └── tavily.py
│   │   │   └── email/
│   │   │       ├── __init__.py
│   │   │       └── smtp_client.py
│   │   │
│   │   └── auth/                         # Authentication implementation
│   │       ├── __init__.py
│   │       ├── jwt_handler.py
│   │       └── password_hasher.py
│   │
│   ├── presentation/                     # 🔴 PRESENTATION LAYER
│   │   ├── __init__.py
│   │   │
│   │   ├── http/                         # HTTP/REST interface
│   │   │   ├── __init__.py
│   │   │   │
│   │   │   ├── api/                      # API versioning
│   │   │   │   ├── __init__.py
│   │   │   │   ├── v1/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── router.py         # v1 router aggregator
│   │   │   │   │   ├── chat.py           # Chat endpoints
│   │   │   │   │   ├── conversations.py  # Conversation endpoints
│   │   │   │   │   ├── files.py          # File endpoints
│   │   │   │   │   ├── upload.py         # Upload endpoints
│   │   │   │   │   ├── ingest.py         # Ingest endpoints
│   │   │   │   │   └── org.py            # Org structure
│   │   │   │   └── health.py             # Health checks
│   │   │   │
│   │   │   ├── schemas/                  # Pydantic request/response
│   │   │   │   ├── __init__.py
│   │   │   │   ├── chat.py
│   │   │   │   ├── conversation.py
│   │   │   │   ├── file.py
│   │   │   │   └── common.py
│   │   │   │
│   │   │   ├── middleware/               # HTTP middleware
│   │   │   │   ├── __init__.py
│   │   │   │   ├── correlation_id.py
│   │   │   │   ├── error_handler.py
│   │   │   │   └── logging.py
│   │   │   │
│   │   │   └── dependencies/             # FastAPI dependencies (thin)
│   │   │       ├── __init__.py
│   │   │       └── auth.py               # Extract user from request
│   │   │
│   │   └── streaming/                    # SSE/Streaming
│   │       ├── __init__.py
│   │       └── sse_response.py
│   │
│   └── setup/                            # 🟣 APPLICATION BOOTSTRAP
│       ├── __init__.py
│       ├── app_factory.py                # FastAPI app creation
│       ├── config.py                     # Pydantic Settings
│       ├── logging.py                    # Logging configuration
│       │
│       └── ioc/                          # Dependency Injection (Dishka)
│           ├── __init__.py
│           ├── container.py              # Main DI container
│           ├── providers/
│           │   ├── __init__.py
│           │   ├── database.py           # DB session provider
│           │   ├── cache.py              # Redis provider
│           │   ├── vector_store.py       # ChromaDB provider
│           │   ├── llm.py                # LLM client provider
│           │   ├── repositories.py       # Repository providers
│           │   ├── services.py           # Service providers
│           │   └── use_cases.py          # Command/Query providers
│           └── scopes.py                 # Custom DI scopes
│
├── tests/                                # Test suite
│   ├── __init__.py
│   ├── conftest.py                       # Pytest fixtures
│   │
│   ├── unit/                             # Unit tests (no I/O)
│   │   ├── domain/
│   │   │   ├── test_entities.py
│   │   │   └── test_value_objects.py
│   │   └── application/
│   │       ├── test_commands.py
│   │       └── test_queries.py
│   │
│   ├── integration/                      # Integration tests
│   │   ├── test_repositories.py
│   │   └── test_api.py
│   │
│   └── e2e/                              # End-to-end tests
│       └── test_chat_flow.py
│
├── prompts/                              # LLM prompts (moved outside src)
│   ├── __init__.py
│   ├── planning.py
│   ├── generation.py
│   ├── evaluation.py
│   └── refinement.py
│
├── alembic/                              # Database migrations
│   ├── env.py
│   └── versions/
│
├── scripts/                              # Utility scripts
│   └── seed_data.py
│
├── .env                                  # Environment variables
├── .env.example
├── pyproject.toml                        # Modern Python config
├── Dockerfile
├── docker-compose.yml
└── run.py                                # Entry point
```

### 2.2 Key Structural Differences

| Current (Flask) | New (Clean Architecture) | Why |
|-----------------|--------------------------|-----|
| `src/routes/chat.py` | `src/presentation/http/api/v1/chat.py` | Clear layer separation |
| `src/services/agent_service.py` | `src/application/services/agent_service.py` | Application layer |
| Business logic in routes | `src/application/commands/` | CQRS pattern |
| `src/config/settings.py` | `src/setup/config.py` | Bootstrap module |
| No repository layer | `src/domain/ports/` + `src/infrastructure/persistence/repositories/` | Repository pattern |
| Flask `g` object | `src/setup/ioc/` (Dishka) | Framework-agnostic DI |

---

## 3. Layer Responsibilities

### 3.1 Domain Layer (Pure Python)

**NO FRAMEWORK DEPENDENCIES**. This is the heart of your application.

```python
# src/domain/entities/conversation.py
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from src.domain.value_objects.conversation_id import ConversationId
from src.domain.value_objects.user_id import UserId


@dataclass
class Conversation:
    """
    Conversation entity - pure domain object.
    No SQLAlchemy, no Pydantic, no framework code.
    """
    id: ConversationId
    user_id: UserId
    title: Optional[str]
    created_at: datetime
    updated_at: datetime

    def update_title(self, new_title: str) -> None:
        """Domain behavior - business rule."""
        if len(new_title) > 200:
            raise ValueError("Title cannot exceed 200 characters")
        self.title = new_title
        self.updated_at = datetime.utcnow()
```

```python
# src/domain/ports/repositories/conversation_repository.py
from abc import ABC, abstractmethod
from typing import Optional, List

from src.domain.entities.conversation import Conversation
from src.domain.value_objects.conversation_id import ConversationId
from src.domain.value_objects.user_id import UserId


class ConversationRepository(ABC):
    """
    Port (interface) for conversation persistence.
    Domain layer defines WHAT it needs, not HOW.
    """

    @abstractmethod
    async def get_by_id(self, id: ConversationId) -> Optional[Conversation]:
        ...

    @abstractmethod
    async def get_by_user(self, user_id: UserId) -> List[Conversation]:
        ...

    @abstractmethod
    async def save(self, conversation: Conversation) -> None:
        ...

    @abstractmethod
    async def delete(self, id: ConversationId) -> bool:
        ...
```

### 3.2 Application Layer (Use Cases)

Based on [CQRS with FastAPI](https://wawaziphil.medium.com/building-a-python-api-using-cqrs-a-simple-guide-3d584b6ead34):

```python
# src/application/commands/conversations/delete_conversation.py
from dataclasses import dataclass

from src.domain.ports.repositories.conversation_repository import ConversationRepository
from src.domain.value_objects.conversation_id import ConversationId
from src.domain.value_objects.user_id import UserId
from src.domain.exceptions.entity_not_found import EntityNotFoundError
from src.domain.exceptions.access_denied import AccessDeniedError


@dataclass(frozen=True)
class DeleteConversationCommand:
    """Command to delete a conversation."""
    conversation_id: str
    user_id: str


class DeleteConversationHandler:
    """
    Command handler - orchestrates the use case.
    No HTTP, no framework code - pure business logic.
    """

    def __init__(self, conversation_repo: ConversationRepository):
        self._repo = conversation_repo

    async def handle(self, command: DeleteConversationCommand) -> bool:
        conversation_id = ConversationId(command.conversation_id)
        user_id = UserId(command.user_id)

        conversation = await self._repo.get_by_id(conversation_id)

        if not conversation:
            raise EntityNotFoundError(f"Conversation {conversation_id} not found")

        if conversation.user_id != user_id:
            raise AccessDeniedError("Cannot delete another user's conversation")

        return await self._repo.delete(conversation_id)
```

```python
# src/application/queries/conversations/list_conversations.py
from dataclasses import dataclass
from typing import List

from src.application.dto.conversation import ConversationDTO
from src.domain.ports.repositories.conversation_repository import ConversationRepository
from src.domain.value_objects.user_id import UserId


@dataclass(frozen=True)
class ListConversationsQuery:
    """Query to list user's conversations."""
    user_id: str


class ListConversationsHandler:
    """
    Query handler - optimized for reads.
    Can return DTOs directly, bypassing full entity loading.
    """

    def __init__(self, conversation_repo: ConversationRepository):
        self._repo = conversation_repo

    async def handle(self, query: ListConversationsQuery) -> List[ConversationDTO]:
        user_id = UserId(query.user_id)
        conversations = await self._repo.get_by_user(user_id)

        return [
            ConversationDTO(
                id=str(c.id),
                title=c.title,
                created_at=c.created_at,
                updated_at=c.updated_at,
            )
            for c in conversations
        ]
```

### 3.3 Infrastructure Layer (Implementations)

```python
# src/infrastructure/persistence/repositories/sqlalchemy_conversation_repo.py
from typing import Optional, List

from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from src.domain.entities.conversation import Conversation
from src.domain.ports.repositories.conversation_repository import ConversationRepository
from src.domain.value_objects.conversation_id import ConversationId
from src.domain.value_objects.user_id import UserId
from src.infrastructure.persistence.models.conversation import ConversationModel


class SqlAlchemyConversationRepository(ConversationRepository):
    """
    SQLAlchemy implementation of ConversationRepository.
    Infrastructure implements domain ports.
    """

    def __init__(self, session: AsyncSession):
        self._session = session

    async def get_by_id(self, id: ConversationId) -> Optional[Conversation]:
        stmt = select(ConversationModel).where(ConversationModel.id == str(id))
        result = await self._session.execute(stmt)
        model = result.scalar_one_or_none()

        if not model:
            return None

        return self._to_entity(model)

    async def get_by_user(self, user_id: UserId) -> List[Conversation]:
        stmt = (
            select(ConversationModel)
            .where(ConversationModel.user_email == str(user_id))
            .order_by(ConversationModel.updated_at.desc())
        )
        result = await self._session.execute(stmt)
        return [self._to_entity(m) for m in result.scalars()]

    async def save(self, conversation: Conversation) -> None:
        model = self._to_model(conversation)
        self._session.add(model)
        await self._session.flush()

    async def delete(self, id: ConversationId) -> bool:
        stmt = delete(ConversationModel).where(ConversationModel.id == str(id))
        result = await self._session.execute(stmt)
        return result.rowcount > 0

    def _to_entity(self, model: ConversationModel) -> Conversation:
        return Conversation(
            id=ConversationId(model.id),
            user_id=UserId(model.user_email),
            title=model.title,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )

    def _to_model(self, entity: Conversation) -> ConversationModel:
        return ConversationModel(
            id=str(entity.id),
            user_email=str(entity.user_id),
            title=entity.title,
            created_at=entity.created_at,
            updated_at=entity.updated_at,
        )
```

### 3.4 Presentation Layer (Thin Controllers)

Based on [FastAPI Best Practices](https://github.com/zhanymkanov/fastapi-best-practices):

> "Don't perform any business logic in your endpoints. Leave it to use-case code nicely tucked in the service layer."

```python
# src/presentation/http/api/v1/conversations.py
from fastapi import APIRouter, status
from dishka.integrations.fastapi import DishkaRoute, FromDishka

from src.application.commands.conversations.delete_conversation import (
    DeleteConversationCommand,
    DeleteConversationHandler,
)
from src.application.queries.conversations.list_conversations import (
    ListConversationsQuery,
    ListConversationsHandler,
)
from src.presentation.http.schemas.conversation import (
    ConversationListResponse,
    ConversationResponse,
)
from src.presentation.http.dependencies.auth import get_current_user
from src.domain.value_objects.user_id import UserId


router = APIRouter(
    prefix="/conversations",
    tags=["conversations"],
    route_class=DishkaRoute,  # Dishka integration
)


@router.get("", response_model=ConversationListResponse)
async def list_conversations(
    handler: FromDishka[ListConversationsHandler],
    current_user: UserId = Depends(get_current_user),
):
    """
    List all conversations for current user.
    Controller is THIN - just maps HTTP to use case.
    """
    query = ListConversationsQuery(user_id=str(current_user))
    conversations = await handler.handle(query)
    return ConversationListResponse(conversations=conversations)


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: str,
    handler: FromDishka[DeleteConversationHandler],
    current_user: UserId = Depends(get_current_user),
):
    """Delete a conversation."""
    command = DeleteConversationCommand(
        conversation_id=conversation_id,
        user_id=str(current_user),
    )
    await handler.handle(command)
```

---

## 4. Dependency Injection with Dishka

### 4.1 Why Dishka over FastAPI Depends?

Based on [Dishka documentation](https://dishka.readthedocs.io/en/stable/alternatives.html) and [Better DI in FastAPI](https://vladiliescu.net/better-dependency-injection-in-fastapi/):

| Feature | FastAPI `Depends` | Dishka |
|---------|-------------------|--------|
| Scope management | Request only | App, Request, Custom |
| Singleton support | Manual via `app.state` | Native |
| Framework coupling | Tight | Framework-agnostic |
| Lazy initialization | No | Yes |
| Finalization | Manual | Automatic |
| Complex graphs | Difficult | Easy |
| Testability | Requires override | Simple mock injection |

### 4.2 Dishka Container Setup

```python
# src/setup/ioc/container.py
from dishka import make_async_container
from dishka.integrations.fastapi import setup_dishka

from src.setup.ioc.providers.database import DatabaseProvider
from src.setup.ioc.providers.cache import CacheProvider
from src.setup.ioc.providers.vector_store import VectorStoreProvider
from src.setup.ioc.providers.repositories import RepositoryProvider
from src.setup.ioc.providers.services import ServiceProvider
from src.setup.ioc.providers.use_cases import UseCaseProvider


def create_container():
    """Create the main DI container with all providers."""
    return make_async_container(
        DatabaseProvider(),
        CacheProvider(),
        VectorStoreProvider(),
        RepositoryProvider(),
        ServiceProvider(),
        UseCaseProvider(),
    )


def setup_di(app):
    """Attach DI container to FastAPI app."""
    container = create_container()
    setup_dishka(container, app)
```

```python
# src/setup/ioc/providers/repositories.py
from dishka import Provider, Scope, provide

from src.domain.ports.repositories.conversation_repository import ConversationRepository
from src.infrastructure.persistence.repositories.sqlalchemy_conversation_repo import (
    SqlAlchemyConversationRepository,
)


class RepositoryProvider(Provider):
    """Provider for repository implementations."""

    scope = Scope.REQUEST  # New instance per request

    @provide
    async def conversation_repo(
        self, session: AsyncSession
    ) -> ConversationRepository:
        return SqlAlchemyConversationRepository(session)
```

```python
# src/setup/ioc/providers/use_cases.py
from dishka import Provider, Scope, provide

from src.application.commands.conversations.delete_conversation import (
    DeleteConversationHandler,
)
from src.application.queries.conversations.list_conversations import (
    ListConversationsHandler,
)
from src.domain.ports.repositories.conversation_repository import ConversationRepository


class UseCaseProvider(Provider):
    """Provider for command/query handlers."""

    scope = Scope.REQUEST

    @provide
    async def delete_conversation_handler(
        self, repo: ConversationRepository
    ) -> DeleteConversationHandler:
        return DeleteConversationHandler(repo)

    @provide
    async def list_conversations_handler(
        self, repo: ConversationRepository
    ) -> ListConversationsHandler:
        return ListConversationsHandler(repo)
```

---

## 5. CQRS Implementation

### 5.1 Command/Query Base Classes

```python
# src/application/common/interfaces.py
from abc import ABC, abstractmethod
from typing import TypeVar, Generic

TCommand = TypeVar("TCommand")
TQuery = TypeVar("TQuery")
TResult = TypeVar("TResult")


class CommandHandler(ABC, Generic[TCommand, TResult]):
    """Base class for command handlers (write operations)."""

    @abstractmethod
    async def handle(self, command: TCommand) -> TResult:
        ...


class QueryHandler(ABC, Generic[TQuery, TResult]):
    """Base class for query handlers (read operations)."""

    @abstractmethod
    async def handle(self, query: TQuery) -> TResult:
        ...
```

### 5.2 Chat Command with Streaming

```python
# src/application/commands/chat/send_message.py
from dataclasses import dataclass
from typing import AsyncGenerator, List, Optional

from src.application.dto.chat import MessageDTO, AttachmentDTO, FilterDTO
from src.application.services.query_supervisor import QuerySupervisor
from src.domain.ports.repositories.conversation_repository import ConversationRepository
from src.domain.value_objects.user_id import UserId
from src.domain.value_objects.conversation_id import ConversationId


@dataclass(frozen=True)
class SendMessageCommand:
    """Command to send a chat message with RAG retrieval."""
    messages: List[MessageDTO]
    user_id: str
    dept_id: str
    conversation_id: Optional[str] = None
    filters: Optional[List[FilterDTO]] = None
    attachments: Optional[List[AttachmentDTO]] = None


class SendMessageHandler:
    """
    Handles chat message with streaming response.
    This is a command because it writes (saves message + updates conversation).
    """

    def __init__(
        self,
        query_supervisor: QuerySupervisor,
        conversation_repo: ConversationRepository,
    ):
        self._supervisor = query_supervisor
        self._conversation_repo = conversation_repo

    async def handle(self, command: SendMessageCommand) -> AsyncGenerator[str, None]:
        """
        Returns async generator for streaming response.
        Caller wraps in StreamingResponse.
        """
        user_id = UserId(command.user_id)
        conversation_id = (
            ConversationId(command.conversation_id)
            if command.conversation_id
            else None
        )

        answer_chunks = []

        try:
            async for chunk in self._supervisor.stream(
                messages=command.messages,
                user_id=user_id,
                dept_id=command.dept_id,
                filters=command.filters,
                attachments=command.attachments,
            ):
                answer_chunks.append(chunk)
                yield chunk

            # Send context marker
            contexts = self._supervisor.get_contexts()
            yield f"\n__CONTEXT__:{contexts}"

        finally:
            # Save assistant message to conversation
            if conversation_id:
                full_answer = "".join(answer_chunks)
                await self._save_message(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=full_answer,
                )

    async def _save_message(
        self,
        conversation_id: ConversationId,
        role: str,
        content: str,
    ) -> None:
        # Implementation
        ...
```

---

## 6. Migration Mapping

### 6.1 Flask → Clean Architecture Mapping

| Current Flask Location | New Clean Architecture Location |
|------------------------|--------------------------------|
| `src/routes/chat.py` | `src/presentation/http/api/v1/chat.py` |
| `src/routes/conversations.py` | `src/presentation/http/api/v1/conversations.py` |
| Route business logic | `src/application/commands/` or `src/application/queries/` |
| `src/services/agent_service.py` | `src/application/services/agent_service.py` |
| `src/services/retrieval.py` | `src/application/services/retrieval_service.py` |
| `src/services/retrieval_evaluator.py` | `src/domain/services/retrieval_evaluator.py` (pure logic) |
| `src/services/conversation_service.py` | `src/infrastructure/persistence/repositories/` |
| `src/config/settings.py` | `src/setup/config.py` |
| `src/config/redis_client.py` | `src/infrastructure/cache/redis_cache.py` |
| `src/middleware/auth.py` | `src/presentation/http/middleware/` + `src/infrastructure/auth/` |
| Flask `g.identity` | Dishka DI with `request.state` |
| `asyncio.run()` workarounds | Native `async/await` throughout |

### 6.2 Pattern Mapping

| Flask Pattern | Clean Architecture Pattern |
|---------------|---------------------------|
| `@bp.post("/chat")` | `@router.post("")` with thin controller |
| `g.identity.get("user_id")` | `FromDishka[Identity]` or `Depends(get_current_user)` |
| `request.get_json()` | Pydantic schema parameter |
| `Response(generate())` | `StreamingResponse(async_generate())` |
| Service with DB calls | Command Handler → Repository Port → SQLAlchemy Implementation |
| Direct Prisma/SQLAlchemy in routes | Repository abstraction in domain layer |

---

## 7. Implementation Details

### 7.1 Streaming Response (SSE)

Based on [sse-starlette](https://github.com/sysid/sse-starlette):

```python
# src/presentation/http/api/v1/chat.py
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from dishka.integrations.fastapi import DishkaRoute, FromDishka

from src.application.commands.chat.send_message import (
    SendMessageCommand,
    SendMessageHandler,
)
from src.presentation.http.schemas.chat import ChatRequest
from src.presentation.http.dependencies.auth import get_current_user


router = APIRouter(prefix="/chat", tags=["chat"], route_class=DishkaRoute)


@router.post("")
async def chat(
    payload: ChatRequest,
    handler: FromDishka[SendMessageHandler],
    current_user: FromDishka[Identity],
):
    """
    Chat endpoint with streaming response.
    Controller is thin - delegates to command handler.
    """
    command = SendMessageCommand(
        messages=payload.messages,
        user_id=current_user.user_id,
        dept_id=current_user.dept_id,
        conversation_id=payload.conversation_id,
        filters=payload.filters,
        attachments=payload.attachments,
    )

    return StreamingResponse(
        handler.handle(command),
        media_type="text/plain; charset=utf-8",
        headers={"X-Accel-Buffering": "no"},  # Disable nginx buffering
    )
```

### 7.2 Error Handling

```python
# src/presentation/http/middleware/error_handler.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.domain.exceptions.entity_not_found import EntityNotFoundError
from src.domain.exceptions.access_denied import AccessDeniedError
from src.domain.exceptions.validation_error import DomainValidationError


def setup_exception_handlers(app: FastAPI):
    """Register domain exception handlers."""

    @app.exception_handler(EntityNotFoundError)
    async def entity_not_found_handler(request: Request, exc: EntityNotFoundError):
        return JSONResponse(
            status_code=404,
            content={"error": str(exc)},
        )

    @app.exception_handler(AccessDeniedError)
    async def access_denied_handler(request: Request, exc: AccessDeniedError):
        return JSONResponse(
            status_code=403,
            content={"error": str(exc)},
        )

    @app.exception_handler(DomainValidationError)
    async def validation_handler(request: Request, exc: DomainValidationError):
        return JSONResponse(
            status_code=422,
            content={"error": str(exc)},
        )
```

### 7.3 Application Factory

```python
# src/setup/app_factory.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.setup.config import get_settings
from src.setup.ioc.container import setup_di
from src.setup.logging import setup_logging
from src.presentation.http.middleware.error_handler import setup_exception_handlers
from src.presentation.http.middleware.correlation_id import CorrelationIdMiddleware
from src.presentation.http.api.v1.router import api_v1_router
from src.presentation.http.api.health import health_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    settings = get_settings()
    setup_logging(settings.log_level)
    # Dishka handles resource lifecycle
    yield


def create_app() -> FastAPI:
    """Application factory - creates and configures FastAPI app."""
    settings = get_settings()

    app = FastAPI(
        title="Agentic RAG API",
        description="Clean Architecture RAG API with LangGraph",
        version="2.0.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.debug else None,
        redoc_url=None,
    )

    # Middleware (order matters - first added = last executed)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(CorrelationIdMiddleware)

    # Dependency Injection
    setup_di(app)

    # Exception Handlers
    setup_exception_handlers(app)

    # Routes
    app.include_router(health_router)
    app.include_router(api_v1_router, prefix="/api/v1")

    return app
```

---

## 8. Technology Stack

### 8.1 Core Dependencies

```txt
# Core Framework
fastapi>=0.115.0
uvicorn[standard]>=0.34.0
starlette>=0.45.0

# Dependency Injection
dishka>=1.4.0

# Pydantic
pydantic>=2.10.0
pydantic-settings>=2.7.0

# Database - Async
sqlalchemy[asyncio]>=2.0.0
asyncpg>=0.30.0
alembic>=1.14.0

# Redis - Async
redis>=5.0.0

# Rate Limiting
slowapi>=0.1.9

# SSE Streaming
sse-starlette>=2.2.0

# Security
PyJWT>=2.10.0
bcrypt>=4.2.0
python-multipart>=0.0.20

# ChromaDB & Embeddings
chromadb>=0.4.0
sentence-transformers>=2.2.2

# LangChain/LangGraph
langchain>=0.3.0
langchain-core>=0.3.0
langchain-openai>=0.2.0
langgraph>=0.2.0

# OpenAI
openai>=1.0.0

# Observability
langsmith>=0.1.0

# Testing
pytest>=8.0.0
pytest-asyncio>=0.24.0
pytest-cov>=6.0.0

# Development
ruff>=0.8.0
mypy>=1.14.0
```

### 8.2 Removed Dependencies

```txt
# Flask ecosystem - completely removed
flask
flask-cors
flask-limiter
werkzeug

# Sync database drivers
psycopg2-binary  # → asyncpg

# Prisma (optional - can keep for migrations)
# prisma  # → SQLAlchemy
```

---

## 9. Migration Phases

### Phase 1: Setup Foundation (Days 1-2)
- [ ] Create new directory structure
- [ ] Set up `pyproject.toml`
- [ ] Create `src/setup/config.py` (Pydantic Settings)
- [ ] Create `src/setup/app_factory.py`
- [ ] Set up Dishka container (`src/setup/ioc/`)
- [ ] Create base classes (`src/application/common/`)

### Phase 2: Domain Layer (Days 3-4)
- [ ] Define entities (`src/domain/entities/`)
- [ ] Define value objects (`src/domain/value_objects/`)
- [ ] Define repository ports (`src/domain/ports/repositories/`)
- [ ] Define other ports (cache, vector store, LLM)
- [ ] Define domain exceptions

### Phase 3: Infrastructure Layer (Days 5-7)
- [ ] Set up async SQLAlchemy (`src/infrastructure/persistence/database.py`)
- [ ] Create ORM models (`src/infrastructure/persistence/models/`)
- [ ] Implement repositories (`src/infrastructure/persistence/repositories/`)
- [ ] Implement Redis cache (`src/infrastructure/cache/`)
- [ ] Implement ChromaDB store (`src/infrastructure/vector_store/`)
- [ ] Set up Alembic migrations

### Phase 4: Application Layer (Days 8-10)
- [ ] Create DTOs (`src/application/dto/`)
- [ ] Create commands for conversations
- [ ] Create commands for chat (with streaming)
- [ ] Create commands for files/upload
- [ ] Create queries for all read operations
- [ ] Migrate agent service
- [ ] Migrate retrieval service

### Phase 5: Presentation Layer (Days 11-13)
- [ ] Create Pydantic schemas (`src/presentation/http/schemas/`)
- [ ] Create thin controllers (`src/presentation/http/api/v1/`)
- [ ] Set up middleware (correlation ID, error handler)
- [ ] Set up auth dependency
- [ ] Configure rate limiting with slowapi

### Phase 6: Testing & Polish (Days 14-16)
- [ ] Unit tests for domain layer
- [ ] Unit tests for application layer (mock repositories)
- [ ] Integration tests for repositories
- [ ] API integration tests
- [ ] Update Dockerfile
- [ ] Update docker-compose.yml
- [ ] Performance testing

---

## 10. References

### Architecture & Patterns
- [FastAPI Clean Architecture Example](https://github.com/ivan-borovets/fastapi-clean-example) - Production-ready Clean Architecture with DDD, CQRS
- [fast-clean-architecture CLI](https://pypi.org/project/fast-clean-architecture/) - CLI tool for scaffolding
- [Architecture Patterns with Python](https://www.cosmicpython.com/book/chapter_02_repository.html) - Repository pattern
- [Python Design Patterns for Clean Architecture](https://www.glukhov.org/post/2025/11/python-design-patterns-for-clean-architecture/)
- [Building Python API with CQRS](https://wawaziphil.medium.com/building-a-python-api-using-cqrs-a-simple-guide-3d584b6ead34)

### FastAPI & Frameworks
- [FastAPI Best Practices](https://github.com/zhanymkanov/fastapi-best-practices)
- [Official FastAPI Template](https://github.com/fastapi/full-stack-fastapi-template)
- [PyCon India 2025: FastAPI for Production](https://cfp.in.pycon.org/2025/talk/LHLX8U/)
- [FastAPI vs Litestar 2025](https://medium.com/@rameshkannanyt0078/fastapi-vs-litestar-2025-which-async-python-web-framework-should-you-choose-8dc05782a276)

### Dependency Injection
- [Dishka Documentation](https://dishka.readthedocs.io/en/stable/)
- [Better Dependency Injection in FastAPI](https://vladiliescu.net/better-dependency-injection-in-fastapi/)
- [Dishka vs Depends Comparison](https://dishka.readthedocs.io/en/stable/alternatives.html)

### Streaming & Rate Limiting
- [sse-starlette](https://github.com/sysid/sse-starlette)
- [slowapi Documentation](https://slowapi.readthedocs.io/)
- [FastAPI Streaming APIs](https://python.plainenglish.io/streaming-apis-for-beginners-python-fastapi-and-async-generators-848b73a8fc06)

### Event Sourcing & CQRS
- [Event Sourcing & CQRS with FastAPI - PyCon Greece 2025](https://dev.to/markoulis/how-i-learned-to-stop-worrying-and-love-raw-events-event-sourcing-cqrs-with-fastapi-and-celery-477e)
- [python-cqrs Library](https://pypi.org/project/python-cqrs/)

---

## Summary

This migration guide represents **true 2025-2026 cutting-edge architecture**, not just a Flask-to-FastAPI port:

| Aspect | What We're Doing |
|--------|------------------|
| **Architecture** | 4-Layer Clean Architecture (Domain → Application → Infrastructure → Presentation) |
| **DI Framework** | Dishka (framework-agnostic, better than FastAPI Depends) |
| **Data Pattern** | CQRS (Command/Query Separation) |
| **Domain Design** | DDD with Entities, Value Objects, Ports |
| **Testing** | Framework-agnostic domain/application testing |
| **Repository Pattern** | Abstract ports in domain, implementations in infrastructure |
| **Async** | Native async/await, no workarounds |

**Estimated effort**: 14-16 days for complete migration

This is the architecture used by **enterprise teams** and recommended at **PyCon 2025** for production FastAPI applications.
