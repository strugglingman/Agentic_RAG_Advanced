"""
Dishka DI Container Setup.

Guidelines:
- Registers all dependencies (repositories, handlers, services)
- Maps abstract interfaces to concrete implementations
- Manages lifecycle (singleton, request-scoped, etc.)

Steps to implement:
1. Create a Provider class that defines how to create dependencies
2. Register Prisma client (singleton, connected at startup)
3. Register repositories (interface → implementation)
4. Register command/query handlers
5. Create container factory function

Dishka concepts:
- Provider: Class that defines how to create dependencies
- @provide: Decorator to mark factory methods
- Scope: Lifecycle of dependency (APP = singleton, REQUEST = per-request)
- make_async_container: Creates the container

Flow:
  Container → provides → PrismaConversationRepository → to → CreateConversationHandler
                                    ↓
                            uses ConversationRepository interface
"""

import logging
from typing import Optional
from dishka import Provider, Scope, make_async_container, provide, AsyncContainer
from prisma import Prisma
from openai import OpenAI
from redis.asyncio import Redis
from src.services.vector_db import VectorDB
from src.services.query_supervisor import QuerySupervisor
from src.domain.ports.repositories import (
    ConversationRepository,
    MessageRepository,
    FileRegistryRepository,
)
from src.infrastructure.persistence import (
    PrismaConversationRepository,
    PrismaMessageRepository,
    PrismaFileRegistryRepository,
)
from src.infrastructure.cache import create_redis_client, CachedMessageRepository
from src.infrastructure.jobs import IngestJobStore
from src.infrastructure.storage import FileStorageService
from src.services.agent_state import AgentSessionStateStore

logger = logging.getLogger(__name__)
from src.application.services import FileService
from src.application.commands.conversations import (
    CreateConversationHandler,
    DeleteConversationHandler,
    UpdateTitleHandler,
)
from src.application.queries.conversations import (
    ListConversationsHandler,
)
from src.application.queries.chat import (
    GetChatHistoryHandler,
)
from src.application.commands.chat.send_message import (
    SendMessageHandler,
)
from src.application.commands.files import (
    UploadFileHandler,
)
from src.application.queries.files import (
    ListFilesHandler,
    GetFileHandler,
)
from src.application.queries.org import (
    GetOrgStructureHandler,
)
from src.config.settings import Config


class AppProvider(Provider):
    """
    Application dependency provider.

    Registers all dependencies and their implementations.

    """

    # ==================== VECTOR DATABASE ====================
    @provide(scope=Scope.APP)
    def get_vector_db(self) -> VectorDB:
        return VectorDB(
            path=Config.CHROMA_PATH,
            embedding_provider=Config.EMBEDDING_PROVIDER,
        )

    # ==================== OPENAI CLIENT ====================
    @provide(scope=Scope.APP)
    def get_openai_client(self) -> OpenAI:
        return OpenAI(api_key=Config.OPENAI_KEY)

    # ==================== REDIS CACHE ====================

    @provide(scope=Scope.APP)
    async def get_redis_client(self) -> Optional[Redis]:
        """
        Provide Redis client (singleton, app-scoped, optional).

        - Scope.APP = created ONCE when app starts, shared across all requests
        - Returns None if Redis is disabled or unavailable (graceful degradation)
        - async because connect and ping are async
        """
        if not Config.REDIS_ENABLED:
            logger.info("[Redis] Caching disabled by config")
            return None
        try:
            client = await create_redis_client()
            return client
        except Exception as e:
            logger.warning(f"[Redis] Unavailable, caching disabled: {e}")
            return None

    # ==================== JOB STORE ====================

    @provide(scope=Scope.APP)
    def get_ingest_job_store(self, redis_client: Optional[Redis]) -> Optional[IngestJobStore]:
        """Provide IngestJobStore for tracking active ingestion jobs."""
        if redis_client:
            return IngestJobStore(redis_client)
        return None

    @provide(scope=Scope.APP)
    def get_agent_state_store(
        self, redis_client: Optional[Redis]
    ) -> Optional[AgentSessionStateStore]:
        """
        Provide AgentSessionStateStore for persisting agent state across messages.

        Returns None if Redis unavailable - agent will fall back to per-request state.
        """
        if redis_client:
            return AgentSessionStateStore(redis_client)
        return None

    # ==================== DATABASE ====================

    @provide(scope=Scope.APP)
    async def get_prisma(self) -> Prisma:
        """
        Provide Prisma client (singleton, app-scoped).

        - Scope.APP = created ONCE when app starts, shared across all requests
        - async because connect() is async
        """
        prisma = Prisma()
        await prisma.connect()
        return prisma

    # ==================== SERVICES ====================
    @provide(scope=Scope.REQUEST)
    def get_query_supervisor(self, openai_client: OpenAI) -> QuerySupervisor:
        return QuerySupervisor(openai_client=openai_client)

    @provide(scope=Scope.APP)
    def get_file_storage_service(self) -> FileStorageService:
        """
        Provide FileStorageService (singleton).

        - Scope.APP = created ONCE when app starts
        - No state, just disk operations
        """
        return FileStorageService()

    @provide(scope=Scope.REQUEST)
    def get_file_service(
        self,
        storage: FileStorageService,
        file_registry_repository: FileRegistryRepository,
    ) -> FileService:
        """
        Provide FileService (per-request).

        - Coordinates disk I/O (FileStorageService) with DB (FileRegistryRepository)
        - Replaces old FileManager
        """
        return FileService(storage=storage, repository=file_registry_repository)

    # ==================== REPOSITORIES ====================

    @provide(scope=Scope.REQUEST)
    def get_conversation_repository(self, prisma: Prisma) -> ConversationRepository:
        """
        Provide ConversationRepository implementation.

        - Return type is ABSTRACT (ConversationRepository)
        - Implementation is CONCRETE (PrismaConversationRepository)
        - Dishka sees: "when someone asks for ConversationRepository, give them this"
        - Scope.REQUEST = new instance per HTTP request
        """
        return PrismaConversationRepository(prisma)

    @provide(scope=Scope.REQUEST)
    def get_message_repository(
        self, prisma: Prisma, redis_client: Optional[Redis]
    ) -> MessageRepository:
        """
        Provide MessageRepository implementation with optional caching.

        - Return type is ABSTRACT (MessageRepository)
        - Implementation is CONCRETE (PrismaMessageRepository or CachedMessageRepository)
        - If Redis available: wraps with CachedMessageRepository (decorator pattern)
        - If Redis unavailable: returns plain PrismaMessageRepository
        - Scope.REQUEST = new instance per HTTP request
        """
        base_repo = PrismaMessageRepository(prisma)
        if redis_client:
            return CachedMessageRepository(base_repo, redis_client)
        return base_repo

    # ==================== HANDLERS ====================

    @provide(scope=Scope.REQUEST)
    def get_create_conversation_handler(
        self, conversation_repository: ConversationRepository
    ) -> CreateConversationHandler:
        """
        Provide CreateConversationHandler.

        - Parameter asks for ConversationRepository (abstract)
        - Dishka automatically resolves it using get_conversation_repository()
        - This is the "magic" of DI - dependencies are auto-wired
        """
        return CreateConversationHandler(conversation_repository)

    @provide(scope=Scope.REQUEST)
    def get_delete_conversation_handler(
        self, conversation_repository: ConversationRepository
    ) -> DeleteConversationHandler:
        return DeleteConversationHandler(conversation_repository)

    @provide(scope=Scope.REQUEST)
    def get_update_title_handler(
        self, conversation_repository: ConversationRepository
    ) -> UpdateTitleHandler:
        return UpdateTitleHandler(conversation_repository)

    @provide(scope=Scope.REQUEST)
    def get_list_conversations_handler(
        self, conversation_repository: ConversationRepository
    ) -> ListConversationsHandler:
        return ListConversationsHandler(conversation_repository)

    @provide(scope=Scope.REQUEST)
    def get_send_message_handler(
        self,
        conversation_repository: ConversationRepository,
        message_repository: MessageRepository,
        query_supervisor: QuerySupervisor,
        file_service: FileService,
        vector_db: VectorDB,
        openai_client: OpenAI,
        agent_state_store: Optional[AgentSessionStateStore],
    ) -> SendMessageHandler:
        return SendMessageHandler(
            conv_repo=conversation_repository,
            msg_repo=message_repository,
            query_supervisor=query_supervisor,
            file_service=file_service,
            vector_db=vector_db,
            openai_client=openai_client,
            agent_state_store=agent_state_store,
        )

    @provide(scope=Scope.REQUEST)
    def get_chat_history_handler(
        self,
        conversation_repository: ConversationRepository,
        message_repository: MessageRepository,
    ) -> GetChatHistoryHandler:
        return GetChatHistoryHandler(
            conv_repo=conversation_repository,
            msg_repo=message_repository,
        )

    # ==================== FILE REPOSITORIES ====================

    @provide(scope=Scope.REQUEST)
    def get_file_registry_repository(self, prisma: Prisma) -> FileRegistryRepository:
        """Provide FileRegistryRepository implementation."""
        return PrismaFileRegistryRepository(prisma)

    # ==================== FILE HANDLERS ====================

    @provide(scope=Scope.REQUEST)
    def get_upload_file_handler(
        self, file_registry_repository: FileRegistryRepository
    ) -> UploadFileHandler:
        """Provide UploadFileHandler."""
        return UploadFileHandler(file_registry_repository)

    @provide(scope=Scope.REQUEST)
    def get_list_files_handler(
        self, file_registry_repository: FileRegistryRepository
    ) -> ListFilesHandler:
        """Provide ListFilesHandler."""
        return ListFilesHandler(file_registry_repository)

    @provide(scope=Scope.REQUEST)
    def get_get_file_handler(
        self, file_registry_repository: FileRegistryRepository
    ) -> GetFileHandler:
        """Provide GetFileHandler."""
        return GetFileHandler(file_registry_repository)

    @provide(scope=Scope.REQUEST)
    def get_get_org_structure_handler(self) -> GetOrgStructureHandler:
        return GetOrgStructureHandler()


async def create_container() -> AsyncContainer:
    """
    Create and configure the DI container.

    - make_async_container() creates the container with all providers
    - Call this ONCE at app startup
    """
    return make_async_container(AppProvider())


# ==================== USAGE IN FASTAPI ====================
"""
In your main.py or app.py:

```python
from fastapi import FastAPI
from dishka.integrations.fastapi import setup_dishka
from src.setup.ioc import create_container

app = FastAPI()

@app.on_event("startup")
async def startup():
    container = await create_container()
    setup_dishka(container, app)

@app.on_event("shutdown")
async def shutdown():
    await app.state.dishka_container.close()
```

Or with lifespan (recommended for FastAPI):

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    container = await create_container()
    setup_dishka(container, app)
    yield
    # Shutdown
    await container.close()

app = FastAPI(lifespan=lifespan)
```
"""
