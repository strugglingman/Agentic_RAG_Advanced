"""
Base interfaces for CQRS pattern.

Usage:
    @dataclass
    class CreateConversationCommand(Command[ConversationId]):
        user_email: str
        title: str

    class CreateConversationHandler(CommandHandler[ConversationId]):
        def __init__(self, repo: ConversationRepository):
            self.repo = repo

        async def execute(self, cmd: CreateConversationCommand) -> ConversationId:
            conversation = Conversation(...)
            await self.repo.save(conversation)
            return conversation.id
"""
from abc import ABC, abstractmethod
from typing import TypeVar, Generic

T = TypeVar("T")
class Command(ABC, Generic[T]):
    """Base class for write operations"""
    pass

class CommandHandler(ABC, Generic[T]):
    @abstractmethod
    async def execute(self, command: Command[T]) -> T:
        """Execute the command and return a result of type T"""
        ...

class Query(ABC, Generic[T]):
    """Base class for read operations"""
    pass

class QueryHandler(ABC, Generic[T]):
    @abstractmethod
    async def execute(self, query: Query[T]) -> T:
        """Execute the query and return a result of type T"""
        ...