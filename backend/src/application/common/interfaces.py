"""
Base interfaces for CQRS pattern.

TODO: Implement Command and Query base classes

Example:
```python
from abc import ABC, abstractmethod
from typing import TypeVar, Generic

T = TypeVar("T")

class Command(ABC, Generic[T]):
    '''Base class for write operations'''
    pass

class CommandHandler(ABC, Generic[T]):
    @abstractmethod
    async def execute(self, command: Command[T]) -> T:
        ...

class Query(ABC, Generic[T]):
    '''Base class for read operations'''
    pass

class QueryHandler(ABC, Generic[T]):
    @abstractmethod
    async def execute(self, query: Query[T]) -> T:
        ...
```

Usage in command:
```python
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
```
"""