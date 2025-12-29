"""
Create Conversation Command.

Guidelines:
- Command: @dataclass(frozen=True) holding input data
- Handler: Business logic, receives repository via __init__ (DI)
- Handler.execute(): Async method that performs the operation
- Returns: ConversationId

Steps to implement:
1. Create CreateConversationCommand(Command[ConversationId]) with fields: user_email, title
2. Create CreateConversationHandler(CommandHandler[ConversationId])
3. Handler.__init__ receives ConversationRepository
4. Handler.execute():
   - Create value objects (ConversationId, UserEmail)
   - Create Conversation entity
   - Save via repository
   - Return conversation_id
"""

from dataclasses import dataclass
from src.domain.entities.conversation import Conversation
from src.domain.ports.repositories import ConversationRepository
from src.domain.value_objects.conversation_id import ConversationId
from src.domain.value_objects.user_email import UserEmail
from src.application.common.interfaces import Command, CommandHandler


@dataclass(frozen=True)
class CreateConversationCommand(Command[ConversationId]):
    user_email: UserEmail
    title: str


class CreateConversationHandler(CommandHandler[ConversationId]):
    _conversation_repository: ConversationRepository

    def __init__(self, conversation_repository: ConversationRepository):
        self._conversation_repository = conversation_repository

    async def execute(self, command: CreateConversationCommand) -> ConversationId:
        conversation = Conversation.create(
            user_email=command.user_email, title=command.title
        )
        await self._conversation_repository.save(conversation)
        return conversation.id
