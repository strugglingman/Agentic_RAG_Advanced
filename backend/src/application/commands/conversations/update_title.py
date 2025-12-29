"""Update Title Command."""

from dataclasses import dataclass
from src.domain.entities.conversation import Conversation
from src.application.common.interfaces import Command, CommandHandler
from src.domain.exceptions import EntityNotFoundError, AccessDeniedError
from src.domain.ports.repositories import ConversationRepository
from src.domain.value_objects.conversation_id import ConversationId
from src.domain.value_objects.user_email import UserEmail


@dataclass(frozen=True)
class UpdateTitleCommand(Command[Conversation]):
    conversation_id: ConversationId
    user_email: UserEmail
    new_title: str


class UpdateTitleHandler(CommandHandler[Conversation]):
    def __init__(self, conversation_repository: ConversationRepository):
        self._conversation_repository = conversation_repository

    async def execute(self, command: UpdateTitleCommand) -> Conversation:
        conversation = await self._conversation_repository.get_by_id(
            command.conversation_id
        )
        if not conversation:
            raise EntityNotFoundError(
                f"Conversation {command.conversation_id.value} not found"
            )
        if conversation.user_email.value != command.user_email.value:
            raise AccessDeniedError(f"User {command.user_email.value} has no access")

        conversation.update_title(command.new_title)
        await self._conversation_repository.save(conversation)

        return conversation
