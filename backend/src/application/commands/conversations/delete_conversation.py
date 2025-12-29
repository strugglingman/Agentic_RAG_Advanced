"""Delete Conversation Command."""

from dataclasses import dataclass
from src.domain.exceptions.access_denied import AccessDeniedError
from src.domain.exceptions.entity_not_found import EntityNotFoundError
from src.domain.value_objects.conversation_id import ConversationId
from src.domain.value_objects.user_email import UserEmail
from src.domain.ports.repositories import ConversationRepository
from src.application.common.interfaces import Command, CommandHandler


@dataclass(frozen=True)
class DeleteConversationCommand(Command[bool]):
    conversation_id: ConversationId
    user_email: UserEmail


class DeleteConversationHandler(CommandHandler[bool]):
    def __init__(self, conversation_repository: ConversationRepository):
        self._conversation_repository = conversation_repository

    async def execute(self, command: DeleteConversationCommand) -> bool:
        conversation_id = command.conversation_id
        user_email = command.user_email

        conversation = await self._conversation_repository.get_by_id(conversation_id)
        if not conversation:
            raise EntityNotFoundError(
                f"Conversation {conversation_id.value} not found."
            )

        if conversation.user_email.value != user_email.value:
            raise AccessDeniedError("User does not own this conversation.")

        return await self._conversation_repository.delete(conversation_id)
