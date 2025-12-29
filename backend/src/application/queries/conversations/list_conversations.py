"""List Conversations Query."""

from dataclasses import dataclass
from src.domain.ports.repositories.conversation_repository import (
    ConversationRepository,
)
from src.application.common.interfaces import Query, QueryHandler
from src.domain.entities.conversation import Conversation
from src.domain.value_objects.user_email import UserEmail


@dataclass(frozen=True)
class ListConversationsQuery(Query[list[Conversation]]):
    user_email: UserEmail
    limit: int = 50


class ListConversationsHandler(QueryHandler[list[Conversation]]):
    def __init__(self, conversation_repository: ConversationRepository):
        self._conversation_repository = conversation_repository

    async def execute(self, query: ListConversationsQuery) -> list[Conversation]:
        return await self._conversation_repository.get_by_user(
            query.user_email, query.limit
        )
