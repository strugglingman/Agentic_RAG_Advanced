"""
Prisma Conversation Repository Implementation.

Guidelines:
- Implements ConversationRepository port from domain layer
- Uses Prisma client for database operations
- Maps between Prisma models and domain entities
- All methods are async

Steps to implement:
1. Import Prisma client and domain types
2. Create PrismaConversationRepository class implementing ConversationRepository
3. Inject Prisma client via __init__
4. Implement each method:
   - get_by_id: Query by id, map to entity or return None
   - get_by_user: Query by user_email with limit, order by updated_at desc
   - save: Use upsert (create or update based on id)
   - delete: Delete by id, return True if deleted

Mapping:
- Prisma model fields: id, user_email, title, created_at, updated_at
- Domain entity: Conversation with value objects (ConversationId, UserEmail)
- Convert str -> ConversationId, str -> UserEmail when reading
- Convert ConversationId.value, UserEmail.value -> str when writing
"""

from typing import Optional
from prisma import Prisma
from prisma.models import Conversation as PrismaConversation
from src.domain.entities.conversation import Conversation
from src.domain.ports.repositories import ConversationRepository
from src.domain.value_objects.conversation_id import ConversationId
from src.domain.value_objects.user_email import UserEmail


class PrismaConversationRepository(ConversationRepository):
    _prisma: Prisma

    def __init__(self, prisma: Prisma):
        self._prisma = prisma

    def _to_entity(self, record: PrismaConversation) -> Conversation:
        """Map Prisma record to domain entity."""
        return Conversation(
            id=ConversationId(record.id),
            user_email=UserEmail(record.user_email),
            title=record.title,
            created_at=record.created_at,
            updated_at=record.updated_at,
            last_message=record.messages[0].content[:50] if record.messages else None,
        )

    async def get_by_id(
        self, conversation_id: ConversationId
    ) -> Optional[Conversation]:
        """Get conversation by ID."""
        record = await self._prisma.conversation.find_unique(
            where={"id": conversation_id.value}
        )
        return self._to_entity(record) if record else None

    async def get_by_user(
        self, user_email: UserEmail, limit: int
    ) -> list[Conversation]:
        """Get conversations for user, ordered by updated_at desc."""
        records = await self._prisma.conversation.find_many(
            where={"user_email": user_email.value},
            order={"updated_at": "desc"},
            take=limit,
            include={
                "messages": {
                    "order_by": {"created_at": "desc"},
                    "take": 1,
                }
            },
        )
        conversations = [self._to_entity(record) for record in records]
        return conversations

    async def save(self, conversation: Conversation) -> None:
        """Save (create or update) conversation."""
        await self._prisma.conversation.upsert(
            where={"id": conversation.id.value},
            data={
                "create": {
                    "id": conversation.id.value,
                    "user_email": conversation.user_email.value,
                    "title": conversation.title,
                    "created_at": conversation.created_at,
                    "updated_at": conversation.updated_at,
                },
                "update": {
                    "title": conversation.title,
                    "updated_at": conversation.updated_at,
                },
            },
        )

    async def delete(self, conversation_id: ConversationId) -> bool:
        """Delete conversation by ID. Returns True if deleted."""
        try:
            await self._prisma.conversation.delete(where={"id": conversation_id.value})
            return True
        except Exception:
            return False
