"""
Conversation Entity - A chat session between user and AI.
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4
from src.domain.value_objects.conversation_id import ConversationId
from src.domain.value_objects.user_email import UserEmail


@dataclass
class Conversation:
    id: ConversationId
    user_email: UserEmail
    title: str
    created_at: datetime
    updated_at: datetime
    last_message: str | None = None
    source_channel_id: str | None = None  # Format: "slack:C0123ABC", "teams:xxx", None for web

    @classmethod
    def create(
        cls, user_email: UserEmail, title: str, source_channel_id: str | None = None
    ) -> Conversation:
        conversation_id = ConversationId(str(uuid4()))
        return cls(
            id=conversation_id,
            user_email=user_email,
            title=title,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            source_channel_id=source_channel_id,
        )

    def update_title(self, new_title: str) -> None:
        if len(new_title) > 50:
            raise ValueError("Title cannot exceed 50 characters")

        self.title = new_title
        self.updated_at = datetime.now(timezone.utc)
