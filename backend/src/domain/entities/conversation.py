"""
Conversation Entity - A chat session between user and AI.
"""

from dataclasses import dataclass
from datetime import datetime, timezone

from src.domain.value_objects.conversation_id import ConversationId
from src.domain.value_objects.user_email import UserEmail


@dataclass
class Conversation:
    id: ConversationId
    user_email: UserEmail
    title: str
    created_at: datetime
    updated_at: datetime

    def update_title(self, new_title: str) -> None:
        if len(new_title) > 50:
            raise ValueError("Title cannot exceed 50 characters")

        self.title = new_title
        self.updated_at = datetime.now(timezone.utc)
