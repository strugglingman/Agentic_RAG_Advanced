"""
Message Entity - A single message in a conversation.
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4
from src.domain.value_objects.message_id import MessageId
from src.domain.value_objects.conversation_id import ConversationId


@dataclass
class Message:
    id: MessageId
    conversation_id: ConversationId
    role: str
    content: str
    created_at: datetime
    tokens_used: Optional[int] = None
    latency_ms: Optional[int] = None

    def __post_init__(self):
        if self.role not in ["user", "assistant", "system"]:
            raise ValueError(f"Invalid role: {self.role}")

    @classmethod
    def create(
        cls,
        conversation_id: ConversationId,
        role: str,
        content: str,
        tokens_used: Optional[int] = None,
        latency_ms: Optional[int] = None,
    ) -> Message:
        """Factory method to create a new Message with a generated ID and timestamp."""
        id = MessageId(str(uuid4()))
        return cls(
            id=id,
            conversation_id=conversation_id,
            role=role,
            content=content,
            created_at=datetime.now(timezone.utc),
            tokens_used=tokens_used,
            latency_ms=latency_ms,
        )
