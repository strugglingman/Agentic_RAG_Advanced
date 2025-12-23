"""
Message Entity - A single message in a conversation.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
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
