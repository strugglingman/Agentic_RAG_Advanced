"""
ConversationId Value Object - UUID wrapper for conversation identity.
"""

from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True)
class ConversationId:
    value: str  # conversation_id, presented as UUID string

    def __post_init__(self):
        if not self.value or not UUID(self.value):
            raise ValueError(f"Invalid conversation ID (UUID): {self.value}")

    def __str__(self) -> str:
        return self.value
