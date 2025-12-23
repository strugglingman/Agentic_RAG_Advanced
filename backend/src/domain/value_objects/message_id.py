"""
MessageId Value Object - UUID wrapper for message identity.
"""

from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True)
class MessageId:
    value: str  # message_id, presented as UUID string

    def __post_init__(self):
        if not self.value:
            raise ValueError("Message ID cannot be empty")
        UUID(self.value)  # raises ValueError if invalid UUID

    def __str__(self) -> str:
        return self.value
