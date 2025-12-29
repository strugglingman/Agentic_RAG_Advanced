"""
ConversationId Value Object - UUID wrapper for conversation identity.
"""

from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True)
class ConversationId:
    value: str  # conversation_id, presented as UUID string (empty string = new conversation)

    def __post_init__(self):
        # Allow empty string for "new conversation" case
        if self.value and not self._is_valid_uuid(self.value):
            raise ValueError(f"Invalid conversation ID (UUID): {self.value}")

    def _is_valid_uuid(self, value: str) -> bool:
        """Check if string is a valid UUID."""
        try:
            UUID(value)
            return True
        except ValueError:
            return False

    def __str__(self) -> str:
        return self.value

    def __bool__(self) -> bool:
        """Return False if empty string (new conversation)."""
        return bool(self.value)
