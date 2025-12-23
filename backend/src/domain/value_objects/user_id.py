"""
UserId Value Object
"""

from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True)
class UserId:
    value: str  # user_id

    def __post_init__(self):
        if not self.value:
            raise ValueError("UserId cannot be empty")

        UUID(self.value)  # Validate UUID format

    def __str__(self) -> str:
        return self.value
