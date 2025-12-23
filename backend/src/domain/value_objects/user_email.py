"""
UserEmail Value Object - Wraps user email with validation.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class UserEmail:
    value: str  # user_email, presented as email

    def __post_init__(self):
        if not self.value or "@" not in self.value:
            raise ValueError(f"Invalid user email: {self.value}")

    def __str__(self) -> str:
        return self.value
