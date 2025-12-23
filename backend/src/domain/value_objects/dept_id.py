"""
DeptId Value Object - Department identifier.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DeptId:
    value: str  # department identifier

    def __post_init__(self):
        if not self.value or not self.value.strip():
            raise ValueError("Department ID cannot be empty")

    def __str__(self) -> str:
        return self.value
