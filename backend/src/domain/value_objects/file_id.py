"""
FileId Value Object - File identifier (cuid format).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class FileId:
    value: str  # file identifier (cuid)

    def __post_init__(self):
        if not self.value or not self.value.strip():
            raise ValueError("FileId cannot be empty")

    def __str__(self) -> str:
        return self.value
