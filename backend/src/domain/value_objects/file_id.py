"""
FileId Value Object - File identifier (cuid format).

CUID (Collision-resistant Unique Identifier) format:
- Starts with 'c' (lowercase)
- 25 characters total
- Example: clz8x0y5a0000qf0gv9z1h2j3
"""

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class FileId:
    value: str  # file identifier (cuid)

    # CUID pattern: starts with 'c', followed by 24 alphanumeric characters
    _CUID_PATTERN = re.compile(r"^c[a-z0-9]{24}$")

    def __post_init__(self):
        if not self.value or not self.value.strip():
            raise ValueError("FileId cannot be empty")
        if not self._CUID_PATTERN.match(self.value):
            raise ValueError(f"Invalid FileId (expected cuid format): {self.value}")

    def __str__(self) -> str:
        return self.value
