"""
FilePath Value Object - Validated file system path.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FilePath:
    value: str  # file system path

    def __post_init__(self):
        if not self.value or not self.value.strip():
            raise ValueError("File path cannot be empty")

    def __str__(self) -> str:
        return self.value

    @property
    def path(self) -> Path:
        return Path(self.value)

    @property
    def filename(self) -> str:
        return Path(self.value).name

    @property
    def extension(self) -> str:
        return Path(self.value).suffix
