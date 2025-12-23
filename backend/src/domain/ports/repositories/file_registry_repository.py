"""
FileRegistry Repository Port - Interface for file registry persistence.
Implementation: src/infrastructure/persistence/prisma_file_registry_repository.py
"""

from abc import ABC, abstractmethod
from typing import Optional

from src.domain.entities.file_registry import FileRegistry
from src.domain.value_objects.file_id import FileId
from src.domain.value_objects.user_email import UserEmail


class FileRegistryRepository(ABC):
    @abstractmethod
    async def get_by_id(self, file_id: FileId) -> Optional[FileRegistry]: ...

    @abstractmethod
    async def get_by_user(
        self, user_email: UserEmail, category: Optional[str] = None, limit: int = 50
    ) -> list[FileRegistry]: ...

    @abstractmethod
    async def save(self, file: FileRegistry) -> None: ...

    @abstractmethod
    async def delete(self, file_id: FileId) -> bool: ...
