"""
FileRegistry Repository Port - Interface for file registry persistence.
Implementation: src/infrastructure/persistence/prisma_file_registry_repository.py
"""

from abc import ABC, abstractmethod
from typing import Optional, List

from src.domain.entities.file_registry import FileRegistry
from src.domain.value_objects.file_id import FileId
from src.domain.value_objects.user_email import UserEmail
from src.domain.value_objects.dept_id import DeptId
from src.domain.value_objects.conversation_id import ConversationId


class FileRegistryRepository(ABC):
    @abstractmethod
    async def get_by_id(self, file_id: FileId) -> Optional[FileRegistry]:
        """Get file by ID."""
        ...

    @abstractmethod
    async def get_by_id_and_user(
        self, file_id: FileId, user_email: UserEmail
    ) -> Optional[FileRegistry]:
        """Get file by ID with user ownership check."""
        ...

    @abstractmethod
    async def get_by_user(
        self,
        user_email: UserEmail,
        category: Optional[str] = None,
        conversation_id: Optional[ConversationId] = None,
        limit: int = 50,
    ) -> list[FileRegistry]:
        """Get files for user with optional filters."""
        ...

    @abstractmethod
    async def find_by_name(
        self,
        user_email: UserEmail,
        original_name: str,
        category: Optional[str] = None,
    ) -> Optional[FileRegistry]:
        """Find file by original name (most recent if multiple)."""
        ...

    @abstractmethod
    async def create(self, file: FileRegistry) -> FileRegistry:
        """
        Create a new file registry record.

        Args:
            file: FileRegistry entity (id can be None, will be auto-generated)

        Returns:
            FileRegistry with generated id and download_url
        """
        ...

    @abstractmethod
    async def update(self, file: FileRegistry) -> None:
        """Update an existing file registry record."""
        ...

    @abstractmethod
    async def delete(self, file_id: FileId) -> bool:
        """Delete file by ID. Returns True if deleted."""
        ...

    @abstractmethod
    async def get_accessible_files(
        self,
        user_email: UserEmail,
        dept_id: DeptId,
        category: Optional[str] = None,
        limit: int = 100,
    ) -> list[FileRegistry]:
        """
        Get files accessible to user: user's own files + shared files in same dept.

        Business logic (matching Flask files.py):
        - Files where user_email matches (user's own files)
        - Files where dept_id matches AND file_for_user=False (shared files)

        Args:
            user_email: User's email
            dept_id: User's department ID
            category: Optional category filter (e.g., "uploaded")
            limit: Maximum number of files to return

        Returns:
            List of accessible FileRegistry entities
        """
        ...

    @abstractmethod
    async def get_accessible_file(
        self,
        file_id: FileId,
        user_email: UserEmail,
        dept_id: DeptId,
    ) -> Optional[FileRegistry]:
        """
        Get a single file by ID if user has access.

        Access control:
        - User owns the file (user_email matches), OR
        - File is shared (file_for_user=False in metadata) AND same dept_id

        Args:
            file_id: File ID to retrieve
            user_email: User's email for ownership check
            dept_id: User's department ID for shared access check

        Returns:
            FileRegistry entity if accessible, None otherwise
        """
        ...

    @abstractmethod
    async def get_uningested_files(
        self, file_ids: Optional[List[FileId]], user_email: UserEmail, dept_id: DeptId
    ) -> list[FileRegistry]:
        """
        Get files that have not been ingested/indexed yet.

        Args:
            file_ids: List of file IDs to check, or None for all uningested files
            user_email: User's email for ownership check
            dept_id: User's department ID for shared access check

        Returns:
            List of uningested FileRegistry entities
        """
        ...

    @abstractmethod
    async def mark_indexed(self, file_id: FileId, collection_name: str) -> None:
        """
        Mark a file as indexed in ChromaDB.

        Args:
            file_id: File ID to mark as indexed
            collection_name: Name of the ChromaDB collection
        """
        ...
