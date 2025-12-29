"""
ListFiles Query - Get files accessible to user.

Returns files the user can access:
- User's own files (any category)
- Shared files in same department (file_for_user=False)

Response format matches Flask /files endpoint:
{
    "files": [
        {
            "file_id": "cuid_xxx",
            "filename": "document.pdf",
            "file_path": "dept123/user@email.com/document.pdf",  # relative path
            "ext": "pdf",
            "size_kb": 123.4,
            "upload_at": "2024-01-01T12:00:00",
            "tags": ["tag1", "tag2"],
            "ingested": false
        }
    ]
}
"""

import os
from dataclasses import dataclass
from typing import Optional

from src.application.common.interfaces import Query, QueryHandler
from src.application.dto.file import FileInfoDTO
from src.domain.entities.file_registry import FileRegistry
from src.domain.ports.repositories.file_registry_repository import (
    FileRegistryRepository,
)
from src.domain.value_objects.user_email import UserEmail
from src.domain.value_objects.dept_id import DeptId
from src.config.settings import Config


# ==================== RESULT ====================


@dataclass
class ListFilesResult:
    """Result of list files query."""

    files: list[FileInfoDTO]


# ==================== QUERY ====================


@dataclass(frozen=True)
class ListFilesQuery(Query[ListFilesResult]):
    """
    Query to list files accessible to user.

    Args:
        user_email: User's email (value object)
        dept_id: User's department ID (value object)
        category: Optional filter by category (e.g., "uploaded")
        limit: Maximum number of files to return
    """

    user_email: UserEmail
    dept_id: DeptId
    category: Optional[str] = None
    limit: int = 100


# ==================== HANDLER ====================


class ListFilesHandler(QueryHandler[ListFilesResult]):
    """
    Handler for list files query.

    Retrieves files from repository and transforms to DTO format.
    """

    def __init__(self, file_registry_repository: FileRegistryRepository):
        self._repo = file_registry_repository

    async def execute(self, query: ListFilesQuery) -> ListFilesResult:
        # Get accessible files from repository
        files = await self._repo.get_accessible_files(
            user_email=query.user_email,
            dept_id=query.dept_id,
            category=query.category,
            limit=query.limit,
        )

        # Transform to DTOs
        file_dtos = [self._to_dto(f) for f in files]

        return ListFilesResult(files=file_dtos)

    def _to_dto(self, file: FileRegistry) -> FileInfoDTO:
        """Transform FileRegistry entity to FileInfoDTO."""
        # Get relative path (strip UPLOAD_BASE)
        storage_path = file.storage_path.value
        try:
            relative_path = os.path.relpath(storage_path, Config.UPLOAD_BASE)
        except ValueError:
            # On Windows, relpath fails if paths are on different drives
            relative_path = storage_path

        # Get extension from filename
        ext = ""
        if "." in file.original_name:
            ext = file.original_name.rsplit(".", 1)[1].lower()

        # Get size in KB
        size_kb = round(file.size_bytes / 1024, 1) if file.size_bytes else 0.0

        # Get upload_at from metadata or created_at
        upload_at = None
        if file.metadata and file.metadata.get("upload_at"):
            upload_at = file.metadata["upload_at"]
        elif file.created_at:
            upload_at = file.created_at.isoformat()

        # Get tags from metadata (stored as comma-separated string)
        tags: list[str] = []
        if file.metadata and file.metadata.get("tags"):
            tags_value = file.metadata["tags"]
            if isinstance(tags_value, str):
                tags = [t.strip() for t in tags_value.split(",") if t.strip()]
            elif isinstance(tags_value, list):
                tags = tags_value

        # Get ingested status from indexed_in_chromadb or metadata
        ingested = file.indexed_in_chromadb
        if not ingested and file.metadata:
            ingested = file.metadata.get("ingested", False)

        return FileInfoDTO(
            file_id=file.id or "",
            filename=file.original_name,
            file_path=relative_path,
            ext=ext,
            size_kb=size_kb,
            upload_at=upload_at,
            tags=tags,
            ingested=ingested,
        )
