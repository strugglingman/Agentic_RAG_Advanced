"""
GetFile Query - Retrieve a single file by ID with access control.

Usage in presentation layer:
    handler: FromDishka[GetFileHandler]
    query = GetFileQuery(
        file_id=FileId(file_id),
        user_email=current_user.email,
        dept_id=current_user.dept,
    )
    result = await handler.execute(query)
    if result.file is None:
        raise HTTPException(404, "File not found")
    # Return FileResponse with result.file.storage_path

Response:
    GetFileResult with FileRegistry entity or None if not accessible
"""

import os
from dataclasses import dataclass
from typing import Optional

from src.application.common.interfaces import Query, QueryHandler
from src.domain.entities.file_registry import FileRegistry
from src.domain.ports.repositories.file_registry_repository import (
    FileRegistryRepository,
)
from src.domain.value_objects.file_id import FileId
from src.domain.value_objects.user_email import UserEmail
from src.domain.value_objects.dept_id import DeptId


# ==================== RESULT ====================


@dataclass
class GetFileResult:
    """Result of get file query."""

    file: Optional[FileRegistry]
    error: Optional[str] = None


# ==================== QUERY ====================


@dataclass
class GetFileQuery(Query):
    """
    Query to get a file by ID with access control.

    Access control:
    - User owns the file (user_email matches), OR
    - File is shared (file_for_user=False) AND same dept_id
    """

    file_id: FileId
    user_email: UserEmail
    dept_id: DeptId


# ==================== HANDLER ====================


class GetFileHandler(QueryHandler[GetFileResult]):
    """Handler for GetFileQuery."""

    def __init__(self, file_registry_repository: FileRegistryRepository):
        self._repository = file_registry_repository

    async def execute(self, query: GetFileQuery) -> GetFileResult:
        """
        Execute the query to get a file by ID.

        Returns:
            GetFileResult with file entity or error message
        """
        # Get file with access control
        file = await self._repository.get_accessible_file(
            file_id=query.file_id,
            user_email=query.user_email,
            dept_id=query.dept_id,
        )

        if file is None:
            return GetFileResult(
                file=None,
                error="File not found or access denied",
            )

        # Verify file exists on disk
        if not os.path.exists(file.storage_path.value):
            return GetFileResult(
                file=None,
                error="File not found on disk",
            )

        return GetFileResult(file=file)
