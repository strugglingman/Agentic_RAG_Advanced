"""
Prisma FileRegistry Repository - Implements FileRegistryRepository port.
"""

import json
from typing import Optional, Any, Dict

from prisma import Prisma

from src.domain.entities.file_registry import FileRegistry
from src.domain.ports.repositories.file_registry_repository import (
    FileRegistryRepository,
)
from src.domain.value_objects.file_id import FileId
from src.domain.value_objects.user_email import UserEmail
from src.domain.value_objects.dept_id import DeptId
from src.domain.value_objects.file_path import FilePath
from src.domain.value_objects.conversation_id import ConversationId


class PrismaFileRegistryRepository(FileRegistryRepository):
    """Prisma implementation of FileRegistryRepository."""

    _prisma: Prisma

    def __init__(self, prisma: Prisma):
        self._prisma = prisma

    def _to_entity(self, record) -> FileRegistry:
        """Map Prisma record to domain entity."""
        # Parse metadata JSON if it's a string
        metadata = record.metadata
        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        return FileRegistry(
            user_email=UserEmail(record.user_email),
            category=record.category,
            original_name=record.original_name,
            storage_path=FilePath(record.storage_path),
            created_at=record.created_at,
            accessed_at=record.accessed_at,
            id=record.id,
            dept_id=DeptId(record.dept_id) if record.dept_id else None,
            download_url=record.download_url,
            mime_type=record.mime_type,
            size_bytes=int(record.size_bytes) if record.size_bytes else None,
            source_tool=record.source_tool,
            source_url=record.source_url,
            conversation_id=(
                ConversationId(record.conversation_id)
                if record.conversation_id
                else None
            ),
            indexed_in_chromadb=record.indexed_in_chromadb or False,
            chromadb_collection=record.chromadb_collection,
            metadata=metadata,
        )

    def _to_create_dict(self, file: FileRegistry) -> Dict[str, Any]:
        """Map domain entity to Prisma dict for create."""
        return {
            "user_email": file.user_email.value,
            "category": file.category,
            "original_name": file.original_name,
            "storage_path": file.storage_path.value,
            "created_at": file.created_at,
            "accessed_at": file.accessed_at,
            "dept_id": file.dept_id.value if file.dept_id else None,
            "download_url": file.download_url,
            "mime_type": file.mime_type,
            "size_bytes": file.size_bytes,
            "source_tool": file.source_tool,
            "source_url": file.source_url,
            "conversation_id": (
                file.conversation_id.value if file.conversation_id else None
            ),
            "indexed_in_chromadb": file.indexed_in_chromadb,
            "chromadb_collection": file.chromadb_collection,
            "metadata": json.dumps(file.metadata) if file.metadata else json.dumps({}),
        }

    def _to_update_dict(self, file: FileRegistry) -> Dict[str, Any]:
        """Map domain entity to Prisma dict for update."""
        return {
            "category": file.category,
            "original_name": file.original_name,
            "storage_path": file.storage_path.value,
            "accessed_at": file.accessed_at,
            "dept_id": file.dept_id.value if file.dept_id else None,
            "download_url": file.download_url,
            "mime_type": file.mime_type,
            "size_bytes": file.size_bytes,
            "source_tool": file.source_tool,
            "source_url": file.source_url,
            "conversation_id": (
                file.conversation_id.value if file.conversation_id else None
            ),
            "indexed_in_chromadb": file.indexed_in_chromadb,
            "chromadb_collection": file.chromadb_collection,
            "metadata": json.dumps(file.metadata) if file.metadata else json.dumps({}),
        }

    async def get_by_id(self, file_id: FileId) -> Optional[FileRegistry]:
        """Get file by ID."""
        record = await self._prisma.fileregistry.find_unique(
            where={"id": file_id.value}
        )
        return self._to_entity(record) if record else None

    async def get_by_id_and_user(
        self, file_id: FileId, user_email: UserEmail
    ) -> Optional[FileRegistry]:
        """Get file by ID with user ownership check."""
        record = await self._prisma.fileregistry.find_first(
            where={"id": file_id.value, "user_email": user_email.value}
        )
        return self._to_entity(record) if record else None

    async def get_by_user(
        self,
        user_email: UserEmail,
        category: Optional[str] = None,
        conversation_id: Optional[ConversationId] = None,
        limit: int = 50,
    ) -> list[FileRegistry]:
        """Get files for user with optional filters."""
        where: Dict[str, Any] = {"user_email": user_email.value}
        if category:
            where["category"] = category
        if conversation_id:
            where["conversation_id"] = conversation_id.value

        records = await self._prisma.fileregistry.find_many(
            where=where,
            order={"created_at": "desc"},
            take=limit,
        )
        return [self._to_entity(r) for r in records]

    async def find_by_name(
        self,
        user_email: UserEmail,
        original_name: str,
        category: Optional[str] = None,
    ) -> Optional[FileRegistry]:
        """Find file by original name (most recent if multiple)."""
        where: Dict[str, Any] = {
            "user_email": user_email.value,
            "original_name": original_name,
        }
        if category:
            where["category"] = category

        record = await self._prisma.fileregistry.find_first(
            where=where,
            order={"created_at": "desc"},
        )
        return self._to_entity(record) if record else None

    async def create(self, file: FileRegistry) -> FileRegistry:
        """
        Create a new file registry record.

        Prisma auto-generates the cuid for id.
        After creation, generates download_url based on category and updates the record.

        Args:
            file: FileRegistry entity (id should be None)

        Returns:
            FileRegistry with generated id and download_url
        """
        # 1. Create record (Prisma generates cuid)
        data = self._to_create_dict(file)
        record = await self._prisma.fileregistry.create(data=data)

        # 2. Generate unified download_url - all files use /api/files/{file_id}
        download_url = f"/api/files/{record.id}"

        # 3. Update record with download_url
        record = await self._prisma.fileregistry.update(
            where={"id": record.id},
            data={"download_url": download_url},
        )

        return self._to_entity(record)

    async def update(self, file: FileRegistry) -> None:
        """Update an existing file registry record."""
        if not file.id:
            raise ValueError("Cannot update file without id")

        data = self._to_update_dict(file)
        await self._prisma.fileregistry.update(
            where={"id": file.id},
            data=data,
        )

    async def delete(self, file_id: FileId) -> bool:
        """Delete file by ID."""
        try:
            await self._prisma.fileregistry.delete(where={"id": file_id.value})
            return True
        except Exception:
            return False

    async def get_accessible_files(
        self,
        user_email: UserEmail,
        dept_id: DeptId,
        category: Optional[str] = None,
        limit: int = 100,
    ) -> list[FileRegistry]:
        """
        Get files accessible to user: user's own files + shared files in same dept.

        Query by dept_id (all files in department), then filter:
        - User's own files (user_email matches) → always include
        - Other users' files → only if file_for_user=False (shared)
        """
        # Query all files in the department
        where_clause: Dict[str, Any] = {"dept_id": dept_id.value}

        # Add category filter if specified
        if category:
            where_clause["category"] = category

        records = await self._prisma.fileregistry.find_many(
            where=where_clause,
            order={"created_at": "desc"},
            take=limit,
        )

        # Filter: user's own files OR shared files (file_for_user=False)
        result = []
        for r in records:
            entity = self._to_entity(r)
            # User's own files - always include
            if entity.user_email.value == user_email.value:
                result.append(entity)
            # Other users' files - only include if shared (file_for_user=False)
            elif entity.metadata and not entity.metadata.get("file_for_user", False):
                result.append(entity)

        return result

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
        """
        # First try to get the file by ID
        record = await self._prisma.fileregistry.find_unique(
            where={"id": file_id.value}
        )

        if not record:
            return None

        entity = self._to_entity(record)

        # Check 1: User owns the file
        if entity.user_email.value == user_email.value:
            return entity

        # Check 2: File is in same dept AND is shared (file_for_user=False)
        if entity.dept_id and entity.dept_id.value == dept_id.value:
            # Check if file is shared (file_for_user=False or not set)
            is_shared = not entity.metadata.get("file_for_user", False) if entity.metadata else True
            if is_shared:
                return entity

        # No access
        return None

    async def get_uningested_files(
        self,
        file_ids: Optional[list[FileId]],
        user_email: UserEmail,
        dept_id: DeptId,
    ) -> list[FileRegistry]:
        """
        Get files that have not been ingested/indexed yet.

        If file_ids is a list, returns only those files (if uningested and accessible).
        If file_ids is None, returns all uningested files accessible to user.

        Access control same as get_accessible_files:
        - User's own files (user_email matches)
        - Shared files in same dept (file_for_user=False)
        """
        where_clause: Dict[str, Any] = {
            "dept_id": dept_id.value,
            "indexed_in_chromadb": False,
            "category": "uploaded",  # Only uploaded files are ingestable
        }

        # If specific file_ids provided, use IN filter
        if file_ids:
            where_clause["id"] = {"in": [fid.value for fid in file_ids]}

        records = await self._prisma.fileregistry.find_many(
            where=where_clause,
            order={"created_at": "desc"},
        )

        # Apply access control filter
        result = []
        for r in records:
            entity = self._to_entity(r)
            # User's own files - always include
            if entity.user_email.value == user_email.value:
                result.append(entity)
            # Other users' files - only include if shared (file_for_user=False)
            elif entity.metadata and not entity.metadata.get("file_for_user", False):
                result.append(entity)

        return result

    async def mark_indexed(
        self,
        file_id: FileId,
        collection_name: str,
    ) -> None:
        """
        Mark a file as indexed in ChromaDB.

        Updates both indexed_in_chromadb column AND ingested field in metadata JSON
        to keep them in sync.

        Args:
            file_id: File ID to mark as indexed
            collection_name: Name of the ChromaDB collection
        """
        # First get current metadata to update ingested field
        record = await self._prisma.fileregistry.find_unique(
            where={"id": file_id.value}
        )

        if record:
            # Parse existing metadata and set ingested=True
            metadata = record.metadata
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            if metadata is None:
                metadata = {}
            metadata["ingested"] = True

            await self._prisma.fileregistry.update(
                where={"id": file_id.value},
                data={
                    "indexed_in_chromadb": True,
                    "chromadb_collection": collection_name,
                    "metadata": json.dumps(metadata),
                },
            )
