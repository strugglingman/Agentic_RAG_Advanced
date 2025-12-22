"""
FileManager Service - Unified file management system.

This service provides a centralized way to manage all files in the system:
- Chat attachments
- Uploaded RAG documents
- Downloaded files
- Created documents

All files are registered in the FileRegistry database table for easy reference and tracking.
"""

import os
import re
import logging
from datetime import datetime
from typing import List, Dict, Optional
from prisma import Prisma
from src.config.settings import Config

logger = logging.getLogger(__name__)


class FileManager:
    """
    Unified file management system.
    Handles storage, registration, and retrieval of all file types.
    """

    # File categories
    CATEGORY_CHAT = "chat"
    CATEGORY_UPLOADED = "uploaded"
    CATEGORY_DOWNLOADED = "downloaded"
    CATEGORY_CREATED = "created"

    VALID_CATEGORIES = {CATEGORY_CHAT, CATEGORY_UPLOADED, CATEGORY_DOWNLOADED, CATEGORY_CREATED}

    def __init__(self):
        """Initialize FileManager. Prisma client created lazily on first connect."""
        self.db = None  # Created lazily in __aenter__ to avoid event loop issues

    async def __aenter__(self):
        """Async context manager entry - create fresh database connection."""
        self.db = Prisma()
        await self.db.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - disconnect and cleanup."""
        if self.db:
            await self.db.disconnect()
            self.db = None

    async def register_file(
        self,
        user_email: str,
        category: str,
        original_name: str,
        storage_path: str,
        source_tool: str,
        mime_type: Optional[str] = None,
        size_bytes: Optional[int] = None,
        conversation_id: Optional[str] = None,
        dept_id: Optional[str] = None,
        source_url: Optional[str] = None,
        indexed_in_chromadb: bool = False,
        chromadb_collection: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Register a file in the system.

        Args:
            user_email: Email of user who owns the file
            category: 'chat', 'uploaded', 'downloaded', 'created'
            original_name: Original filename
            storage_path: Absolute path to file on disk
            source_tool: Tool that created/uploaded file
            mime_type: MIME type
            size_bytes: File size in bytes
            conversation_id: Chat session ID (if applicable)
            dept_id: Department ID for multi-tenancy
            source_url: Original URL (for downloaded files)
            indexed_in_chromadb: Whether file is indexed in ChromaDB
            chromadb_collection: ChromaDB collection name
            metadata: Additional metadata as dict

        Returns:
            file_id: Unique identifier for this file

        Raises:
            ValueError: If category is invalid
        """
        if category not in self.VALID_CATEGORIES:
            raise ValueError(
                f"Invalid category '{category}'. Must be one of: {self.VALID_CATEGORIES}"
            )

        # Create file registry record (download_url will be set after we have file_id)
        # Prisma requires Json fields to be serialized
        import json as json_lib

        file_record = await self.db.fileregistry.create(
            data={
                "user_email": user_email,
                "dept_id": dept_id,
                "category": category,
                "original_name": original_name,
                "storage_path": storage_path,
                "download_url": None,  # Will be updated below
                "mime_type": mime_type,
                "size_bytes": size_bytes,
                "source_tool": source_tool,
                "source_url": source_url,
                "conversation_id": conversation_id,
                "indexed_in_chromadb": indexed_in_chromadb,
                "chromadb_collection": chromadb_collection,
                "metadata": json_lib.dumps(metadata) if metadata else json_lib.dumps({}),
            }
        )

        # Generate download URL based on category
        filename = os.path.basename(storage_path)
        if category == "uploaded":
            # Uploaded files use the secure /api/files/{file_id} endpoint
            download_url = f"/api/files/{file_record.id}"
        else:
            # Downloaded, created, chat files use /api/downloads/ path (unified API pattern)
            download_url = f"/api/downloads/{user_email}/{filename}"

        # Update the record with the correct download_url
        file_record = await self.db.fileregistry.update(
            where={"id": file_record.id},
            data={"download_url": download_url}
        )

        logger.info(
            f"[FILE_MANAGER] Registered file: {file_record.id} | "
            f"user={user_email} | category={category} | name={original_name} | "
            f"download_url={download_url}"
        )

        # Return both file_id and download_url for tool use
        return {
            "file_id": file_record.id,
            "download_url": download_url
        }

    async def get_file_path(self, file_ref: str, user_email: str) -> str:
        """
        Resolve file reference to absolute storage path.

        Supports multiple reference formats:
        - File ID: "file_abc123xyz"
        - Category:name: "chat:report.pdf", "created:summary.pdf"
        - Just filename: "report.pdf" (searches all categories, most recent)

        Args:
            file_ref: File reference string
            user_email: User email for security check

        Returns:
            Absolute path to file on disk

        Raises:
            FileNotFoundError: File doesn't exist or user doesn't have access
            PermissionError: User trying to access another user's file
        """
        file_record = None

        # Case 1: File ID lookup (e.g., "file_abc123")
        if file_ref.startswith("file_") or len(file_ref) == 25:  # cuid length
            file_record = await self.db.fileregistry.find_first(
                where={"id": file_ref, "user_email": user_email}
            )

        # Case 2: Category:name lookup (e.g., "chat:report.pdf")
        elif ":" in file_ref:
            category, name = file_ref.split(":", 1)
            file_record = await self.db.fileregistry.find_first(
                where={
                    "user_email": user_email,
                    "category": category,
                    "original_name": name,
                },
                order={"created_at": "desc"},  # Most recent first
            )

        # Case 3: Just filename (e.g., "report.pdf")
        else:
            file_record = await self.db.fileregistry.find_first(
                where={"user_email": user_email, "original_name": file_ref},
                order={"created_at": "desc"},  # Most recent first
            )

        if not file_record:
            raise FileNotFoundError(
                f"File not found: {file_ref} for user {user_email}"
            )

        # Verify file exists on disk
        if not os.path.exists(file_record.storage_path):
            logger.error(
                f"[FILE_MANAGER] File in registry but missing on disk: {file_record.storage_path}"
            )
            raise FileNotFoundError(
                f"File registered but not found on disk: {file_ref}"
            )

        # Update accessed_at timestamp
        await self.db.fileregistry.update(
            where={"id": file_record.id}, data={"accessed_at": datetime.now()}
        )

        return file_record.storage_path

    async def list_files(
        self,
        user_email: str,
        category: Optional[str] = None,
        conversation_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict]:
        """
        List files available to user.
        Used to populate LLM context about available files.

        Args:
            user_email: User email
            category: Filter by category (optional)
            conversation_id: Filter by conversation (optional)
            limit: Maximum number of files to return

        Returns:
            List of file records as dictionaries
        """
        where_clause = {"user_email": user_email}

        if category:
            where_clause["category"] = category

        if conversation_id:
            where_clause["conversation_id"] = conversation_id

        files = await self.db.fileregistry.find_many(
            where=where_clause, order={"created_at": "desc"}, take=limit
        )

        # Convert to dicts for easier handling
        return [
            {
                "id": f.id,
                "category": f.category,
                "original_name": f.original_name,
                "download_url": f.download_url,
                "mime_type": f.mime_type,
                "size_bytes": f.size_bytes,
                "created_at": f.created_at.isoformat() if f.created_at else None,
                "source_tool": f.source_tool,
            }
            for f in files
        ]

    async def save_chat_attachment(
        self,
        user_email: str,
        filename: str,
        content: bytes,
        mime_type: str,
        conversation_id: Optional[str] = None,
        dept_id: Optional[str] = None,
    ) -> str:
        """
        Save chat attachment to disk and register in database.
        Replaces Redis-based temporary storage.

        Args:
            user_email: User email
            filename: Original filename
            content: File content as bytes
            mime_type: MIME type
            conversation_id: Chat session ID
            dept_id: Department ID

        Returns:
            file_id: Unique identifier for the saved file
        """
        # Create user's chat directory
        user_chat_dir = os.path.join(Config.DOWNLOAD_BASE, user_email, "chat")
        os.makedirs(user_chat_dir, exist_ok=True)

        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
        safe_filename = re.sub(r"[^\w\-_\. ]", "_", filename)
        unique_filename = f"{timestamp}_{safe_filename}"
        storage_path = os.path.join(user_chat_dir, unique_filename)

        # Save file to disk
        with open(storage_path, "wb") as f:
            f.write(content)

        # Register in database
        result = await self.register_file(
            user_email=user_email,
            category=self.CATEGORY_CHAT,
            original_name=filename,
            storage_path=storage_path,
            source_tool="chat_upload",
            mime_type=mime_type,
            size_bytes=len(content),
            conversation_id=conversation_id,
            dept_id=dept_id,
        )

        file_id = result["file_id"]

        logger.info(
            f"[FILE_MANAGER] Saved chat attachment: {file_id} | "
            f"user={user_email} | file={filename} | size={len(content)} bytes"
        )

        return file_id

    async def delete_file(self, file_id: str, user_email: str) -> bool:
        """
        Delete a file from both database and disk.

        Args:
            file_id: File ID to delete
            user_email: User email for security check

        Returns:
            True if deleted successfully, False otherwise

        Raises:
            PermissionError: User trying to delete another user's file
        """
        file_record = await self.db.fileregistry.find_first(
            where={"id": file_id, "user_email": user_email}
        )

        if not file_record:
            raise PermissionError(
                f"File not found or access denied: {file_id} for user {user_email}"
            )

        # Delete from disk if exists
        if os.path.exists(file_record.storage_path):
            try:
                os.remove(file_record.storage_path)
                logger.info(f"[FILE_MANAGER] Deleted file from disk: {file_record.storage_path}")
            except Exception as e:
                logger.error(f"[FILE_MANAGER] Failed to delete file from disk: {e}")

        # Delete from database
        await self.db.fileregistry.delete(where={"id": file_id})
        logger.info(f"[FILE_MANAGER] Deleted file from registry: {file_id}")

        return True

    async def get_file_by_id(self, file_id: str, user_email: str) -> Optional[Dict]:
        """
        Get file metadata by ID.

        Args:
            file_id: File ID
            user_email: User email for security check

        Returns:
            File metadata dict or None if not found
        """
        file_record = await self.db.fileregistry.find_first(
            where={"id": file_id, "user_email": user_email}
        )

        if not file_record:
            return None

        return {
            "id": file_record.id,
            "category": file_record.category,
            "original_name": file_record.original_name,
            "storage_path": file_record.storage_path,
            "download_url": file_record.download_url,
            "mime_type": file_record.mime_type,
            "size_bytes": file_record.size_bytes,
            "created_at": file_record.created_at.isoformat()
            if file_record.created_at
            else None,
            "source_tool": file_record.source_tool,
            "conversation_id": file_record.conversation_id,
        }
