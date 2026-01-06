"""
FileStorageService - Pure disk I/O operations.

This service handles all file system operations:
- Save files to disk
- Read files from disk
- Delete files from disk
- Create directories

This is a SYNC service - no database, no async.
For database operations, use FileRegistryRepository.
For coordinated operations (disk + DB), use FileService in application layer.
"""

import os
import re
import logging
from datetime import datetime
from typing import Optional

from src.config.settings import Config

logger = logging.getLogger(__name__)


class FileStorageService:
    """
    Pure file system operations service.

    All methods are synchronous since file I/O in Python is sync.
    For async file I/O, consider using aiofiles (not needed for most cases).
    """

    # File categories (for directory structure)
    CATEGORY_CHAT = "chat"
    CATEGORY_UPLOADED = "uploaded"
    CATEGORY_DOWNLOADED = "downloaded"
    CATEGORY_CREATED = "created"

    def __init__(
        self,
        upload_base: str = None,
        download_base: str = None,
    ):
        """
        Initialize FileStorageService.

        Args:
            upload_base: Base directory for uploads (default: Config.UPLOAD_BASE)
            download_base: Base directory for downloads (default: Config.DOWNLOAD_BASE)
        """
        self.upload_base = upload_base or Config.UPLOAD_BASE
        self.download_base = download_base or Config.DOWNLOAD_BASE

    def save_file(
        self,
        content: bytes,
        directory: str,
        filename: str,
        make_unique: bool = True,
    ) -> str:
        """
        Save file content to disk.

        Args:
            content: File content as bytes
            directory: Target directory (will be created if not exists)
            filename: Original filename
            make_unique: If True, prepend timestamp to make filename unique

        Returns:
            Absolute path to saved file
        """
        # Ensure directory exists
        os.makedirs(directory, exist_ok=True)

        # Sanitize and optionally make unique
        safe_filename = self._sanitize_filename(filename)
        if make_unique:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
            safe_filename = f"{timestamp}_{safe_filename}"

        file_path = os.path.join(directory, safe_filename)

        # Write file
        with open(file_path, "wb") as f:
            f.write(content)

        logger.debug(f"[FileStorage] Saved file: {file_path} ({len(content)} bytes)")
        return file_path

    def read_file(self, file_path: str) -> bytes:
        """
        Read file content from disk.

        Args:
            file_path: Absolute path to file

        Returns:
            File content as bytes

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "rb") as f:
            content = f.read()

        logger.debug(f"[FileStorage] Read file: {file_path} ({len(content)} bytes)")
        return content

    def delete_file(self, file_path: str) -> bool:
        """
        Delete file from disk.

        Args:
            file_path: Absolute path to file

        Returns:
            True if deleted, False if file didn't exist
        """
        if not os.path.exists(file_path):
            logger.warning(f"[FileStorage] File not found for deletion: {file_path}")
            return False

        try:
            os.remove(file_path)
            logger.debug(f"[FileStorage] Deleted file: {file_path}")
            return True
        except Exception as e:
            logger.error(f"[FileStorage] Failed to delete file {file_path}: {e}")
            return False

    def file_exists(self, file_path: str) -> bool:
        """Check if file exists on disk."""
        return os.path.exists(file_path)

    def get_file_size(self, file_path: str) -> int:
        """
        Get file size in bytes.

        Args:
            file_path: Absolute path to file

        Returns:
            File size in bytes

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        return os.path.getsize(file_path)

    def create_upload_dir(
        self,
        dept_id: str,
        user_id: str,
    ) -> str:
        """
        Create upload directory for user.

        Directory structure: {upload_base}/{dept_id}/{user_id}/

        Args:
            dept_id: Department ID
            user_id: User ID or email

        Returns:
            Absolute path to created directory
        """
        directory = os.path.join(self.upload_base, dept_id, user_id)
        os.makedirs(directory, exist_ok=True)
        logger.debug(f"[FileStorage] Created upload dir: {directory}")
        return directory

    def create_download_dir(
        self,
        user_id: str,
        subdirectory: Optional[str] = None,
    ) -> str:
        """
        Create download directory for user.

        Directory structure: {download_base}/{user_id}/{subdirectory}/

        Args:
            user_id: User ID or email
            subdirectory: Optional subdirectory (e.g., "chat", "created")

        Returns:
            Absolute path to created directory
        """
        if subdirectory:
            directory = os.path.join(self.download_base, user_id, subdirectory)
        else:
            directory = os.path.join(self.download_base, user_id)

        os.makedirs(directory, exist_ok=True)
        return directory

    def get_chat_attachment_dir(self, user_email: str) -> str:
        """Get directory for chat attachments."""
        return self.create_download_dir(user_email, self.CATEGORY_CHAT)

    def get_created_files_dir(self, user_email: str) -> str:
        """Get directory for created files (documents, reports)."""
        return self.create_download_dir(user_email, self.CATEGORY_CREATED)

    def get_downloaded_files_dir(self, user_email: str) -> str:
        """Get directory for downloaded files."""
        return self.create_download_dir(user_email, self.CATEGORY_DOWNLOADED)

    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename to remove unsafe characters.

        Args:
            filename: Original filename

        Returns:
            Sanitized filename safe for file system
        """
        # Replace unsafe characters with underscore
        safe = re.sub(r"[^\w\-_\. ]", "_", filename)
        # Remove leading/trailing whitespace
        safe = safe.strip()
        # Ensure not empty
        if not safe:
            safe = "unnamed_file"
        return safe

    def get_extension(self, filename: str) -> str:
        """
        Get file extension from filename.

        Args:
            filename: Filename with extension

        Returns:
            Extension without dot (lowercase), or empty string
        """
        if "." in filename:
            return filename.rsplit(".", 1)[1].lower()
        return ""
