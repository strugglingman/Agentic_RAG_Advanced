"""File DTOs for API request/response."""

from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class FileDTO(BaseModel):
    """General file DTO with all fields."""

    id: str
    original_name: str
    category: str
    download_url: str
    mime_type: Optional[str] = None
    size_bytes: Optional[int] = None
    created_at: datetime


class FileInfoDTO(BaseModel):
    """
    File info DTO for list files response.

    Matches Flask /files endpoint response format:
    - file_id: cuid from database
    - filename: original filename
    - file_path: relative path (UPLOAD_BASE stripped)
    - ext: file extension
    - size_kb: size in kilobytes
    - upload_at: upload timestamp ISO string
    - tags: list of tags
    - ingested: whether file is indexed in ChromaDB
    """

    file_id: str
    filename: str
    file_path: str
    ext: str
    size_kb: float
    upload_at: Optional[str] = None
    tags: Optional[list[str]] = None
    ingested: bool = False