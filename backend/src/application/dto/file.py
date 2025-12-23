"""File DTOs for API request/response."""

from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class FileDTO(BaseModel):
    id: str
    original_name: str
    category: str
    download_url: str
    mime_type: Optional[str] = None
    size_bytes: Optional[int] = None
    created_at: datetime