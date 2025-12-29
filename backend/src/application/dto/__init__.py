"""
DTOs - Data Transfer Objects

DTOs for transferring data between layers:
- chat.py → MessageDTO
- file.py → FileDTO, FileInfoDTO

Note: These are different from domain entities.
DTOs are for API input/output, entities are for business logic.
"""

from src.application.dto.chat import MessageDTO
from src.application.dto.file import FileDTO, FileInfoDTO

__all__ = [
    "MessageDTO",
    "FileDTO",
    "FileInfoDTO",
]
