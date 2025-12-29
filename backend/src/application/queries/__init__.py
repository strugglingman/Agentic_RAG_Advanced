"""
QUERIES - Read operations (CQRS)

Queries retrieve data without modifying state. Each query has:
- Query class: Parameters for the read
- Handler class: Executes the read

Subfolders:
- chat/          → get_chat_history
- conversations/ → list_conversations, get_conversation
- files/         → list_files, get_file
"""

from src.application.queries.files import (
    ListFilesQuery,
    ListFilesHandler,
    ListFilesResult,
    FileInfoDTO,
)

__all__ = [
    # files
    "ListFilesQuery",
    "ListFilesHandler",
    "ListFilesResult",
    "FileInfoDTO",
]