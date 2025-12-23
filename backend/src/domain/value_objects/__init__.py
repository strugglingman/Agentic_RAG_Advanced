"""
VALUE OBJECTS - Immutable domain types

Each value object:
- Has no identity (compared by value, not by ID)
- Is immutable (frozen dataclass)
- Validates itself on creation
- Pure Python (no framework dependencies)
"""

from src.domain.value_objects.user_id import UserId
from src.domain.value_objects.user_email import UserEmail
from src.domain.value_objects.dept_id import DeptId
from src.domain.value_objects.conversation_id import ConversationId
from src.domain.value_objects.message_id import MessageId
from src.domain.value_objects.file_path import FilePath
from src.domain.value_objects.file_id import FileId

__all__ = [
    "UserId",
    "UserEmail",
    "DeptId",
    "ConversationId",
    "MessageId",
    "FileId",
    "FilePath",
]
