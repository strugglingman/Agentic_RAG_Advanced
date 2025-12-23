"""
ENTITIES - Business objects with identity

Each entity:
- Has a unique identifier
- Has behavior (methods)
- Can change state over time
- Pure Python dataclasses (no ORM, no Pydantic)
"""

from src.domain.entities.conversation import Conversation
from src.domain.entities.message import Message
from src.domain.entities.user import User
from src.domain.entities.file_registry import FileRegistry

__all__ = [
    "Conversation",
    "Message",
    "User",
    "FileRegistry",
]
