"""
REPOSITORY PORTS - Data persistence interfaces

Each repository port:
- Is an abstract base class (ABC)
- Defines methods the domain needs
- Does NOT specify implementation (Prisma, SQLAlchemy, etc.)

Infrastructure layer provides implementations.
"""

from src.domain.ports.repositories.conversation_repository import ConversationRepository
from src.domain.ports.repositories.message_repository import MessageRepository
from src.domain.ports.repositories.user_repository import UserRepository
from src.domain.ports.repositories.file_registry_repository import FileRegistryRepository

__all__ = [
    "ConversationRepository",
    "MessageRepository",
    "UserRepository",
    "FileRegistryRepository",
]
