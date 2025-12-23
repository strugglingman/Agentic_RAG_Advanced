"""
User Repository Port - Interface for user persistence.
Implementation: src/infrastructure/persistence/prisma_user_repository.py
"""

from abc import ABC, abstractmethod
from typing import Optional
from src.domain.entities.user import User
from src.domain.value_objects.user_email import UserEmail
from src.domain.value_objects.user_id import UserId


class UserRepository(ABC):
    @abstractmethod
    async def get_by_id(self, user_id: UserId) -> Optional[User]: ...

    @abstractmethod
    async def get_by_email(self, email: UserEmail) -> Optional[User]: ...

    @abstractmethod
    async def save(self, user: User) -> None: ...

    @abstractmethod
    async def delete(self, user_id: UserId) -> bool: ...
