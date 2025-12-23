"""
User Entity - A system user.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from src.domain.value_objects.dept_id import DeptId
from src.domain.value_objects.user_id import UserId
from src.domain.value_objects.user_email import UserEmail


@dataclass
class User:
    id: UserId
    email: UserEmail
    dept: Optional[DeptId] = None
    password_hash: str
    name: Optional[str] = None
    role: str
    created_at: datetime
    updated_at: datetime

    def __post_init__(self):
        valid_roles = ["staff", "admin", "manager"]
        if self.role not in valid_roles:
            raise ValueError(
                f"Invalid role: {self.role}. Must be one of {valid_roles}."
            )
