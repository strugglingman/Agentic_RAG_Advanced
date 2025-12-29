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
    # Required fields (no defaults) - must come first
    id: UserId
    email: UserEmail
    password_hash: str
    role: str
    created_at: datetime
    updated_at: datetime
    # Optional fields (with defaults) - must come last
    dept: Optional[DeptId] = None
    name: Optional[str] = None

    def __post_init__(self):
        valid_roles = ["staff", "admin", "manager"]
        if self.role not in valid_roles:
            raise ValueError(
                f"Invalid role: {self.role}. Must be one of {valid_roles}."
            )
