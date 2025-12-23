"""
DOMAIN EXCEPTIONS - Business rule violations

These exceptions are raised by domain logic and caught by presentation layer.
Presentation layer maps them to HTTP status codes.
"""

from src.domain.exceptions.entity_not_found import EntityNotFoundError
from src.domain.exceptions.access_denied import AccessDeniedError
from src.domain.exceptions.validation_error import DomainValidationError

__all__ = [
    "EntityNotFoundError",
    "AccessDeniedError",
    "DomainValidationError",
]
