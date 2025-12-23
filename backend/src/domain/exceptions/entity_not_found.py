"""
EntityNotFoundError - Raised when a requested entity does not exist.
Maps to: HTTP 404 Not Found
"""


class EntityNotFoundError(Exception):
    """Exception raised when a requested entity is not found."""

    def __init__(self, message: str = "The requested entity was not found."):
        super().__init__(message)
