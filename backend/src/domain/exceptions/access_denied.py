"""
AccessDeniedError - Raised when user lacks permission to access a resource.
Maps to: HTTP 403 Forbidden
"""


class AccessDeniedError(Exception):
    """Raised when user lacks permission to access a resource"""

    def __init__(self, message: str = "Access denied"):
        super().__init__(message)
