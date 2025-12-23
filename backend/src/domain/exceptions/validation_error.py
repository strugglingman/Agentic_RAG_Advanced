"""
DomainValidationError - Raised when a business rule is violated.
Maps to: HTTP 422 Unprocessable Entity
"""


class DomainValidationError(Exception):
    """Exception raised for domain validation errors."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message
