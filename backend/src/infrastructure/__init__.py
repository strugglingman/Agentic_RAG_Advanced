"""
Infrastructure Layer - Technical implementations of domain ports.

This layer contains:
- persistence/: Database implementations (Prisma repositories)
- storage/: File system operations (FileStorageService)
- external/: External service integrations (email, etc.)
"""

from src.infrastructure.storage import FileStorageService

__all__ = ["FileStorageService"]