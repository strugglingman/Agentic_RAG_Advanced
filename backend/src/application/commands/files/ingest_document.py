"""
IngestDocument Command - Index document into ChromaDB.

Flow:
1. Query FileRegistry for uningested files (single file or ALL)
2. For each file:
   a. Read file content from disk
   b. Split into chunks
   c. Upsert chunks to ChromaDB with metadata
   d. Mark file as indexed in FileRegistry
3. Rebuild BM25 index for user/dept

Maps from: src/routes/ingest.py and src/services/ingestion.py
"""

from dataclasses import dataclass, field
from typing import Optional
from chromadb.api.models.Collection import Collection
from src.application.common.interfaces import Command, CommandHandler
from src.domain.ports.repositories import FileRegistryRepository
from src.domain.value_objects.file_id import FileId
from src.domain.value_objects.user_email import UserEmail
from src.domain.value_objects.dept_id import DeptId
from src.services.ingestion import ingest_file, IngestResult
from src.services.retrieval import build_bm25
from logging import getLogger

logger = getLogger(__name__)


# ==================== RESULT ====================


@dataclass
class IngestDocumentResult:
    """Result of document ingestion."""

    success: bool
    message: str
    total_files: int = 0
    ingested_files: int = 0
    total_chunks: int = 0
    results: list[dict] = field(default_factory=list)


# ==================== COMMAND ====================


@dataclass(frozen=True)
class IngestDocumentCommand(Command[IngestDocumentResult]):
    """
    Command to ingest documents into ChromaDB.

    Args:
        file_id: Specific file ID to ingest, or None to ingest ALL uningested files
        user_email: User's email for access control
        dept_id: User's department ID for access control
    """

    file_id: Optional[FileId]  # None means ingest ALL uningested files
    user_email: UserEmail
    dept_id: DeptId


# ==================== HANDLER ====================


class IngestDocumentHandler(CommandHandler[IngestDocumentResult]):
    """Handler for IngestDocumentCommand."""

    def __init__(
        self,
        file_repo: FileRegistryRepository,
        collection: Collection,
    ):
        self._file_repo = file_repo
        self._collection = collection

    async def execute(self, command: IngestDocumentCommand) -> IngestDocumentResult:
        """
        Execute document ingestion.

        1. Get uningested files from FileRegistry
        2. For each file, call ingest_file() to chunk and index
        3. Mark each file as indexed in FileRegistry
        4. Rebuild BM25 index
        5. Return summary result
        """
        user_email = command.user_email.value
        dept_id = command.dept_id.value

        # 1. Get files to ingest
        uningested_files = await self._file_repo.get_uningested_files(
            file_id=command.file_id,
            user_email=command.user_email,
            dept_id=command.dept_id,
        )

        if not uningested_files:
            return IngestDocumentResult(
                success=True,
                message="No files to ingest.",
                total_files=0,
                ingested_files=0,
                total_chunks=0,
            )

        # 2. Ingest each file
        results: list[dict] = []
        total_chunks = 0
        ingested_count = 0

        for file in uningested_files:
            # Call the new ingest_file function (no .meta.json)
            result: IngestResult = ingest_file(
                collection=self._collection,
                file=file,
                user_email=user_email,
                dept_id=dept_id,
            )

            results.append({
                "file_id": result.file_id,
                "filename": result.filename,
                "chunks_count": result.chunks_count,
                "success": result.success,
                "error": result.error,
            })

            if result.success and result.chunks_count > 0:
                # 3. Mark file as indexed in FileRegistry
                collection_name = (
                    self._collection.name
                    if hasattr(self._collection, "name")
                    else "docs"
                )
                await self._file_repo.mark_indexed(
                    file_id=FileId(file.id),
                    collection_name=collection_name,
                )
                total_chunks += result.chunks_count
                ingested_count += 1
                logger.info(
                    f"[INGEST] Successfully indexed file {file.id} ({file.original_name}): "
                    f"{result.chunks_count} chunks"
                )
            elif result.error:
                logger.warning(
                    f"[INGEST] Failed to ingest file {file.id} ({file.original_name}): "
                    f"{result.error}"
                )

        # 4. Rebuild BM25 index
        try:
            build_bm25(self._collection, dept_id, user_email)
            logger.info(f"[INGEST] Rebuilt BM25 index for dept={dept_id}, user={user_email}")
        except Exception as e:
            logger.error(f"[INGEST] Failed to rebuild BM25 index: {e}")

        # 5. Build result message
        if ingested_count > 0:
            message = (
                f"Ingestion completed: {ingested_count}/{len(uningested_files)} files, "
                f"{total_chunks} total chunks indexed."
            )
        else:
            message = "No new content ingested."

        return IngestDocumentResult(
            success=True,
            message=message,
            total_files=len(uningested_files),
            ingested_files=ingested_count,
            total_chunks=total_chunks,
            results=results,
        )
