"""
Ingest API Router - FastAPI endpoint for document ingestion.

Endpoints:
- POST /ingest - Ingest documents into ChromaDB vector database

Maps from: src/routes/ingest.py
"""

import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from dishka.integrations.fastapi import FromDishka, inject
from pydantic import BaseModel

from src.application.commands.files.ingest_document import (
    IngestDocumentCommand,
    IngestDocumentHandler,
    IngestDocumentResult,
)
from src.domain.value_objects.file_id import FileId
from src.presentation.dependencies.auth import AuthUser, get_current_user

logger = logging.getLogger(__name__)


# ==================== REQUEST/RESPONSE MODELS ====================


class IngestRequest(BaseModel):
    """
    Request body for document ingestion.

    Matches frontend:
    {
        "file_id": "clx_abc123"  // specific file ID or "ALL" for all uningested files
    }
    """

    file_id: str  # File ID (cuid) or "ALL" to ingest all uningested files


class IngestFileResult(BaseModel):
    """Result for a single file ingestion."""

    file_id: str
    filename: str
    chunks_count: int
    success: bool
    error: Optional[str] = None


class IngestResponse(BaseModel):
    """Response for document ingestion."""

    success: bool
    message: str
    total_files: int
    ingested_files: int
    total_chunks: int
    results: list[IngestFileResult]


# ==================== ROUTER ====================

router = APIRouter(prefix="/ingest", tags=["ingest"])


# ==================== ENDPOINTS ====================


@router.post("", response_model=IngestResponse)
@inject
async def ingest_documents(
    request: IngestRequest,
    handler: FromDishka[IngestDocumentHandler],
    current_user: AuthUser = Depends(get_current_user),
):
    """
    Ingest documents into ChromaDB vector database.

    - If file_id is a specific ID: ingest only that file
    - If file_id is "ALL": ingest all uningested files accessible to user

    Access control:
    - User can ingest their own files
    - User can ingest shared files in their department (file_for_user=False)

    Returns:
        IngestResponse with summary of ingestion results
    """
    try:
        # Determine if ingesting single file or all files
        file_id_value = request.file_id.strip()

        if file_id_value.upper() == "ALL":
            # Ingest all uningested files
            file_id = None
        else:
            # Ingest specific file - validate it's a valid cuid
            try:
                file_id = FileId(file_id_value)
            except ValueError as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid file_id: {str(e)}",
                )

        # Create command
        command = IngestDocumentCommand(
            file_id=file_id,
            user_email=current_user.email,
            dept_id=current_user.dept,
        )

        # Execute handler
        result: IngestDocumentResult = await handler.execute(command)

        # Map to response
        return IngestResponse(
            success=result.success,
            message=result.message,
            total_files=result.total_files,
            ingested_files=result.ingested_files,
            total_chunks=result.total_chunks,
            results=[
                IngestFileResult(
                    file_id=r["file_id"],
                    filename=r["filename"],
                    chunks_count=r["chunks_count"],
                    success=r["success"],
                    error=r.get("error"),
                )
                for r in result.results
            ],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ingest error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}",
        )
