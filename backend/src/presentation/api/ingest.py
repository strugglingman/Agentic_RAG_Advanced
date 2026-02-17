"""
Ingest API Router - FastAPI endpoint for document ingestion with SSE progress.

Endpoints:
- POST /ingest - Ingest documents with SSE progress streaming
- POST /ingest/cancel - Cancel an active ingestion job
- GET /ingest/active - Get list of files currently being ingested
"""

import asyncio
import logging
import uuid
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from dishka.integrations.fastapi import FromDishka, inject
from pydantic import BaseModel

from src.domain.ports.repositories import FileRegistryRepository
from src.utils.stream_utils import sse_event
from src.domain.value_objects.file_id import FileId
from src.services.ingestion import ingest_file
from src.services.retrieval import build_bm25
from src.services.vector_db import VectorDB
from src.infrastructure.jobs import IngestJobStore
from src.presentation.dependencies.auth import AuthUser, get_current_user

logger = logging.getLogger(__name__)


# ==================== REQUEST/RESPONSE MODELS ====================


class IngestRequest(BaseModel):
    """
    Request body for document ingestion.

    Examples:
        {"file_ids": ["ALL"]}  - Ingest all uningested files
        {"file_ids": ["cuid123"]}  - Ingest single file
        {"file_ids": ["cuid1", "cuid2", "cuid3"]}  - Batch ingest
    """

    file_ids: List[str]  # List of file IDs or ["ALL"] for all uningested files


class CancelRequest(BaseModel):
    """Request body for cancellation."""

    job_id: str


class CancelResponse(BaseModel):
    """Response for cancel request."""

    success: bool
    message: str


class ActiveFilesResponse(BaseModel):
    """Response with list of files currently being ingested."""

    file_ids: List[str]


# ==================== ROUTER ====================

router = APIRouter(prefix="/ingest", tags=["ingest"])


# ==================== ENDPOINTS ====================


@router.post("")
@inject
async def ingest_documents(
    request: IngestRequest,
    file_repo: FromDishka[FileRegistryRepository],
    vector_db: FromDishka[VectorDB],
    job_store: FromDishka[Optional[IngestJobStore]],
    current_user: AuthUser = Depends(get_current_user),
):
    """
    Ingest documents with SSE progress streaming.

    Returns Server-Sent Events stream with progress updates.
    Frontend should use EventSource to consume this.

    Events:
    - progress: {job_id, current_file, current_index, total_files, total_chunks, status}
    - complete: {job_id, total_files, ingested_files, total_chunks, status, results}
    - cancelled: {job_id, processed_files, total_files, status}
    - error: {job_id, error, status}
    """
    # Parse file_ids - check if "ALL" or specific IDs
    is_all = len(request.file_ids) == 1 and request.file_ids[0].upper() == "ALL"
    file_id_list = None if is_all else [FileId(fid) for fid in request.file_ids]

    # Get files to ingest
    uningested_files = await file_repo.get_uningested_files(
        file_ids=file_id_list,  # None means all, list means specific files
        user_email=current_user.email,
        dept_id=current_user.dept,
    )

    if not uningested_files:
        # Return immediate JSON response if no files
        return StreamingResponse(
            iter(
                [
                    sse_event(
                        "complete",
                        {
                            "job_id": "",
                            "total_files": 0,
                            "ingested_files": 0,
                            "total_chunks": 0,
                            "status": "completed",
                            "message": "No files to ingest",
                            "results": [],
                        },
                    )
                ]
            ),
            media_type="text/event-stream",
        )

    # Generate job ID
    job_id = str(uuid.uuid4())
    file_ids = [f.id for f in uningested_files]

    # Track job in Redis (if available)
    if job_store:
        await job_store.start_job(job_id, current_user.email.value, file_ids)

    async def generate_events():
        """Generator that yields SSE events during ingestion."""
        total_chunks = 0
        ingested_count = 0
        results = []
        cancelled = False

        try:
            for i, file in enumerate(uningested_files):
                # Check for cancellation before each file
                if job_store and await job_store.is_cancelled(job_id):
                    cancelled = True
                    yield sse_event(
                        "cancelled",
                        {
                            "job_id": job_id,
                            "processed_files": i,
                            "total_files": len(uningested_files),
                            "status": "cancelled",
                        },
                    )
                    break

                # Ingest the file
                result = await asyncio.get_running_loop().run_in_executor(
                    None,
                    ingest_file,
                    vector_db,
                    file,
                    current_user.email.value,
                    current_user.dept.value,
                )

                results.append(
                    {
                        "file_id": result.file_id,
                        "filename": result.filename,
                        "chunks_count": result.chunks_count,
                        "success": result.success,
                        "error": result.error,
                    }
                )

                # Track completed file ID for real-time UI update
                completed_file_id = None
                if result.success and result.chunks_count > 0:
                    # Mark file as indexed
                    collection_name = getattr(vector_db.collection, "name", "docs")
                    await file_repo.mark_indexed(
                        file_id=FileId(file.id),
                        collection_name=collection_name,
                    )
                    total_chunks += result.chunks_count
                    ingested_count += 1
                    completed_file_id = file.id

                # Send progress event AFTER processing (includes completed file ID)
                yield sse_event(
                    "progress",
                    {
                        "job_id": job_id,
                        "current_file": file.original_name,
                        "current_index": i + 1,
                        "total_files": len(uningested_files),
                        "total_chunks": total_chunks,
                        "status": "processing",
                        "completed_file_id": completed_file_id,
                    },
                )

            # Build BM25 index if we ingested anything
            if ingested_count > 0:
                try:
                    await asyncio.get_running_loop().run_in_executor(
                        None,
                        build_bm25,
                        vector_db,
                        current_user.dept.value,
                        current_user.email.value,
                    )
                except Exception as e:
                    logger.error(f"Failed to rebuild BM25: {e}")

            # Send completion event (if not cancelled)
            if not cancelled:
                yield sse_event(
                    "complete",
                    {
                        "job_id": job_id,
                        "total_files": len(uningested_files),
                        "ingested_files": ingested_count,
                        "total_chunks": total_chunks,
                        "status": "completed",
                        "results": results,
                    },
                )

        except Exception as e:
            logger.error(f"Ingestion error: {e}")
            yield sse_event(
                "error",
                {
                    "job_id": job_id,
                    "error": "Ingestion failed",
                    "status": "error",
                },
            )

        finally:
            # Cleanup job tracking
            if job_store:
                await job_store.complete_job(job_id)

    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Job-ID": job_id,
        },
    )


@router.post("/cancel", response_model=CancelResponse)
@inject
async def cancel_ingestion(
    request: CancelRequest,
    job_store: FromDishka[Optional[IngestJobStore]],
    current_user: AuthUser = Depends(get_current_user),
):
    """
    Cancel an active ingestion job.

    The job will stop after the current file finishes processing.
    Already ingested files remain indexed (upsert is idempotent).
    """
    if not job_store:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Job tracking not available (Redis required)",
        )

    # Verify ownership
    owner = await job_store.get_job_owner(request.job_id)
    if not owner:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found or already completed",
        )

    if owner != current_user.email.value:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only cancel your own jobs",
        )

    success = await job_store.cancel_job(request.job_id)

    return CancelResponse(
        success=success,
        message="Cancellation requested" if success else "Job not found",
    )


@router.get("/active", response_model=ActiveFilesResponse)
@inject
async def get_active_files(
    job_store: FromDishka[Optional[IngestJobStore]],
    current_user: AuthUser = Depends(get_current_user),
):
    """
    Get list of file IDs currently being ingested.

    Used by frontend to disable ingest buttons for files in progress.
    """
    if not job_store:
        return ActiveFilesResponse(file_ids=[])

    file_ids = await job_store.get_all_active_file_ids()
    return ActiveFilesResponse(file_ids=file_ids)
