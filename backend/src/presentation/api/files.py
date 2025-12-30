"""
Files API Router - FastAPI endpoints for file operations.

Endpoints:
- POST /upload - Upload a file (multipart/form-data)
- GET /files - List files for current user
- GET /files/{file_id} - Download a file by ID

List Files Endpoint Guidelines:
================================
1. Create ListFilesQuery in src/application/queries/files/list_files.py
2. Query uses FileRegistryRepository.get_by_user()
3. Returns files for user's dept (both user-specific and shared)

Response format (match Flask):
{
    "files": [
        {
            "file_id": "cuid_xxx",
            "filename": "document.pdf",
            "file_path": "dept123/user@email.com/document.pdf",  # relative path
            "ext": "pdf",
            "size_kb": 123.4,
            "upload_at": "2024-01-01T12:00:00",
            "tags": ["tag1", "tag2"],
            "ingested": false
        }
    ]
}

Download File Endpoint Guidelines:
==================================
1. Get file by ID from FileRegistryRepository
2. Verify user owns the file (same user_email or shared file in same dept)
3. Return file using FileResponse from fastapi.responses
"""

import json
import logging
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from fastapi.responses import FileResponse
from dishka.integrations.fastapi import FromDishka, inject
from pydantic import BaseModel

from src.application.commands.files import (
    UploadFileCommand,
    UploadFileHandler,
)
from src.application.queries.files import (
    ListFilesQuery,
    ListFilesHandler,
    GetFileQuery,
    GetFileHandler,
)
from src.application.dto.file import FileInfoDTO
from src.domain.value_objects.file_id import FileId
from src.presentation.dependencies.auth import AuthUser, get_current_user
from src.config.settings import Config

logger = logging.getLogger(__name__)

# Maximum upload size in bytes (from Config)
MAX_UPLOAD_BYTES = int(Config.MAX_UPLOAD_MB * 1024 * 1024)


# ==================== RESPONSE MODELS ====================
class UploadFileResponse(BaseModel):
    """Response model for file upload."""

    file_id: str
    msg: str


class ListFilesResponse(BaseModel):
    """Response model for file list."""

    files: list[FileInfoDTO]


# ==================== ROUTERS ====================
# Two routers: /upload for upload, /files for list/download
upload_router = APIRouter(prefix="/upload", tags=["files"])
files_router = APIRouter(prefix="/files", tags=["files"])


# ==================== UPLOAD ENDPOINT ====================
@upload_router.post("", response_model=UploadFileResponse)
@inject
async def upload_file(
    handler: FromDishka[UploadFileHandler],
    current_user: AuthUser = Depends(get_current_user),
    file: UploadFile = File(...),
    tags: str = Form(default=""),
    file_for_user: str = Form(default="0"),
):
    """
    Upload a file.

    Request: multipart/form-data
    - file: binary file data (required)
    - tags: JSON string array, e.g. '["tag1", "tag2"]' (optional)
    - file_for_user: "1" for user-specific, "0" for shared (optional)

    Max file size: {MAX_UPLOAD_MB} MB (from Config.MAX_UPLOAD_MB)
    """
    # Parse tags from JSON string
    tags_list: list[str] = []
    if tags:
        try:
            tags_list = json.loads(tags)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid tags format. Expected JSON array.",
            )

    # Read file content
    content = await file.read()

    # Validate file size (matches Flask MAX_CONTENT_LENGTH)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum upload size is {Config.MAX_UPLOAD_MB} MB.",
        )

    # Create command
    command = UploadFileCommand(
        user_email=current_user.email,
        dept_id=current_user.dept,
        filename=file.filename or "unknown",
        content=content,
        content_type=file.content_type,
        tags=tags_list,
        file_for_user=(file_for_user == "1"),
    )

    try:
        result = await handler.execute(command)
        return UploadFileResponse(
            file_id=result.file_id,
            msg="File uploaded successfully",
        )
    except ValueError as ve:
        logger.warning(f"Upload validation error: {ve}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve),
        )
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


# ==================== LIST FILES ENDPOINT ====================
@files_router.get("", response_model=ListFilesResponse)
@inject
async def list_files(
    handler: FromDishka[ListFilesHandler],
    current_user: AuthUser = Depends(get_current_user),
):
    """
    List all files accessible to the current user.

    Returns files the user can access:
    - User's own files
    - Shared files in same department (file_for_user=False)

    Response format matches Flask /files endpoint.
    """
    user_email = current_user.email
    dept_id = current_user.dept

    if not user_email.value or not dept_id.value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User email or department ID missing",
        )

    try:
        # Create query and execute
        query = ListFilesQuery(
            user_email=user_email,
            dept_id=dept_id,
            category="uploaded",
        )
        result = await handler.execute(query)

        # result.files already contains FileInfoDTO objects
        return ListFilesResponse(files=result.files)

    except Exception as e:
        logger.error(f"List files error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list files",
        )


# ==================== DOWNLOAD FILE ENDPOINT ====================
@files_router.get("/{file_id}")
@inject
async def download_file(
    file_id: str,
    handler: FromDishka[GetFileHandler],
    current_user: AuthUser = Depends(get_current_user),
):
    """
    Download a file by its ID.

    Access control:
    - User owns the file (user_email matches), OR
    - File is shared (file_for_user=False) AND user in same dept

    Returns:
        FileResponse with the file content as attachment
    """
    # Create query with access control parameters
    query = GetFileQuery(
        file_id=FileId(file_id),
        user_email=current_user.email,
        dept_id=current_user.dept,
    )

    try:
        result = await handler.execute(query)

        if result.file is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=result.error or "File not found",
            )

        # Return file as download
        return FileResponse(
            path=result.file.storage_path.value,
            filename=result.file.original_name,
            media_type=result.file.mime_type or "application/octet-stream",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download file error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to download file",
        )
