"""
Ingestion service for document processing and ChromaDB storage.
Handles reading, chunking, and upserting documents to vector database.
"""

import os
import json
import hashlib
import asyncio
import logging
from typing import Optional
from dataclasses import dataclass
from src.services.document_processor import read_text, make_chunks
from src.services.file_manager import FileManager
from src.domain.entities.file_registry import FileRegistry

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    """Result of ingesting a single file."""

    file_id: str
    filename: str
    chunks_count: int
    success: bool
    error: Optional[str] = None


def make_id(text):
    """Generate MD5 hash for a text string."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def ingest_one(
    collection, info: Optional[dict], app_user_id: str, app_dept_id: str
) -> Optional[str]:
    """
    Ingest a single document into the vector database.

    Args:
        collection: ChromaDB collection
        info: File metadata dictionary from .meta.json
        app_user_id: Current user ID (from auth)
        app_dept_id: Current department ID (from auth)

    Returns:
        File ID if successfully ingested, None otherwise
    """
    if not info:
        return None

    dept_id = info.get("dept_id", "")
    user_id = info.get("user_id", "")
    if not dept_id or not user_id:
        return None

    file_for_user = info.get("file_for_user", False)
    if file_for_user and (dept_id != app_dept_id or user_id != app_user_id):
        return None

    ingested = info.get("ingested", False)
    if ingested:
        return None

    file_path = info.get("file_path", "")
    if not os.path.exists(file_path):
        return None

    pages_text = read_text(file_path)
    if not pages_text:
        return None

    # Chunking - now returns list of (page_num, chunk_text) tuples
    chunks_with_pages = make_chunks(pages_text)
    filename = info.get("filename", os.path.basename(file_path))

    # Upsert to chroma
    ids, docs, metas = [], [], []
    seen = set()
    for page_num, chunk in chunks_with_pages:
        # Incorporate page number into the ID seed to avoid collisions
        if file_for_user:
            seed = f"{dept_id}|{user_id}|{filename}|p{page_num}|{chunk}"
        else:
            seed = f"{dept_id}|{filename}|p{page_num}|{chunk}"
        chunk_id = make_id(seed)
        ids.append(chunk_id)
        docs.append(chunk)
        metas.append(
            {
                "dept_id": info.get("dept_id", ""),
                "user_id": info.get("user_id", ""),
                "file_for_user": file_for_user,
                "chunk_id": chunk_id,
                "source": filename,
                "ext": filename.split(".")[-1].lower(),
                "file_id": info.get("file_id", ""),
                "size_kb": info.get("size_kb", 0),
                "tags": info.get("tags", "").lower(),
                "upload_at": info.get("upload_at", ""),
                "uploaded_at_ts": info.get("uploaded_at_ts", 0),
                "page": page_num,
            }
        )

        if chunk_id in seen:
            print(
                f"Duplicate chunk detected even with page in ID: {chunk_id}, "
                f"file: {filename}, page: {page_num}, first 80 chars: {chunk[:80]}"
            )
        else:
            seen.add(chunk_id)

    if docs:
        collection.upsert(ids=ids, documents=docs, metadatas=metas)

    # Set ingested flag
    with open(file_path + ".meta.json", "w", encoding="utf-8") as info_f:
        info["ingested"] = True
        json.dump(info, info_f, indent=2)

    # Update FileRegistry to mark file as indexed in ChromaDB and sync metadata
    registry_file_id = info.get("registry_file_id")
    if registry_file_id and docs:
        try:

            async def update_chromadb_status():
                async with FileManager() as fm:
                    # Get the file record
                    file_record = await fm.db.fileregistry.find_first(
                        where={"id": registry_file_id}
                    )
                    if file_record:
                        # Update indexed_in_chromadb flag and sync metadata with .meta.json
                        await fm.db.fileregistry.update(
                            where={"id": registry_file_id},
                            data={
                                "indexed_in_chromadb": True,
                                "chromadb_collection": (
                                    collection.name
                                    if hasattr(collection, "name")
                                    else "default"
                                ),
                                "metadata": json.dumps(
                                    info
                                ),  # Sync complete file_info including ingested=True
                            },
                        )
                        logger.info(
                            f"[INGESTION] Marked file {registry_file_id} as indexed in ChromaDB and synced metadata"
                        )

            asyncio.run(update_chromadb_status())
        except Exception as e:
            logger.error(
                f"[INGESTION] Failed to update FileRegistry indexed status: {e}"
            )

    return info.get("file_id", "") if docs else None


def ingest_file(
    collection,
    file: FileRegistry,
    user_email: str,
    dept_id: str,
) -> IngestResult:
    """
    Ingest a single file from FileRegistry into ChromaDB.

    This is the new version that uses FileRegistry directly instead of .meta.json files.

    Args:
        collection: ChromaDB collection
        file: FileRegistry entity with file metadata
        user_email: Current user email (for access control)
        dept_id: Current department ID (for access control)

    Returns:
        IngestResult with success status and chunk count
    """
    file_id = file.id
    filename = file.original_name
    file_path = file.storage_path.value
    file_for_user = file.metadata.get("file_for_user", False) if file.metadata else False
    file_dept_id = file.dept_id.value if file.dept_id else ""
    file_user_email = file.user_email.value
    tags = file.metadata.get("tags", "") if file.metadata else ""
    size_kb = (file.size_bytes // 1024) if file.size_bytes else 0
    created_at = file.created_at.isoformat() if file.created_at else ""
    created_at_ts = int(file.created_at.timestamp()) if file.created_at else 0

    # Access control check
    if file_for_user and (file_dept_id != dept_id or file_user_email != user_email):
        return IngestResult(
            file_id=file_id,
            filename=filename,
            chunks_count=0,
            success=False,
            error="Access denied: file is private to another user",
        )

    # Already indexed check
    if file.indexed_in_chromadb:
        return IngestResult(
            file_id=file_id,
            filename=filename,
            chunks_count=0,
            success=False,
            error="File already indexed",
        )

    # File existence check
    if not os.path.exists(file_path):
        return IngestResult(
            file_id=file_id,
            filename=filename,
            chunks_count=0,
            success=False,
            error=f"File not found on disk: {file_path}",
        )

    # Read file content
    try:
        pages_text = read_text(file_path)
    except Exception as e:
        return IngestResult(
            file_id=file_id,
            filename=filename,
            chunks_count=0,
            success=False,
            error=f"Failed to read file: {str(e)}",
        )

    if not pages_text:
        return IngestResult(
            file_id=file_id,
            filename=filename,
            chunks_count=0,
            success=False,
            error="No text content extracted from file",
        )

    # Chunking - returns list of (page_num, chunk_text) tuples
    chunks_with_pages = make_chunks(pages_text)

    # Build chunk IDs, documents, and metadata for ChromaDB
    ids, docs, metas = [], [], []
    seen = set()

    for page_num, chunk in chunks_with_pages:
        # Generate unique chunk ID incorporating page number
        if file_for_user:
            seed = f"{file_dept_id}|{file_user_email}|{filename}|p{page_num}|{chunk}"
        else:
            seed = f"{file_dept_id}|{filename}|p{page_num}|{chunk}"

        chunk_id = make_id(seed)

        # Skip duplicates
        if chunk_id in seen:
            logger.warning(
                f"Duplicate chunk detected: {chunk_id}, file: {filename}, page: {page_num}"
            )
            continue

        seen.add(chunk_id)
        ids.append(chunk_id)
        docs.append(chunk)
        metas.append({
            "dept_id": file_dept_id,
            "user_id": file_user_email,
            "file_for_user": file_for_user,
            "chunk_id": chunk_id,
            "source": filename,
            "ext": filename.split(".")[-1].lower() if "." in filename else "",
            "file_id": file_id,  # This is now the FileRegistry.id (cuid)
            "size_kb": size_kb,
            "tags": tags.lower() if tags else "",
            "upload_at": created_at,
            "uploaded_at_ts": created_at_ts,
            "page": page_num,
        })

    # Upsert to ChromaDB
    if docs:
        try:
            collection.upsert(ids=ids, documents=docs, metadatas=metas)
            logger.info(
                f"[INGESTION] Indexed {len(docs)} chunks for file {file_id} ({filename})"
            )
        except Exception as e:
            return IngestResult(
                file_id=file_id,
                filename=filename,
                chunks_count=0,
                success=False,
                error=f"ChromaDB upsert failed: {str(e)}",
            )

    return IngestResult(
        file_id=file_id,
        filename=filename,
        chunks_count=len(docs),
        success=True,
    )
