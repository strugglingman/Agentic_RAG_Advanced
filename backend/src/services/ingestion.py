"""
Ingestion service for document processing and ChromaDB storage.
Handles reading, chunking, and upserting documents to vector database.
"""

from __future__ import annotations
import os
import hashlib
import logging
from typing import TYPE_CHECKING, Optional
from dataclasses import dataclass
from src.services.document_processor import read_text
from src.services.langchain_processor import make_chunks_langchain
from src.domain.entities.file_registry import FileRegistry
from src.config.settings import Config
from src.observability.metrics import increment_error, MetricsErrorType

if TYPE_CHECKING:
    from src.services.vector_db import VectorDB

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


def ingest_file(
    vector_db: VectorDB,
    file: FileRegistry,
    user_email: str,
    dept_id: str,
) -> IngestResult:
    """
    Ingest a single file from FileRegistry into ChromaDB.

    This is the new version that uses FileRegistry directly instead of .meta.json files.

    Args:
        vector_db: VectorDB instance for vector storage
        file: FileRegistry entity with file metadata
        user_email: Current user email (for access control)
        dept_id: Current department ID (for access control)

    Returns:
        IngestResult with success status and chunk count
    """
    file_id = file.id
    filename = file.original_name
    file_path = file.storage_path.value
    file_for_user = (
        file.metadata.get("file_for_user", False) if file.metadata else False
    )
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
        pages_text = read_text(file_path, text_max=Config.TEXT_MAX)
    except Exception as e:
        increment_error(MetricsErrorType.INGESTION_FAILED)
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

    # Concatenate all pages into full document text for full-document chunking.
    # Per-page chunking destroys cross-page context (up to 9% recall gap).
    full_text = "\n\n".join(text for _, text in pages_text if text.strip())

    # Chunking - chunk the full document as one unit
    chunks_with_pages = make_chunks_langchain(
        [(0, full_text)],
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        strategy=Config.CHUNKING_STRATEGY,
    )

    # Contextual Retrieval: enrich chunks with LLM-generated document context
    # When enabled, each chunk gets a context preamble prepended before embedding.
    # The original chunk text is preserved in metadata for display.
    use_contextual = Config.CONTEXTUAL_RETRIEVAL_ENABLED and chunks_with_pages
    if use_contextual:
        from src.services.contextual_retrieval import contextualize_chunks

        contextualized = contextualize_chunks(full_text, chunks_with_pages, filename)
        logger.info(
            f"[INGESTION] Contextual retrieval: {len(contextualized)} chunks contextualized for {filename}"
        )
    else:
        # No contextual retrieval: convert to same 3-tuple format (page, chunk, chunk)
        contextualized = [(pg, chunk, chunk) for pg, chunk in chunks_with_pages]

    # Build chunk IDs, documents, and metadata for ChromaDB
    ids, docs, metas = [], [], []
    seen = set()

    for page_num, doc_text, original_text in contextualized:
        # Chunk IDs use ORIGINAL text so re-indexing with/without contextual
        # retrieval doesn't create duplicates
        if file_for_user:
            seed = f"{file_dept_id}|{file_user_email}|{filename}|p{page_num}|{original_text}"
        else:
            seed = f"{file_dept_id}|{filename}|p{page_num}|{original_text}"

        chunk_id = make_id(seed)

        # Skip duplicates
        if chunk_id in seen:
            logger.warning(
                f"Duplicate chunk detected: {chunk_id}, file: {filename}, page: {page_num}"
            )
            continue

        seen.add(chunk_id)
        ids.append(chunk_id)
        docs.append(doc_text)  # Contextualized text for embedding + BM25
        meta = {
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
        }
        # Store original text in metadata when contextual retrieval is active
        if use_contextual:
            meta["original_text"] = original_text
        metas.append(meta)

    # Upsert to ChromaDB
    if docs:
        try:
            vector_db.upsert(ids=ids, documents=docs, metadatas=metas)
            logger.info(
                f"[INGESTION] Indexed {len(docs)} chunks for file {file_id} ({filename})"
            )
        except Exception as e:
            increment_error(MetricsErrorType.INGESTION_FAILED)
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
