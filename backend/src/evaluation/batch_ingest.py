"""
Batch ingestion script for evaluation documents.

Uploads all files from a specified folder directly into Qdrant,
bypassing the FileRegistry for quick evaluation testing.

Usage:
    python -m src.evaluation.batch_ingest --folder eval_data/financial_docs --dept-id "TEST|eval|finance"
    python -m src.evaluation.batch_ingest --folder eval_data/financial_docs --clear  # Clear existing and re-ingest
"""

import os
import sys
import argparse
import asyncio
import hashlib
import logging
from pathlib import Path
from datetime import datetime

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.services.vector_db_qdrant import QdrantVectorDB
from src.services.document_processor import read_text
from src.services.langchain_processor import make_chunks_langchain
from src.config.settings import Config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def make_id(text: str) -> str:
    """Generate MD5 hash for a text string."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def get_supported_extensions() -> set:
    """Return set of supported file extensions."""
    return {".txt", ".pdf", ".docx", ".pptx", ".html", ".htm", ".xlsx", ".csv", ".json", ".md"}


async def ingest_file_direct(
    vector_db: QdrantVectorDB,
    file_path: str,
    dept_id: str,
    user_id: str,
    use_langchain: bool = True,
) -> tuple[int, str]:
    """
    Ingest a single file directly into Qdrant.

    Args:
        vector_db: QdrantVectorDB instance
        file_path: Path to the file
        dept_id: Department ID for metadata
        user_id: User ID for metadata
        use_langchain: Whether to use LangChain chunking

    Returns:
        Tuple of (chunks_count, error_message)
    """
    filename = os.path.basename(file_path)

    # Read file content
    try:
        pages_text = read_text(file_path, text_max=Config.TEXT_MAX)
    except Exception as e:
        return 0, f"Failed to read file: {str(e)}"

    if not pages_text:
        return 0, "No text extracted from file"

    # Chunk the text
    try:
        if use_langchain:
            chunks_with_pages = make_chunks_langchain(
                pages_text,
                chunk_size=Config.CHUNK_SIZE,
                chunk_overlap=Config.CHUNK_OVERLAP,
                strategy=Config.CHUNKING_STRATEGY,
            )
        else:
            from src.services.document_processor import make_chunks

            chunks_with_pages = make_chunks(
                pages_text,
                target=Config.CHUNK_SIZE,
                overlap=Config.CHUNK_OVERLAP,
            )
    except Exception as e:
        return 0, f"Failed to chunk file: {str(e)}"

    if not chunks_with_pages:
        return 0, "No chunks generated"

    # Prepare for upsert
    ids, docs, metas = [], [], []
    file_id = make_id(f"{dept_id}|{filename}|{datetime.now().isoformat()}")

    for page_num, chunk in chunks_with_pages:
        seed = f"{dept_id}|{filename}|p{page_num}|{chunk}"
        chunk_id = make_id(seed)

        ids.append(chunk_id)
        docs.append(chunk)
        metas.append(
            {
                "dept_id": dept_id,
                "user_id": user_id,
                "file_for_user": False,
                "chunk_id": chunk_id,
                "source": filename,
                "ext": filename.split(".")[-1].lower(),
                "file_id": file_id,
                "size_kb": os.path.getsize(file_path) // 1024,
                "tags": "evaluation",
                "upload_at": datetime.now().isoformat(),
                "uploaded_at_ts": int(datetime.now().timestamp()),
                "page": page_num,
            }
        )

    # Upsert to Qdrant
    try:
        await vector_db.upsert(ids=ids, documents=docs, metadatas=metas)
    except Exception as e:
        return 0, f"Failed to upsert to Qdrant: {str(e)}"

    return len(docs), ""


async def batch_ingest(
    folder_path: str,
    dept_id: str,
    user_id: str,
    clear_existing: bool = False,
    use_langchain: bool = True,
) -> dict:
    """
    Batch ingest all supported files from a folder.

    Args:
        folder_path: Path to folder containing documents
        dept_id: Department ID for all documents
        user_id: User ID for all documents
        clear_existing: Whether to clear existing documents for this dept_id first
        use_langchain: Whether to use LangChain chunking

    Returns:
        Dictionary with ingestion statistics
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise ValueError(f"Folder not found: {folder_path}")

    # Initialize QdrantVectorDB
    logger.info("Initializing QdrantVectorDB...")
    vector_db = QdrantVectorDB(
        url=Config.QDRANT_URL,
        api_key=Config.QDRANT_API_KEY or None,
        collection_name=Config.QDRANT_COLLECTION_NAME,
        embedding_provider=Config.EMBEDDING_PROVIDER,
    )
    await vector_db.ensure_collection()

    # Clear existing documents for this dept_id if requested
    if clear_existing:
        logger.info(f"Clearing existing documents for dept_id: {dept_id}")
        try:
            deleted = await vector_db.delete_by_filter({"dept_id": dept_id})
            logger.info(f"Deleted {deleted} existing chunks")
        except Exception as e:
            logger.warning(f"Failed to clear existing documents: {e}")

    # Find all supported files
    supported_extensions = get_supported_extensions()
    files_to_ingest = []

    for ext in supported_extensions:
        files_to_ingest.extend(folder.glob(f"*{ext}"))

    if not files_to_ingest:
        logger.warning(f"No supported files found in {folder_path}")
        return {"total_files": 0, "success": 0, "failed": 0, "total_chunks": 0}

    logger.info(f"Found {len(files_to_ingest)} files to ingest")

    # Process each file
    stats = {
        "total_files": len(files_to_ingest),
        "success": 0,
        "failed": 0,
        "total_chunks": 0,
        "errors": [],
    }

    for i, file_path in enumerate(files_to_ingest, 1):
        filename = file_path.name
        logger.info(f"[{i}/{len(files_to_ingest)}] Processing: {filename}")

        chunks_count, error = await ingest_file_direct(
            vector_db=vector_db,
            file_path=str(file_path),
            dept_id=dept_id,
            user_id=user_id,
            use_langchain=use_langchain,
        )

        if error:
            stats["failed"] += 1
            stats["errors"].append({"file": filename, "error": error})
            logger.error(f"  Failed: {error}")
        else:
            stats["success"] += 1
            stats["total_chunks"] += chunks_count
            logger.info(f"  Success: {chunks_count} chunks")

    # Final stats
    total_in_db = await vector_db.count()
    logger.info(f"\n{'='*50}")
    logger.info(f"Batch ingestion complete!")
    logger.info(f"  Files processed: {stats['total_files']}")
    logger.info(f"  Successful: {stats['success']}")
    logger.info(f"  Failed: {stats['failed']}")
    logger.info(f"  Total chunks added: {stats['total_chunks']}")
    logger.info(f"  Total chunks in DB: {total_in_db}")
    logger.info(f"{'='*50}")

    await vector_db.client.close()
    return stats


async def async_main():
    parser = argparse.ArgumentParser(
        description="Batch ingest documents for evaluation"
    )
    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Path to folder containing documents to ingest",
    )
    parser.add_argument(
        "--dept-id",
        type=str,
        default="MYHB|software|ml",
        help="Department ID for ingested documents (default: MYHB|software|ml)",
    )
    parser.add_argument(
        "--user-id",
        type=str,
        default="strugglingman@gmail.com",
        help="User ID for ingested documents (default: strugglingman@gmail.com)",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing documents for the dept-id before ingesting",
    )
    parser.add_argument(
        "--no-langchain",
        action="store_true",
        help="Use original chunking instead of LangChain",
    )

    args = parser.parse_args()

    try:
        stats = await batch_ingest(
            folder_path=args.folder,
            dept_id=args.dept_id,
            user_id=args.user_id,
            clear_existing=args.clear,
            use_langchain=not args.no_langchain,
        )

        if stats["failed"] > 0:
            print(f"\nWarning: {stats['failed']} files failed to ingest")
            for err in stats["errors"]:
                print(f"  - {err['file']}: {err['error']}")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Batch ingestion failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(async_main())
