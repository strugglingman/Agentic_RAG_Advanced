"""
Batch ingestion script for evaluation documents.

Uploads all files from a specified folder directly into Qdrant,
bypassing the FileRegistry for quick evaluation testing.

Usage:
    python -m src.evaluation.batch_ingest --folder eval_data/financial_docs --dept-id "TEST|eval|finance"
    python -m src.evaluation.batch_ingest --folder eval_data/financial_docs --clear  # Clear existing and re-ingest
    python -m src.evaluation.batch_ingest --folder eval_data/financial_docs --no-contextual  # Skip contextual retrieval
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

from openai import AsyncOpenAI
from httpx import Timeout
from src.services.vector_db_qdrant import QdrantVectorDB
from src.services.document_processor import read_text
from src.services.langchain_processor import make_chunks_langchain
from src.services.contextual_retrieval import contextualize_chunks
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
    use_contextual: bool = True,
    openai_client: AsyncOpenAI | None = None,
    chunk_size: int = Config.CHUNK_SIZE,
    chunk_overlap: int = Config.CHUNK_OVERLAP,
) -> tuple[int, str]:
    """
    Ingest a single file directly into Qdrant.

    Args:
        vector_db: QdrantVectorDB instance
        file_path: Path to the file
        dept_id: Department ID for metadata
        user_id: User ID for metadata
        use_langchain: Whether to use LangChain chunking
        use_contextual: Whether to use contextual retrieval
        openai_client: AsyncOpenAI client (required when use_contextual=True)

    Returns:
        Tuple of (chunks_count, error_message)
    """
    filename = os.path.basename(file_path)

    # Read file content
    try:
        pages_text = await asyncio.to_thread(read_text, file_path, Config.TEXT_MAX)
    except Exception as e:
        return 0, f"Failed to read file: {str(e)}"

    if not pages_text:
        return 0, "No text extracted from file"

    # Concatenate all pages into full document text
    full_text = "\n\n".join(text for _, text in pages_text if text.strip())

    # Chunk the full document as one unit
    try:
        if use_langchain:
            chunks_with_pages = make_chunks_langchain(
                [(0, full_text)],
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                strategy=Config.CHUNKING_STRATEGY,
            )
        else:
            from src.services.document_processor import make_chunks

            chunks_with_pages = make_chunks(
                [(0, full_text)],
                target=chunk_size,
                overlap=chunk_overlap,
            )
    except Exception as e:
        return 0, f"Failed to chunk file: {str(e)}"

    if not chunks_with_pages:
        return 0, "No chunks generated"

    # Contextual Retrieval: enrich chunks with LLM-generated document context
    if use_contextual and openai_client:
        try:
            contextualized = await contextualize_chunks(
                full_text, chunks_with_pages, filename, openai_client=openai_client,
            )
            logger.info(f"  Contextualized {len(contextualized)} chunks")
        except Exception as e:
            logger.warning(f"  Contextual retrieval failed, using raw chunks: {e}")
            contextualized = [(pg, chunk, chunk) for pg, chunk in chunks_with_pages]
    else:
        contextualized = [(pg, chunk, chunk) for pg, chunk in chunks_with_pages]

    # Prepare for upsert
    ids, docs, metas = [], [], []
    file_id = make_id(f"{dept_id}|{filename}|{datetime.now().isoformat()}")

    for page_num, doc_text, original_text in contextualized:
        # Chunk IDs use ORIGINAL text so re-indexing with/without contextual
        # retrieval doesn't create duplicates
        seed = f"{dept_id}|{filename}|p{page_num}|{original_text}"
        chunk_id = make_id(seed)

        ids.append(chunk_id)
        docs.append(doc_text)  # Contextualized text for embedding + BM25
        meta = {
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
        if use_contextual and openai_client:
            meta["original_text"] = original_text
        metas.append(meta)

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
    use_contextual: bool = True,
    chunk_size: int = Config.CHUNK_SIZE,
    chunk_overlap: int = Config.CHUNK_OVERLAP,
) -> dict:
    """
    Batch ingest all supported files from a folder.

    Args:
        folder_path: Path to folder containing documents
        dept_id: Department ID for all documents
        user_id: User ID for all documents
        clear_existing: Whether to clear existing documents for this dept_id first
        use_langchain: Whether to use LangChain chunking
        use_contextual: Whether to use contextual retrieval

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
        prefer_grpc=Config.QDRANT_PREFER_GRPC,
    )
    await vector_db.ensure_collection()

    # Initialize OpenAI client for contextual retrieval
    openai_client = None
    if use_contextual:
        logger.info(
            f"Contextual retrieval enabled (model={Config.CONTEXTUAL_RETRIEVAL_MODEL}, "
            f"workers={Config.CONTEXTUAL_RETRIEVAL_MAX_WORKERS})"
        )
        openai_client = AsyncOpenAI(
            api_key=Config.OPENAI_KEY,
            max_retries=Config.LLM_MAX_RETRIES,
            timeout=Timeout(Config.LLM_TIMEOUT, connect=Config.LLM_CONNECT_TIMEOUT),
        )
    else:
        logger.info("Contextual retrieval disabled")

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
    logger.info(f"Chunk config: size={chunk_size} tokens, overlap={chunk_overlap} tokens")

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
            use_contextual=use_contextual,
            openai_client=openai_client,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
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
    if use_contextual:
        logger.info(f"  Contextual retrieval: enabled")
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
    parser.add_argument(
        "--no-contextual",
        action="store_true",
        help="Skip contextual retrieval (faster, no LLM cost)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=Config.CHUNK_SIZE,
        help=f"Chunk size in tokens (default: {Config.CHUNK_SIZE})",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=Config.CHUNK_OVERLAP,
        help=f"Chunk overlap in tokens (default: {Config.CHUNK_OVERLAP})",
    )

    args = parser.parse_args()

    try:
        stats = await batch_ingest(
            folder_path=args.folder,
            dept_id=args.dept_id,
            user_id=args.user_id,
            clear_existing=args.clear,
            use_langchain=not args.no_langchain,
            use_contextual=not args.no_contextual,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
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
