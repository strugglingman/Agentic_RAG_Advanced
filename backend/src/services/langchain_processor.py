"""
LangChain Document Processor - Skeleton & Guidelines

=============================================================================
INSTALLATION REQUIRED
=============================================================================
pip install langchain-text-splitters langchain-experimental langchain-openai

=============================================================================
STRATEGIES
=============================================================================
1. RECURSIVE (default): Fast, no API cost, uses smart separators
2. SEMANTIC: Best accuracy, requires OpenAI embeddings (~$0.02/100 pages)

=============================================================================
RECOMMENDED PARAMETERS
=============================================================================
Content Type     | chunk_size | chunk_overlap | Strategy
-----------------|------------|---------------|----------
General docs     | 500        | 100 (20%)     | recursive
Technical docs   | 300        | 60 (20%)      | recursive
Books/narratives | 600-800    | 100-120       | semantic

=============================================================================
INTEGRATION STEPS
=============================================================================
1. Add to settings.py:
   CHUNKING_STRATEGY = os.getenv("CHUNKING_STRATEGY", "recursive")

2. Update ingestion.py:
   - Replace: from src.services.document_processor import make_chunks
   - With:    from src.services.langchain_processor import make_chunks_langchain
   - Call:    make_chunks_langchain(pages_text, chunk_size, chunk_overlap, strategy, api_key)

3. Re-index: Clear ChromaDB, re-upload docs, run RAGAS eval
"""

from __future__ import annotations
import logging
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Literal
from src.config.settings import Config

logger = logging.getLogger(__name__)


class ChunkingStrategy(str, Enum):
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"


@dataclass
class ChunkingConfig:
    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE
    chunk_size: int = Config.CHUNK_SIZE
    chunk_overlap: int = Config.CHUNK_OVERLAP
    # Semantic settings
    breakpoint_threshold_type: Literal[
        "percentile", "standard_deviation", "interquartile", "gradient"
    ] = "percentile"
    breakpoint_threshold_amount: float = 90.0
    embedding_model: str = Config.LANGCHAIN_EMBEDDING_MODEL


class LangChainProcessor:
    """Document processor using LangChain text splitters."""

    def __init__(
        self,
        config: Optional[ChunkingConfig] = None,
        openai_api_key: Optional[str] = Config.OPENAI_KEY,
    ):
        self.config = config or ChunkingConfig()
        self.openai_api_key = openai_api_key
        self._embedding_model = self.config.embedding_model
        self._splitter = None
        self._init_splitter()

    def _init_splitter(self):
        """Initialize splitter based on strategy."""
        if self.config.strategy == ChunkingStrategy.RECURSIVE:
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            self._splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
                add_start_index=True,
                strip_whitespace=True,
            )
        elif self.config.strategy == ChunkingStrategy.SEMANTIC:
            from langchain_experimental.text_splitter import SemanticChunker
            from langchain_openai import OpenAIEmbeddings

            embeddings = OpenAIEmbeddings(
                api_key=self.openai_api_key, model=self._embedding_model
            )
            self._splitter = SemanticChunker(
                embeddings=embeddings,
                breakpoint_threshold_type=self.config.breakpoint_threshold_type,
                breakpoint_threshold_amount=self.config.breakpoint_threshold_amount,
            )

    def process(self, pages_text: list[tuple[int, str]]) -> list[tuple[int, str]]:
        """
        Process pages and return chunks.

        Args:
            pages_text: List of (page_num, text) from read_text()

        Returns:
            List of (page_num, chunk_text) - same format as make_chunks()
        """
        all_chunks = []
        for page_num, text in pages_text:
            if not text.strip():
                continue
            chunks = self._splitter.split_text(text)
            for chunk in chunks:
                if chunk.strip():
                    all_chunks.append((page_num, chunk.strip()))

        return all_chunks


def make_chunks_langchain(
    pages_text: list[tuple[int, str]],
    chunk_size: int = 600,
    chunk_overlap: int = 120,
    strategy: str = "recursive",
    openai_api_key: Optional[str] = Config.OPENAI_KEY,
    embedding_model: str = Config.LANGCHAIN_EMBEDDING_MODEL,
) -> list[tuple[int, str]]:
    """
    Drop-in replacement for document_processor.make_chunks().

    Args:
        pages_text: List of (page_num, text) from read_text()
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks
        strategy: "recursive" or "semantic"
        openai_api_key: Required for semantic strategy
        embedding_model: Embedding model for semantic chunking

    Returns:
        List of (page_num, chunk_text) tuples
    """
    config = ChunkingConfig(
        strategy=ChunkingStrategy(strategy),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=embedding_model,
    )

    processor = LangChainProcessor(config=config, openai_api_key=openai_api_key)

    return processor.process(pages_text)
