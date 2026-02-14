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
import re
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
    # Token-based measurement encoding (cl100k_base for OpenAI text-embedding-3-*,
    # o200k_base for GPT-4o). Only used when strategy=recursive.
    tiktoken_encoding: str = Config.TIKTOKEN_ENCODING
    # Semantic settings
    breakpoint_threshold_type: Literal[
        "percentile", "standard_deviation", "interquartile", "gradient"
    ] = "percentile"
    breakpoint_threshold_amount: float = 90.0
    embedding_model: str = Config.OPENAI_EMBEDDING_MODEL


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

            # Token-based measurement: chunk_size/chunk_overlap are in tokens,
            # but splitting still uses recursive separators for structural awareness.
            # This aligns chunk sizes with embedding model token limits and gives
            # more predictable, language-agnostic results than character counting.
            # See: Chroma Research, Microsoft Azure AI Search, Firecrawl 2025 best practices.
            self._splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name=self.config.tiktoken_encoding,
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

    @staticmethod
    def _has_markdown_headers(text: str) -> bool:
        """Detect if text contains Markdown headers (#, ##, ###, etc.)."""
        return bool(re.search(r"^#{1,6}\s", text, re.MULTILINE))

    def process(self, pages_text: list[tuple[int, str]]) -> list[tuple[int, str]]:
        """
        Process pages and return chunks.

        Two-stage pipeline for Markdown content:
          Stage 1: MarkdownHeaderTextSplitter splits by section boundaries
          Stage 2: RecursiveCharacterTextSplitter enforces size limits within sections

        For non-Markdown content, uses the configured splitter directly.

        Args:
            pages_text: List of (page_num, text) from read_text()

        Returns:
            List of (page_num, chunk_text) - same format as make_chunks()
        """
        all_chunks = []
        for page_num, text in pages_text:
            if not text.strip():
                continue

            # Two-stage split for Markdown content (Docling output, .md files)
            if self._has_markdown_headers(text) and self.config.strategy == ChunkingStrategy.RECURSIVE:
                chunks = self._split_markdown_two_stage(text)
            else:
                chunks = self._splitter.split_text(text)

            for chunk in chunks:
                if chunk.strip():
                    all_chunks.append((page_num, chunk.strip()))

        return all_chunks

    def _split_markdown_two_stage(self, text: str) -> list[str]:
        """
        Stage 1: Split by Markdown headers into semantic sections.
        Stage 2: Apply RecursiveCharacterTextSplitter for size control.

        When Stage 2 splits a large section into multiple sub-chunks, only the
        first sub-chunk naturally contains the header text. We fix this by
        checking each sub-chunk and prepending headers from metadata when the
        chunk content doesn't already start with them.
        """
        from langchain_text_splitters import MarkdownHeaderTextSplitter

        md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ],
            strip_headers=False,
        )

        # Stage 1: split by headers → list of Documents with header metadata
        header_docs = md_splitter.split_text(text)

        if not header_docs:
            return self._splitter.split_text(text)

        # Stage 2: enforce size limits within each section
        sized_docs = self._splitter.split_documents(header_docs)

        # Rebuild header prefix from metadata for sub-chunks that lost it.
        # With strip_headers=False, the first sub-chunk of a section already
        # contains its OWN header (e.g. "## Costs") but NOT parent headers
        # (e.g. "# Report"). Continuation sub-chunks have NO headers at all.
        # We detect which header the content starts with and prepend only
        # the missing parent headers above it.
        header_keys = [("Header 1", "#"), ("Header 2", "##"), ("Header 3", "###")]
        results = []
        for doc in sized_docs:
            content = doc.page_content
            # Build full header hierarchy from metadata
            prefix_lines = []
            for meta_key, md_marker in header_keys:
                if meta_key in doc.metadata:
                    prefix_lines.append(f"{md_marker} {doc.metadata[meta_key]}")

            if prefix_lines:
                # Find which header the content already starts with (if any)
                prepend_lines = prefix_lines  # default: prepend all (continuation chunk)
                for i, line in enumerate(prefix_lines):
                    if content.startswith(line):
                        # Content already has this header; only prepend parents above it
                        prepend_lines = prefix_lines[:i]
                        break

                if prepend_lines:
                    content = "\n".join(prepend_lines) + "\n" + content

            results.append(content)

        return results


def make_chunks_langchain(
    pages_text: list[tuple[int, str]],
    chunk_size: int = 600,
    chunk_overlap: int = 120,
    strategy: str = "recursive",
    openai_api_key: Optional[str] = Config.OPENAI_KEY,
    embedding_model: str = Config.OPENAI_EMBEDDING_MODEL,
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
