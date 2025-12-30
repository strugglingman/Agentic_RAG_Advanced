"""Vector database service"""

import os
import logging
import chromadb
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
    OpenAIEmbeddingFunction,
)
from src.config.settings import Config

logger = logging.getLogger(__name__)


class VectorDB:
    """ChromaDB wrapper with support for multiple embedding providers."""

    def __init__(
        self,
        path: str,
        embedding_provider: str = "local",
    ):
        """
        Initialize VectorDB with configurable embedding provider.

        Args:
            path: ChromaDB storage path
            embedding_provider: "local" for SentenceTransformer, "openai" for OpenAI
        """
        self.embedding_provider = embedding_provider.lower()

        if self.embedding_provider == "openai":
            api_key = Config.OPENAI_KEY
            if not api_key:
                raise ValueError("OpenAI API key required for OpenAI embeddings")
            self.embedding_fun = OpenAIEmbeddingFunction(
                api_key=api_key,
                model_name=Config.OPENAI_EMBEDDING_MODEL,
            )
            logger.info(f"[VectorDB] Using OpenAI embeddings: {Config.OPENAI_EMBEDDING_MODEL}")
        else:
            self.embedding_fun = SentenceTransformerEmbeddingFunction(
                model_name=Config.EMBEDDING_MODEL_NAME,
            )
            logger.info(f"[VectorDB] Using local embeddings: {Config.EMBEDDING_MODEL_NAME}")

        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(
            name="docs",
            metadata={"hnsw:space": "cosine"},
            embedding_function=self.embedding_fun,
        )

    def upsert(self, ids: list, documents: list, metadatas: list):
        """Insert or update documents"""
        self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    def query(
        self,
        query_texts: list,
        n_results: int,
        where: dict = None,
        include: list = None,
    ):
        """Query similar documents"""
        return self.collection.query(
            query_texts=query_texts,
            n_results=n_results,
            where=where,
            include=include or ["documents", "metadatas", "distances"],
        )

    def get(self, include: list = None):
        """Get all documents"""
        return self.collection.get(include=include or ["documents", "metadatas"])

    def delete_collection(self):
        """Delete the entire collection (useful for re-indexing with new embeddings)"""
        self.client.delete_collection("docs")
        logger.info("[VectorDB] Collection 'docs' deleted")

    def recreate_collection(self):
        """Delete and recreate collection (for embedding model changes)"""
        self.delete_collection()
        self.collection = self.client.get_or_create_collection(
            name="docs",
            metadata={"hnsw:space": "cosine"},
            embedding_function=self.embedding_fun,
        )
        logger.info("[VectorDB] Collection 'docs' recreated")
