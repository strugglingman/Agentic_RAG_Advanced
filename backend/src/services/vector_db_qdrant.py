"""Qdrant vector database service — drop-in replacement for ChromaDB VectorDB.

Stores both dense (OpenAI / SentenceTransformer) and sparse (BM25 via fastembed)
vectors per chunk.  Query uses Qdrant's hybrid search with server-side RRF fusion,
eliminating the need for in-memory BM25 index in retrieval.py.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any  # used by _parse_condition return type

from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseVector,
    VectorParams,
    SparseVectorParams,
    Modifier,
)
from fastembed import SparseTextEmbedding
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log

from src.config.settings import Config

logger = logging.getLogger(__name__)

# Retry decorator: 3 attempts, exponential backoff (1s, 2s, 4s)
_qdrant_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)

# Embedding dimensions per OpenAI model
_OPENAI_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

# Local model defaults (intfloat/multilingual-e5-base = 768)
_LOCAL_DIM_DEFAULT = 768


class QdrantVectorDB:
    """Qdrant wrapper matching VectorDB's public interface.

    Public methods:
        upsert(ids, documents, metadatas, embeddings=None)
        query(query_texts, n_results, where=None, include=None)
        query_hybrid(query_text, n_results, where=None, dense_embedding=None)
        get(include=None, where=None)
        delete_by_file_id(file_id) -> int
        delete_collection()
        recreate_collection()
    """

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        collection_name: str | None = None,
        embedding_provider: str = "openai",
        prefer_grpc: bool = False,
    ):
        self.url = url or Config.QDRANT_URL
        self.api_key = api_key or Config.QDRANT_API_KEY or None
        self.collection_name = collection_name or Config.QDRANT_COLLECTION_NAME
        self.embedding_provider = embedding_provider.lower()
        self._dense_dim = self._resolve_dense_dim()

        # Async Qdrant client (non-blocking for FastAPI)
        self.client = AsyncQdrantClient(
            url=self.url,
            api_key=self.api_key,
            prefer_grpc=prefer_grpc,
            timeout=30,
        )

        # Dense embedding client (async for non-blocking API calls)
        if self.embedding_provider == "openai":
            self._openai_client = AsyncOpenAI(api_key=Config.OPENAI_KEY)
            self._embedding_model = Config.OPENAI_EMBEDDING_MODEL
        else:
            from sentence_transformers import SentenceTransformer

            self._local_model = SentenceTransformer(Config.EMBEDDING_MODEL_NAME)
            self._embedding_model = Config.EMBEDDING_MODEL_NAME

        # Sparse embedding model (fastembed BM25)
        self._sparse_model = SparseTextEmbedding(
            model_name=Config.QDRANT_SPARSE_MODEL,
        )

        logger.info(
            "[QdrantVectorDB] Initialized for %s, collection=%s, "
            "dense_dim=%d, sparse_model=%s",
            self.url,
            self.collection_name,
            self._dense_dim,
            Config.QDRANT_SPARSE_MODEL,
        )

    # ── Private helpers ──────────────────────────────────────────────────

    def _resolve_dense_dim(self) -> int:
        if self.embedding_provider == "openai":
            return _OPENAI_DIMS.get(Config.OPENAI_EMBEDDING_MODEL, 3072)
        return _LOCAL_DIM_DEFAULT

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Compute dense embeddings for a batch of texts.

        Uses the same provider (OpenAI / local SentenceTransformer) configured
        for this instance.  Called by ingestion (documents) and retrieval (query).
        """
        if not texts:
            return []

        logger.debug(
            "[QdrantVectorDB] Embedding %d texts via %s (%s)",
            len(texts), self.embedding_provider, self._embedding_model,
        )

        if self.embedding_provider == "openai":
            response = await self._openai_client.embeddings.create(
                input=texts,
                model=self._embedding_model,
            )
            return [item.embedding for item in response.data]
        else:
            embeddings = await asyncio.to_thread(
                self._local_model.encode, texts, normalize_embeddings=True,
            )
            return embeddings.tolist()

    async def embed_query(self, text: str) -> list[float]:
        """Compute dense embedding for a single query text."""
        if not text or not text.strip():
            raise ValueError("embed_query() requires a non-empty text string")
        return (await self.embed([text]))[0]

    async def ensure_collection(self) -> None:
        """Create collection with dense + sparse vector config if it doesn't exist.

        Must be called once after construction (async init).
        Collection uses:
        - Named dense vectors with cosine distance and on-disk storage
        - Named sparse vectors with IDF modifier (required for fastembed BM25)
        - Scalar INT8 quantization for 4x memory reduction (~1% accuracy loss)
        """
        if await self.client.collection_exists(self.collection_name):
            return
        await self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": VectorParams(
                    size=self._dense_dim,
                    distance=Distance.COSINE,
                    on_disk=True,
                ),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    modifier=Modifier.IDF,
                ),
            },
            quantization_config=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True,
                ),
            ),
        )
        # Index commonly filtered metadata fields for fast filtering
        for field_name, field_type in [
            ("dept_id", models.PayloadSchemaType.KEYWORD),
            ("user_id", models.PayloadSchemaType.KEYWORD),
            ("file_id", models.PayloadSchemaType.KEYWORD),
            ("file_for_user", models.PayloadSchemaType.BOOL),
        ]:
            await self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=field_type,
            )
        logger.info(
            "[QdrantVectorDB] Created collection '%s' with dense(%d) + sparse + IDF + INT8 quantization",
            self.collection_name,
            self._dense_dim,
        )

    async def _generate_sparse(self, texts: list[str]) -> list[SparseVector]:
        """Generate sparse BM25 vectors for a batch of texts via fastembed."""
        results = await asyncio.to_thread(lambda: list(self._sparse_model.embed(texts)))
        return [
            SparseVector(
                indices=r.indices.tolist(),
                values=r.values.tolist(),
            )
            for r in results
        ]

    def _build_qdrant_filter(self, where: dict | None) -> models.Filter | None:
        """Convert ChromaDB-style where clause to Qdrant filter.

        Supports:
            {"field": value}                    → simple match
            {"$and": [{...}, {...}]}            → AND of conditions
            {"$or": [{...}, {...}]}             → OR of conditions
        """
        if not where:
            return None

        conditions = self._parse_condition(where)
        if isinstance(conditions, list):
            return models.Filter(must=conditions)
        return models.Filter(must=[conditions])

    def _parse_condition(self, clause: dict) -> Any:
        """Recursively parse a ChromaDB-style filter into Qdrant conditions.

        Handles mixed clauses like {"$and": [...], "field": "value"} by
        processing both operators and sibling plain-field conditions.
        """
        # Collect sibling plain-field conditions (keys that aren't $and/$or)
        sibling_keys = {k: v for k, v in clause.items() if k not in ("$and", "$or")}
        sibling_conditions = self._parse_field_conditions(sibling_keys) if sibling_keys else []

        if "$and" in clause:
            sub_conditions = list(sibling_conditions)
            for sub in clause["$and"]:
                parsed = self._parse_condition(sub)
                if isinstance(parsed, list):
                    sub_conditions.extend(parsed)
                else:
                    sub_conditions.append(parsed)
            return sub_conditions

        if "$or" in clause:
            or_conditions = []
            for sub in clause["$or"]:
                parsed = self._parse_condition(sub)
                if isinstance(parsed, list):
                    or_conditions.extend(parsed)
                else:
                    or_conditions.append(parsed)
            or_filter = models.Filter(should=or_conditions)
            # If there are sibling conditions, wrap both in a must list
            if sibling_conditions:
                return sibling_conditions + [or_filter]
            return or_filter

        # Simple field match only: {"field_name": value, ...}
        return sibling_conditions if len(sibling_conditions) > 1 else sibling_conditions[0]

    def _parse_field_conditions(self, fields: dict) -> list:
        """Convert plain key-value pairs to Qdrant FieldCondition list."""
        conditions = []
        for key, value in fields.items():
            if isinstance(value, bool):
                conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value),
                    )
                )
            elif isinstance(value, (int, float)):
                conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value),
                    )
                )
            else:
                conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=str(value)),
                    )
                )
        return conditions

    # ── Public API (ChromaDB-compatible interface) ───────────────────────

    async def upsert(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict],
        embeddings: list[list[float]] | None = None,
    ) -> None:
        """Upsert documents with dense + sparse vectors.

        Args:
            ids: Chunk IDs
            documents: Chunk text content (used for sparse vector + stored in payload)
            metadatas: Per-chunk metadata dicts
            embeddings: Pre-computed dense embeddings (required — computed by ingestion)
        """
        if not ids:
            logger.debug("[QdrantVectorDB] upsert called with empty ids, skipping")
            return

        logger.info(
            "[QdrantVectorDB] Upserting %d chunks (embeddings %s)",
            len(ids),
            "pre-computed" if embeddings else "will be computed",
        )

        if embeddings is None:
            embeddings = await self.embed(documents)

        batch_size = Config.UPSERT_BATCH_SIZE
        total = len(ids)
        for i in range(0, total, batch_size):
            end = min(i + batch_size, total)
            await self._upsert_batch(
                ids=ids[i:end],
                documents=documents[i:end],
                metadatas=metadatas[i:end],
                embeddings=embeddings[i:end],
            )
            if total > batch_size:
                logger.info(
                    "[QdrantVectorDB] Upserted batch %d/%d (%d chunks)",
                    i // batch_size + 1,
                    (total + batch_size - 1) // batch_size,
                    end - i,
                )

        logger.info("[QdrantVectorDB] Upsert complete: %d chunks total", total)

    @_qdrant_retry
    async def _upsert_batch(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict],
        embeddings: list[list[float]],
    ) -> None:
        """Upsert a single batch with retry."""
        n = len(ids)
        if len(documents) != n or len(metadatas) != n or len(embeddings) != n:
            raise ValueError(
                f"_upsert_batch input length mismatch: "
                f"ids={n}, documents={len(documents)}, "
                f"metadatas={len(metadatas)}, embeddings={len(embeddings)}"
            )
        sparse_vectors = await self._generate_sparse(documents)

        points = []
        for chunk_id, doc, meta, dense_emb, sparse_vec in zip(
            ids, documents, metadatas, embeddings, sparse_vectors
        ):
            payload = {**meta, "document": doc}
            point = PointStruct(
                id=self._to_uuid(chunk_id),
                vector={
                    "dense": dense_emb,
                    "sparse": sparse_vec,
                },
                payload=payload,
            )
            points.append(point)

        await self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    @staticmethod
    def _to_uuid(chunk_id: str) -> str:
        """Convert an MD5 hex string to a UUID-compatible format for Qdrant."""
        try:
            return str(uuid.UUID(chunk_id))
        except ValueError:
            # Generate a deterministic UUID from the chunk_id string
            return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

    @_qdrant_retry
    async def query(
        self,
        query_texts: list[str],
        n_results: int,
        where: dict | None = None,
        include: list | None = None,
        query_embedding: list[float] | None = None,
    ) -> dict:
        """Dense-only semantic query. Returns ChromaDB-compatible result dict.

        Args:
            query_texts: List of query strings (first one used)
            n_results: Number of results to return
            where: ChromaDB-style filter dict
            include: Ignored (all fields always returned)
            query_embedding: Pre-computed query dense embedding (required)
        """
        if not query_texts or not query_texts[0].strip():
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        if query_embedding is None:
            query_embedding = await self.embed_query(query_texts[0])

        qdrant_filter = self._build_qdrant_filter(where)

        logger.debug(
            "[QdrantVectorDB] Dense query: n_results=%d, filter=%s",
            n_results, bool(qdrant_filter),
        )

        results = await self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            using="dense",
            limit=n_results,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        logger.info(
            "[QdrantVectorDB] Dense query returned %d results", len(results.points),
        )
        return self._to_chroma_format(results.points, is_hybrid=False)

    @_qdrant_retry
    async def query_hybrid(
        self,
        query_text: str,
        n_results: int,
        where: dict | None = None,
        dense_embedding: list[float] | None = None,
    ) -> dict:
        """Hybrid search: dense + sparse with server-side RRF fusion.

        This replaces the custom BM25 + RRF logic in retrieval.py.

        Args:
            query_text: Query string (used for sparse vector generation)
            n_results: Number of results to return
            where: ChromaDB-style filter dict
            dense_embedding: Pre-computed query dense embedding (required)

        Returns:
            ChromaDB-compatible result dict with keys:
            ids, documents, metadatas, distances
        """
        if not query_text or not query_text.strip():
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        if dense_embedding is None:
            dense_embedding = await self.embed_query(query_text)

        qdrant_filter = self._build_qdrant_filter(where)

        # Generate sparse vector for query
        sparse_vectors = await self._generate_sparse([query_text])
        query_sparse = sparse_vectors[0]

        prefetch_limit = max(Config.CANDIDATES, n_results)
        logger.debug(
            "[QdrantVectorDB] Hybrid query: n_results=%d, prefetch_limit=%d, "
            "rrf_k=%d, filter=%s",
            n_results, prefetch_limit, Config.RRF_K, bool(qdrant_filter),
        )

        # Hybrid query: prefetch dense + sparse, fuse with RRF
        results = await self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=dense_embedding,
                    using="dense",
                    limit=max(Config.CANDIDATES, n_results),
                    filter=qdrant_filter,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=query_sparse.indices,
                        values=query_sparse.values,
                    ),
                    using="sparse",
                    limit=max(Config.CANDIDATES, n_results),
                    filter=qdrant_filter,
                ),
            ],
            query=models.RrfQuery(rrf=models.Rrf(k=Config.RRF_K)),
            limit=n_results,
            with_payload=True,
        )

        logger.info(
            "[QdrantVectorDB] Hybrid query returned %d results", len(results.points),
        )
        return self._to_chroma_format(results.points, is_hybrid=True)

    def _to_chroma_format(self, points: list, is_hybrid: bool = False) -> dict:
        """Convert Qdrant ScoredPoints to ChromaDB-compatible dict format.

        Args:
            points: Qdrant ScoredPoint list
            is_hybrid: True for RRF fusion results (synthetic scores),
                       False for dense-only results (cosine similarity)
        """
        ids = []
        documents = []
        metadatas = []
        distances = []

        for point in points:
            payload = point.payload or {}
            doc = payload.get("document", "")
            meta = {k: v for k, v in payload.items() if k != "document"}
            ids.append(meta.get("chunk_id", str(point.id)))
            documents.append(doc)
            metadatas.append(meta)
            if is_hybrid:
                # RRF fusion produces synthetic ranked scores, not cosine similarity.
                # Pass raw score through; downstream retrieval.py handles interpretation.
                distances.append(point.score)
            else:
                # Dense-only: Qdrant cosine score is similarity [-1,1]
                # (typically [0,1] for normalized embeddings).
                # ChromaDB returns distance = 1 - similarity
                distances.append(max(0, 1.0 - point.score))

        return {
            "ids": [ids],
            "documents": [documents],
            "metadatas": [metadatas],
            "distances": [distances],
        }

    @_qdrant_retry
    async def get(
        self,
        include: list | None = None,
        where: dict | None = None,
    ) -> dict:
        """Get all documents matching filter. Returns ChromaDB-compatible dict.

        Note: Uses scroll to handle large result sets without loading everything at once.
        """
        qdrant_filter = self._build_qdrant_filter(where)
        logger.debug("[QdrantVectorDB] get: filter=%s", bool(qdrant_filter))

        all_ids = []
        all_documents = []
        all_metadatas = []

        offset = None
        while True:
            results, next_offset = await self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=qdrant_filter,
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            for point in results:
                payload = point.payload or {}
                doc = payload.get("document", "")
                meta = {k: v for k, v in payload.items() if k != "document"}
                all_ids.append(str(point.id))
                all_documents.append(doc)
                all_metadatas.append(meta)

            if next_offset is None:
                break
            offset = next_offset

        logger.info("[QdrantVectorDB] get returned %d documents", len(all_ids))
        return {
            "ids": all_ids,
            "documents": all_documents,
            "metadatas": all_metadatas,
        }

    @_qdrant_retry
    async def delete_by_file_id(self, file_id: str) -> int:
        """Delete all chunks belonging to a specific file.

        Returns:
            Number of chunks deleted
        """
        # Count before delete
        count_result = await self.client.count(
            collection_name=self.collection_name,
            count_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="file_id",
                        match=models.MatchValue(value=file_id),
                    )
                ]
            ),
        )
        count_before = count_result.count

        if count_before == 0:
            logger.info("[QdrantVectorDB] No chunks found for file_id=%s", file_id)
            return 0

        await self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="file_id",
                            match=models.MatchValue(value=file_id),
                        )
                    ]
                )
            ),
        )
        logger.info(
            "[QdrantVectorDB] Deleted %d chunks for file_id=%s",
            count_before,
            file_id,
        )
        return count_before

    async def delete_collection(self) -> None:
        """Delete the entire collection."""
        if await self.client.collection_exists(self.collection_name):
            await self.client.delete_collection(self.collection_name)
            logger.info(
                "[QdrantVectorDB] Collection '%s' deleted", self.collection_name
            )

    async def recreate_collection(self) -> None:
        """Delete and recreate collection (for embedding model changes)."""
        await self.delete_collection()
        await self.ensure_collection()
        logger.info("[QdrantVectorDB] Collection '%s' recreated", self.collection_name)
