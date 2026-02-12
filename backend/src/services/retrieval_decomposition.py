"""
Query Decomposition for Multi-Hop RAG Retrieval.

This module implements deterministic query decomposition for multi-entity
comparison queries, following 2025-2026 best practices from:
- ACL 2025: "Question Decomposition for Retrieval-Augmented Generation"
- Anthropic: Multi-agent research system patterns
- Haystack/Deepset: Query decomposition for RAG

Architecture:
============
User Query: "Compare Delta and Starbucks revenue 2023"
    ↓
decompose_query() → ["Delta revenue 2023", "Starbucks revenue 2023"]
    ↓
parallel_retrieve() → ThreadPoolExecutor runs retrieve() for each sub-query
    ↓
merge_with_balanced_topk() → Deduplicated, balanced contexts with sub_query labels
    ↓
Return to caller (agent_tools.py or langgraph_nodes.py)

Key Design Decisions:
====================
1. SYNC function - Uses ThreadPoolExecutor, not asyncio (keeps agent_tools sync)
2. Early return for simple queries - No overhead when len(sub_queries) == 1
3. Balanced merge - Each sub-query gets fair representation (prevents entity starvation)
4. Flat context structure - Adds sub_query field, doesn't change list[dict] format
5. No answer synthesis - Returns contexts, not summaries (preserves citations)

Reference: src/evaluation/eval_data/multihop_test.jsonl for example decompositions
"""

import asyncio
import json
import logging
import atexit
import time
from typing import Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.services.vector_db import VectorDB
from src.services.retrieval import retrieve
from src.services.llm_client import chat_completion_json
from src.config.settings import Config
from src.observability.metrics import observe_retrieval_latency, increment_error, MetricsErrorType

logger = logging.getLogger(__name__)


# =============================================================================
# SHARED THREAD POOL (Prevents thread explosion under load)
# =============================================================================
#
# Why shared pool:
# - Without limit: 100 users × 2 sub-queries = 200 threads → crash
# - With shared pool: Max 8 concurrent retrieves, others queue
#
# Future migration to async:
# - When ChromaDB supports async, replace with asyncio.gather()
# - For now, ThreadPoolExecutor is the pragmatic choice

_RETRIEVAL_EXECUTOR: Optional[ThreadPoolExecutor] = None


def get_retrieval_executor() -> ThreadPoolExecutor:
    """
    Get or create shared thread pool for parallel retrieval.

    Uses lazy initialization to avoid creating pool if decomposition is disabled.
    Pool is shared across all requests to limit total thread count.

    Returns:
        ThreadPoolExecutor with limited workers
    """
    global _RETRIEVAL_EXECUTOR
    if _RETRIEVAL_EXECUTOR is None:
        max_workers = Config.DECOMPOSITION_MAX_WORKERS
        _RETRIEVAL_EXECUTOR = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="retrieval_decomp"
        )
        logger.info(
            f"[DECOMPOSITION] Created shared thread pool with {max_workers} workers"
        )
    return _RETRIEVAL_EXECUTOR


def shutdown_executor():
    """Cleanup thread pool on application shutdown."""
    global _RETRIEVAL_EXECUTOR
    if _RETRIEVAL_EXECUTOR is not None:
        _RETRIEVAL_EXECUTOR.shutdown(wait=False)
        logger.info("[DECOMPOSITION] Shut down shared thread pool")


# Register cleanup on process exit
atexit.register(shutdown_executor)


# =============================================================================
# SECTION 1: FEW-SHOT EXAMPLES FOR DECOMPOSITION
# =============================================================================

# Load examples from multihop_test.jsonl or define inline
# These guide the LLM to produce consistent decomposition format
FEW_SHOT_EXAMPLES = """
Example 1:
Query: "Compare the 2023 revenues of Delta Air Lines and Starbucks Corporation"
Output: ["Delta Air Lines total revenue 2023", "Starbucks total revenue 2023"]

Example 2:
Query: "How did Chipotle's net income change from 2021 to 2023?"
Output: ["Chipotle net income 2021", "Chipotle net income 2023"]

Example 3:
Query: "What is the relationship between BNSF Railway and Berkshire Hathaway, and how many employees does BNSF have?"
Output: ["BNSF Railway Berkshire Hathaway ownership", "BNSF Railway employees 2023"]

Example 4:
Query: "What is our Q3 revenue?"
Output: ["Q3 revenue"]

Example 5:
Query: "Tell me about the vacation policy"
Output: ["vacation policy"]

Example 6:
Query: "Compare Berkshire Hathaway and UnitedHealth Group's share repurchase and dividend per share in 2023"
Output: ["Berkshire Hathaway share repurchase 2023", "Berkshire Hathaway dividend per share 2023", "UnitedHealth Group share repurchase 2023", "UnitedHealth Group dividend per share 2023"]

Example 7:
Query: "Compare Apple, Microsoft, and Google's revenue growth in 2023"
Output: ["Apple revenue growth 2023", "Microsoft revenue growth 2023", "Google revenue growth 2023"]
"""


# =============================================================================
# SECTION 2: DECOMPOSITION PROMPT
# =============================================================================

DECOMPOSITION_SYSTEM_PROMPT = """You are a query decomposition assistant for a RAG system.

Your task: Analyze the user's query and determine if it needs to be split into
multiple sub-queries for better retrieval results.

WHEN TO DECOMPOSE:
- Comparison queries: "Compare A and B" → ["A metric", "B metric"]
- Multi-entity queries: "What are X and Y's revenues?" → ["X revenue", "Y revenue"]
- Multi-aspect queries: "What is A's relationship to B and how many employees?" → ["A B relationship", "A employees"]
- Time-range queries: "How did X change from 2021 to 2023?" → ["X 2021", "X 2023"]
- Multi-entity multi-attribute: "Compare A and B on X and Y" → ["A X", "A Y", "B X", "B Y"] (cover ALL combinations)

WHEN NOT TO DECOMPOSE (return single query):
- Simple factual: "What is our Q3 revenue?" → ["Q3 revenue"]
- Single entity: "Tell me about Delta's performance" → ["Delta performance"]
- Already focused: "BNSF Railway employee count" → ["BNSF Railway employee count"]

OUTPUT FORMAT:
- Return a JSON array of strings
- Each string is a SHORT, KEYWORD-FOCUSED search query (not a sentence)
- Keep entity names intact (e.g., "Delta Air Lines" not just "Delta")
- Include relevant year/period if mentioned
- Maximum 6 sub-queries (covers most multi-entity multi-attribute comparisons)

{few_shot_examples}
"""

DECOMPOSITION_USER_PROMPT = """Query: "{query}"

Analyze this query and return a JSON object with a "queries" key containing an array of sub-queries.
If no decomposition needed, return a single-element array with the original query converted to keyword format.

Output ONLY valid JSON object:
{{"queries": ["sub-query1", "sub-query2", ...]}}
"""


# =============================================================================
# SECTION 3: CORE FUNCTIONS
# =============================================================================


async def decompose_query(
    query: str,
    openai_client,
    model: str = Config.OPENAI_MODEL,
    temperature: float = 0.0,
) -> list[str]:
    """
    Decompose a complex query into sub-queries using LLM.

    Args:
        query: Original user query
        openai_client: OpenAI client instance
        model: Model to use (default: Config.DECOMPOSITION_MODEL or gpt-4o-mini)
        temperature: LLM temperature (0.0 for deterministic)

    Returns:
        List of sub-queries (1 element if no decomposition needed)

    Implementation Steps:
    --------------------
    1. Build prompt with few-shot examples
    2. Call LLM with JSON mode or structured output
    3. Parse response as JSON array
    4. Validate: non-empty, max 4 sub-queries
    5. Fallback: return [query] if parsing fails

    Example:
        >>> decompose_query("Compare Delta and Starbucks revenue", client)
        ["Delta Air Lines total revenue 2023", "Starbucks total revenue 2023"]

        >>> decompose_query("What is our Q3 revenue?", client)
        ["Q3 revenue"]
    """
    if not Config.DECOMPOSITION_ENABLED:
        return [query]

    messages = [
        {
            "role": "system",
            "content": DECOMPOSITION_SYSTEM_PROMPT.format(
                few_shot_examples=FEW_SHOT_EXAMPLES
            ),
        },
        {
            "role": "user",
            "content": DECOMPOSITION_USER_PROMPT.format(query=query),
        },
    ]
    try:
        response = await chat_completion_json(
            client=openai_client,
            messages=messages,
            model=model,
            temperature=temperature,
        )
        content = response.choices[0].message.content
        parsed = json.loads(content)

        # Extract queries from JSON object (response_format=json_object returns {})
        if isinstance(parsed, dict):
            sub_queries = parsed.get("queries", [])
        elif isinstance(parsed, list):
            # Fallback if LLM returns array directly
            sub_queries = parsed
        else:
            raise ValueError(f"Unexpected response type: {type(parsed)}")

        if not isinstance(sub_queries, list):
            raise ValueError("'queries' field is not a list")
        if len(sub_queries) == 0:
            raise ValueError("Decomposition returned empty list")
        if len(sub_queries) > 6:
            sub_queries = sub_queries[:6]

        logger.info(f"[DECOMPOSITION] '{query}' → {sub_queries}")
        print(f"         Decomposed into: {sub_queries}")

        return sub_queries
    except Exception as e:
        logger.warning(
            f"[DECOMPOSITION] Failed for '{query}': {e}, returning original query"
        )
        return [query]


def parallel_retrieve(
    sub_queries: list[str],
    vector_db: VectorDB,
    dept_id: str,
    user_id: str,
    top_k: int,
    where: Optional[dict],
    use_hybrid: bool,
    use_reranker: bool,
) -> list[tuple[list[dict], str]]:
    """
    Execute retrieve() for each sub-query in parallel using ThreadPoolExecutor.

    Args:
        sub_queries: List of sub-queries from decompose_query()
        vector_db: VectorDB instance
        dept_id: Department ID for filtering
        user_id: User ID for filtering
        top_k: Number of results per sub-query
        where: ChromaDB where clause
        use_hybrid: Enable hybrid search
        use_reranker: Enable reranker

    Returns:
        List of (contexts, sub_query) tuples, one per sub-query

    Implementation Steps:
    --------------------
    1. Import retrieve from src.services.retrieval
    2. Create ThreadPoolExecutor with max_workers=len(sub_queries)
    3. Submit retrieve() call for each sub-query
    4. Collect results with as_completed()
    5. Return list of (contexts, sub_query) tuples

    Why ThreadPoolExecutor (not asyncio):
    - retrieve() is sync function
    - agent_tools.execute_search_documents is sync
    - ThreadPoolExecutor runs in OS threads, no event loop issues

    Example:
        >>> results = parallel_retrieve(
        ...     ["Delta revenue", "Starbucks revenue"],
        ...     vector_db, dept_id, user_id, top_k=5, ...
        ... )
        >>> len(results)
        2
        >>> results[0]  # (contexts_list, "Delta revenue")
    """
    if len(sub_queries) == 1:
        ctx, _ = retrieve(
            vector_db=vector_db,
            query=sub_queries[0],
            dept_id=dept_id,
            user_id=user_id,
            top_k=top_k,
            where=where,
            use_hybrid=use_hybrid,
            use_reranker=use_reranker,
        )
        return [(ctx or [], sub_queries[0])]

    executor = get_retrieval_executor()
    future_to_sq = {
        executor.submit(
            retrieve,
            vector_db=vector_db,
            query=sq,
            dept_id=dept_id,
            user_id=user_id,
            top_k=top_k,
            where=where,
            use_hybrid=use_hybrid,
            use_reranker=use_reranker,
        ): sq
        for sq in sub_queries
    }

    results = []
    sq = ""
    try:
        for future in as_completed(future_to_sq, timeout=60):
            sq = future_to_sq.get(future, "")
            ctx, _ = future.result()
            results.append((ctx or [], sq))
            logger.debug(f"[PARALLEL_RETRIEVE] '{sq}': {len(ctx or [])} contexts")
    except TimeoutError as te:
        logger.error(f"[PARALLEL_RETRIEVE] Timeout: {te}")
        increment_error(MetricsErrorType.TIMEOUT)
        results.append(([], sq))
    except Exception as e:
        logger.error(f"[PARALLEL_RETRIEVE] Failed for '{sq}': {e}")
        results.append(([], sq))

    return results


def merge_with_balanced_topk(
    results: list[tuple[list[dict], str]],
    total_k: int = None,
    original_query: str = None,
) -> list[dict]:
    """
    Merge contexts from multiple sub-queries with balanced representation.

    Args:
        results: List of (contexts, sub_query) from parallel_retrieve()
        total_k: Total number of contexts to return
        original_query: Original query (for potential reranking)

    Returns:
        Merged, deduplicated contexts with sub_query labels

    Key Design: Balanced Selection
    -----------------------------
    Problem: Global top-K can starve one entity (e.g., 8 Delta, 2 Starbucks)
    Solution: Guarantee minimum contexts per sub-query, then fill with best remaining

    Implementation Steps:
    --------------------
    1. Calculate per_query_k = max(2, total_k // n_subqueries)
    2. First pass: Take top per_query_k from each sub-query (with dedup)
    3. Second pass: Fill remaining slots with best unused chunks
    4. Add sub_query label to each context
    5. Optionally rerank merged set for ordering (but don't drop)

    Deduplication Key:
    - Primary: chunk_id (if available)
    - Fallback: file_id + chunk[:150] (matches existing unique_snippet logic)

    Example:
        >>> results = [
        ...     ([ctx1, ctx2, ctx3, ctx4, ctx5], "Delta revenue"),
        ...     ([ctx6, ctx7, ctx8, ctx9, ctx10], "Starbucks revenue"),
        ... ]
        >>> merged = merge_with_balanced_topk(results, total_k=8, ...)
        >>> len(merged)
        8
        >>> # Guaranteed: at least 4 from each sub-query (balanced)

    Note:
        Current implementation uses simple merge with deduplication.
        Full balanced allocation (per_query_k) not yet implemented.
        May cause entity starvation in some multi-entity queries.
    """
    if not results:
        return []

    SUB_QUERY_LABEL = "sub_query"
    merged_contexts = []
    seen = set()
    for contexts, sq in results:
        for ctx in contexts:
            ctx_key = get_dedup_key(ctx)
            if ctx_key not in seen:
                ctx[SUB_QUERY_LABEL] = sq
                merged_contexts.append(ctx)
                seen.add(ctx_key)

    return merged_contexts


# =============================================================================
# SECTION 4: MAIN ENTRY POINT
# =============================================================================


async def retrieve_with_decomposition(
    vector_db: VectorDB,
    openai_client,
    query: str,
    dept_id: str,
    user_id: str,
    top_k: int = Config.TOP_K,
    where: Optional[dict] = None,
    use_hybrid: bool = Config.USE_HYBRID,
    use_reranker: bool = Config.USE_RERANKER,
) -> Tuple[list[dict], Optional[str]]:
    """
    Main entry point: Retrieve with automatic query decomposition.

    This function wraps the existing retrieve() function, adding:
    1. Automatic detection of multi-entity queries
    2. LLM-based query decomposition
    3. Parallel retrieval for sub-queries
    4. Balanced merging with deduplication

    Args:
        vector_db: VectorDB instance
        query: User's search query
        dept_id: Department ID for filtering
        user_id: User ID for filtering
        openai_client: OpenAI client for decomposition LLM call
        top_k: Number of results to return (default: Config.TOP_K)
        where: ChromaDB where clause
        use_hybrid: Enable hybrid search (BM25 + semantic)
        use_reranker: Enable cross-encoder reranking

    Returns:
        Tuple of (contexts_list, error_message)
        - contexts_list: List of context dicts, each with added "sub_query" field
        - error_message: None on success, error string on failure

    Signature matches retrieve() for drop-in replacement.

    Implementation Steps:
    --------------------
    1. Early return if decomposition disabled
    2. Call decompose_query() to get sub-queries
    3. If single sub-query, call retrieve() directly (no overhead)
    4. If multiple sub-queries, call parallel_retrieve()
    5. Merge results with balanced top-K
    6. Return merged contexts

    Usage in agent_tools.py:
        # OLD:
        ctx, err = retrieve(vector_db, query, ...)

        # NEW:
        ctx, err = retrieve_with_decomposition(vector_db, query, ..., openai_client)

    Usage in langgraph_nodes.py:
        # OLD:
        ctx, err = retrieve(query, vector_db, ...)

        # NEW:
        ctx, err = retrieve_with_decomposition(vector_db, query, ..., openai_client)
    """
    start_time = time.time()
    search_type = "single"  # Default, updated if decomposition happens

    def _get_search_mode() -> str:
        """Build search mode string for metrics label."""
        mode = "hybrid" if use_hybrid else "semantic"
        if use_reranker:
            mode += "_reranker"
        return mode

    try:
        if not Config.DECOMPOSITION_ENABLED:
            search_type = _get_search_mode()
            return await asyncio.to_thread(
                retrieve,
                vector_db=vector_db,
                query=query,
                dept_id=dept_id,
                user_id=user_id,
                top_k=top_k,
                where=where,
                use_hybrid=use_hybrid,
                use_reranker=use_reranker,
            )

        sub_queries = await decompose_query(
            query, openai_client=openai_client, model=Config.OPENAI_MODEL, temperature=0
        )

        # Set search_type based on decomposition result
        if len(sub_queries) > 1:
            search_type = f"decomposed_{_get_search_mode()}"
        else:
            search_type = _get_search_mode()

        results = await asyncio.to_thread(
            parallel_retrieve,
            sub_queries=sub_queries,
            vector_db=vector_db,
            dept_id=dept_id,
            user_id=user_id,
            top_k=top_k,
            where=where,
            use_hybrid=use_hybrid,
            use_reranker=use_reranker,
        )
        final_contexts = merge_with_balanced_topk(results=results)
        log_decomposition_result(
            original_query=query,
            sub_queries=sub_queries,
            merged=final_contexts,
        )
        return final_contexts, None
    except Exception as e:
        logger.warning(f"[DECOMPOSITION] Failed: {e}, falling back to original query")
        search_type = _get_search_mode()
        return await asyncio.to_thread(
            retrieve,
            vector_db=vector_db,
            query=query,
            dept_id=dept_id,
            user_id=user_id,
            top_k=top_k,
            where=where,
            use_hybrid=use_hybrid,
            use_reranker=use_reranker,
        )
    finally:
        duration = time.time() - start_time
        observe_retrieval_latency(search_type, duration)


# =============================================================================
# SECTION 5: UTILITY FUNCTIONS
# =============================================================================


def get_dedup_key(ctx: dict) -> str:
    """
    Generate deduplication key for a context dict.

    Priority:
    1. chunk_id (most reliable, from ChromaDB)
    2. file_id + chunk prefix (fallback, matches unique_snippet logic)

    Args:
        ctx: Context dictionary

    Returns:
        String key for deduplication
    """
    chunk_id = ctx.get("chunk_id")
    if chunk_id:
        return chunk_id

    return f"{ctx.get('file_id', '')}:{ctx.get('chunk', '')[:150]}"


def log_decomposition_result(
    original_query: str,
    sub_queries: list[str],
    merged: list[dict],
) -> None:
    """
    Log decomposition results for debugging and monitoring.

    Args:
        original_query: Original user query
        sub_queries: Decomposed sub-queries
        merged: Final merged contexts
    """
    logger.info(f"[DECOMPOSITION] Original: '{original_query}'")
    logger.info(f"[DECOMPOSITION] Sub-queries: {sub_queries}")

    # Count contexts by sub-query
    sub_query_counts = {}
    for ctx in merged:
        sq = ctx.get("sub_query", "unknown")
        sub_query_counts[sq] = sub_query_counts.get(sq, 0) + 1
    logger.info(f"[DECOMPOSITION] Context counts: {sub_query_counts}")

    # Pretty print each context
    logger.info(f"[DECOMPOSITION] Merged Contexts ({len(merged)} total):")
    for i, ctx in enumerate(merged, 1):
        chunk_id = ctx.get("chunk_id", "N/A")
        chunk_preview = ctx.get("chunk", "")[:100].replace("\n", " ")
        source = ctx.get("source", "unknown")
        sub_query = ctx.get("sub_query", "unknown")

        logger.info(
            f"  [{i}] chunk_id: {chunk_id}\n"
            f"      source: {source}\n"
            f"      sub_query: {sub_query}\n"
            f"      content: {chunk_preview}..."
        )
