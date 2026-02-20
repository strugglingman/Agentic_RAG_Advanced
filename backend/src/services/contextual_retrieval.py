"""
Contextual Retrieval — Anthropic's technique for enriching chunks with document context.

Uses an LLM to generate a short context preamble per chunk at ingestion time.
The contextualized text (context + original) is what gets embedded and BM25-indexed.
Original chunk text is preserved in metadata for display.

Reference: https://www.anthropic.com/news/contextual-retrieval
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from typing import Optional

from src.config.settings import Config

logger = logging.getLogger(__name__)

# ── Cache ────────────────────────────────────────────────────────────────────
_CACHE_DIR = os.path.join(Config.CHROMA_PATH, ".contextual_cache")


def _cache_key(full_text: str, model: str) -> str:
    """Deterministic hash of document content + model so cache invalidates on either change."""
    raw = f"{model}|{full_text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_cache(key: str) -> Optional[list[tuple[int, str, str]]]:
    """Load cached contextualized chunks if they exist."""
    path = os.path.join(_CACHE_DIR, f"{key}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [(item["page"], item["ctx"], item["orig"]) for item in data]
    except Exception as e:
        logger.warning(f"[CONTEXTUAL] Cache read failed ({key[:12]}…): {e}")
        return None


def _save_cache(key: str, results: list[tuple[int, str, str]]) -> None:
    """Persist contextualized chunks to disk."""
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        path = os.path.join(_CACHE_DIR, f"{key}.json")
        data = [{"page": pg, "ctx": ctx, "orig": orig} for pg, ctx, orig in results]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"[CONTEXTUAL] Cache write failed ({key[:12]}…): {e}")


# ── OpenAI client ────────────────────────────────────────────────────────────
# Lazy singleton — avoid creating a client per call
_client: Optional[object] = None


def _get_client():
    """Get or create the async OpenAI client."""
    global _client
    if _client is None:
        from openai import AsyncOpenAI

        _client = AsyncOpenAI(api_key=Config.OPENAI_KEY)
    return _client


_PROMPT_TEMPLATE = """\
<document>
{document}
</document>
Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>
Please give a short succinct context to situate this chunk within the overall document \
for the purposes of improving search retrieval of the chunk. \
Answer only with the succinct context and nothing else."""

# Truncate full document to ~120k chars (~30k tokens) to fit context windows
_MAX_DOC_CHARS = 120_000


async def _generate_context(
    full_text: str,
    chunk_text: str,
    filename: str,
    model: str,
) -> str:
    """
    Call the LLM to generate a context preamble for a single chunk.

    Returns the context string, or a rule-based fallback on failure.
    """
    try:
        client = _get_client()
        prompt = _PROMPT_TEMPLATE.format(
            document=full_text[:_MAX_DOC_CHARS],
            chunk=chunk_text,
        )
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.0,
        )
        context = response.choices[0].message.content.strip()
        if context:
            return context
    except Exception as e:
        logger.warning(f"[CONTEXTUAL] LLM call failed for chunk in {filename}: {e}")

    # Rule-based fallback
    return f"From {filename}."


async def contextualize_chunks(
    full_document_text: str,
    chunks: list[tuple[int, str]],
    filename: str,
) -> list[tuple[int, str, str]]:
    """
    Enrich chunks with LLM-generated document context.

    Args:
        full_document_text: The entire document text (all pages concatenated).
        chunks: List of (page_num, chunk_text) from the chunker.
        filename: Original filename (used in fallback context and logging).

    Returns:
        List of (page_num, contextualized_text, original_text) tuples.
        contextualized_text = "{context}\n\n{original_text}"
    """
    if not chunks:
        return []

    model = Config.CONTEXTUAL_RETRIEVAL_MODEL
    max_workers = Config.CONTEXTUAL_RETRIEVAL_MAX_WORKERS

    # Check cache — same document content + model = identical contextualization
    key = _cache_key(full_document_text, model)
    cached = _load_cache(key)
    if cached is not None:
        logger.info(
            f"[CONTEXTUAL] Cache hit for {filename} ({key[:12]}…), "
            f"{len(cached)} chunks"
        )
        return cached

    logger.info(
        f"[CONTEXTUAL] Contextualizing {len(chunks)} chunks for {filename} "
        f"(model={model}, concurrency={max_workers})"
    )

    semaphore = asyncio.Semaphore(max_workers)

    async def _process_chunk(idx: int, page_num: int, chunk_text: str):
        async with semaphore:
            try:
                context = await _generate_context(
                    full_document_text, chunk_text, filename, model,
                )
                contextualized = f"{context}\n\n{chunk_text}"
                return idx, page_num, contextualized, chunk_text
            except Exception as e:
                logger.warning(
                    f"[CONTEXTUAL] Worker failed for chunk {idx} in {filename}: {e}"
                )
                fallback = f"From {filename}.\n\n{chunk_text}"
                return idx, page_num, fallback, chunk_text

    results = await asyncio.gather(
        *(_process_chunk(i, pg, txt) for i, (pg, txt) in enumerate(chunks))
    )

    # Sort by original index to preserve chunk order
    results = sorted(results, key=lambda x: x[0])

    final = [(pg, ctx, orig) for _, pg, ctx, orig in results]

    # Persist to cache for future re-ingestion
    _save_cache(key, final)

    logger.info(
        f"[CONTEXTUAL] Done contextualizing {len(final)} chunks for {filename}"
    )

    return final
