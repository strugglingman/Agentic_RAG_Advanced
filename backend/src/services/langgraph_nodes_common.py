"""Shared helpers for LangGraph node modules."""

from typing import Dict, Any
from dataclasses import dataclass
import asyncio
import logging

from src.services.langgraph_state import AgentState, RuntimeContext
from src.models.evaluation import EvaluationResult, QualityLevel, RecommendationAction
from src.utils.file_content_extractor import extract_file_content as _extract_attachment_content
from src.services.llm_client import chat_completion
from src.config.settings import Config

logger = logging.getLogger(__name__)

def _clone_step_contexts(state: AgentState) -> dict:
    """
    Defensive copy for step_contexts to avoid mutating prior state snapshots in-place.
    """
    raw = state.get("step_contexts", {}) or {}
    return {
        step: list(items) if isinstance(items, list) else []
        for step, items in raw.items()
    }

def _clone_tool_results(state: AgentState) -> dict:
    """
    Defensive copy for tool_results to avoid mutating prior state snapshots in-place.
    """
    raw = state.get("tool_results", {}) or {}
    return {
        key: list(items) if isinstance(items, list) else []
        for key, items in raw.items()
    }

def evaluation_result_to_dict(result: EvaluationResult) -> dict:
    """Convert EvaluationResult object to serializable dict."""
    return {
        "quality": result.quality.value,
        "confidence": result.confidence,
        "coverage": result.coverage,
        "recommendation": result.recommendation.value,
        "reasoning": result.reasoning,
    }

def dict_to_evaluation_result(d: dict) -> EvaluationResult:
    """Convert dict back to EvaluationResult object."""
    if d is None:
        return None
    return EvaluationResult(
        quality=QualityLevel(d["quality"]),
        confidence=d["confidence"],
        coverage=d["coverage"],
        recommendation=RecommendationAction(d["recommendation"]),
        reasoning=d["reasoning"],
    )

class PreviousStepContext:
    """
    Container for all previous step information.

    Provides everything to LLM so it can extract what it needs.
    """

    def __init__(
        self,
        text: str = "",
        file_ids: list = None,
        urls: list = None,
    ):
        self.text = text  # Combined step_answers + raw step_contexts results
        self.file_ids = file_ids or []  # Pre-extracted file_ids for convenience
        self.urls = urls or []  # Pre-extracted URLs for convenience

def build_previous_step_context(
    state: AgentState, current_step: int
) -> PreviousStepContext:
    """
    Build complete context from ALL previous steps.

    This unified helper collects everything so LLM can extract what it needs:
    1. step_answers: Human-readable verified answers per step
    2. step_contexts: Raw tool/doc results (contains URLs, file content, etc.)
    3. file_ids: Pre-extracted from files_created for convenience
    4. urls: Pre-extracted from web_search results for convenience

    Args:
        state: Current agent state
        current_step: Current step number (collects from steps < current_step)

    Returns:
        PreviousStepContext with text, file_ids, and urls
    """
    import re

    step_answers = state.get("step_answers", [])
    step_contexts = _clone_step_contexts(state)

    text_parts = []
    file_ids = []
    urls = []

    # 1. Collect from step_answers (verified human-readable answers)
    for ans in step_answers:
        step_num = ans.get("step", "?")
        question = ans.get("question", "")
        answer = ans.get("answer", "")
        text_parts.append(f"Step {step_num} ({question}):\n{answer}")

    # 2. Collect from step_contexts (raw tool/doc data)
    for step_num in sorted(step_contexts.keys()):
        # Only look at PREVIOUS steps (not current or future)
        if step_num >= current_step:
            continue

        for ctx in step_contexts[step_num]:
            ctx_type = ctx.get("type", "")
            tool_name = ctx.get("tool_name", "")
            result = ctx.get("result", "")

            # Extract file_ids from files_created
            files_created = ctx.get("files_created", [])
            for f in files_created:
                fid = f.get("file_id") if isinstance(f, dict) else f
                if fid and fid not in file_ids:
                    file_ids.append(fid)

            # Extract URLs from web_search results
            if tool_name == "web_search" and result:
                # Match http:// and https:// URLs
                url_pattern = r'https?://[^\s<>"\')\]]+(?=[^\w]|$)'
                found_urls = re.findall(url_pattern, result)
                for url in found_urls:
                    # Clean up trailing punctuation
                    url = url.rstrip(".,;:!?")
                    if url and url not in urls:
                        urls.append(url)

            # Add raw result to text (for full context)
            if ctx_type == "tool" and result:
                text_parts.append(f"{tool_name} result:\n{result}")
            elif ctx_type == "document":
                docs = ctx.get("docs", [])
                if docs:
                    doc_texts = []
                    for i, doc in enumerate(docs[:5]):  # Limit to 5 docs
                        content = doc.get("content", "")[:500]  # Truncate
                        source = doc.get("source", "unknown")
                        doc_texts.append(f"  [{i+1}] {source}: {content}...")
                    text_parts.append(f"Retrieved documents:\n" + "\n".join(doc_texts))

    logger.info(
        f"[BUILD_PREV_CTX] step={current_step}, "
        f"answers={len(step_answers)}, "
        f"file_ids={len(file_ids)}, "
        f"urls={len(urls)}"
    )

    return PreviousStepContext(
        text="\n\n".join(text_parts) if text_parts else "",
        file_ids=file_ids,
        urls=urls,
    )

class AttachmentInfo:
    """Single attachment metadata and optional content."""

    file_id: str
    filename: str
    mime_type: str
    content: str = ""  # Populated only when extract_content=True

async def get_attachment_context(
    runtime: RuntimeContext,
    extract_content: bool = False,
) -> tuple[list[AttachmentInfo], str]:
    """
    Get attachment info from runtime context.

    Unified helper for any node to access attachment info:
    - extract_content=False: Returns metadata only (for planning)
    - extract_content=True: Returns full extracted content (for document creation)

    Handles both images (Vision API) and text files (PDF, DOCX, etc.)

    Args:
        runtime: RuntimeContext with attachment_file_ids, file_service, etc.
        extract_content: If True, extract file content. If False, just metadata.

    Returns:
        Tuple of (list of AttachmentInfo, formatted string for LLM prompt)
    """
    attachment_file_ids = runtime.get("attachment_file_ids", [])
    if not attachment_file_ids:
        return [], ""

    # Get runtime dependencies once (not inside loop)
    file_service = runtime.get("file_service")
    user_id = runtime.get("user_id")
    dept_id = runtime.get("dept_id")
    openai_client = runtime.get("openai_client")

    attachments = []
    text_parts = []

    for att in attachment_file_ids:
        file_id = att.get("file_id", "")
        filename = att.get("filename", "unknown")
        mime_type = att.get("mime_type", "")

        info = AttachmentInfo(
            file_id=file_id,
            filename=filename,
            mime_type=mime_type,
        )

        if extract_content:
            # Extract full content from file
            if file_service:
                try:
                    file_path = await file_service.get_file_path(
                        file_id, user_id, dept_id=dept_id
                    )
                    content = await _extract_attachment_content(
                        file_path, mime_type, openai_client
                    )
                    info.content = content
                    text_parts.append(
                        f"--- Attached File: {filename} (file_id: {file_id}) ---\n{content}"
                    )
                except Exception as e:
                    logger.warning(f"[ATTACHMENT] Failed to extract {filename}: {e}")
                    text_parts.append(
                        f"--- Attached File: {filename} (file_id: {file_id}) ---\n[Error: {e}]"
                    )
            else:
                text_parts.append(
                    f"--- Attached File: {filename} (file_id: {file_id}) ---\n[FileService not available]"
                )
        else:
            # Metadata only
            text_parts.append(f"- {filename} (file_id: {file_id}, type: {mime_type})")

        attachments.append(info)

    # Format output with consistent header
    header = "[ATTACHED FILE CONTENTS]" if extract_content else "[ATTACHED FILES]"
    formatted = f"{header}\n" + "\n".join(text_parts)

    return attachments, formatted

async def _optimize_step_query(step_query: str, tool_type: str, openai_client) -> str:
    """
    Optimize a planner-generated step query for the target tool.

    For "retrieve": Balanced optimization - expand abbreviations AND remove filler words.
    For "web_search": Shorten to stay under Tavily 400 char limit.

    Skip if refined_query exists (handled by refine_node).

    Args:
        step_query: The verbose query from planner
        tool_type: "retrieve" or "web_search"
        openai_client: OpenAI client for LLM optimization

    Returns:
        Optimized query string
    """
    if not step_query or not openai_client:
        return step_query

    try:
        if tool_type == "retrieve":
            # Balanced optimization: expand abbreviations + remove filler words
            prompt = f"""Optimize this query for document retrieval.

Input: {step_query}

Tasks (do ALL):
1. EXPAND common abbreviations and acronyms:
   - PTO → PTO paid time off
   - Q1/Q2/Q3/Q4 → Q1 first quarter (keep both forms)
   - YoY → YoY year over year
   - HR → HR human resources
   - ROI → ROI return on investment
   - KPI → KPI key performance indicator
   - OKR → OKR objectives key results
   - Rev → revenue
   - FY → FY fiscal year

2. REMOVE filler phrases (if present):
   - "Can you help me find..."
   - "I would like to know..."
   - "Search for information about..."
   - "Tell me about..."

3. KEEP important terms:
   - Names, dates, numbers, percentages
   - Domain-specific keywords
   - Key entities

Output the optimized query with abbreviations expanded. Do not truncate or shorten entity names.
Output ONLY the optimized query, nothing else."""

        elif tool_type == "web_search":
            # Optimize for web search - focus on keywords
            prompt = f"""Optimize this query for web search.

Input: {step_query}

Rules:
1. Keep essential keywords and entity names intact
2. Remove filler phrases and redundant descriptions
3. Do not truncate or shorten entity names

Output ONLY the optimized query."""
        else:
            return step_query

        response = await chat_completion(
            client=openai_client,
            model=Config.OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=300,
        )

        optimized = response.choices[0].message.content.strip().strip('"').strip("'")
        if len(optimized) >= 5:
            logger.info(
                f"[QUERY_OPT] {tool_type}: '{step_query[:50]}...' → '{optimized}'"
            )
            return optimized
        return step_query

    except Exception as e:
        logger.warning(f"[QUERY_OPT] Failed: {e}")
        return step_query
