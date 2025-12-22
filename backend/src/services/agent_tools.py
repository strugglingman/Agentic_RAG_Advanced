"""
Agent tools for the ReAct agent.

This module defines:
1. Tool schemas (what the LLM sees - OpenAI function calling format)
2. Tool execution functions (what actually runs when LLM calls a tool)
3. Tool registry (maps tool names to execution functions)

Each tool has two parts:
- SCHEMA: Describes the tool to the LLM (name, description, parameters)
- execute_xxx: The Python function that actually executes the tool
"""

import os
import re
import asyncio
from datetime import datetime
from typing import Dict, Any
import logging
from urllib.parse import urlparse
import urllib.parse
import requests
from langsmith import traceable
from src.services.retrieval import retrieve, build_where
from src.config.settings import Config
from src.utils.safety import scrub_context
from src.services.retrieval_evaluator import RetrievalEvaluator
from src.models.evaluation import ReflectionConfig, EvaluationCriteria
from src.services.query_refiner import QueryRefiner
from src.services.clarification_helper import ClarificationHelper
from src.services.web_search import WebSearchService
from src.utils.send_email import send_email
from src.services.file_manager import FileManager

logger = logging.getLogger(__name__)

# ============================================================================
# TOOL 1: SEARCH DOCUMENTS
# ============================================================================

SEARCH_DOCUMENTS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_documents",
        "description": (
            "Search internal company documents for information. "
            "Use this when the user asks about company data, reports, policies, "
            "or any information that might be in uploaded documents. "
            "Returns relevant document chunks with sources and page numbers."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The search query. Be specific and use keywords from the user's question. "
                        "For example: 'Q1 2024 revenue', 'employee vacation policy', 'sales targets'"
                    ),
                },
                "top_k": {
                    "type": "integer",
                    "description": (
                        "Number of document chunks to retrieve. "
                        "Use 3-5 for quick lookups, 8-10 for comprehensive answers. Default is 5."
                    ),
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}


@traceable
def execute_search_documents(args: Dict[str, Any], context: Dict[str, Any]) -> str:
    """
    Execute the search_documents tool.

    Args:
        args: Arguments from LLM containing:
            - query (str): Search query
            - top_k (int, optional): Number of results to return

        context: System context containing:
            - collection: ChromaDB collection
            - dept_id: Department ID from auth
            - user_id: User ID from auth
            - request: Flask request object (for filters)

    Returns:
        Formatted string result for LLM containing document chunks

    Flow with refinement loop:
    1. Retrieve contexts with original query
    2. Evaluate quality
    3. If REFINE recommended and attempts < MAX:
       - Refine query
       - Retrieve again
       - Re-evaluate
       - Repeat until ANSWER or max attempts
    4. Return best contexts found
    """
    query = args.get("query", "")
    top_k = args.get("top_k", 5)
    collection = context.get("collection", None)
    dept_id = context.get("dept_id", "")
    user_id = context.get("user_id", "")
    use_hybrid = context.get("use_hybrid", Config.USE_HYBRID)
    use_reranker = context.get("use_reranker", Config.USE_RERANKER)
    request_data = context.get("request_data", {})
    if not dept_id or not user_id:
        return "Error: No organization ID or user ID provided"
    if not collection:
        return "Error: No document collection available"
    try:
        # Build where clause
        where = build_where(request_data, dept_id, user_id)

        # Initial retrieval
        current_query = query
        ctx, _ = retrieve(
            collection=collection,
            query=current_query,
            dept_id=dept_id,
            user_id=user_id,
            top_k=top_k,
            where=where,
            use_hybrid=use_hybrid,
            use_reranker=use_reranker,
        )

        if not ctx:
            return "No relevant documents found."

        # Store raw contexts in context dict for agent to access later
        context["_retrieved_contexts"] = ctx

        # Initialize messages (set by self-reflection based on quality)
        clarification_msg = None
        external_msg = None

        if Config.USE_SELF_REFLECTION:
            config = ReflectionConfig.from_settings(Config)
            client = context.get("openai_client", None)
            evaluator = RetrievalEvaluator(config=config, openai_client=client)

            # Initial evaluation
            criteria = EvaluationCriteria(
                query=current_query,
                contexts=ctx,
                search_metadata={
                    "hybrid": use_hybrid,
                    "reranker": use_reranker,
                    "top_k": top_k,
                },
                mode=config.mode,
            )
            eval_result = evaluator.evaluate(criteria)
            context["_last_evaluation"] = eval_result

            # Log initial evaluation
            print(
                f"[SELF-REFLECTION] Quality: {eval_result.quality.value}, "
                f"Confidence: {eval_result.confidence:.2f}, "
                f"Recommendation: {eval_result.recommendation.value}"
            )
            print(f"[SELF-REFLECTION] Reasoning: {eval_result.reasoning}")
            if eval_result.issues:
                print(f"[SELF-REFLECTION] Issues: {', '.join(eval_result.issues)}")
            if eval_result.missing_aspects:
                print(
                    f"[SELF-REFLECTION] Missing aspects: {', '.join(eval_result.missing_aspects)}"
                )

            # =================================================================
            # Handle Direct CLARIFY (confidence < 0.5) - skip refinement
            # =================================================================
            if eval_result.should_clarify:
                clarifier = ClarificationHelper(openai_client=client)
                clarification_msg = clarifier.generate_clarification(
                    query=query,
                    eval_result=eval_result,
                    max_attempts_reached=False,
                    context_hint="uploaded documents",
                )
                print(f"[CLARIFICATION] Direct clarify (confidence < 0.5)")

            # =================================================================
            # Handle EXTERNAL Recommendation (Week 3 - Day 6)
            # =================================================================
            # Signal agent to use web_search tool (don't execute automatically)
            if eval_result.should_search_external:
                print(f"[SELF-REFLECTION] EXTERNAL recommended - suggesting web search")
                external_msg = (
                    f"[EXTERNAL SEARCH SUGGESTED]\n"
                    f'The query "{query}" appears to require external information '
                    f"not found in the uploaded documents.\n\n"
                    f"Reason: {eval_result.reasoning}\n\n"
                    f"Recommendation: Use the web_search tool to find current or external information.\n\n"
                    f"---\n\n"
                )

            # Refinement loop with local counter (only if not already CLARIFY)
            refinement_count = 0
            if Config.REFLECTION_AUTO_REFINE and eval_result.should_refine:
                max_attempts = Config.REFLECTION_MAX_REFINEMENT_ATTEMPTS
                openai_model = context.get("model", Config.OPENAI_MODEL)
                temperature = context.get("temperature", Config.OPENAI_TEMPERATURE)
                query_refiner = QueryRefiner(
                    openai_client=client, model=openai_model, temperature=temperature
                )

                # Track refinement history for this tool call (for logging)
                refinement_history = []

                while refinement_count < max_attempts:

                    refinement_count += 1
                    print(
                        f"[QUERY_REFINER] Refinement attempt {refinement_count}/{max_attempts}"
                    )

                    # Refine the query (use current_query, not original)
                    refined_query = query_refiner.refine_query(
                        original_query=current_query,
                        eval_result=eval_result,
                        context_hint=" ".join([c.get("chunk", "")[:100] for c in ctx]),
                    )

                    # Track for logging
                    refinement_history.append(
                        {
                            "attempt": refinement_count,
                            "from": current_query,
                            "to": refined_query,
                        }
                    )
                    print(f"[QUERY_REFINER] '{current_query}' -> '{refined_query}'")

                    # Update current query for next iteration
                    current_query = refined_query

                    # Retrieve with refined query
                    ctx, _ = retrieve(
                        collection=collection,
                        query=current_query,
                        dept_id=dept_id,
                        user_id=user_id,
                        top_k=top_k,
                        where=where,
                        use_hybrid=use_hybrid,
                        use_reranker=use_reranker,
                    )

                    if not ctx:
                        print(
                            f"[QUERY_REFINER] No results for refined query, keeping previous results"
                        )
                        # Restore previous contexts
                        ctx = context["_retrieved_contexts"]
                        break

                    # Update stored contexts
                    context["_retrieved_contexts"] = ctx

                    # Re-evaluate
                    criteria = EvaluationCriteria(
                        query=current_query,
                        contexts=ctx,
                        search_metadata={
                            "hybrid": use_hybrid,
                            "reranker": use_reranker,
                            "top_k": top_k,
                        },
                        mode=config.mode,
                    )
                    eval_result = evaluator.evaluate(criteria)
                    context["_last_evaluation"] = eval_result

                    # Log refined evaluation
                    print(
                        f"[SELF-REFLECTION] (Attempt {refinement_count}) Quality: {eval_result.quality.value}, "
                        f"Confidence: {eval_result.confidence:.2f}, "
                        f"Recommendation: {eval_result.recommendation.value}"
                    )

                # Log final status
                if refinement_count > 0:
                    if refinement_count >= max_attempts:
                        print(
                            f"[QUERY_REFINER] Max refinement attempts ({max_attempts}) reached"
                        )

                        # =================================================================
                        # Progressive Fallback: max attempts reached → clarification
                        # =================================================================
                        clarifier = ClarificationHelper(openai_client=client)
                        clarification_msg = clarifier.generate_clarification(
                            query=query,  # Original query (not refined)
                            eval_result=eval_result,
                            max_attempts_reached=True,
                            context_hint="uploaded documents",
                        )
                        print(
                            f"[CLARIFICATION] Max attempts reached, returning clarification"
                        )
                    else:
                        print(
                            f"[QUERY_REFINER] Refinement complete after {refinement_count} attempt(s)"
                        )

                    # Store history in context for debugging (optional)
                    context["_refinement_history"] = refinement_history

        # Continue with normal flow: format and return contexts
        for c in ctx:
            c["chunk"] = scrub_context(c.get("chunk", ""))

        # Format contexts for LLM using the same high-quality formatting as build_prompt()
        # This creates detailed context headers with source, page, and scores
        context_str = "\n\n".join(
            (
                f"Context {i+1} (Source: {c.get('source', 'unknown')}"
                + (f", Page: {c['page']}" if c.get("page", 0) > 0 else "")
                + f"):\n{c.get('chunk', '')}\n"
                + (
                    f"Hybrid score: {c['hybrid']:.2f}"
                    if c.get("hybrid") is not None
                    else ""
                )
                + (
                    f", Rerank score: {c['rerank']:.2f}"
                    if c.get("rerank") is not None
                    else ""
                )
            )
            for i, c in enumerate(ctx)
        )

        # Return formatted contexts with instructions for the LLM
        result = (
            f"Found {len(ctx)} relevant document(s):\n\n"
            f"{context_str}\n\n"
            f"Instructions for DOCUMENT CONTEXTS ABOVE (Context 1-{len(ctx)}):\n"
            f"- CRITICAL: Every sentence MUST include at least one citation like [1], [2] that refers to the numbered Context items\n"
            f"- Place bracket citations [n] IMMEDIATELY AFTER each sentence\n"
            f"- Example: 'The revenue was $50M [1]. Sales increased by 20% [2].'\n"
            f"- Use ONLY the information from these contexts to answer\n"
            f"- At the end of your answer, cite the sources you used. For each source file, list the specific page numbers "
            f"from the contexts you referenced (look at the 'Page:' information in each context header). "
            f"Format: 'Sources: filename1.pdf (pages 15, 23), filename2.pdf (page 7)'\n"
            f"- If you also have web_search results in other tool responses, answer those naturally WITHOUT citations\n"
        )

        # Prepend external search suggestion if recommended
        if external_msg:
            print(
                "[SEARCH_DOCUMENTS] BUT!!!!!!!!!!!!!! Prepending external search suggestion"
            )
            result = external_msg + result
            print(f"[SEARCH_DOCUMENTS] External search suggestion:\n{result}")

        # Prepend clarification message if quality was poor
        if clarification_msg:
            result = (
                f"[QUALITY WARNING]\n{clarification_msg}\n\n" f"---\n\n" f"{result}"
            )

        return result
    except Exception as e:
        return f"Error during document retrieval: {str(e)}"


# ============================================================================
# TOOL 2: CALCULATOR
# ============================================================================

CALCULATOR_SCHEMA = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": (
            "Perform mathematical calculations. "
            "Use this for arithmetic operations, percentages, or any numerical computation. "
            "Supports basic operations (+, -, *, /), percentages, and more."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": (
                        "Mathematical expression to evaluate. "
                        "Examples: '2+2', '15% of 100', '(100-80)/80 * 100', '1000 * 1.15'"
                    ),
                },
            },
            "required": ["expression"],
        },
    },
}


@traceable
def execute_calculator(args: Dict[str, Any], context: Dict[str, Any]) -> str:
    """
    Execute the calculator tool.

    Args:
        args: Arguments from LLM containing:
            - expression (str): Mathematical expression to evaluate

        context: System context (not used for calculator)

    Returns:
        String result of the calculation
    """
    expression = args.get("expression", "").strip()

    if not expression:
        return "Error: No expression provided"

    try:
        # Handle percentage expressions
        # "15% of 100" -> "0.15 * 100"
        # "20%" -> "0.20"
        processed = _handle_percentage(expression)

        # Safely evaluate the expression
        result = _safe_eval(processed)

        # Format result nicely
        if isinstance(result, float):
            # Remove unnecessary decimals for whole numbers
            if result.is_integer():
                return str(int(result))
            else:
                return f"{result:.6f}".rstrip("0").rstrip(".")
        return str(result)

    except ZeroDivisionError:
        return "Error: Division by zero"
    except Exception as e:
        return f"Error: Invalid mathematical expression - {str(e)}"


def _handle_percentage(expr: str) -> str:
    """
    Convert percentage expressions to Python math.

    Examples:
        "15% of 100" -> "0.15 * 100"
        "20%" -> "0.20"
        "increase by 10%" -> handled in context
    """
    # Pattern: "X% of Y" -> "X/100 * Y"
    expr = re.sub(
        r"(\d+(?:\.\d+)?)\s*%\s+of\s+(\d+(?:\.\d+)?)",
        r"(\1/100) * \2",
        expr,
        flags=re.IGNORECASE,
    )

    # Pattern: "X%" -> "X/100"
    expr = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", expr)

    return expr


def _safe_eval(expr: str) -> float:
    """
    Safely evaluate a mathematical expression.

    Security: Only allows numbers and basic math operators.
    """
    # Whitelist: only allow numbers, operators, parentheses, dots, spaces
    if not re.match(r"^[\d\s\+\-\*\/\(\)\.\%]+$", expr):
        raise ValueError("Expression contains invalid characters")

    # Additional safety: limit length
    if len(expr) > 200:
        raise ValueError("Expression too long")

    # Evaluate using Python's eval (safe because we validated input)
    # Note: In production, consider using numexpr or ast-based parser
    result = eval(expr, {"__builtins__": {}}, {})

    return float(result)


# ============================================================================
# TOOL 3: WEB SEARCH
# ============================================================================
WEB_SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for up-to-date information when not found in internal documents. "
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The search query. Be specific and use keywords from the user's question. "
                        "For example: 'Q1 2024 revenue', 'employee vacation policy', 'sales targets'"
                    ),
                },
                "max_results": {
                    "type": "integer",
                    "description": (
                        "Number of web search results to return. "
                        "Use 3-5 for quick lookups, 8-10 for comprehensive answers. Default is 5."
                    ),
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}

# ============================================================================
# TOOL 4: DOWNLOAD FILE
# ============================================================================
DOWNLOAD_FILE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "download_file",
        "description": ("Download a file from a given URL or internal document link."),
        "parameters": {
            "type": "object",
            "properties": {
                "file_urls": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "description": "URL of the file to download.",
                    },
                    "description": (
                        "The list of URLs of the files to download. Can be external URLs or internal document links."
                    ),
                },
            },
            "required": ["file_urls"],
        },
    },
}

# ============================================================================
# TOOL 5: SEND EMAIL
# ============================================================================
SEND_EMAIL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "send_email",
        "description": "Send an email with optional attachments.",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {
                    "type": "array",
                    "items": {"type": "string", "format": "email"},
                    "description": "Recipient email addresses",
                },
                "subject": {"type": "string", "description": "Email subject"},
                "body": {"type": "string", "description": "Email body content"},
                "attachments": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional file references to attach. ALWAYS use file_id (e.g., 'cmjg8yrab0000xspw5ip79qcb') from tool responses or Available Files list. "
                        "DO NOT use download URLs like '/api/downloads/...' or '/api/files/...'. "
                        "Accepted formats: file_id (recommended), category:filename, or just filename"
                    ),
                    "minItems": 0,
                },
            },
            "required": ["to", "subject", "body"],
        },
    },
}

# ============================================================================
# TOOL 6: CREATE DOCUMENTS
# ============================================================================

CREATE_DOCUMENTS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "create_documents",
        "description": (
            "Create one or more formatted documents (TXT, CSV, HTML, PDF, DOCX etc.) from provided content. "
            "Use this when user asks to generate reports, summaries, or comparison documents. "
            "Each generated document will be saved and download links returned."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "documents": {
                    "type": "array",
                    "description": "List of documents to create",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": (
                                    "The main content to include in the document. "
                                    "Can include markdown formatting (headings, lists, bold, etc.)"
                                ),
                            },
                            "filename": {
                                "type": "string",
                                "description": (
                                    "Optional filename for the document (without extension). "
                                    "If not provided, title will be used."
                                ),
                            },
                            "title": {
                                "type": "string",
                                "description": "Document title (will be used as filename and header)",
                            },
                            "format": {
                                "type": "string",
                                "enum": [
                                    "pdf",
                                    "docx",
                                    "txt",
                                    "md",
                                    "html",
                                    "csv",
                                    "xlsx",
                                ],
                                "description": (
                                    "Output format:\n"
                                    "- 'pdf': PDF document (formatted, professional)\n"
                                    "- 'docx': Microsoft Word document\n"
                                    "- 'txt': Plain text file\n"
                                    "- 'md': Markdown file (preserves markdown formatting)\n"
                                    "- 'html': HTML file (web-ready)\n"
                                    "- 'csv': CSV file (for tabular data)\n"
                                    "- 'xlsx': Excel spreadsheet (for tabular data)"
                                ),
                                "default": "pdf",
                            },
                            "metadata": {
                                "type": "object",
                                "description": "Optional metadata (author, date, description, etc.)",
                                "properties": {
                                    "author": {"type": "string"},
                                    "subject": {"type": "string"},
                                    "keywords": {"type": "string"},
                                },
                            },
                        },
                        "required": ["content", "title"],
                    },
                    "minItems": 1,
                },
            },
            "required": ["documents"],
        },
    },
}


@traceable
def execute_web_search(args: Dict[str, Any], context: Dict[str, Any]) -> str:
    """
    Execute the web_search tool.

    Args:
        args: Arguments from LLM containing:
            - query (str): Search query
            - top_k (int, optional): Number of results to return
    """
    query = args.get("query", "")
    max_results = args.get("max_results", 5) or Config.WEB_SEARCH_MAX_RESULTS
    try:
        web_search_service = WebSearchService(
            provider=Config.WEB_SEARCH_PROVIDER,
            max_results=max_results,
            tavily_api_key=Config.TAVILY_API_KEY,
        )
        web_search_results = web_search_service.search(
            query=query, max_results=max_results
        )
        if web_search_results:
            return web_search_service.format_for_agent(web_search_results, query)
        else:
            print("[WEB_SEARCH] No results found, falling back to document contexts")
            return "No relevant web results found."
    except Exception as e:
        return f"Error during web search: {str(e)}"


@traceable
def execute_download_file(args: Dict[str, Any], context: Dict[str, Any]) -> str:
    """
    Execute the download_file tool - Downloads files from URLs to server and returns download links.

    Architecture:
    =============
    User asks "download this PDF" → LLM calls download_file → Server downloads →
    Server saves to downloads/{user_id}/{timestamp}_{filename} → Returns download URL →
    User sees clickable link in chat

    Args:
        args: Arguments from LLM containing:
            - file_urls (list[str]): List of URLs to download

        context: System context containing:
            - user_id (str): Current user ID for directory organization
            - dept_id (str): Department ID (optional, for multi-tenancy)

    Returns:
        str: Success message with download links OR error message

    Example output:
        "Downloaded 2 files:
        👉 [南京著名景点简介.pdf](/api/files/clx_abc123)
        👉 [销售报告.xlsx](/api/files/clx_def456)"

    TODO Implementation Steps:
    ==========================
    Step 1: Validate inputs
        - Check if file_urls list is not empty
        - Get user_id from context (required for directory structure)
        - Return error if user_id missing: "Error: User ID not found in context"

    Step 2: Setup downloads directory
        - Create base downloads directory: DOWNLOAD_BASE = "downloads"
        - Create user-specific subdirectory: downloads/{user_id}/
        - Use os.makedirs(path, exist_ok=True) to ensure directory exists
        - Example: downloads/user123/

    Step 3: Download each file with safety checks
        For each file_url in file_urls:
            a) Validate URL format
                - Use urllib.parse.urlparse to check scheme is http/https
                - Reject file:// or other dangerous schemes
                - Return error: "Invalid URL scheme: {url}"

            b) Check file size before downloading (HEAD request)
                - Make HEAD request: requests.head(url, timeout=10)
                - Get Content-Length header
                - MAX_DOWNLOAD_SIZE = 100 * 1024 * 1024  # 100MB
                - If size > MAX_DOWNLOAD_SIZE: skip with error message

            c) Download file content
                - Use requests.get(url, stream=True, timeout=30)
                - Check response.status_code == 200
                - IMPORTANT: Must use stream=True to enable chunked reading

            d) Extract filename and sanitize
                - Try Content-Disposition header first
                - Fallback to URL path: url.split("/")[-1]
                - Decode URL encoding: urllib.parse.unquote(filename)
                - Sanitize: Remove dangerous characters, path traversal attempts
                - Replace spaces/special chars with underscores

            e) Generate unique filename
                - Add timestamp prefix: datetime.now().strftime("%Y%m%d_%H%M%S")
                - Format: {timestamp}_{sanitized_filename}
                - Example: "20250118_120530_report.pdf"

            f) Save file to disk
                - Full path: downloads/{user_id}/{timestamped_filename}
                - MUST use chunked writing to preserve file size and save memory:
                  with open(path, "wb") as f:
                      for chunk in response.iter_content(chunk_size=8192):
                          if chunk:
                              f.write(chunk)
                - DO NOT use: f.write(response.content) - loads entire file into memory

            g) Register file in FileRegistry database
                - Call _register_file_sync() to get file_id and download_url
                - Format: /api/files/{file_id} (unified for all file types)
                - Store in results list: (original_url, download_url, filename, file_id)

    Step 4: Build response message for LLM
        - Count successful downloads
        - Format markdown links: [filename](download_url)
        - Add emoji: "👉 [filename.pdf](url)"
        - Handle errors: "⚠️ Failed to download: {url} - {error}"
        - Return formatted string that LLM will show to user

    Step 5: Error handling
        - Wrap entire function in try/except
        - Catch requests.RequestException for network errors
        - Catch IOError for file system errors
        - Return user-friendly error messages
        - Log errors with logger.error() for debugging

    Security Considerations:
    ========================
    - Validate URL schemes (only http/https)
    - Sanitize filenames (no path traversal: ../, \\)
    - Check file size limits (prevent disk fill attacks)
    - Use user-specific directories (prevent file access across users)
    - Set download timeout (prevent hanging requests)
    - Validate content types if needed (optional)

    Directory Structure:
    ====================
    downloads/
    ├── user123/
    │   ├── 20250118_120530_report.pdf
    │   ├── 20250118_120545_image.png
    │   └── ...
    └── user456/
        └── ...

    Example Usage:
    ==============
    User: "Please download this PDF: https://example.com/report.pdf"
    LLM calls: download_file(file_urls=["https://example.com/report.pdf"])
    Server downloads → Saves to downloads/user123/20250118_120530_report.pdf
    Server registers in FileRegistry → file_id: clx_abc123, download_url: /api/files/clx_abc123
    Returns: "Downloaded 1 file:\\n👉 [report.pdf](/api/files/clx_abc123) [file:clx_abc123]"
    LLM shows: "Downloaded successfully! 👉 [report.pdf](/api/files/clx_abc123)"
    """
    # TODO: Implement download_file logic following the steps above
    # For now, return placeholder
    file_list = args.get("file_urls", [])
    if not file_list:
        return "No file URLs provided"

    user_id = context.get("user_id")
    if not user_id:
        return "Error: User ID not found in context"

    os.makedirs(f"{Config.DOWNLOAD_BASE}/{user_id}/", exist_ok=True)
    results = []
    errors = []

    for file_url in file_list:
        try:
            logger.debug(f"[DOWNLOAD_FILE] File URL: {file_url}")
            parsed_url = urlparse(file_url)
            if parsed_url.scheme not in ("http", "https"):
                errors.append(f"⚠️ Invalid URL scheme: {file_url}")
                continue

            # Check file size before downloading
            try:
                request_header = requests.head(
                    file_url, timeout=10, allow_redirects=True
                )
                content_length = request_header.headers.get("Content-Length", 0)
                if Config.MAX_DOWNLOAD_SIZE_MB and content_length:
                    file_size_mb = int(content_length) / (1024 * 1024)
                    if file_size_mb > Config.MAX_DOWNLOAD_SIZE_MB:
                        errors.append(
                            f"⚠️ File too large: {file_url} ({file_size_mb:.1f}MB > {Config.MAX_DOWNLOAD_SIZE_MB}MB)"
                        )
                        continue
            except requests.RequestException:
                # HEAD request failed, try downloading anyway
                pass

            # Download file
            res = requests.get(
                file_url,
                stream=True,
                timeout=Config.DOWNLOAD_TIMEOUT,
                allow_redirects=True,
            )
            if res.status_code != 200:
                errors.append(f"⚠️ Download failed: {file_url} (HTTP {res.status_code})")
                continue

            # Extract filename
            filename = ""
            if "Content-Disposition" in res.headers:
                cd = res.headers.get("Content-Disposition")
                fname_match = re.findall('filename="?([^\'";]+)"?', cd)
                if fname_match:
                    filename = fname_match[0]

            if not filename:
                filename = os.path.basename(parsed_url.path)

            if not filename:
                filename = "downloaded_file"

            filename = urllib.parse.unquote(filename)
            filename = re.sub(r"[^\w\-_\. ]", "_", filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_filename = f"{timestamp}_{filename}"
            download_path = os.path.join(Config.DOWNLOAD_BASE, user_id, unique_filename)

            # Save file in chunks
            with open(download_path, "wb") as f:
                for chunk in res.iter_content(chunk_size=8192):
                    if chunk:  # Filter out keep-alive chunks
                        f.write(chunk)

            # Register file in FileRegistry database
            file_size = os.path.getsize(download_path)
            mime_type = res.headers.get("Content-Type", "application/octet-stream")

            result = _register_file_sync(
                user_email=user_id,
                category="downloaded",
                original_name=filename,
                storage_path=download_path,
                source_tool="download_file",
                mime_type=mime_type,
                size_bytes=file_size,
                source_url=file_url,
                conversation_id=context.get("conversation_id"),
            )

            file_id = result["file_id"]
            download_url = result[
                "download_url"
            ]  # Use unified /api/files/{file_id} URL
            # Put file_id FIRST and most prominent for attachment operations
            results.append(
                f"✅ Downloaded: {filename}\n"
                f"   File ID: {file_id}\n"
                f"   Download: [{filename}]({download_url})\n"
                f"   → Use file ID '{file_id}' for email attachments"
            )

        except requests.RequestException as e:
            errors.append(f"⚠️ Network error: {file_url} - {str(e)}")
            logger.error(f"[DOWNLOAD_FILE] Request error for {file_url}: {e}")
        except IOError as e:
            errors.append(f"⚠️ File write error: {file_url} - {str(e)}")
            logger.error(f"[DOWNLOAD_FILE] IO error for {file_url}: {e}")
        except Exception as e:
            errors.append(f"⚠️ Unexpected error: {file_url} - {str(e)}")
            logger.error(f"[DOWNLOAD_FILE] Unexpected error for {file_url}: {e}")

    # Build response message
    if not results and not errors:
        return "No files were downloaded"

    response_parts = []
    if results:
        response_parts.append(
            f"Downloaded {len(results)} file(s):\n" + "\n".join(results)
        )
    if errors:
        response_parts.append("\nErrors:\n" + "\n".join(errors))

    return "\n".join(response_parts)


@traceable
def execute_send_email(args: Dict[str, Any], context: Dict[str, Any]) -> str:
    """
    Execute the send_email tool with unified file resolution.

    Args:
        args: Arguments from LLM containing:
            - to (list[str]): Recipient email addresses
            - subject (str): Email subject
            - body (str): Email body content
            - attachments (list[str], optional): Attachment identifiers
              Supports multiple formats:
                - "file_abc123" or "clx_abc123": File ID from FileRegistry (recommended)
                - "chat:report.pdf": Category:filename format
                - "report.pdf": Filename only (searches user's files)
                - Direct path format (legacy, for backwards compat)

        context: System context containing:
            - user_id (str): Current user email
            - attachments (list[dict]): Legacy chat attachments (base64)
            - chat_attachment_ids (list[str]): New file IDs for chat attachments

    Returns:
        Success/error message string
    """
    to_addresses = args.get("to", [])
    subject = args.get("subject", "")
    body = args.get("body", "")
    if not to_addresses or not subject or not body:
        return "Missing required email fields"

    user_email = context.get("user_id")
    if not user_email:
        return "Error: User email not found in context"

    # Get attachment references from LLM args
    attachment_refs = args.get("attachments", [])
    logger.debug(f"[SEND_EMAIL] Attachment references: {attachment_refs}")

    real_attachments = []
    errors = []

    # Resolve each attachment reference using unified file system
    # All files (chat, uploaded, downloaded, created) are in FileRegistry
    for ref in attachment_refs:
        try:
            file_path = _resolve_file_sync(ref, user_email)
            real_attachments.append(file_path)
        except FileNotFoundError as e:
            errors.append(f"⚠️ File not found: {ref}")
            logger.error(f"[SEND_EMAIL] {e}")
        except Exception as e:
            errors.append(f"⚠️ Error resolving {ref}: {str(e)}")
            logger.error(f"[SEND_EMAIL] Unexpected error resolving {ref}: {e}")

    if errors:
        return "Failed to send email:\n" + "\n".join(errors)

    logger.debug(f"[SEND_EMAIL] Resolved attachments: {real_attachments}")

    # Send email with resolved file paths
    result = send_email(to_addresses, subject, body, real_attachments)

    if result.get("status", "failed") == "failed":
        logger.error(
            f"[SEND_EMAIL] Failed to send email: {result.get('error', 'Unknown error')}"
        )
        return f"Failed to send email: {result.get('error', 'Unknown error')}"

    return f"Email sent successfully to recipients: {', '.join(to_addresses)} with subject: {subject}"


def _resolve_file_sync(file_ref: str, user_email: str) -> str:
    """
    Synchronous wrapper to resolve file reference using FileManager.
    Handles event loop properly by using thread pool when needed.

    Args:
        file_ref: File reference (file_id, category:name, path, filename, or URL)
        user_email: User email for security

    Returns:
        Absolute path to file on disk

    Raises:
        FileNotFoundError: File not found or no access
    """
    import nest_asyncio

    nest_asyncio.apply()  # Allow nested event loops

    async def _do_resolve():
        return await _resolve_file_async(file_ref, user_email)

    loop = asyncio.get_event_loop()
    if loop.is_running():
        # If loop is running, create task and wait
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(lambda: asyncio.run(_do_resolve())).result()
    else:
        return asyncio.run(_do_resolve())


async def _resolve_file_async(file_ref: str, user_email: str) -> str:
    """
    Async helper to resolve file reference using FileManager.

    Args:
        file_ref: File reference (file_id, category:name, path, filename, or URL)
        user_email: User email for security

    Returns:
        Absolute path to file on disk

    Raises:
        FileNotFoundError: File not found or no access
    """
    # Handle download URLs from LLM
    # Convert /api/downloads/{user}/{filename} → direct file path
    # Convert /api/files/{file_id} → file_id
    if file_ref.startswith("/api/downloads/"):
        # Extract user and filename from /api/downloads/{user}/{filename}
        parts = file_ref.replace("/api/downloads/", "").split("/", 1)
        if len(parts) == 2:
            user_id, filename = parts
            # Construct direct file path
            import urllib.parse

            filename = urllib.parse.unquote(filename)
            file_path = os.path.join(Config.DOWNLOAD_BASE, user_id, filename)
            if os.path.exists(file_path):
                return file_path
            # Fallback: try to find in FileRegistry
            file_ref = filename
    elif file_ref.startswith("/api/files/"):
        # Extract file_id from /api/files/{file_id}
        file_ref = file_ref.replace("/api/files/", "")

    async with FileManager() as fm:
        return await fm.get_file_path(file_ref, user_email)


def _register_file_sync(
    user_email: str,
    category: str,
    original_name: str,
    storage_path: str,
    source_tool: str,
    **kwargs,
) -> str:
    """
    Synchronous wrapper to register a file using FileManager.
    Handles event loop properly by creating fresh async context.

    Args:
        user_email: User email
        category: File category
        original_name: Original filename
        storage_path: Path to file on disk
        source_tool: Tool that created the file
        **kwargs: Additional arguments (mime_type, size_bytes, etc.)

    Returns:
        file_id: Unique identifier for registered file
    """
    import nest_asyncio

    nest_asyncio.apply()  # Allow nested event loops

    async def _do_register():
        async with FileManager() as fm:
            return await fm.register_file(
                user_email=user_email,
                category=category,
                original_name=original_name,
                storage_path=storage_path,
                source_tool=source_tool,
                **kwargs,
            )

    loop = asyncio.get_event_loop()
    if loop.is_running():
        # If loop is running, create task and wait
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(lambda: asyncio.run(_do_register())).result()
    else:
        return asyncio.run(_do_register())


async def _register_file_async(
    user_email: str,
    category: str,
    original_name: str,
    storage_path: str,
    source_tool: str,
    **kwargs,
) -> str:
    """
    Async helper to register a file using FileManager.

    Args:
        user_email: User email
        category: File category
        original_name: Original filename
        storage_path: Path to file on disk
        source_tool: Tool that created the file
        **kwargs: Additional arguments (mime_type, size_bytes, etc.)

    Returns:
        file_id: Unique identifier for registered file
    """
    async with FileManager() as fm:
        return await fm.register_file(
            user_email=user_email,
            category=category,
            original_name=original_name,
            storage_path=storage_path,
            source_tool=source_tool,
            **kwargs,
        )


@traceable
def execute_create_documents(args: Dict[str, Any], context: Dict[str, Any]) -> str:
    """
    Execute the create_documents tool - supports creating multiple documents in one call.

    Args:
        args: Arguments from LLM containing:
            - documents (list): List of document objects, each containing:
                - content (str): Main document content (supports markdown)
                - title (str): Document title
                - format (str): Output format - pdf/docx/txt/md/html/csv/xlsx (default: 'pdf')
                - metadata (dict, optional): Author, subject, keywords, etc.

        context: System context containing:
            - user_id (str): User ID for file ownership

    Returns:
        Formatted string with download links for all created documents

    Implementation TODO:

    SUPPORTED FORMATS (7 total):

    1. PDF - reportlab (pip install reportlab)
       - Professional formatted documents
       - Markdown → styled PDF with fonts/sizes
       - Page numbers, headers, metadata

    2. DOCX - python-docx (pip install python-docx)
       - Microsoft Word documents
       - Markdown → Word styles (headings, lists, bold)
       - Document properties

    3. TXT - built-in
       - Plain text, strip markdown or keep it
       - open(path, "w", encoding="utf-8")

    4. MD - built-in
       - Markdown file (save content as-is)
       - open(path, "w", encoding="utf-8")

    5. HTML - markdown2 (pip install markdown2)
       - Convert markdown → HTML
       - Wrap in proper HTML structure
       - Add CSS styling

    6. CSV - built-in csv module
       - For tabular data
       - Parse content, write with csv.writer
       - Headers, rows

    7. XLSX - openpyxl (pip install openpyxl)
       - Excel spreadsheets
       - Parse content for tables
       - Format cells (bold headers, borders)

    IMPLEMENTATION:
    Loop through documents → validate → generate based on format → collect results

    LIBRARIES:
    pip install reportlab python-docx markdown2 openpyxl
    """
    documents = args.get("documents", [])
    if not documents:
        return "No documents provided for creation"

    user_id = context.get("user_id")
    if not user_id:
        return "Error: User ID not found in context"

    # Create user downloads directory
    user_dir = os.path.join(Config.DOWNLOAD_BASE, user_id)
    os.makedirs(user_dir, exist_ok=True)

    results = []
    errors = []

    for doc in documents:
        try:
            content = doc.get("content", "")
            title = doc.get("title", "untitled")
            custom_filename = doc.get("filename", "")
            format_type = doc.get("format", "pdf").lower()
            metadata = doc.get("metadata", {})

            if format_type not in ("pdf", "docx", "txt", "md", "html", "csv", "xlsx"):
                errors.append(f"⚠️ Unsupported format: {format_type} for {title}")
                continue

            # Sanitize filename
            base_filename = custom_filename if custom_filename else title
            base_filename = re.sub(r"[^\w\-_\. ]", "_", base_filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Generate file based on format
            if format_type == "txt":
                filename = f"{timestamp}_{base_filename}.txt"
                filepath = os.path.join(user_dir, filename)
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)

            elif format_type == "md":
                filename = f"{timestamp}_{base_filename}.md"
                filepath = os.path.join(user_dir, filename)
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)

            elif format_type == "html":
                filename = f"{timestamp}_{base_filename}.html"
                filepath = os.path.join(user_dir, filename)
                try:
                    import markdown2

                    html_content = markdown2.markdown(content)
                    full_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto; padding: 20px; }}
        h1, h2, h3 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    {html_content}
</body>
</html>"""
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(full_html)
                except ImportError:
                    errors.append(f"⚠️ HTML format requires markdown2 library: {title}")
                    continue

            elif format_type == "pdf":
                filename = f"{timestamp}_{base_filename}.pdf"
                filepath = os.path.join(user_dir, filename)
                try:
                    from reportlab.lib.pagesizes import letter
                    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                    from reportlab.lib.units import inch
                    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

                    doc_pdf = SimpleDocTemplate(filepath, pagesize=letter)
                    styles = getSampleStyleSheet()
                    story = []

                    # Add title
                    title_style = ParagraphStyle(
                        "CustomTitle",
                        parent=styles["Heading1"],
                        fontSize=24,
                        textColor="#333333",
                        spaceAfter=30,
                    )
                    story.append(Paragraph(title, title_style))
                    story.append(Spacer(1, 0.2 * inch))

                    # Add content (simple paragraph, markdown not fully rendered)
                    for line in content.split("\n"):
                        if line.strip():
                            story.append(Paragraph(line, styles["BodyText"]))
                            story.append(Spacer(1, 0.1 * inch))

                    doc_pdf.build(story)
                except ImportError:
                    errors.append(f"⚠️ PDF format requires reportlab library: {title}")
                    continue

            elif format_type == "docx":
                filename = f"{timestamp}_{base_filename}.docx"
                filepath = os.path.join(user_dir, filename)
                try:
                    from docx import Document

                    doc_docx = Document()

                    # Add title
                    doc_docx.add_heading(title, 0)

                    # Add content (basic paragraph formatting)
                    for line in content.split("\n"):
                        if line.strip():
                            doc_docx.add_paragraph(line)

                    # Set metadata if provided
                    core_props = doc_docx.core_properties
                    if metadata.get("author"):
                        core_props.author = metadata["author"]
                    if metadata.get("subject"):
                        core_props.subject = metadata["subject"]

                    doc_docx.save(filepath)
                except ImportError:
                    errors.append(
                        f"⚠️ DOCX format requires python-docx library: {title}"
                    )
                    continue

            elif format_type == "csv":
                filename = f"{timestamp}_{base_filename}.csv"
                filepath = os.path.join(user_dir, filename)
                import csv

                # Simple CSV: treat each line as a row, split by commas or tabs
                with open(filepath, "w", encoding="utf-8", newline="") as f:
                    writer = csv.writer(f)
                    for line in content.split("\n"):
                        if line.strip():
                            # Try tab-separated first, then comma-separated
                            if "\t" in line:
                                cells = [cell.strip() for cell in line.split("\t")]
                            else:
                                cells = [cell.strip() for cell in line.split(",")]
                            writer.writerow(cells)

            elif format_type == "xlsx":
                filename = f"{timestamp}_{base_filename}.xlsx"
                filepath = os.path.join(user_dir, filename)
                try:
                    from openpyxl import Workbook

                    wb = Workbook()
                    ws = wb.active
                    ws.title = title[:31]  # Excel sheet name limit

                    # Parse content as simple table (comma or tab-separated)
                    row_num = 1
                    for line in content.split("\n"):
                        if line.strip():
                            if "\t" in line:
                                cells = [cell.strip() for cell in line.split("\t")]
                            else:
                                cells = [cell.strip() for cell in line.split(",")]
                            for col_num, cell_value in enumerate(cells, 1):
                                ws.cell(row=row_num, column=col_num, value=cell_value)
                            row_num += 1

                    wb.save(filepath)
                except ImportError:
                    errors.append(f"⚠️ XLSX format requires openpyxl library: {title}")
                    continue

            # Register file in FileRegistry database
            file_size = os.path.getsize(filepath)
            mime_type_map = {
                "pdf": "application/pdf",
                "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "txt": "text/plain",
                "md": "text/markdown",
                "html": "text/html",
                "csv": "text/csv",
                "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            }

            result = asyncio.run(
                _register_file_async(
                    user_email=user_id,
                    category="created",
                    original_name=f"{base_filename}.{format_type}",
                    storage_path=filepath,
                    source_tool="create_documents",
                    mime_type=mime_type_map.get(
                        format_type, "application/octet-stream"
                    ),
                    size_bytes=file_size,
                    conversation_id=context.get("conversation_id"),
                    metadata={"title": title, "format": format_type},
                )
            )

            file_id = result["file_id"]
            download_url = result[
                "download_url"
            ]  # Use unified /api/files/{file_id} URL
            # Put file_id FIRST and most prominent for attachment operations
            results.append(
                f"✅ Created: {base_filename}.{format_type}\n"
                f"   File ID: {file_id}\n"
                f"   Download: [{base_filename}.{format_type}]({download_url})\n"
                f"   → Use file ID '{file_id}' for email attachments"
            )
            logger.info(
                f"[CREATE_DOCUMENTS] Created {format_type} document: {filepath} | file_id: {file_id}"
            )

        except Exception as e:
            errors.append(
                f"⚠️ Failed to create document '{doc.get('title', 'unknown')}': {str(e)}"
            )
            logger.error(f"[CREATE_DOCUMENTS] Error creating document: {e}")

    # Build response message
    if not results and not errors:
        return "No documents were created"

    response_parts = []
    if results:
        response_parts.append(
            f"Created {len(results)} document(s):\n" + "\n".join(results)
        )
    if errors:
        response_parts.append("\nErrors:\n" + "\n".join(errors))

    return "\n".join(response_parts)


# ============================================================================
# TOOL REGISTRY
# ============================================================================

TOOL_REGISTRY = {
    "search_documents": execute_search_documents,
    "web_search": execute_web_search,
    "calculator": execute_calculator,
    "download_file": execute_download_file,
    "send_email": execute_send_email,
    "create_documents": execute_create_documents,
}
"""
Maps tool names to their execution functions.
When LLM calls a tool, we lookup the function here and execute it.
"""


# ============================================================================
# ALL TOOLS LIST
# ============================================================================

ALL_TOOLS = [
    SEARCH_DOCUMENTS_SCHEMA,
    CALCULATOR_SCHEMA,
    WEB_SEARCH_SCHEMA,
    DOWNLOAD_FILE_SCHEMA,
    SEND_EMAIL_SCHEMA,
    CREATE_DOCUMENTS_SCHEMA,
]
"""
List of all tool schemas to send to OpenAI API.
This tells the LLM what tools are available.
"""


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def get_tool_executor(tool_name: str):
    """
    Get the execution function for a tool by name.

    Args:
        tool_name: Name of the tool (e.g., "search_documents")

    Returns:
        Execution function or None if tool not found

    Example:
        executor = get_tool_executor("calculator")
        result = executor({"expression": "2+2"}, {})
    """
    return TOOL_REGISTRY.get(tool_name)


@traceable
def execute_tool_call(
    tool_name: str, tool_args: Dict[str, Any], context: Dict[str, Any]
) -> str:
    """
    Execute a tool call from the LLM.

    Args:
        tool_name: Name of the tool to execute
        tool_args: Arguments from LLM (parsed from JSON)
        context: System context (collection, auth, etc.)

    Returns:
        String result from tool execution

    Raises:
        ValueError: If tool doesn't exist
    """
    # Look up tool executor
    executor = get_tool_executor(tool_name)

    if not executor:
        raise ValueError(
            f"Tool '{tool_name}' not found in registry. Available tools: {list(TOOL_REGISTRY.keys())}"
        )

    # Ensure args is a dict
    executor_args = tool_args if tool_args else {}

    try:
        # Execute the tool
        result = executor(executor_args, context)
        return result
    except Exception as e:
        # Log error and return error message
        error_msg = f"Error executing tool '{tool_name}': {str(e)}"
        print(error_msg)  # For debugging
        return error_msg
