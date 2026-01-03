"""
LangGraph agent node implementations.

Each node is created via a factory function that binds RuntimeContext.
Pattern: create_xxx_node(runtime) -> async node_function(state) -> updated_state

This allows nodes to access non-serializable objects (vector_db, openai_client)
without storing them in the checkpointable AgentState.

All nodes are ASYNC to support async tool execution and file operations.
"""

from typing import Dict, Any, Callable, Coroutine
import json
import logging
from langchain_core.messages import AIMessage
from src.services.langgraph_state import AgentState, RuntimeContext
from src.services.retrieval import retrieve, build_where
from src.services.agent_tools import execute_tool_call
from src.services.retrieval_evaluator import RetrievalEvaluator
from src.models.evaluation import (
    EvaluationCriteria,
    ReflectionMode,
    ReflectionConfig,
    EvaluationResult,
    QualityLevel,
    RecommendationAction,
)
from src.services.query_refiner import QueryRefiner
from src.utils.safety import enforce_citations
from src.utils.sanitizer import sanitize_text
from src.config.settings import Config
from src.prompts import PlanningPrompts, GenerationPrompts, ToolPrompts
from src.prompts.generation import ContextType

logger = logging.getLogger(__name__)

# ==================== HELPER: EvaluationResult <-> dict ====================


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


# ==================== HELPER: Extract Attachment Content ====================


def _describe_image_with_vision(
    file_path: str, mime_type: str, openai_client=None
) -> str:
    """
    Use Vision API to describe an image.

    Args:
        file_path: Absolute path to image file
        mime_type: MIME type (image/png, image/jpeg, etc.)
        openai_client: OpenAI client instance

    Returns:
        Text description of the image from Vision API
    """
    import base64
    import os

    if not openai_client:
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        return f"[Image file ({mime_type}) - {file_size} bytes - Vision API client not available]"

    try:
        # Read and encode image as base64
        with open(file_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        # Build multimodal message for Vision API
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Describe this image in detail. Include:\n"
                            "- What the image shows (objects, people, scenes)\n"
                            "- Any text visible in the image\n"
                            "- Charts/graphs: describe the data and trends\n"
                            "- Documents: summarize the content\n"
                            "Be concise but thorough."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_data}",
                            "detail": "auto",
                        },
                    },
                ],
            }
        ]

        # Call Vision API
        response = openai_client.chat.completions.create(
            model=Config.OPENAI_VISION_MODEL,
            messages=messages,
            max_tokens=1000,
            temperature=0.3,
        )

        description = response.choices[0].message.content.strip()
        logger.info(f"[LANGGRAPH] Vision API described image: {file_path[:50]}...")
        return f"[IMAGE DESCRIPTION]\n{description}"

    except Exception as e:
        logger.error(f"[LANGGRAPH] Vision API failed for {file_path}: {e}")
        # Fallback: return basic info about the image
        try:
            file_size = os.path.getsize(file_path)
            return f"[Image file ({mime_type}) - {file_size} bytes - Vision API error: {e}]"
        except Exception:
            return f"[Image file ({mime_type}) - Vision API unavailable]"


async def _extract_attachment_content(
    file_path: str, mime_type: str, openai_client=None
) -> str:
    """
    Extract text content from attachment file (async wrapper).

    Supports PDF, DOCX, Excel, text files, and images (via Vision API).
    Similar to agent_service._extract_file_content but async-compatible.

    Args:
        file_path: Absolute path to file on disk
        mime_type: MIME type of the file
        openai_client: Optional OpenAI client for Vision API (images)

    Returns:
        Extracted text content (truncated if too long)
    """
    import asyncio
    from functools import partial

    # Run sync extraction in thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    func = partial(_extract_file_content_sync, file_path, mime_type, openai_client)
    return await loop.run_in_executor(None, func)


def _extract_file_content_sync(
    file_path: str, mime_type: str, openai_client=None
) -> str:
    """
    Synchronous file content extraction.

    Args:
        file_path: Absolute path to file on disk
        mime_type: MIME type of the file
        openai_client: Optional OpenAI client for Vision API (images)

    Returns:
        Extracted text content
    """
    try:
        max_chars = 50000  # Limit to prevent overwhelming context

        # Image files - use Vision API
        if mime_type.startswith("image/"):
            return _describe_image_with_vision(file_path, mime_type, openai_client)

        # PDF files
        elif mime_type == "application/pdf":
            try:
                import PyPDF2

                with open(file_path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    text_parts = []
                    for page in reader.pages[:50]:  # Limit to first 50 pages
                        text_parts.append(page.extract_text())
                    content = "\n".join(text_parts)
                    return content[:max_chars] + (
                        "..." if len(content) > max_chars else ""
                    )
            except Exception as e:
                logger.error(f"[LANGGRAPH] Failed to extract PDF content: {e}")
                return f"[PDF file - text extraction failed: {e}]"

        # DOCX files
        elif (
            mime_type
            == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ):
            try:
                import docx

                doc = docx.Document(file_path)
                text_parts = [paragraph.text for paragraph in doc.paragraphs]
                content = "\n".join(text_parts)
                return content[:max_chars] + ("..." if len(content) > max_chars else "")
            except Exception as e:
                logger.error(f"[LANGGRAPH] Failed to extract DOCX content: {e}")
                return f"[DOCX file - text extraction failed]"

        # Excel files
        elif mime_type in [
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/vnd.ms-excel",
        ]:
            try:
                import openpyxl

                wb = openpyxl.load_workbook(file_path, data_only=True)
                text_parts = []
                for sheet in wb.worksheets[:5]:  # Limit to first 5 sheets
                    text_parts.append(f"Sheet: {sheet.title}")
                    for row in list(sheet.iter_rows(values_only=True))[:100]:
                        row_text = "\t".join(
                            str(cell) if cell is not None else "" for cell in row
                        )
                        if row_text.strip():
                            text_parts.append(row_text)
                content = "\n".join(text_parts)
                return content[:max_chars] + ("..." if len(content) > max_chars else "")
            except Exception as e:
                logger.error(f"[LANGGRAPH] Failed to extract Excel content: {e}")
                return f"[Excel file - text extraction failed]"

        # Text files (plain text, markdown, CSV, etc.)
        elif mime_type.startswith("text/"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    return content[:max_chars] + (
                        "..." if len(content) > max_chars else ""
                    )
            except UnicodeDecodeError:
                try:
                    with open(file_path, "r", encoding="latin-1") as f:
                        content = f.read()
                        return content[:max_chars] + (
                            "..." if len(content) > max_chars else ""
                        )
                except Exception as e:
                    logger.error(f"[LANGGRAPH] Failed to read text file: {e}")
                    return f"[Text file - encoding error]"

        # Unsupported file type
        else:
            import os

            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
            return f"[File type {mime_type} not supported - {file_size} bytes]"

    except Exception as e:
        logger.error(f"[LANGGRAPH] Unexpected error extracting file content: {e}")
        return f"[Error reading file: {str(e)}]"


# ==================== PLANNING NODE ====================


def create_plan_node(
    runtime: RuntimeContext,
) -> Callable[[AgentState], Coroutine[Any, Any, Dict[str, Any]]]:
    """
    Factory function to create plan_node with runtime context bound.

    Args:
        runtime: RuntimeContext with openai_client

    Returns:
        Async plan_node function
    """

    async def plan_node(state: AgentState) -> Dict[str, Any]:
        """
        Create execution plan before taking action.

        This is the "thinking" step - decompose complex query into steps.

        IMPORTANT: Only creates plan on first call. On subsequent calls (when looping back
        from verify_node), it returns existing plan without re-planning.

        Args:
            state: Current agent state

        Returns:
            Updated state with plan
        """
        # Check if plan already exists (subsequent call from loop)
        existing_plan = state.get("plan")
        if existing_plan:
            # Plan already exists, don't re-plan, just return state as-is
            return {
                "plan": existing_plan,
                "current_step": state.get("current_step", 0),
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": state.get("messages", [])
                + [
                    AIMessage(
                        content=f"Continuing with existing plan, step {state.get('current_step', 0) + 1}/{len(existing_plan)}"
                    )
                ],
            }

        query = state.get("query", "")

        # Build conversation context summary for reference resolution
        conversation_history = runtime.get("conversation_history", [])
        conversation_context = ""
        if conversation_history:
            # Build a concise summary of recent conversation for context
            recent_messages = conversation_history[-10:]  # Last 10 messages max
            context_parts = []
            for msg in recent_messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")[:500]  # Truncate long messages
                context_parts.append(f"{role}: {content}")
            conversation_context = "\n".join(context_parts)

        # Get available files and attachments from runtime context
        available_files = runtime.get("available_files", [])
        attachment_file_ids = runtime.get("attachment_file_ids", [])

        # Extract attachment content (similar to agent_service)
        # This ensures LangGraph can actually read uploaded file contents
        attachment_contents = ""
        file_service = runtime.get("file_service")
        user_id = runtime.get("user_id")
        dept_id = runtime.get("dept_id")
        openai_client = runtime.get("openai_client")  # For Vision API (images)
        if attachment_file_ids and file_service:
            content_parts = []
            for att in attachment_file_ids:
                try:
                    file_path = await file_service.get_file_path(
                        att.get("file_id"),
                        user_id,
                        dept_id=dept_id,
                    )
                    extracted = await _extract_attachment_content(
                        file_path, att.get("mime_type", ""), openai_client
                    )
                    if extracted:
                        content_parts.append(
                            f"\n\n--- Attached File: {att.get('filename', 'unknown')} "
                            f"(file_id: {att.get('file_id')}) ---\n{extracted}"
                        )
                except Exception as e:
                    logger.warning(f"[PLAN] Failed to extract attachment: {e}")
            if content_parts:
                attachment_contents = "\n".join(content_parts)

        # Append attachment contents to query so planner can see them
        query_with_attachments = query
        if attachment_contents:
            query_with_attachments = (
                f"{query}\n\n[ATTACHED FILE CONTENTS]{attachment_contents}"
            )

        planning_prompt = PlanningPrompts.create_plan(
            query=query_with_attachments,
            conversation_context=conversation_context,
            available_files=available_files,
            attachment_file_ids=attachment_file_ids,
        )
        try:
            client = runtime.get("openai_client")
            if not client:
                raise ValueError("OpenAI client is required for planning node.")

            response = client.chat.completions.create(
                model=Config.OPENAI_MODEL,
                messages=[{"role": "user", "content": planning_prompt}],
                temperature=Config.OPENAI_TEMPERATURE,
                response_format={"type": "json_object"},
            )

            plan_data = {}
            if response.choices and response.choices[0].message:
                plan_data = json.loads(response.choices[0].message.content)

            plans = []
            if plan_data:
                plans = plan_data.get(
                    "steps", [f"Retrieve documents for: {query}", "Generate answer"]
                )
        except Exception as e:
            plans = [
                f"Retrieve documents for the query: {query}",
                "Generate answer from retrieved documents",
            ]  # Fallback plan

        return {
            "plan": plans,
            "current_step": 0,
            "iteration_count": state.get("iteration_count", 0) + 1,
            "messages": state.get("messages", [])
            + [
                AIMessage(
                    content=f"""
                         Plan created with {len(plans)} steps:\n
                         {"\n".join(["step " + str(i+1) + ": " + p for i, p in enumerate(plans)])}
                         """
                )
            ],
        }

    return plan_node


# ==================== RETRIEVAL NODE ====================


def create_retrieve_node(
    runtime: RuntimeContext,
) -> Callable[[AgentState], Coroutine[Any, Any, Dict[str, Any]]]:
    """
    Factory function to create retrieve_node with runtime context bound.

    Args:
        runtime: RuntimeContext with vector_db, dept_id, user_id, request_data

    Returns:
        Async retrieve_node function
    """

    async def retrieve_node(state: AgentState) -> Dict[str, Any]:
        """
        Retrieve documents from ChromaDB.

        Args:
            state: Current agent state

        Returns:
            Updated state with retrieved documents
        """
        plan = state.get("plan", [])
        current_step = state.get("current_step", 0)
        is_detour = state.get("evaluation_result") is not None
        if not plan:
            return {
                "retrieved_docs": [],
                "current_step": current_step,
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": state.get("messages", [])
                + [AIMessage(content="No plan or over max steps in plan.")],
            }
        if current_step >= len(plan):
            return {
                "retrieved_docs": state.get("retrieved_docs", []),
                "current_step": current_step,
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": state.get("messages", [])
                + [AIMessage(content="Over maximum steps in plan.")],
            }
        action = plan[current_step].lower()
        if (
            "retrieve" not in action
            and "search" not in action
            and "document" not in action
            and "find" not in action
        ):
            return {
                "retrieved_docs": state.get("retrieved_docs", []),
                "current_step": current_step,
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": state.get("messages", [])
                + [AIMessage(content="Current step is not to retrieve documents.")],
            }

        # Get runtime context
        vector_db = runtime.get("vector_db")
        dept_id = runtime.get("dept_id", "")
        user_id = runtime.get("user_id", "")
        request_data = runtime.get("request_data")

        if not vector_db:
            return {
                "retrieved_docs": state.get("retrieved_docs", []),
                "current_step": current_step,
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": state.get("messages", [])
                + [AIMessage(content="No vector database available for retrieval.")],
            }
        if not dept_id or not user_id:
            return {
                "retrieved_docs": [],
                "current_step": current_step,
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": state.get("messages", [])
                + [
                    AIMessage(
                        content="Missing department or user context for retrieval."
                    )
                ],
            }

        try:
            # Extract query from current plan step
            # Format: "retrieve: search for information about X" or just the action text
            current_plan_step = plan[current_step]

            # Try to extract query after colon (e.g., "retrieve: query text")
            if ":" in current_plan_step:
                step_query = current_plan_step.split(":", 1)[1].strip()
            else:
                # Fallback: use the step text as-is, removing action keywords
                step_query = current_plan_step
                for keyword in ["retrieve", "search", "find", "document", "documents"]:
                    step_query = step_query.replace(keyword, "").strip()

            # Use refined query if available (from refinement loop), otherwise use step-specific query
            query = state.get("refined_query") or step_query or state.get("query")

            where = build_where(request_data, dept_id, user_id)
            ctx, err = retrieve(
                query=query,
                vector_db=vector_db,
                dept_id=dept_id,
                user_id=user_id,
                top_k=Config.TOP_K,
                where=where,
                use_hybrid=Config.USE_HYBRID,
                use_reranker=Config.USE_RERANKER,
            )
            if not ctx:
                return {
                    "retrieved_docs": [],
                    "current_step": current_step,
                    "iteration_count": state.get("iteration_count", 0) + 1,
                    "messages": state.get("messages", [])
                    + [AIMessage(content="No relevant documents found.")],
                }

            # Store retrieved docs PER STEP to avoid mixing contexts from different questions
            # Replace old retrieval context if exists (from refinement loop), keep only latest
            step_contexts = state.get("step_contexts", {})
            if current_step not in step_contexts:
                step_contexts[current_step] = []

            # Remove any existing retrieval context (from previous refinement attempt)
            step_contexts[current_step] = [
                ctx
                for ctx in step_contexts[current_step]
                if ctx.get("type") != "retrieval"
            ]

            # Add new retrieval context
            step_contexts[current_step].append(
                {
                    "type": "retrieval",
                    "docs": ctx,
                    "plan_step": plan[current_step] if current_step < len(plan) else "",
                }
            )

            return {
                "retrieved_docs": ctx,  # Keep for backward compatibility with reflection
                "step_contexts": step_contexts,
                "tools_used": state.get("tools_used", []) + ["search_documents"],
                "current_step": current_step,
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": state.get("messages", [])
                + [AIMessage(content=f"Retrieved {len(ctx)} documents.")],
            }
        except Exception as e:
            return {
                "retrieved_docs": [],
                "current_step": current_step,
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": state.get("messages", [])
                + [AIMessage(content="Error during document retrieval.")],
            }

    return retrieve_node


# ==================== REFLECTION NODE ====================


def create_reflect_node(
    runtime: RuntimeContext,
) -> Callable[[AgentState], Coroutine[Any, Any, Dict[str, Any]]]:
    """
    Factory function to create reflect_node with runtime context bound.

    Args:
        runtime: RuntimeContext with openai_client

    Returns:
        Async reflect_node function
    """

    async def reflect_node(state: AgentState) -> Dict[str, Any]:
        """
        Evaluate retrieval quality (self-reflection).

        Args:
            state: Current agent state

        Returns:
            Updated state with quality assessment (evaluation_result as dict)
        """
        try:
            # Use step-specific query from plan, not full original query
            plan = state.get("plan", [])
            current_step = state.get("current_step", 0)

            # Extract step-specific query from plan (current_step points directly to executing step)
            step_query = state.get("query", "")  # Default to full query
            if plan and current_step < len(plan):
                current_plan_step = plan[current_step]
                # Extract query after colon (e.g., "retrieve: The Man Called Ove" → "The Man Called Ove")
                if ":" in current_plan_step:
                    step_query = current_plan_step.split(":", 1)[1].strip()
                else:
                    # Fallback: use the step text as-is, removing action keywords
                    step_query = current_plan_step
                    for keyword in [
                        "retrieve",
                        "search",
                        "find",
                        "document",
                        "documents",
                    ]:
                        step_query = step_query.replace(keyword, "").strip()

            # Use refined query if available (from refinement loop), otherwise use step-specific query
            query = state.get("refined_query") or step_query
            retrieved_docs = state.get("retrieved_docs", [])

            # Create evaluation criteria
            evaluator_criteria = EvaluationCriteria(
                query=query,
                contexts=retrieved_docs,
                mode=ReflectionMode.BALANCED,
            )
            reflection_config = ReflectionConfig.from_settings(Config)
            openai_client = runtime.get("openai_client")

            if not openai_client:
                raise ValueError("OpenAI client is required for reflection node.")

            evaluator = RetrievalEvaluator(
                config=reflection_config,
                openai_client=openai_client,
            )
            evaluation_result = evaluator.evaluate(evaluator_criteria)

            # Convert EvaluationResult to dict for serialization
            return {
                "evaluation_result": evaluation_result_to_dict(evaluation_result),
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": state.get("messages", [])
                + [
                    AIMessage(
                        content=f"Retrieval quality: {evaluation_result.quality.value} (confidence: {evaluation_result.confidence:.2f}). Recommendation: {evaluation_result.recommendation.value}."
                    )
                ],
            }
        except Exception as e:
            # Fallback to default values (as dict)
            fallback_result = EvaluationResult(
                quality=QualityLevel.PARTIAL,
                confidence=0.5,
                coverage=0.5,
                recommendation=RecommendationAction.ANSWER,
                reasoning="Reflection failed due to error, proceeding with default assessment.",
            )
            return {
                "evaluation_result": evaluation_result_to_dict(fallback_result),
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": state.get("messages", [])
                + [
                    AIMessage(
                        content=f"Reflection failed: {str(e)}. Proceeding with default assessment."
                    )
                ],
            }

    return reflect_node


# ==================== TOOL EXECUTOR NODE ====================


def create_tool_calculator_node(
    runtime: RuntimeContext,
) -> Callable[[AgentState], Coroutine[Any, Any, Dict[str, Any]]]:
    """
    Factory function to create tool_calculator_node with runtime context bound.

    Args:
        runtime: RuntimeContext with openai_client, vector_db, dept_id, user_id

    Returns:
        Async tool_calculator_node function
    """

    async def tool_calculator_node(state: AgentState) -> Dict[str, Any]:
        """
        Execute non-retrieval tools (calculator) using LLM function calling.

        This node uses OpenAI function calling to let the LLM decide tool arguments,
        similar to agent_service.py approach.

        This node can be called in two ways:
        1. PLANNED: From route_after_planning (part of the original plan) → increments current_step
        2. DETOUR: From route_after_reflection (ad-hoc tool call) → does NOT increment current_step

        Args:
            state: Current agent state

        Returns:
            Updated state with tool results
        """
        try:
            plan = state.get("plan", [])
            current_step = state.get("current_step", 0)
            query = state.get("query", "")

            # Get OpenAI client from runtime
            client = runtime.get("openai_client")
            if not client:
                return {
                    "tools_used": state.get("tools_used", []),
                    "tool_results": state.get("tool_results", {}),
                    "current_step": current_step,  # Don't increment on error
                    "iteration_count": state.get("iteration_count", 0) + 1,
                    "error": "OpenAI client required for tool execution",
                }

            # Determine if this is a DETOUR call BEFORE building prompt
            is_detour = state.get("evaluation_result") is not None

            # Build prompt for LLM tool calling
            # For planned calls: use plan[current_step]
            # For detour calls: prefer refined_query (specific to current step), else plan[current_step]
            if plan and current_step < len(plan) and not is_detour:
                # PLANNED call - use current step
                action_step = plan[current_step]
                # Extract clean query (remove tool name prefix)
                clean_query = (
                    action_step.split(":", 1)[1].strip()
                    if ":" in action_step
                    else action_step
                )
                prompt = ToolPrompts.calculator_prompt(clean_query, is_detour=False)
            elif is_detour:
                # DETOUR call - prefer refined_query (from refine_node), else current step
                refined_query = state.get("refined_query")
                if refined_query:
                    # refined_query is specific to the step we're supplementing
                    task_query = refined_query
                elif plan and current_step < len(plan):
                    # Fallback to current step
                    task_query = plan[current_step]
                else:
                    # Final fallback
                    task_query = query
                prompt = ToolPrompts.calculator_prompt(task_query, is_detour=True)
            else:
                # True fallback - no plan, no detour
                prompt = ToolPrompts.fallback_prompt(query, "calculator")

            response = client.chat.completions.create(
                model=Config.OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                tools=TOOL_CALCULATOR,
                tool_choice="auto",
                temperature=0.1,
            )

            # Check if LLM called a tool
            if not response.choices[0].message.tool_calls:
                return {
                    "tools_used": state.get("tools_used", []),
                    "tool_results": state.get("tool_results", {}),
                    "current_step": current_step,
                    "iteration_count": state.get("iteration_count", 0) + 1,
                    "messages": state.get("messages", [])
                    + [AIMessage(content="No tool was called by the LLM.")],
                }

            # Execute the tool call (same pattern as agent_service.py)
            tool_call = response.choices[0].message.tool_calls[0]
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            # Build context for tool execution from runtime
            context = {
                "vector_db": runtime.get("vector_db"),
                "dept_id": runtime.get("dept_id"),
                "user_id": runtime.get("user_id"),
                "openai_client": client,
                "request_data": runtime.get("request_data") or {},
                "file_service": runtime.get("file_service"),
            }

            # Execute the tool (async)
            result = await execute_tool_call(tool_name, tool_args, context)

            # Update tool results
            tool_results = state.get("tool_results", {})
            tool_key = f"{tool_name}_step_{current_step}"
            if tool_key not in tool_results:
                tool_results[tool_key] = []
            tool_results[tool_key].append(
                {
                    "step": current_step,
                    "args": tool_args,
                    "result": result,
                    "query": query,
                }
            )

            # Store tool result PER STEP to avoid mixing contexts
            # Replace old calculator context if exists (from refinement loop), keep only latest
            step_contexts = state.get("step_contexts", {})
            if current_step not in step_contexts:
                step_contexts[current_step] = []

            # Remove any existing calculator context (from previous refinement attempt)
            step_contexts[current_step] = [
                ctx
                for ctx in step_contexts[current_step]
                if not (
                    ctx.get("type") == "tool" and ctx.get("tool_name") == "calculator"
                )
            ]

            # Add new calculator context
            step_contexts[current_step].append(
                {
                    "type": "tool",
                    "tool_name": tool_name,
                    "result": result,
                    "args": tool_args,
                    "plan_step": (
                        plan[current_step] if plan and current_step < len(plan) else ""
                    ),
                }
            )

            return {
                "tools_used": state.get("tools_used", []) + [tool_name],
                "tool_results": tool_results,
                "step_contexts": step_contexts,
                "current_step": current_step,
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": state.get("messages", [])
                + [
                    AIMessage(
                        content=f"Executed {tool_name} with result: {result[:200]}..."
                    )
                ],
            }

        except Exception as e:
            # On error, keep current_step constant
            return {
                "tools_used": state.get("tools_used", []),
                "tool_results": state.get("tool_results", {}),
                "current_step": current_step,
                "iteration_count": state.get("iteration_count", 0) + 1,
                "error": f"Tool execution failed: {str(e)}",
                "messages": state.get("messages", [])
                + [AIMessage(content=f"Tool execution failed: {str(e)}")],
            }

    return tool_calculator_node


def create_tool_web_search_node(
    runtime: RuntimeContext,
) -> Callable[[AgentState], Coroutine[Any, Any, Dict[str, Any]]]:
    """
    Factory function to create tool_web_search_node with runtime context bound.

    Args:
        runtime: RuntimeContext with openai_client, vector_db, dept_id, user_id

    Returns:
        Async tool_web_search_node function
    """

    async def tool_web_search_node(state: AgentState) -> Dict[str, Any]:
        """
        Execute web search tool using LLM function calling.

        This node can be called in two ways:
        1. PLANNED: From route_after_planning → increments current_step
        2. DETOUR: From route_after_reflection → does NOT increment current_step

        Args:
            state: Current agent state

        Returns:
            Updated state with tool results
        """
        try:
            plan = state.get("plan", [])
            current_step = state.get("current_step", 0)
            query = state.get("query", "")

            # Get OpenAI client from runtime
            client = runtime.get("openai_client")
            if not client:
                return {
                    "tools_used": state.get("tools_used", []),
                    "tool_results": state.get("tool_results", {}),
                    "current_step": current_step,
                    "iteration_count": state.get("iteration_count", 0) + 1,
                    "error": "OpenAI client required for tool execution",
                }

            # Determine if this is a DETOUR call BEFORE building prompt
            is_detour = state.get("evaluation_result") is not None

            # Build prompt for LLM tool calling
            if plan and current_step < len(plan) and not is_detour:
                action_step = plan[current_step]
                # Extract clean query (remove tool name prefix)
                clean_query = (
                    action_step.split(":", 1)[1].strip()
                    if ":" in action_step
                    else action_step
                )
                prompt = ToolPrompts.web_search_prompt(clean_query, is_detour=False)
            elif is_detour:
                refined_query = state.get("refined_query")
                if refined_query:
                    task_query = refined_query
                elif plan and current_step < len(plan):
                    task_query = plan[current_step]
                else:
                    task_query = query
                prompt = ToolPrompts.web_search_prompt(task_query, is_detour=True)
            else:
                prompt = ToolPrompts.fallback_prompt(query, "web_search")

            response = client.chat.completions.create(
                model=Config.OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                tools=TOOL_WEB_SEARCH,
                tool_choice="auto",
                temperature=0.1,
            )

            if not response.choices[0].message.tool_calls:
                return {
                    "tools_used": state.get("tools_used", []),
                    "tool_results": state.get("tool_results", {}),
                    "current_step": current_step,
                    "iteration_count": state.get("iteration_count", 0) + 1,
                    "messages": state.get("messages", [])
                    + [AIMessage(content="No tool was called by the LLM.")],
                }

            tool_call = response.choices[0].message.tool_calls[0]
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            # Build context for tool execution from runtime
            context = {
                "vector_db": runtime.get("vector_db"),
                "dept_id": runtime.get("dept_id"),
                "user_id": runtime.get("user_id"),
                "openai_client": client,
                "request_data": runtime.get("request_data") or {},
                "file_service": runtime.get("file_service"),
            }

            result = await execute_tool_call(tool_name, tool_args, context)

            tool_results = state.get("tool_results", {})
            tool_key = f"{tool_name}_step_{current_step}"
            if tool_key not in tool_results:
                tool_results[tool_key] = []
            tool_results[tool_key].append(
                {
                    "step": current_step,
                    "args": tool_args,
                    "result": result,
                    "query": query,
                }
            )

            # Replace old web_search context if exists (from refinement loop), keep only latest
            step_contexts = state.get("step_contexts", {})
            if current_step not in step_contexts:
                step_contexts[current_step] = []

            # Remove any existing web_search context (from previous refinement attempt)
            step_contexts[current_step] = [
                ctx
                for ctx in step_contexts[current_step]
                if not (
                    ctx.get("type") == "tool" and ctx.get("tool_name") == "web_search"
                )
            ]

            # Add new web_search context
            step_contexts[current_step].append(
                {
                    "type": "tool",
                    "tool_name": tool_name,
                    "result": result,
                    "args": tool_args,
                    "plan_step": (
                        plan[current_step] if plan and current_step < len(plan) else ""
                    ),
                }
            )

            return {
                "tools_used": state.get("tools_used", []) + [tool_name],
                "tool_results": tool_results,
                "step_contexts": step_contexts,
                "current_step": current_step,
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": state.get("messages", [])
                + [
                    AIMessage(
                        content=f"Executed {tool_name} with result: {result[:200]}..."
                    )
                ],
            }

        except Exception as e:
            return {
                "tools_used": state.get("tools_used", []),
                "tool_results": state.get("tool_results", {}),
                "current_step": current_step,
                "iteration_count": state.get("iteration_count", 0) + 1,
                "error": f"Tool execution failed: {str(e)}",
                "messages": state.get("messages", [])
                + [AIMessage(content=f"Tool execution failed: {str(e)}")],
            }

    return tool_web_search_node


# ==================== DOWNLOAD FILE NODE ====================


def create_tool_download_file_node(
    runtime: RuntimeContext,
) -> Callable[[AgentState], Coroutine[Any, Any, Dict[str, Any]]]:
    """
    Factory function to create tool_download_file_node with runtime context bound.

    Args:
        runtime: RuntimeContext with openai_client, file_service, user_id, dept_id

    Returns:
        Async tool_download_file_node function
    """

    async def tool_download_file_node(state: AgentState) -> Dict[str, Any]:
        """
        Execute download_file tool using LLM function calling.

        This node downloads files from URLs and stores them in the file registry.
        The file_id from the result can be used by subsequent tools (e.g., send_email).

        Args:
            state: Current agent state

        Returns:
            Updated state with tool results including files_created for chaining
        """
        try:
            plan = state.get("plan", [])
            current_step = state.get("current_step", 0)
            query = state.get("query", "")

            # Get OpenAI client from runtime
            client = runtime.get("openai_client")
            if not client:
                return {
                    "tools_used": state.get("tools_used", []),
                    "tool_results": state.get("tool_results", {}),
                    "current_step": current_step,
                    "iteration_count": state.get("iteration_count", 0) + 1,
                    "error": "OpenAI client required for tool execution",
                }

            # Build prompt for LLM tool calling
            if plan and current_step < len(plan):
                action_step = plan[current_step]
                # Extract URLs from plan step
                clean_query = (
                    action_step.split(":", 1)[1].strip()
                    if ":" in action_step
                    else action_step
                )
                prompt = ToolPrompts.download_file_prompt(clean_query)
            else:
                prompt = ToolPrompts.fallback_prompt(query, "download_file")

            response = client.chat.completions.create(
                model=Config.OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                tools=TOOL_DOWNLOAD_FILE,
                tool_choice="auto",
                temperature=0.1,
            )

            if not response.choices[0].message.tool_calls:
                return {
                    "tools_used": state.get("tools_used", []),
                    "tool_results": state.get("tool_results", {}),
                    "current_step": current_step,
                    "iteration_count": state.get("iteration_count", 0) + 1,
                    "messages": state.get("messages", [])
                    + [
                        AIMessage(content="No tool was called by the LLM for download.")
                    ],
                }

            tool_call = response.choices[0].message.tool_calls[0]
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            # Build context for tool execution from runtime
            context = {
                "vector_db": runtime.get("vector_db"),
                "dept_id": runtime.get("dept_id"),
                "user_id": runtime.get("user_id"),
                "openai_client": client,
                "request_data": runtime.get("request_data") or {},
                "file_service": runtime.get("file_service"),
                "conversation_id": (runtime.get("request_data") or {}).get(
                    "conversation_id"
                ),
            }

            result = await execute_tool_call(tool_name, tool_args, context)

            tool_results = state.get("tool_results", {})
            tool_key = f"{tool_name}_step_{current_step}"
            if tool_key not in tool_results:
                tool_results[tool_key] = []
            tool_results[tool_key].append(
                {
                    "step": current_step,
                    "args": tool_args,
                    "result": result,
                    "query": query,
                }
            )

            # Extract file_ids from result for chaining to subsequent tools
            # Format from execute_download_file: "File ID: {file_id}\n..."
            files_created = []
            for line in result.split("\n"):
                if "File ID:" in line:
                    file_id = line.split("File ID:")[1].strip()
                    files_created.append(
                        {"file_id": file_id, "source": "download_file"}
                    )

            # Store tool result with files_created for chaining
            step_contexts = state.get("step_contexts", {})
            if current_step not in step_contexts:
                step_contexts[current_step] = []

            # Remove any existing download_file context
            step_contexts[current_step] = [
                ctx
                for ctx in step_contexts[current_step]
                if not (
                    ctx.get("type") == "tool"
                    and ctx.get("tool_name") == "download_file"
                )
            ]

            # Add new download_file context with files_created for chaining
            step_contexts[current_step].append(
                {
                    "type": "tool",
                    "tool_name": tool_name,
                    "result": result,
                    "args": tool_args,
                    "files_created": files_created,  # For chaining to send_email
                    "plan_step": (
                        plan[current_step] if plan and current_step < len(plan) else ""
                    ),
                }
            )

            return {
                "tools_used": state.get("tools_used", []) + [tool_name],
                "tool_results": tool_results,
                "step_contexts": step_contexts,
                "draft_answer": result,  # Set draft_answer for verify_node
                "current_step": current_step,
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": state.get("messages", [])
                + [
                    AIMessage(
                        content=f"Downloaded files: {len(files_created)} file(s) created"
                    )
                ],
            }

        except Exception as e:
            return {
                "tools_used": state.get("tools_used", []),
                "tool_results": state.get("tool_results", {}),
                "draft_answer": f"Download failed: {str(e)}",
                "current_step": current_step,
                "iteration_count": state.get("iteration_count", 0) + 1,
                "error": f"Download file failed: {str(e)}",
                "messages": state.get("messages", [])
                + [AIMessage(content=f"Download file failed: {str(e)}")],
            }

    return tool_download_file_node


# ==================== CREATE DOCUMENTS NODE ====================


def create_tool_create_documents_node(
    runtime: RuntimeContext,
) -> Callable[[AgentState], Coroutine[Any, Any, Dict[str, Any]]]:
    """
    Factory function to create tool_create_documents_node with runtime context bound.

    Args:
        runtime: RuntimeContext with openai_client, file_service, user_id

    Returns:
        Async tool_create_documents_node function
    """

    async def tool_create_documents_node(state: AgentState) -> Dict[str, Any]:
        """
        Execute create_documents tool using LLM function calling.

        This node creates documents (PDF, DOCX, TXT, CSV, XLSX, HTML, MD) from content.
        The file_id from the result can be used by subsequent tools (e.g., send_email).

        For multi-step queries, this node can access previous step_answers to include
        retrieved content in the created document.

        Args:
            state: Current agent state

        Returns:
            Updated state with tool results including files_created for chaining
        """
        try:
            plan = state.get("plan", [])
            current_step = state.get("current_step", 0)
            query = state.get("query", "")

            # Get OpenAI client from runtime
            client = runtime.get("openai_client")
            if not client:
                return {
                    "tools_used": state.get("tools_used", []),
                    "tool_results": state.get("tool_results", {}),
                    "current_step": current_step,
                    "iteration_count": state.get("iteration_count", 0) + 1,
                    "error": "OpenAI client required for tool execution",
                }

            # Gather context from previous steps for document content
            step_answers = state.get("step_answers", [])
            step_contexts = state.get("step_contexts", {})

            # Build content context from previous step answers
            previous_content = ""
            if step_answers:
                content_parts = []
                for ans in step_answers:
                    content_parts.append(
                        f"## {ans.get('question', 'Step Result')}\n{ans.get('answer', '')}"
                    )
                previous_content = "\n\n".join(content_parts)

            # Build prompt for LLM tool calling
            if plan and current_step < len(plan):
                action_step = plan[current_step]
                clean_query = (
                    action_step.split(":", 1)[1].strip()
                    if ":" in action_step
                    else action_step
                )
                # Include previous step content in prompt for document creation
                prompt = ToolPrompts.create_documents_prompt(
                    clean_query, previous_content=previous_content
                )
            else:
                prompt = ToolPrompts.fallback_prompt(query, "create_documents")

            response = client.chat.completions.create(
                model=Config.OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                tools=TOOL_CREATE_DOCUMENTS,
                tool_choice="auto",
                temperature=0.1,
            )

            if not response.choices[0].message.tool_calls:
                return {
                    "tools_used": state.get("tools_used", []),
                    "tool_results": state.get("tool_results", {}),
                    "current_step": current_step,
                    "iteration_count": state.get("iteration_count", 0) + 1,
                    "messages": state.get("messages", [])
                    + [
                        AIMessage(
                            content="No tool was called by the LLM for document creation."
                        )
                    ],
                }

            tool_call = response.choices[0].message.tool_calls[0]
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            # Build context for tool execution from runtime
            context = {
                "vector_db": runtime.get("vector_db"),
                "dept_id": runtime.get("dept_id"),
                "user_id": runtime.get("user_id"),
                "openai_client": client,
                "request_data": runtime.get("request_data") or {},
                "file_service": runtime.get("file_service"),
                "conversation_id": (runtime.get("request_data") or {}).get(
                    "conversation_id"
                ),
            }

            result = await execute_tool_call(tool_name, tool_args, context)

            tool_results = state.get("tool_results", {})
            tool_key = f"{tool_name}_step_{current_step}"
            if tool_key not in tool_results:
                tool_results[tool_key] = []
            tool_results[tool_key].append(
                {
                    "step": current_step,
                    "args": tool_args,
                    "result": result,
                    "query": query,
                }
            )

            # Extract file_ids from result for chaining to subsequent tools
            # Format from execute_create_documents: "File ID: {file_id}\n..."
            files_created = []
            for line in result.split("\n"):
                if "File ID:" in line:
                    file_id = line.split("File ID:")[1].strip()
                    files_created.append(
                        {"file_id": file_id, "source": "create_documents"}
                    )

            # Store tool result with files_created for chaining
            if current_step not in step_contexts:
                step_contexts[current_step] = []

            # Remove any existing create_documents context
            step_contexts[current_step] = [
                ctx
                for ctx in step_contexts[current_step]
                if not (
                    ctx.get("type") == "tool"
                    and ctx.get("tool_name") == "create_documents"
                )
            ]

            # Add new create_documents context with files_created for chaining
            step_contexts[current_step].append(
                {
                    "type": "tool",
                    "tool_name": tool_name,
                    "result": result,
                    "args": tool_args,
                    "files_created": files_created,  # For chaining to send_email
                    "plan_step": (
                        plan[current_step] if plan and current_step < len(plan) else ""
                    ),
                }
            )

            return {
                "tools_used": state.get("tools_used", []) + [tool_name],
                "tool_results": tool_results,
                "step_contexts": step_contexts,
                "draft_answer": result,  # Set draft_answer for verify_node
                "current_step": current_step,
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": state.get("messages", [])
                + [
                    AIMessage(
                        content=f"Created documents: {len(files_created)} file(s)"
                    )
                ],
            }

        except Exception as e:
            return {
                "tools_used": state.get("tools_used", []),
                "tool_results": state.get("tool_results", {}),
                "draft_answer": f"Document creation failed: {str(e)}",
                "current_step": current_step,
                "iteration_count": state.get("iteration_count", 0) + 1,
                "error": f"Create documents failed: {str(e)}",
                "messages": state.get("messages", [])
                + [AIMessage(content=f"Create documents failed: {str(e)}")],
            }

    return tool_create_documents_node


# ==================== SEND EMAIL NODE ====================


def create_tool_send_email_node(
    runtime: RuntimeContext,
) -> Callable[[AgentState], Coroutine[Any, Any, Dict[str, Any]]]:
    """
    Factory function to create tool_send_email_node with runtime context bound.

    Args:
        runtime: RuntimeContext with openai_client, file_service, user_id

    Returns:
        Async tool_send_email_node function
    """

    async def tool_send_email_node(state: AgentState) -> Dict[str, Any]:
        """
        Execute send_email tool using LLM function calling.

        This node sends emails with optional attachments. It can access files_created
        from previous steps (download_file, create_documents) to attach them.

        Key feature: Extracts file_ids from previous step_contexts for attachments.

        Args:
            state: Current agent state

        Returns:
            Updated state with tool results
        """
        try:
            plan = state.get("plan", [])
            current_step = state.get("current_step", 0)
            query = state.get("query", "")

            # Get OpenAI client from runtime
            client = runtime.get("openai_client")
            if not client:
                return {
                    "tools_used": state.get("tools_used", []),
                    "tool_results": state.get("tool_results", {}),
                    "current_step": current_step,
                    "iteration_count": state.get("iteration_count", 0) + 1,
                    "error": "OpenAI client required for tool execution",
                }

            # Collect all files_created from previous steps for potential attachments
            step_contexts = state.get("step_contexts", {})
            available_file_ids = []
            for step_num, contexts in step_contexts.items():
                if step_num < current_step:  # Only look at previous steps
                    for ctx in contexts:
                        files_created = ctx.get("files_created", [])
                        for f in files_created:
                            available_file_ids.append(f["file_id"])

            # Also include files from available_files in runtime (user's existing files)
            available_files = runtime.get("available_files", [])
            available_file_info = []
            for f in available_files:
                available_file_info.append(
                    {
                        "file_id": f.get("file_id"),
                        "name": f.get("original_name"),
                        "category": f.get("category"),
                    }
                )

            # Build prompt for LLM tool calling with file context
            if plan and current_step < len(plan):
                action_step = plan[current_step]
                clean_query = (
                    action_step.split(":", 1)[1].strip()
                    if ":" in action_step
                    else action_step
                )
                prompt = ToolPrompts.send_email_prompt(
                    clean_query,
                    available_file_ids=available_file_ids,
                    available_files=available_file_info,
                )
            else:
                prompt = ToolPrompts.fallback_prompt(query, "send_email")

            response = client.chat.completions.create(
                model=Config.OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                tools=TOOL_SEND_EMAIL,
                tool_choice="auto",
                temperature=0.1,
            )

            if not response.choices[0].message.tool_calls:
                return {
                    "tools_used": state.get("tools_used", []),
                    "tool_results": state.get("tool_results", {}),
                    "current_step": current_step,
                    "iteration_count": state.get("iteration_count", 0) + 1,
                    "messages": state.get("messages", [])
                    + [
                        AIMessage(
                            content="No tool was called by the LLM for send email."
                        )
                    ],
                }

            tool_call = response.choices[0].message.tool_calls[0]
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            # Build context for tool execution from runtime
            context = {
                "vector_db": runtime.get("vector_db"),
                "dept_id": runtime.get("dept_id"),
                "user_id": runtime.get("user_id"),
                "openai_client": client,
                "request_data": runtime.get("request_data") or {},
                "file_service": runtime.get("file_service"),
            }

            result = await execute_tool_call(tool_name, tool_args, context)

            tool_results = state.get("tool_results", {})
            tool_key = f"{tool_name}_step_{current_step}"
            if tool_key not in tool_results:
                tool_results[tool_key] = []
            tool_results[tool_key].append(
                {
                    "step": current_step,
                    "args": tool_args,
                    "result": result,
                    "query": query,
                }
            )

            # Store tool result in step_contexts
            if current_step not in step_contexts:
                step_contexts[current_step] = []

            # Remove any existing send_email context
            step_contexts[current_step] = [
                ctx
                for ctx in step_contexts[current_step]
                if not (
                    ctx.get("type") == "tool" and ctx.get("tool_name") == "send_email"
                )
            ]

            # Add new send_email context
            step_contexts[current_step].append(
                {
                    "type": "tool",
                    "tool_name": tool_name,
                    "result": result,
                    "args": tool_args,
                    "plan_step": (
                        plan[current_step] if plan and current_step < len(plan) else ""
                    ),
                }
            )

            return {
                "tools_used": state.get("tools_used", []) + [tool_name],
                "tool_results": tool_results,
                "step_contexts": step_contexts,
                "draft_answer": result,  # Set draft_answer for verify_node
                "current_step": current_step,
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": state.get("messages", [])
                + [AIMessage(content=f"Email sent: {result[:100]}...")],
            }

        except Exception as e:
            return {
                "tools_used": state.get("tools_used", []),
                "tool_results": state.get("tool_results", {}),
                "draft_answer": f"Email sending failed: {str(e)}",
                "current_step": current_step,
                "iteration_count": state.get("iteration_count", 0) + 1,
                "error": f"Send email failed: {str(e)}",
                "messages": state.get("messages", [])
                + [AIMessage(content=f"Send email failed: {str(e)}")],
            }

    return tool_send_email_node


# ==================== DIRECT ANSWER NODE ====================
def create_direct_answer_node(
    runtime: RuntimeContext,
) -> Callable[[AgentState], Coroutine[Any, Any, Dict[str, Any]]]:
    """
    Factory function to create direct_answer_node with runtime context bound.
    """

    async def direct_answer_node(state: AgentState) -> Dict[str, Any]:
        # - Get OpenAI client from runtime
        # - Extract question from plan[current_step]
        # - Call LLM with simple prompt (no tools, no retrieval)
        # - Store result in step_contexts with type="direct_answer"
        # - Return updated state
        openai_client = runtime.get("openai_client", None)
        if not openai_client:
            raise ValueError("OpenAI client is required for direct answer node.")

        plan = state.get("plan", [])
        current_step = state.get("current_step", 0)
        if not plan or current_step >= len(plan):
            return {
                "direct_answer": "No valid plan step for direct answer.",
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": state.get("messages", [])
                + [AIMessage(content="No valid plan step for direct answer.")],
            }
        action_step = plan[current_step]
        step_query = (
            action_step.split(":", 1)[1].strip() if ":" in action_step else action_step
        )

        try:
            # Build prompts for LLM
            system_prompt = GenerationPrompts.get_system_prompt(
                ContextType.DIRECT_ANSWER
            )

            # Add available files context for "give me the link" type queries
            available_files = runtime.get("available_files", [])
            files_context = ""
            if available_files:
                file_lines = []
                for f in available_files:
                    file_lines.append(
                        f"- [{f.get('original_name')}]({f.get('download_url')}) "
                        f"(category: {f.get('category')})"
                    )
                files_context = (
                    "\n\nAVAILABLE FILES (use these download links if user asks):\n"
                    + "\n".join(file_lines)
                )
                system_prompt += files_context

            openai_messages = [{"role": "system", "content": system_prompt}]
            conversation_history = runtime.get("conversation_history", [])
            if conversation_history:
                for h in conversation_history:
                    sanitized_msg = {
                        "role": h.get("role", "user"),
                        "content": sanitize_text(
                            h.get("content", ""),
                            max_length=Config.ONE_HISTORY_MAX_TOKENS,
                        ),
                    }
                    openai_messages.append(sanitized_msg)
            user_message = GenerationPrompts.build_user_message(step_query)
            openai_messages.append({"role": "user", "content": user_message})

            response = openai_client.chat.completions.create(
                model=Config.OPENAI_MODEL,
                messages=openai_messages,
                max_completion_tokens=Config.CHAT_MAX_TOKENS,
                temperature=Config.OPENAI_TEMPERATURE,
            )
            direct_answer = ""
            if response.choices and response.choices[0].message:
                direct_answer = response.choices[0].message.content

            # Store direct answer in step_contexts
            # Replace if exists (unlikely for direct_answer, but keep consistent)
            step_contexts = state.get("step_contexts", {})
            if current_step not in step_contexts:
                step_contexts[current_step] = []

            # Remove any existing direct_answer context
            step_contexts[current_step] = [
                ctx
                for ctx in step_contexts[current_step]
                if ctx.get("type") != "direct_answer"
            ]

            # Add new direct_answer context
            step_contexts[current_step].append(
                {
                    "type": "direct_answer",
                    "answer": direct_answer,
                    "plan_step": action_step,
                }
            )

            return {
                "draft_answer": direct_answer,
                "current_step": current_step,
                "step_contexts": step_contexts,
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": state.get("messages", [])
                + [AIMessage(content="Generated answer from direct_answer directly.")],
            }
        except Exception as e:
            return {
                "draft_answer": "",
                "current_step": current_step,
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": state.get("messages", [])
                + [AIMessage(content="Error during direct answer generation.")],
            }

    return direct_answer_node


# ==================== REFINEMENT NODE ====================
def create_refine_node(
    runtime: RuntimeContext,
) -> Callable[[AgentState], Coroutine[Any, Any, Dict[str, Any]]]:
    """
    Factory function to create refine_node with runtime context bound.

    Args:
        runtime: RuntimeContext with openai_client

    Returns:
        Async refine_node function
    """

    async def refine_node(state: AgentState) -> Dict[str, Any]:
        """
        Refine query based on reflection feedback.

        Args:
            state: Current agent state

        Returns:
            Updated state with refined query
        """
        # Use step-specific query from plan, not full original query
        plan = state.get("plan", [])
        current_step = state.get("current_step", 0)

        # Extract step-specific query from plan (current_step points directly to executing step)
        step_query = state.get("query", "")  # Default to full query
        if plan and current_step < len(plan):
            current_plan_step = plan[current_step]
            # Extract query after colon (e.g., "retrieve: The Man Called Ove" → "The Man Called Ove")
            if ":" in current_plan_step:
                step_query = current_plan_step.split(":", 1)[1].strip()
            else:
                # Fallback: use the step text as-is, removing action keywords
                step_query = current_plan_step
                for keyword in ["retrieve", "search", "find", "document", "documents"]:
                    step_query = step_query.replace(keyword, "").strip()

        # Use existing refined query if available, otherwise use step-specific query
        current_query = state.get("refined_query") or step_query
        try:
            openai_client = runtime.get("openai_client")
            if not openai_client:
                raise ValueError("OpenAI client is required for refinement node.")

            refiner = QueryRefiner(
                openai_client=openai_client,
                model=Config.OPENAI_MODEL,
                temperature=Config.OPENAI_TEMPERATURE,
            )
            # Convert dict back to EvaluationResult for refiner
            evaluation_result_dict = state.get("evaluation_result")
            if not evaluation_result_dict:
                raise ValueError("Evaluation result is required for query refinement.")

            evaluation_result = dict_to_evaluation_result(evaluation_result_dict)

            refined_query = refiner.refine_query(
                original_query=current_query,
                eval_result=evaluation_result,
            )

            current_refinement_count = state.get("refinement_count", 0)
            return {
                "refined_query": refined_query,
                "refinement_count": current_refinement_count + 1,
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": state.get("messages", [])
                + [AIMessage(content=f"Refined query to: {refined_query}")],
            }
        except Exception as e:
            return {
                "refined_query": current_query,
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": state.get("messages", [])
                + [AIMessage(content="Error during query refinement.")],
            }

    return refine_node


# ==================== GENERATION NODE ====================


def create_generate_node(
    runtime: RuntimeContext,
) -> Callable[[AgentState], Coroutine[Any, Any, Dict[str, Any]]]:
    """
    Factory function to create generate_node with runtime context.

    Args:
        runtime: Runtime context with non-serializable objects

    Returns:
        Async generate_node function with runtime bound via closure
    """

    async def generate_node(state: AgentState) -> Dict[str, Any]:
        """
        Generate answer from retrieved documents.

        Args:
            state: Current agent state

        Returns:
            Updated state with generated answer
        """
        try:
            openai_client = runtime.get("openai_client", None)
            if not openai_client:
                raise ValueError("OpenAI client is required for generation node.")

            # Check if this is a CLARIFY recommendation from reflection
            evaluation_result_dict = state.get("evaluation_result")
            evaluation_result = dict_to_evaluation_result(evaluation_result_dict)
            if (
                evaluation_result
                and evaluation_result.recommendation == RecommendationAction.CLARIFY
            ):
                # Generate clear clarification request message using prompt registry
                clarification_message = GenerationPrompts.clarification_message(
                    evaluation_result.reasoning
                )
                return {
                    "draft_answer": clarification_message,
                    "iteration_count": state.get("iteration_count", 0) + 1,
                    "messages": state.get("messages", [])
                    + [AIMessage(content=clarification_message)],
                }

            # Get ONLY current step's context (per-step isolation)
            current_step = state.get("current_step", 0)
            step_contexts = state.get("step_contexts", {})

            # Get context for the current step being executed (now a list)
            if current_step not in step_contexts or not step_contexts[current_step]:
                return {
                    "draft_answer": "",
                    "iteration_count": state.get("iteration_count", 0) + 1,
                    "messages": state.get("messages", [])
                    + [
                        AIMessage(
                            content="No context available to generate answer from."
                        )
                    ],
                }

            step_ctx_list = step_contexts[current_step]  # Now a list of contexts
            # Get plan_step from first context (all should have same plan_step)
            plan_step = step_ctx_list[0].get("plan_step", "")

            # Detect if ANY context is a web search result
            is_web_search = any(
                ctx.get("type") == "tool" and ctx.get("tool_name") == "web_search"
                for ctx in step_ctx_list
            )

            # Build numbered context from ALL contexts in this step (handles retrieve + web_search + calculator)
            contexts = []
            context_num = 1

            # Iterate through all contexts for this step
            for step_ctx in step_ctx_list:
                if step_ctx["type"] == "retrieval":
                    # Retrieved documents from this step only
                    docs = step_ctx.get("docs", [])
                    for doc in docs:
                        chunk = doc.get("chunk", str(doc))
                        source = doc.get("source", "unknown")
                        page = doc.get("page", 0)

                        header = f"Context {context_num} (Source: {source}"
                        if page > 0:
                            header += f", Page: {page}"
                        header += "):\n"

                        score_info = ""
                        if doc.get("hybrid") is not None:
                            score_info += f"Hybrid score: {doc['hybrid']:.2f}"
                        if doc.get("rerank") is not None:
                            if score_info:
                                score_info += ", "
                            score_info += f"Rerank score: {doc['rerank']:.2f}"

                        context_entry = f"{header}{chunk}"
                        if score_info:
                            context_entry += f"\n{score_info}"

                        contexts.append(context_entry)
                        context_num += 1

                elif step_ctx["type"] == "tool":
                    # Tool result from this step only
                    tool_name = step_ctx.get("tool_name", "unknown")
                    result_text = step_ctx.get("result", "")
                    args = step_ctx.get("args", {})

                    header = f"Context {context_num} (Tool: {tool_name}, Step: {current_step}):\n"
                    if args:
                        header += f"Arguments: {args}\n"

                    context_entry = f"{header}Result: {result_text}"
                    contexts.append(context_entry)
                    context_num += 1

            if not contexts:
                logger.warning(
                    f"[GENERATE_NODE] No contexts found for step {current_step}. "
                    f"step_ctx_list={step_ctx_list}"
                )
                raise ValueError(
                    "No context available from retrieved documents or tool results."
                )

            final_context = "\n\n".join(contexts)

            # Build system prompt - different rules for web search vs document retrieval
            # Use prompt registry for context-aware prompts
            if is_web_search:
                context_type = ContextType.WEB_SEARCH
            else:
                context_type = ContextType.DOCUMENT

            system_prompt = GenerationPrompts.get_system_prompt(context_type)

            # Add source file download links for retrieved documents
            # Match retrieved doc sources with available_files to provide links
            available_files = runtime.get("available_files", [])
            if available_files and not is_web_search:
                # Build file_id to download_url mapping
                file_map = {}
                for f in available_files:
                    file_map[f.get("file_id")] = {
                        "name": f.get("original_name"),
                        "url": f.get("download_url"),
                    }

                # Collect unique source files from retrieved docs
                source_files = set()
                for step_ctx in step_ctx_list:
                    if step_ctx["type"] == "retrieval":
                        for doc in step_ctx.get("docs", []):
                            file_id = doc.get("file_id")
                            if file_id and file_id in file_map:
                                source_files.add(file_id)

                # Add source file links to system prompt
                if source_files:
                    source_links = []
                    for fid in source_files:
                        info = file_map[fid]
                        source_links.append(f"- [{info['name']}]({info['url']})")
                    system_prompt += (
                        "\n\nSOURCE FILE DOWNLOADS (include these links in your answer):\n"
                        + "\n".join(source_links)
                    )

            # Use the SPECIFIC plan step as the question (not the full multi-part query)
            # This ensures the LLM answers ONLY what this step is about
            refined_query = state.get("refined_query", None)

            # Extract the task from plan step (format: "tool_name: description")
            step_question = plan_step
            if ":" in plan_step:
                step_question = plan_step.split(":", 1)[1].strip()

            # Build user message using prompt registry
            user_message_with_context = GenerationPrompts.build_user_message(
                question=step_question,
                context=final_context,
                refined_query=refined_query,
            )
            # Comment out too much citation prompt
            # Include bracket citations [n] for every sentence that uses information.
            # At the end of your answer, cite the sources you used. For each source file, list the specific page numbers
            # from the contexts you referenced. Format: 'Sources: filename.pdf (pages 15, 23), filename2.pdf (page 7)'

            # Build messages list: system + conversation_history + current query with contexts
            openai_messages = [{"role": "system", "content": system_prompt}]

            # Use pre-loaded conversation history from runtime (loaded in chat.py, avoids async issues)
            conversation_history = runtime.get("conversation_history", [])
            if conversation_history:
                for h in conversation_history:
                    sanitized_msg = {
                        "role": h.get("role", "user"),
                        "content": sanitize_text(
                            h.get("content", ""),
                            max_length=Config.ONE_HISTORY_MAX_TOKENS,
                        ),
                    }
                    openai_messages.append(sanitized_msg)

            # Add current query with contexts
            openai_messages.append(
                {"role": "user", "content": user_message_with_context}
            )

            response = openai_client.chat.completions.create(
                model=Config.OPENAI_MODEL,
                messages=openai_messages,
                max_completion_tokens=Config.CHAT_MAX_TOKENS,
                temperature=Config.OPENAI_TEMPERATURE,
            )
            draft_answer = ""
            if response.choices and response.choices[0].message:
                draft_answer = response.choices[0].message.content

            # Log if answer seems too short (potential issue)
            if draft_answer and len(draft_answer) < 10:
                logger.warning(
                    f"[GENERATE_NODE] Very short answer generated ({len(draft_answer)} chars): "
                    f"'{draft_answer[:100]}'. Context length was {len(final_context)} chars."
                )

            return {
                "draft_answer": draft_answer,
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": state.get("messages", [])
                + [AIMessage(content="Generated answer successfully.")],
            }

        except Exception as e:
            return {
                "draft_answer": "",
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": state.get("messages", [])
                + [AIMessage(content="Error during answer generation.")],
            }

    return generate_node


# ==================== VERIFICATION NODE ====================


def create_verify_node(
    runtime: RuntimeContext,
) -> Callable[[AgentState], Coroutine[Any, Any, Dict[str, Any]]]:
    """
    Factory function to create verify_node with runtime context.

    Args:
        runtime: Runtime context with non-serializable objects

    Returns:
        Async verify_node function with runtime bound via closure
    """

    async def verify_node(state: AgentState) -> Dict[str, Any]:
        """
        Verify citations and route to next step or finalize answer.

        This node handles two scenarios:
        1. INTERMEDIATE: More plan steps remain → verify and continue to next step
        2. FINAL: All plan steps complete → verify and create final_answer

        The key insight from Plan-Execute pattern:
        - Data gathering phase: Execute plan steps, accumulate results
        - Generation phase: ONE final answer at end synthesizing all data

        Args:
            state: Current agent state

        Returns:
            Updated state with verified answer
        """
        draft_answer = state.get("draft_answer", "")
        plan = state.get("plan", [])
        current_step = state.get("current_step", 0)

        # Check if there are more plan steps remaining after this one
        has_more_steps = plan and current_step + 1 < len(plan)

        if not draft_answer:
            return {
                "final_answer": (
                    "" if not has_more_steps else None
                ),  # Only set final_answer at end
                "draft_answer": "",
                "current_step": current_step + 1,
                "evaluation_result": None,  # Clear evaluation_result for next cycle
                "refined_query": None,  # Clear refined_query for next step
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": state.get("messages", [])
                + [AIMessage(content="No draft answer to verify.")],
            }

        # Check if this is a CLARIFY recommendation - pass through without verification
        evaluation_result_dict = state.get("evaluation_result")
        evaluation_result = dict_to_evaluation_result(evaluation_result_dict)
        if (
            evaluation_result
            and evaluation_result.recommendation == RecommendationAction.CLARIFY
        ):
            # Clarification messages don't need citation enforcement
            # CLARIFY always ends the flow (user needs to provide input)
            # Set current_step to end of plan to ensure should_continue() returns "end"
            return {
                "final_answer": draft_answer,  # CLARIFY always creates final_answer (should_continue checks this)
                "current_step": len(plan),  # Explicitly signal end of execution
                "draft_answer": "",
                "evaluation_result": None,  # Clear evaluation_result for next cycle
                "refined_query": None,  # Clear refined_query for next step
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": state.get("messages", [])
                + [AIMessage(content="Clarification request prepared.")],
            }

        try:
            # Calculate valid context IDs from both sources
            retrieved_docs = state.get("retrieved_docs", [])
            tool_results = state.get("tool_results", {})

            step_contexts = state.get("step_contexts", {})
            step_ctx_list = step_contexts.get(current_step, [])
            if not step_ctx_list:
                raise ValueError(f"No context found for step {current_step}")

            # Get types from all contexts (can have multiple: retrieve + web_search)
            step_types = [ctx.get("type", "unknown") for ctx in step_ctx_list]

            # Also get tool names for action tools
            tool_names = [
                ctx.get("tool_name", "")
                for ctx in step_ctx_list
                if ctx.get("type") == "tool"
            ]

            # Action tools that skip citation enforcement (direct results, not RAG)
            action_tools = {"download_file", "create_documents", "send_email"}
            is_action_tool = bool(set(tool_names) & action_tools)

            clean_answer = ""

            # Direct answer and action tool steps skip citation enforcement
            valid_ids = []  # Initialize for all cases
            if "direct_answer" in step_types or is_action_tool:
                clean_answer = draft_answer
            else:
                context_num = 1

                # Add IDs for retrieved documents
                for _ in retrieved_docs:
                    valid_ids.append(context_num)
                    context_num += 1

                # Add IDs for tool results
                for _, results in tool_results.items():
                    for _ in results:
                        valid_ids.append(context_num)
                        context_num += 1

                if not valid_ids:
                    error_message = "I apologize, but I couldn't find any relevant information to answer your question. Please try rephrasing your query or providing more details."
                    return {
                        "final_answer": error_message,  # Always set final_answer to END
                        "current_step": current_step + 1,
                        "draft_answer": "",
                        "evaluation_result": None,  # Clear evaluation_result for next cycle
                        "refined_query": None,  # Clear refined_query for next step
                        "iteration_count": state.get("iteration_count", 0) + 1,
                        "messages": state.get("messages", [])
                        + [
                            AIMessage(
                                content="Warning: No contexts to verify citations against."
                            )
                        ],
                    }

                # Optional: Enforce citations - drops sentences without valid citations
                clean_answer, _ = (
                    enforce_citations(draft_answer, valid_ids)
                    if Config.ENFORCE_CITATIONS
                    else (draft_answer, True)
                )

            # Store answer for THIS STEP
            # Get plan_step from first context (all should have same plan_step)
            plan_step_desc = step_ctx_list[0].get("plan_step", f"Step {current_step}")

            # Extract clean question (remove tool name prefix like "retrieve:", "web_search:")
            clean_question = plan_step_desc
            if ":" in plan_step_desc:
                clean_question = plan_step_desc.split(":", 1)[1].strip()

            step_answers = state.get("step_answers", [])
            step_answers.append(
                {
                    "step": current_step,
                    "question": clean_question,
                    "answer": clean_answer,
                }
            )

            # If all steps complete, concatenate all step answers
            if not has_more_steps:
                # Build final answer from all step answers
                if len(step_answers) == 1:
                    # Single step - return answer directly
                    final_answer = step_answers[0]["answer"]
                else:
                    # Multiple steps - format with sections
                    answer_parts = []
                    for step_ans in step_answers:
                        # Extract task description (remove tool name prefix)
                        task = step_ans["question"]
                        if ":" in task:
                            task = task.split(":", 1)[1].strip()

                        answer_parts.append(
                            f"**{task.capitalize()}**\n{step_ans['answer']}"
                        )

                    final_answer = "\n\n".join(answer_parts)
            else:
                final_answer = None

            logger.debug(
                f"""
                Verified answer at current step {current_step}. Has more steps: {has_more_steps}
                Step contexts: {step_ctx_list}
                Step answers so far: {step_answers}
                Draft answer: {draft_answer}
                Final answer so far: {final_answer}
                """
            )

            return {
                "final_answer": final_answer,
                "step_answers": step_answers,
                "current_step": current_step + 1,
                "draft_answer": "",
                "evaluation_result": None,  # Clear evaluation_result for next cycle
                "refined_query": None,  # Clear refined_query for next step
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": state.get("messages", [])
                + [AIMessage(content="Answer verified and citations checked.")],
            }
        except Exception as e:
            return {
                "final_answer": draft_answer if not has_more_steps else None,
                "current_step": current_step + 1,
                "draft_answer": "",
                "evaluation_result": None,  # Clear evaluation_result for next cycle
                "refined_query": None,  # Clear refined_query for next step
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": state.get("messages", [])
                + [AIMessage(content=f"Error during verification: {str(e)}")],
            }

    return verify_node


# ==================== ERROR HANDLER NODE ====================
async def error_handler_node(state: AgentState) -> Dict[str, Any]:
    """
    Handle errors gracefully.

    Args:
        state: Current agent state

    Returns:
        Updated state with error message
    """
    error_message = state.get("error", "An unknown error occurred.")
    return {
        "final_answer": f"Error: {error_message}",
        "iteration_count": state.get("iteration_count", 0) + 1,
        "messages": state.get("messages", [])
        + [AIMessage(content=f"Handled error: {error_message}")],
    }


# Define available tools for function calling
TOOL_CALCULATOR = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Perform mathematical calculations. Evaluates mathematical expressions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate (e.g., '15 * 0.2', '(100 + 50) / 2')",
                    }
                },
                "required": ["expression"],
            },
        },
    },
]

TOOL_WEB_SEARCH = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information. Use when internal documents don't have the answer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for finding information on the web",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of search results to return",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
]

TOOL_DOWNLOAD_FILE = [
    {
        "type": "function",
        "function": {
            "name": "download_file",
            "description": "Download files from URLs. Returns file_id for each downloaded file that can be used for attachments.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of URLs to download files from",
                    },
                },
                "required": ["file_urls"],
            },
        },
    },
]

TOOL_CREATE_DOCUMENTS = [
    {
        "type": "function",
        "function": {
            "name": "create_documents",
            "description": "Create documents (PDF, DOCX, TXT, CSV, XLSX, HTML, MD) from content. Returns file_id for each created file.",
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
                                    "description": "Document content (supports markdown)",
                                },
                                "title": {
                                    "type": "string",
                                    "description": "Document title",
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
                                    "description": "Output format (default: pdf)",
                                },
                                "filename": {
                                    "type": "string",
                                    "description": "Optional custom filename (without extension)",
                                },
                            },
                            "required": ["content", "title"],
                        },
                    },
                },
                "required": ["documents"],
            },
        },
    },
]

TOOL_SEND_EMAIL = [
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send email with optional attachments. Use file_id from previous tool results for attachments.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Recipient email addresses",
                    },
                    "subject": {
                        "type": "string",
                        "description": "Email subject",
                    },
                    "body": {
                        "type": "string",
                        "description": "Email body content",
                    },
                    "attachments": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "File IDs to attach (from download_file or create_documents results)",
                    },
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
]
