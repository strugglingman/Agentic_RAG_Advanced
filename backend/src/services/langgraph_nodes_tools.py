"""Tool node implementations."""

from typing import Dict, Any, Callable, Coroutine
import json
import asyncio
import logging

from langchain_core.messages import AIMessage

from src.services.langgraph_state import AgentState, RuntimeContext
from src.services.agent_tools import (
    execute_tool_call,
    TOOL_WEB_SEARCH,
    TOOL_DOWNLOAD_FILE,
    TOOL_CREATE_DOCUMENTS,
    TOOL_SEND_EMAIL,
    TOOL_CODE_EXECUTION,
)
from src.services.llm_client import chat_completion_with_tools
from src.config.settings import Config
from src.prompts import ToolPrompts
from src.observability.metrics import increment_query_routing, increment_error, MetricsErrorType
from src.services.langgraph_nodes_common import (
    _clone_step_contexts,
    _clone_tool_results,
    _optimize_step_query,
    build_previous_step_context,
    get_attachment_context,
)

logger = logging.getLogger(__name__)

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
        increment_query_routing("web_search")
        try:
            async with asyncio.timeout(Config.AGENT_TOOL_TIMEOUT):
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
                    # Optimize verbose planner query for web search (stay under Tavily 400 char limit)
                    logger.info(f"[WEB_SEARCH] Optimizing step_query: '{clean_query}'")
                    clean_query = await _optimize_step_query(clean_query, "web_search", client)
                    logger.info(f"[WEB_SEARCH] Optimized query: '{clean_query}'")
                    prompt = ToolPrompts.web_search_prompt(clean_query, is_detour=False)
                elif is_detour:
                    refined_query = state.get("refined_query")
                    if refined_query:
                        task_query = refined_query
                        logger.info(
                            f"[WEB_SEARCH] Using refined_query (detour): '{task_query}'"
                        )
                    elif plan and current_step < len(plan):
                        task_query = plan[current_step]
                        logger.info(
                            f"[WEB_SEARCH] Using plan step (detour): '{task_query}'"
                        )
                    else:
                        task_query = query
                        logger.info(
                            f"[WEB_SEARCH] Using original query (detour): '{task_query}'"
                        )
                    prompt = ToolPrompts.web_search_prompt(task_query, is_detour=True)
                else:
                    prompt = ToolPrompts.fallback_prompt(query, "web_search")

                response = await chat_completion_with_tools(
                    client=client,
                    model=Config.OPENAI_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    tools=TOOL_WEB_SEARCH,
                    tool_choice="auto",
                    temperature=0.1,
                )

                if not response.choices[0].message.tool_calls:
                    # Save step_contexts so verify_node doesn't throw and can combine all step_answers
                    step_contexts = _clone_step_contexts(state)
                    if current_step not in step_contexts:
                        step_contexts[current_step] = []
                    step_contexts[current_step].append(
                        {
                            "type": "tool",
                            "tool_name": "web_search",
                            "result": "Web search not called - no results",
                            "args": {},
                            "plan_step": (
                                plan[current_step]
                                if plan and current_step < len(plan)
                                else ""
                            ),
                        }
                    )
                    return {
                        "tools_used": state.get("tools_used", []),
                        "tool_results": state.get("tool_results", {}),
                        "step_contexts": step_contexts,
                        "current_step": current_step,
                        "iteration_count": state.get("iteration_count", 0) + 1,
                        "draft_answer": "Web search was not performed.",
                        "messages": [AIMessage(content="No tool was called by the LLM.")],
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

                tool_results = _clone_tool_results(state)
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
                step_contexts = _clone_step_contexts(state)
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
                    "messages": [
                        AIMessage(
                            content=f"Executed {tool_name} with result: {result[:200]}..."
                        )
                    ],
                }

        except TimeoutError:
            logger.warning("[WEB_SEARCH] Node timed out after %ds", Config.AGENT_TOOL_TIMEOUT)
            increment_error(MetricsErrorType.NODE_TIMEOUT)
            return {
                "tools_used": state.get("tools_used", []),
                "tool_results": state.get("tool_results", {}),
                "current_step": current_step,
                "iteration_count": state.get("iteration_count", 0) + 1,
                "error": f"Web search timed out after {Config.AGENT_TOOL_TIMEOUT}s",
                "messages": [AIMessage(content=f"Web search timed out after {Config.AGENT_TOOL_TIMEOUT}s.")],
            }
        except Exception as e:
            return {
                "tools_used": state.get("tools_used", []),
                "tool_results": state.get("tool_results", {}),
                "current_step": current_step,
                "iteration_count": state.get("iteration_count", 0) + 1,
                "error": f"Tool execution failed: {str(e)}",
                "messages": [AIMessage(content=f"Tool execution failed: {str(e)}")],
            }

    return tool_web_search_node

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
        increment_query_routing("download_file")
        try:
            async with asyncio.timeout(Config.AGENT_TOOL_TIMEOUT):
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

                # Use unified helper to get all previous step context
                prev_ctx = build_previous_step_context(state, current_step)
                logger.info(
                    f"[DOWNLOAD_FILE_NODE] prev_ctx: urls={len(prev_ctx.urls)}, "
                    f"file_ids={len(prev_ctx.file_ids)}, text_len={len(prev_ctx.text)}"
                )

                # Build prompt for LLM tool calling
                if plan and current_step < len(plan):
                    action_step = plan[current_step]
                    # Extract URLs from plan step
                    clean_query = (
                        action_step.split(":", 1)[1].strip()
                        if ":" in action_step
                        else action_step
                    )
                    prompt = ToolPrompts.download_file_prompt(
                        clean_query, previous_step_context=prev_ctx.text
                    )
                else:
                    prompt = ToolPrompts.fallback_prompt(query, "download_file")

                response = await chat_completion_with_tools(
                    client=client,
                    model=Config.OPENAI_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    tools=TOOL_DOWNLOAD_FILE,
                    tool_choice="auto",
                    temperature=0.1,
                )

                if not response.choices[0].message.tool_calls:
                    # Save step_contexts so verify_node doesn't throw and can combine all step_answers
                    step_contexts = _clone_step_contexts(state)
                    if current_step not in step_contexts:
                        step_contexts[current_step] = []
                    step_contexts[current_step].append(
                        {
                            "type": "tool",
                            "tool_name": "download_file",
                            "result": "Download not performed - no files downloaded",
                            "args": {},
                            "plan_step": (
                                plan[current_step]
                                if plan and current_step < len(plan)
                                else ""
                            ),
                        }
                    )
                    return {
                        "tools_used": state.get("tools_used", []),
                        "tool_results": state.get("tool_results", {}),
                        "step_contexts": step_contexts,
                        "current_step": current_step,
                        "iteration_count": state.get("iteration_count", 0) + 1,
                        "draft_answer": "Download was not performed.",
                        "messages": [
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
                    "conversation_id": runtime.get("conversation_id"),
                }

                result = await execute_tool_call(tool_name, tool_args, context)

                tool_results = _clone_tool_results(state)
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
                step_contexts = _clone_step_contexts(state)
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
                    "messages": [
                        AIMessage(
                            content=f"Downloaded files: {len(files_created)} file(s) created"
                        )
                    ],
                }

        except TimeoutError:
            logger.warning("[DOWNLOAD_FILE] Node timed out after %ds", Config.AGENT_TOOL_TIMEOUT)
            increment_error(MetricsErrorType.NODE_TIMEOUT)
            return {
                "tools_used": state.get("tools_used", []),
                "tool_results": state.get("tool_results", {}),
                "draft_answer": f"Download timed out after {Config.AGENT_TOOL_TIMEOUT}s.",
                "current_step": current_step,
                "iteration_count": state.get("iteration_count", 0) + 1,
                "error": f"Download file timed out after {Config.AGENT_TOOL_TIMEOUT}s",
                "messages": [AIMessage(content=f"Download file timed out after {Config.AGENT_TOOL_TIMEOUT}s.")],
            }
        except Exception as e:
            return {
                "tools_used": state.get("tools_used", []),
                "tool_results": state.get("tool_results", {}),
                "draft_answer": f"Download failed: {str(e)}",
                "current_step": current_step,
                "iteration_count": state.get("iteration_count", 0) + 1,
                "error": f"Download file failed: {str(e)}",
                "messages": [AIMessage(content=f"Download file failed: {str(e)}")],
            }

    return tool_download_file_node

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
        increment_query_routing("create_documents")
        try:
            async with asyncio.timeout(Config.AGENT_TOOL_TIMEOUT):
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

                # Use unified helper to get all previous step context
                prev_ctx = build_previous_step_context(state, current_step)

                # Get attachment content (extract full content for document creation)
                _, attachment_content = await get_attachment_context(
                    runtime, extract_content=True
                )

                # Combine previous step context with attachment content
                combined_content = prev_ctx.text
                if attachment_content:
                    if combined_content:
                        combined_content = f"{combined_content}\n\n{attachment_content}"
                    else:
                        combined_content = attachment_content

                logger.info(
                    f"[CREATE_DOCUMENTS_NODE] prev_ctx: urls={len(prev_ctx.urls)}, "
                    f"file_ids={len(prev_ctx.file_ids)}, text_len={len(prev_ctx.text)}, "
                    f"attachment_len={len(attachment_content)}"
                )

                # Get step_contexts for writing results (will store files_created)
                step_contexts = _clone_step_contexts(state)

                # Build prompt for LLM tool calling
                if plan and current_step < len(plan):
                    action_step = plan[current_step]
                    clean_query = (
                        action_step.split(":", 1)[1].strip()
                        if ":" in action_step
                        else action_step
                    )
                    # Include previous step content AND attachment content in prompt
                    prompt = ToolPrompts.create_documents_prompt(
                        clean_query, previous_content=combined_content
                    )
                else:
                    prompt = ToolPrompts.fallback_prompt(query, "create_documents")

                response = await chat_completion_with_tools(
                    client=client,
                    model=Config.OPENAI_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    tools=TOOL_CREATE_DOCUMENTS,
                    tool_choice="auto",
                    temperature=0.1,
                )

                if not response.choices[0].message.tool_calls:
                    # Save step_contexts so verify_node doesn't throw and can combine all step_answers
                    if current_step not in step_contexts:
                        step_contexts[current_step] = []
                    step_contexts[current_step].append(
                        {
                            "type": "tool",
                            "tool_name": "create_documents",
                            "result": "Document creation not performed",
                            "args": {},
                            "plan_step": (
                                plan[current_step]
                                if plan and current_step < len(plan)
                                else ""
                            ),
                        }
                    )
                    return {
                        "tools_used": state.get("tools_used", []),
                        "tool_results": state.get("tool_results", {}),
                        "step_contexts": step_contexts,
                        "current_step": current_step,
                        "iteration_count": state.get("iteration_count", 0) + 1,
                        "draft_answer": "Document creation was not performed.",
                        "messages": [
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
                    "conversation_id": runtime.get("conversation_id"),
                }

                result = await execute_tool_call(tool_name, tool_args, context)

                tool_results = _clone_tool_results(state)
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
                    "messages": [
                        AIMessage(
                            content=f"Created documents: {len(files_created)} file(s)"
                        )
                    ],
                }

        except TimeoutError:
            logger.warning("[CREATE_DOCUMENTS] Node timed out after %ds", Config.AGENT_TOOL_TIMEOUT)
            increment_error(MetricsErrorType.NODE_TIMEOUT)
            return {
                "tools_used": state.get("tools_used", []),
                "tool_results": state.get("tool_results", {}),
                "draft_answer": f"Document creation timed out after {Config.AGENT_TOOL_TIMEOUT}s.",
                "current_step": current_step,
                "iteration_count": state.get("iteration_count", 0) + 1,
                "error": f"Create documents timed out after {Config.AGENT_TOOL_TIMEOUT}s",
                "messages": [AIMessage(content=f"Create documents timed out after {Config.AGENT_TOOL_TIMEOUT}s.")],
            }
        except Exception as e:
            return {
                "tools_used": state.get("tools_used", []),
                "tool_results": state.get("tool_results", {}),
                "draft_answer": f"Document creation failed: {str(e)}",
                "current_step": current_step,
                "iteration_count": state.get("iteration_count", 0) + 1,
                "error": f"Create documents failed: {str(e)}",
                "messages": [AIMessage(content=f"Create documents failed: {str(e)}")],
            }

    return tool_create_documents_node

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
        increment_query_routing("send_email")
        try:
            async with asyncio.timeout(Config.AGENT_TOOL_TIMEOUT):
                logger.info("[SEND_EMAIL_NODE] Starting send_email node execution")
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

                # Use unified helper to get all previous step context
                # This gives us file_ids from this plan AND full text context
                prev_ctx = build_previous_step_context(state, current_step)
                logger.info(
                    f"[SEND_EMAIL_NODE] prev_ctx: urls={len(prev_ctx.urls)}, "
                    f"file_ids={len(prev_ctx.file_ids)}, text_len={len(prev_ctx.text)}"
                )

                # Get step_contexts for writing results later
                step_contexts = _clone_step_contexts(state)

                # Files from THIS PLAN (high priority) - from helper
                session_file_ids = prev_ctx.file_ids

                # User's EXISTING files (low priority) - from runtime, keep separate
                available_files = runtime.get("available_files", [])
                available_file_info = []
                for f in available_files:
                    available_file_info.append(
                        {
                            "file_id": f.get(
                                "id"
                            ),  # list_files returns "id", not "file_id"
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
                        available_file_ids=session_file_ids,  # Files from THIS plan (high priority)
                        available_files=available_file_info,  # User's existing files (low priority)
                        previous_step_context=prev_ctx.text,  # Full context from previous steps
                    )
                else:
                    prompt = ToolPrompts.fallback_prompt(query, "send_email")

                # Build messages with conversation history (like plan_node/generate_node)
                # This allows LLM to see prior context when user says "confirm" or references earlier messages
                messages = []
                conversation_history = runtime.get("conversation_history", [])
                if conversation_history:
                    # Add recent history for context (reference resolution for confirmations)
                    recent_history = conversation_history[-Config.PLANNING_CONTEXT_LIMIT :]
                    for h in recent_history:
                        messages.append(
                            {"role": h.get("role", "user"), "content": h.get("content", "")}
                        )
                messages.append({"role": "user", "content": prompt})

                logger.info(
                    f"[SEND_EMAIL_NODE] Calling LLM with session_file_ids={session_file_ids}, history_len={len(conversation_history)}"
                )
                response = await chat_completion_with_tools(
                    client=client,
                    model=Config.OPENAI_MODEL,
                    messages=messages,
                    tools=TOOL_SEND_EMAIL,
                    tool_choice="auto",
                    temperature=0.1,
                )
                logger.info(
                    f"[SEND_EMAIL_NODE] LLM response received, has_tool_calls={bool(response.choices[0].message.tool_calls)}"
                )

                if not response.choices[0].message.tool_calls:
                    # Like AgentService: capture LLM's text response (clarification question)
                    llm_text = response.choices[0].message.content or ""
                    logger.info(
                        f"[SEND_EMAIL_NODE] No tool called, LLM text: {llm_text[:200]}..."
                    )

                    # CRITICAL: Save step_contexts even when tool is not called
                    # This ensures verify_node doesn't throw and can combine all step_answers
                    # (including previous download step with markdown links)
                    if current_step not in step_contexts:
                        step_contexts[current_step] = []
                    step_contexts[current_step].append(
                        {
                            "type": "tool",
                            "tool_name": "send_email",
                            "result": llm_text or "Email not sent - awaiting confirmation",
                            "args": {},
                            "plan_step": (
                                plan[current_step]
                                if plan and current_step < len(plan)
                                else ""
                            ),
                        }
                    )

                    if llm_text.strip():
                        # Return LLM's clarification as draft_answer (like AgentService line 98-99)
                        return {
                            "tools_used": state.get("tools_used", []),
                            "tool_results": state.get("tool_results", {}),
                            "step_contexts": step_contexts,  # Include step_contexts!
                            "current_step": current_step,
                            "iteration_count": state.get("iteration_count", 0) + 1,
                            "draft_answer": llm_text,
                            "messages": [AIMessage(content=llm_text)],
                        }

                    return {
                        "tools_used": state.get("tools_used", []),
                        "tool_results": state.get("tool_results", {}),
                        "step_contexts": step_contexts,  # Include step_contexts!
                        "current_step": current_step,
                        "iteration_count": state.get("iteration_count", 0) + 1,
                        "draft_answer": "Email not sent - no response from LLM.",
                        "messages": [
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

                tool_results = _clone_tool_results(state)
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
                    "messages": [AIMessage(content=f"Email sent: {result[:100]}...")],
                }

        except TimeoutError:
            logger.warning("[SEND_EMAIL] Node timed out after %ds", Config.AGENT_TOOL_TIMEOUT)
            increment_error(MetricsErrorType.NODE_TIMEOUT)
            return {
                "tools_used": state.get("tools_used", []),
                "tool_results": state.get("tool_results", {}),
                "draft_answer": f"Email sending timed out after {Config.AGENT_TOOL_TIMEOUT}s.",
                "current_step": current_step,
                "iteration_count": state.get("iteration_count", 0) + 1,
                "error": f"Send email timed out after {Config.AGENT_TOOL_TIMEOUT}s",
                "messages": [AIMessage(content=f"Send email timed out after {Config.AGENT_TOOL_TIMEOUT}s.")],
            }
        except Exception as e:
            logger.error(f"[SEND_EMAIL_NODE] Exception: {type(e).__name__}: {str(e)}")
            import traceback

            logger.error(f"[SEND_EMAIL_NODE] Traceback: {traceback.format_exc()}")
            return {
                "tools_used": state.get("tools_used", []),
                "tool_results": state.get("tool_results", {}),
                "draft_answer": f"Email sending failed: {str(e)}",
                "current_step": current_step,
                "iteration_count": state.get("iteration_count", 0) + 1,
                "error": f"Send email failed: {str(e)}",
                "messages": [AIMessage(content=f"Send email failed: {str(e)}")],
            }

    return tool_send_email_node

def create_tool_code_execution_node(
    runtime: RuntimeContext,
) -> Callable[[AgentState], Coroutine[Any, Any, Dict[str, Any]]]:
    """
    Factory function to create tool_code_execution_node with runtime context bound.

    Args:
        runtime: RuntimeContext with openai_client, vector_db, dept_id, user_id

    Returns:
        Async tool_code_execution_node function
    """

    async def tool_code_execution_node(state: AgentState) -> Dict[str, Any]:
        """
        Execute code_execution tool using LLM function calling.

        This node runs Python code in a secure E2B sandbox for data analysis,
        calculations, and file processing tasks.

        Args:
            state: Current agent state

        Returns:
            Updated state with tool results
        """
        increment_query_routing("code_execution")
        try:
            async with asyncio.timeout(Config.AGENT_TOOL_TIMEOUT):
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

                # Determine if this is a DETOUR call
                is_detour = state.get("evaluation_result") is not None

                # Build context from previous steps for code generation
                prev_ctx = build_previous_step_context(state, current_step)

                # Get attachment content (extract full content for code execution)
                _, attachment_content = await get_attachment_context(
                    runtime, extract_content=True
                )

                # Combine previous step context with attachment content
                combined_context = prev_ctx.text
                if attachment_content:
                    if combined_context:
                        combined_context = f"{combined_context}\n\n{attachment_content}"
                    else:
                        combined_context = attachment_content

                logger.info(
                    f"[CODE_EXECUTION_NODE] prev_ctx: text_len={len(prev_ctx.text)}, "
                    f"attachment_len={len(attachment_content)}"
                )

                # Build prompt for LLM tool calling
                if plan and current_step < len(plan) and not is_detour:
                    action_step = plan[current_step]
                    clean_query = (
                        action_step.split(":", 1)[1].strip()
                        if ":" in action_step
                        else action_step
                    )
                    prompt = ToolPrompts.code_execution_prompt(
                        clean_query,
                        previous_step_context=combined_context,
                        is_detour=False,
                    )
                elif is_detour:
                    refined_query = state.get("refined_query")
                    task_query = (
                        refined_query
                        if refined_query
                        else (
                            plan[current_step]
                            if plan and current_step < len(plan)
                            else query
                        )
                    )
                    prompt = ToolPrompts.code_execution_prompt(
                        task_query,
                        previous_step_context=combined_context,
                        is_detour=True,
                    )
                else:
                    prompt = ToolPrompts.fallback_prompt(query, "code_execution")

                response = await chat_completion_with_tools(
                    client=client,
                    model=Config.OPENAI_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    tools=TOOL_CODE_EXECUTION,
                    tool_choice="auto",
                    temperature=0.1,
                )

                # Check if LLM called a tool
                step_contexts = _clone_step_contexts(state)
                if not response.choices[0].message.tool_calls:
                    if current_step not in step_contexts:
                        step_contexts[current_step] = []
                    step_contexts[current_step].append(
                        {
                            "type": "tool",
                            "tool_name": "code_execution",
                            "result": "Code execution not called - no code generated",
                            "args": {},
                            "plan_step": (
                                plan[current_step]
                                if plan and current_step < len(plan)
                                else ""
                            ),
                        }
                    )
                    return {
                        "tools_used": state.get("tools_used", []),
                        "tool_results": state.get("tool_results", {}),
                        "step_contexts": step_contexts,
                        "current_step": current_step,
                        "iteration_count": state.get("iteration_count", 0) + 1,
                        "draft_answer": "Code execution was not called.",
                        "messages": [AIMessage(content="No tool was called by the LLM.")],
                    }

                # Execute the tool call
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
                tool_results = _clone_tool_results(state)
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

                # Remove any existing code_execution context (from previous refinement)
                step_contexts[current_step] = [
                    ctx
                    for ctx in step_contexts[current_step]
                    if not (
                        ctx.get("type") == "tool"
                        and ctx.get("tool_name") == "code_execution"
                    )
                ]

                # Add new code_execution context
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
                    "draft_answer": result,
                    "current_step": current_step,
                    "iteration_count": state.get("iteration_count", 0) + 1,
                    "messages": [AIMessage(content=f"Executed code with result: {result[:200]}...")],
                }

        except TimeoutError:
            logger.warning("[CODE_EXECUTION] Node timed out after %ds", Config.AGENT_TOOL_TIMEOUT)
            increment_error(MetricsErrorType.NODE_TIMEOUT)
            return {
                "tools_used": state.get("tools_used", []),
                "tool_results": state.get("tool_results", {}),
                "draft_answer": f"Code execution timed out after {Config.AGENT_TOOL_TIMEOUT}s.",
                "current_step": current_step,
                "iteration_count": state.get("iteration_count", 0) + 1,
                "error": f"Code execution timed out after {Config.AGENT_TOOL_TIMEOUT}s",
                "messages": [AIMessage(content=f"Code execution timed out after {Config.AGENT_TOOL_TIMEOUT}s.")],
            }
        except Exception as e:
            logger.error(
                f"[CODE_EXECUTION_NODE] Exception: {type(e).__name__}: {str(e)}"
            )
            return {
                "tools_used": state.get("tools_used", []),
                "tool_results": state.get("tool_results", {}),
                "draft_answer": f"Code execution failed: {str(e)}",
                "current_step": current_step,
                "iteration_count": state.get("iteration_count", 0) + 1,
                "error": f"Code execution failed: {str(e)}",
                "messages": [AIMessage(content=f"Code execution failed: {str(e)}")],
            }

    return tool_code_execution_node
