"""Answer generation/verification node implementations."""

from typing import Dict, Any, Callable, Coroutine
import asyncio
import logging

from langchain_core.messages import AIMessage

from src.services.langgraph_state import AgentState, RuntimeContext
from src.services.llm_client import chat_completion
from src.utils.safety import (
    enforce_citations,
    add_sources_from_citations,
    renumber_citations,
    sanitize_text,
)
from src.config.settings import Config
from src.prompts import GenerationPrompts
from src.prompts.generation import ContextType
from src.observability.metrics import increment_query_routing, increment_error, MetricsErrorType
from src.models.evaluation import RecommendationAction
from src.services.langgraph_nodes_common import (
    _clone_step_contexts,
    dict_to_evaluation_result,
)

logger = logging.getLogger(__name__)

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
        increment_query_routing("direct_answer")
        plan = state.get("plan", [])
        current_step = state.get("current_step", 0)

        openai_client = runtime.get("openai_client", None)
        if not openai_client:
            step_contexts = _clone_step_contexts(state)
            if current_step not in step_contexts:
                step_contexts[current_step] = []
            step_contexts[current_step].append({
                "type": "direct_answer",
                "plan_step": plan[current_step] if plan and current_step < len(plan) else "",
            })
            return {
                "draft_answer": "I'm sorry, I'm unable to process this request right now. Please try again later.",
                "step_contexts": step_contexts,
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": [AIMessage(content="OpenAI client not available for direct answer.")],
            }
        if not plan or current_step >= len(plan):
            return {
                "direct_answer": "No valid plan step for direct answer.",
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": [AIMessage(content="No valid plan step for direct answer.")],
            }
        action_step = plan[current_step]
        step_query = (
            action_step.split(":", 1)[1].strip() if ":" in action_step else action_step
        )

        try:
            async with asyncio.timeout(Config.AGENT_TOOL_TIMEOUT):
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

                response = await chat_completion(
                    client=openai_client,
                    model=Config.OPENAI_MODEL,
                    messages=openai_messages,
                    max_tokens=Config.CHAT_MAX_TOKENS,
                    temperature=Config.OPENAI_TEMPERATURE,
                )
                direct_answer = ""
                if response.choices and response.choices[0].message:
                    direct_answer = response.choices[0].message.content

                # Store direct answer in step_contexts
                # Replace if exists (unlikely for direct_answer, but keep consistent)
                step_contexts = _clone_step_contexts(state)
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
                    "messages": [AIMessage(content="Generated answer from direct_answer directly.")],
                }
        except TimeoutError:
            logger.warning("[DIRECT_ANSWER] Node timed out after %ds", Config.AGENT_TOOL_TIMEOUT)
            increment_error(MetricsErrorType.NODE_TIMEOUT)
            return {
                "draft_answer": "",
                "current_step": current_step,
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": [AIMessage(content=f"Direct answer timed out after {Config.AGENT_TOOL_TIMEOUT}s.")],
            }
        except Exception as e:
            return {
                "draft_answer": "",
                "current_step": current_step,
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": [AIMessage(content="Error during direct answer generation.")],
            }

    return direct_answer_node

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
            async with asyncio.timeout(Config.AGENT_TOOL_TIMEOUT):
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
                        "messages": [AIMessage(content=clarification_message)],
                    }

                # Get ONLY current step's context (per-step isolation)
                current_step = state.get("current_step", 0)
                step_contexts = _clone_step_contexts(state)

                # Get context for the current step being executed (now a list)
                if current_step not in step_contexts or not step_contexts[current_step]:
                    return {
                        "draft_answer": "",
                        "iteration_count": state.get("iteration_count", 0) + 1,
                        "messages": [
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

                # Build numbered context from ALL contexts in this step (handles retrieve + web_search + code_execution)
                contexts = []
                context_num = 1

                # Track if decomposition was used (for instructions later)
                has_decomposition = False
                num_sub_queries = 0

                # Iterate through all contexts for this step
                for step_ctx in step_ctx_list:
                    if step_ctx["type"] == "retrieval":
                        # Retrieved documents from this step only
                        docs = step_ctx.get("docs", [])

                        # Check if contexts have sub_query labels (indicates decomposition was used)
                        # IMPORTANT: Use list with dict.fromkeys() to preserve insertion order from docs
                        # (set doesn't guarantee order, causing citation number mismatch with frontend)
                        sub_queries = list(
                            dict.fromkeys(
                                d.get("sub_query") for d in docs if d.get("sub_query")
                            )
                        )
                        has_decomposition = len(sub_queries) > 1
                        num_sub_queries = len(sub_queries)

                        if has_decomposition:
                            # Group contexts by sub-query for clearer presentation to LLM
                            # Get the original query from state
                            original_query = state.get("query", "")
                            contexts.append(
                                f'Original Query: "{original_query}"\n'
                                f"Decomposed into {num_sub_queries} sub-queries for better retrieval:\n"
                            )

                            for sq in sub_queries:
                                sq_docs = [d for d in docs if d.get("sub_query") == sq]
                                contexts.append(
                                    f'=== Sub-query: "{sq}" ({len(sq_docs)} results) ===\n'
                                )

                                for doc in sq_docs:
                                    chunk = doc.get("chunk", str(doc))
                                    source = doc.get("source", "unknown")
                                    page = doc.get("page", 0)

                                    header = f"Context {context_num} (Source: {source}"
                                    if page > 0:
                                        header += f", Page: {page}"
                                    header += "):\n"

                                    contexts.append(f"{header}{chunk}")
                                    context_num += 1
                        else:
                            # Original flat format (no decomposition or single sub-query)
                            for doc in docs:
                                chunk = doc.get("chunk", str(doc))
                                source = doc.get("source", "unknown")
                                page = doc.get("page", 0)

                                header = f"Context {context_num} (Source: {source}"
                                if page > 0:
                                    header += f", Page: {page}"
                                header += "):\n"

                                contexts.append(f"{header}{chunk}")
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

                # Add decomposition instructions if query was decomposed
                if has_decomposition and not is_web_search:
                    decomp_instruction = (
                        f"\n\nDECOMPOSITION INSTRUCTIONS:\n"
                        f"- The original query was decomposed into {num_sub_queries} sub-queries for better retrieval.\n"
                        f"- Contexts are grouped by sub-query above.\n"
                        f"- Use information from ALL sub-query groups to fully answer the ORIGINAL query.\n"
                        f"- When comparing entities, ensure you include data from each relevant sub-query group."
                    )
                    system_prompt += decomp_instruction

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

                response = await chat_completion(
                    client=openai_client,
                    model=Config.OPENAI_MODEL,
                    messages=openai_messages,
                    max_tokens=Config.CHAT_MAX_TOKENS,
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
                    "messages": [AIMessage(content="Generated answer successfully.")],
                }

        except TimeoutError:
            logger.warning("[GENERATE] Node timed out after %ds", Config.AGENT_TOOL_TIMEOUT)
            increment_error(MetricsErrorType.NODE_TIMEOUT)
            return {
                "draft_answer": "",
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": [AIMessage(content=f"Answer generation timed out after {Config.AGENT_TOOL_TIMEOUT}s.")],
            }
        except Exception as e:
            return {
                "draft_answer": "",
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": [AIMessage(content="Error during answer generation.")],
            }

    return generate_node

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
                "messages": [AIMessage(content="No draft answer to verify.")],
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
                "messages": [AIMessage(content="Clarification request prepared.")],
            }

        try:
            # Calculate valid context IDs from step contexts (matches generate_node numbering)
            step_contexts = _clone_step_contexts(state)
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

            # Tools that skip citation enforcement (direct results, not RAG)
            skip_citation_tools = {
                "web_search",
                "download_file",
                "create_documents",
                "send_email",
                "code_execution",
            }
            is_skip_citation_tool = bool(set(tool_names) & skip_citation_tools)

            clean_answer = ""

            valid_ids = []  # Initialize for all cases
            # Direct answer and skip-citation tools bypass citation enforcement
            if "direct_answer" in step_types or is_skip_citation_tool:
                clean_answer = draft_answer
            else:
                context_num = 1

                # Build valid_ids from step_ctx_list (same source as generate_node)
                # to ensure citation numbers match what the LLM saw in the prompt
                for step_ctx in step_ctx_list:
                    if step_ctx.get("type") == "retrieval":
                        for _ in step_ctx.get("docs", []):
                            valid_ids.append(context_num)
                            context_num += 1
                    elif step_ctx.get("type") == "tool":
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
                        "messages": [
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

            # Copy before append to avoid in-place mutation of prior state snapshots.
            step_answers = list(state.get("step_answers", []))
            step_answers.append(
                {
                    "step": current_step,
                    "question": clean_question,
                    "answer": clean_answer,
                }
            )

            # If all steps complete, concatenate all step answers
            if not has_more_steps:
                # Get step_contexts for doc counting and sources
                step_contexts = _clone_step_contexts(state)

                # Build final answer from all step answers
                if len(step_answers) == 1:
                    # Single step - return answer directly (no renumbering needed)
                    final_answer = step_answers[0]["answer"]
                else:
                    # Multiple steps - renumber citations globally and format
                    # Calculate doc counts per step for cumulative offset
                    step_doc_counts = {}
                    for step_num in sorted(step_contexts.keys()):
                        doc_count = 0
                        for ctx in step_contexts.get(step_num, []):
                            if ctx.get("type") == "retrieval":
                                doc_count += len(ctx.get("docs", []))
                        step_doc_counts[step_num] = doc_count

                    # Build answer parts with renumbered citations
                    answer_parts = []
                    cumulative_offset = 0

                    for step_ans in step_answers:
                        step_num = step_ans["step"]

                        # Renumber citations with cumulative offset
                        renumbered_answer = renumber_citations(
                            step_ans["answer"], cumulative_offset
                        )

                        # Extract task description (remove tool name prefix)
                        task = step_ans["question"]
                        if ":" in task:
                            task = task.split(":", 1)[1].strip()

                        answer_parts.append(
                            f"**{task.capitalize()}**\n{renumbered_answer}"
                        )

                        # Add this step's doc count to offset for next step
                        cumulative_offset += step_doc_counts.get(step_num, 0)

                    final_answer = "\n\n".join(answer_parts)

                # Add programmatic Sources line based on all contexts
                all_docs = []
                for step_num in sorted(step_contexts.keys()):
                    for ctx in step_contexts.get(step_num, []):
                        if ctx.get("type") == "retrieval":
                            all_docs.extend(ctx.get("docs", []))

                if all_docs:
                    final_answer, cited_files = add_sources_from_citations(
                        final_answer, all_docs
                    )
                    if cited_files:
                        logger.info(f"[VERIFY] Final sources: {cited_files}")
            else:
                final_answer = None

            # Summarize step contexts for logging (show doc count instead of full content)
            step_ctx_summary = [
                {
                    "type": ctx.get("type", "unknown"),
                    "docs": (
                        len(ctx.get("docs", []))
                        if isinstance(ctx.get("docs"), list)
                        else 0
                    ),
                    f"plan_step_{i}": ctx.get("plan_step", ""),
                }
                for i, ctx in enumerate(step_ctx_list)
            ]
            # logger.debug(
            #     f"""
            #     Verified answer at current step {current_step}. Has more steps: {has_more_steps}
            #     Step contexts: {step_ctx_summary}
            #     Step answers so far: {step_answers}
            #     Draft answer: {draft_answer}
            #     Final answer so far: {final_answer}
            #     """
            # )

            return {
                "final_answer": final_answer,
                "step_answers": step_answers,
                "current_step": current_step + 1,
                "draft_answer": "",
                "evaluation_result": None,  # Clear evaluation_result for next cycle
                "refined_query": None,  # Clear refined_query for next step
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": [AIMessage(content="Answer verified and citations checked.")],
            }
        except Exception as e:
            return {
                "final_answer": draft_answer if not has_more_steps else None,
                "current_step": current_step + 1,
                "draft_answer": "",
                "evaluation_result": None,  # Clear evaluation_result for next cycle
                "refined_query": None,  # Clear refined_query for next step
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": [AIMessage(content=f"Error during verification: {str(e)}")],
            }

    return verify_node
