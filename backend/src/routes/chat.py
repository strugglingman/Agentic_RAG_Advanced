"""Chat routes"""

import json
from flask import Blueprint, request, jsonify, Response, g
from openai import OpenAI
from src.services.query_supervisor import QuerySupervisor
from src.middleware.auth import require_identity
from src.services.retrieval import retrieve, build_where, build_prompt
from src.config.settings import Config
from src.utils.safety import looks_like_injection, scrub_context
from src.utils.stream_utils import stream_text_smart
from src.services.conversation_service import ConversationService

chat_bp = Blueprint("chat", __name__)

# OpenAI client
openai_client = OpenAI(api_key=Config.OPENAI_KEY)
# Redis + PostgreSQL
conversation_client = ConversationService()
# Query supervisor (created once, reused across all requests)
supervisor = QuerySupervisor(openai_client=openai_client)


@chat_bp.post("/chat")
@require_identity
async def chat(collection):
    """Chat endpoint with RAG retrieval."""
    dept_id = g.identity.get("dept_id", "")
    user_id = g.identity.get("user_id", "")
    print("0" * 20)

    if not dept_id or not user_id:
        return jsonify({"error": "No organization ID or user ID provided"}), 400

    payload = request.get_json(force=True)
    msgs = payload.get("messages", [])
    conversation_id = payload.get("conversation_id", "")
    if not conversation_id:
        conversation_id = await conversation_client.create_conversation(
            user_id, "New Chat"
        )

    latest_user_msg = None
    if msgs and isinstance(msgs[-1], dict) and msgs[-1].get("role") == "user":
        latest_user_msg = msgs[-1]

    if not latest_user_msg:
        return jsonify({"error": "No user message found"}), 400
    if not latest_user_msg.get("content").strip():
        return jsonify({"error": "Empty user message"}), 400

    # Check for prompt injection
    injection_result, injection_error = looks_like_injection(
        latest_user_msg.get("content", "")
    )
    if injection_result:
        return jsonify({"error": f"Input rejected: {injection_error}"}), 400

    query = latest_user_msg.get("content").strip()

    try:
        # Build where clause
        where = build_where(payload, dept_id, user_id)
        print(f"Where clause: {where}")

        # Retrieve relevant documents
        ctx, _ = retrieve(
            collection=collection,
            query=query,
            dept_id=dept_id,
            user_id=user_id,
            top_k=Config.TOP_K,
            where=where,
            use_hybrid=Config.USE_HYBRID,
            use_reranker=Config.USE_RERANKER,
        )

        # try MCP to use external knowledge if no context found
        if not ctx:
            # Append latest user message to session history even if no answer found
            if latest_user_msg:
                await conversation_client.save_message(
                    conversation_id,
                    latest_user_msg.get("role", "user"),
                    latest_user_msg.get("content", ""),
                )
            no_answer = "Based on the provided documents, I don't have enough information to answer your question."
            await conversation_client.save_message(
                conversation_id, "assistant", no_answer
            )
            return Response(stream_text_smart(no_answer), mimetype="text/plain")

        # Filter tags
        filters = payload.get("filters", [])
        tags_filter = next(
            (
                f.get("tags")
                for f in filters
                if "tags" in f and isinstance(f.get("tags"), list)
            ),
            None,
        )
        if tags_filter:
            ctx = [
                c
                for c in ctx
                if any(
                    tag in c.get("tags", "").lower().split(",") for tag in tags_filter
                )
            ]

        # Scrub context chunks before sending to LLM
        for c in ctx:
            c["chunk"] = scrub_context(c.get("chunk", ""))

        system, user = build_prompt(query, ctx, use_ctx=True)
        history = await conversation_client.get_sanitized_latest_history(
            conversation_id, Config.REDIS_CACHE_LIMIT
        )
        messages = (
            [{"role": "system", "content": system}]
            + history
            + [{"role": "user", "content": user}]
        )
        messages = [
            m
            for m in messages
            if m.get("content") and m.get("role") in {"system", "user", "assistant"}
        ]

        def generate():
            import asyncio

            answer = []
            try:
                resp = openai_client.chat.completions.create(
                    model=Config.OPENAI_MODEL,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=Config.CHAT_MAX_TOKENS,
                    stream=True,
                )

                for chunk in resp:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield delta
                        answer.append(delta)
            except Exception as e:
                print(f"Error: {e}")
                yield f"\n[upstream_error] {type(e).__name__}: {e}"
            finally:
                # Update session history with latest query and assistant answer
                if latest_user_msg:
                    asyncio.run(
                        conversation_client.save_message(
                            conversation_id,
                            latest_user_msg.get("role", "user"),
                            latest_user_msg.get("content", ""),
                        )
                    )
                raw_answer = "".join(answer)

                if answer:
                    asyncio.run(
                        conversation_client.save_message(
                            conversation_id, "assistant", raw_answer
                        )
                    )
                yield f"\n__CONTEXT__:{json.dumps(ctx)}"

        return Response(generate(), mimetype="text/plain; charset=utf-8")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@chat_bp.post("/chat/agent")
@require_identity
async def chat_agent(collection):
    """Agentic chat endpoint with tool calling and streaming."""
    # EXACT same validation as /chat
    dept_id = g.identity.get("dept_id", "")
    user_id = g.identity.get("user_id", "")

    if not dept_id or not user_id:
        return jsonify({"error": "No organization ID or user ID provided"}), 400

    payload = request.get_json(force=True)
    msgs = payload.get("messages", [])

    latest_user_msg = None
    if msgs and isinstance(msgs[-1], dict) and msgs[-1].get("role") == "user":
        latest_user_msg = msgs[-1]

    if not latest_user_msg:
        return jsonify({"error": "No user message found"}), 400
    if not latest_user_msg.get("content").strip():
        return jsonify({"error": "Empty user message"}), 400

    # EXACT same security check as /chat
    injection_result, injection_error = looks_like_injection(
        latest_user_msg.get("content", "")
    )
    if injection_result:
        return jsonify({"error": f"Input rejected: {injection_error}"}), 400

    query = latest_user_msg.get("content").strip()

    conversation_id = payload.get("conversation_id", "")
    if not conversation_id:
        conversation_id = await conversation_client.create_conversation(
            user_id, query[:20] + "..."
        )

    try:
        # Pre-load conversation history (needed for both agent_service and langgraph)
        conversation_history = await conversation_client.get_sanitized_latest_history(
            conversation_id, Config.REDIS_CACHE_LIMIT
        )

        # Build context for agent (all system parameters)
        agent_context = {
            "collection": collection,
            "dept_id": dept_id,
            "user_id": user_id,
            "conversation_id": conversation_id,  # For langGraph checkpointing
            "conversation_history": conversation_history,  # Pre-loaded history
            "request_data": payload,  # For build_where and filters, but can not pass request directly, active request context changed.
            "use_hybrid": Config.USE_HYBRID,
            "use_reranker": Config.USE_RERANKER,
            "openai_client": openai_client,  # For self-reflection LLM calls
            "model": Config.OPENAI_MODEL,
            "temperature": Config.OPENAI_TEMPERATURE,
        }

        # Create agent
        # agent = AgentService(
        #     openai_client,
        #     max_iterations=int(Config.AGENT_MAX_ITERATIONS),
        #     model=Config.OPENAI_MODEL,
        #     temperature=Config.OPENAI_TEMPERATURE,
        # )

        def generate():
            import asyncio

            answer_parts = []
            try:
                # Run agent with streaming
                # results = asyncio.run(agent.run_stream(query, agent_context))
                final_answer, contexts = asyncio.run(
                    supervisor.process_query(query, agent_context)
                )

                print("############################################")
                print(f"Final Answer: {final_answer}")
                print(f"Contexts: {contexts}")

                # Stream the answer chunks with UTF-8 encoding
                for chunk in stream_text_smart(final_answer):
                    # Ensure chunk is properly encoded
                    if isinstance(chunk, str):
                        yield chunk.encode("utf-8", errors="ignore").decode("utf-8")
                    else:
                        yield chunk
                    answer_parts.append(chunk)

                # After streaming answer, send context (sanitize for encoding)
                context_chunk = (
                    f"\n__CONTEXT__:{json.dumps(contexts, ensure_ascii=False)}"
                )
                yield context_chunk.encode("utf-8", errors="ignore").decode("utf-8")

            except Exception as e:
                error_msg = f"\n[upstream_error] {type(e).__name__}: {str(e)}"
                print(error_msg, flush=True)
                yield error_msg.encode("utf-8", errors="ignore").decode("utf-8")
            finally:
                # Save messages to conversation history
                if latest_user_msg:
                    asyncio.run(
                        conversation_client.save_message(
                            conversation_id,
                            latest_user_msg.get("role", "user"),
                            latest_user_msg.get("content", ""),
                        )
                    )

                raw_answer = "".join(
                    [p for p in answer_parts if not p.startswith("\n__CONTEXT__:")]
                )

                if raw_answer:
                    asyncio.run(
                        conversation_client.save_message(
                            conversation_id, "assistant", raw_answer
                        )
                    )

        return Response(generate(), mimetype="text/plain; charset=utf-8")

    except Exception as e:
        return jsonify({"error": str(e)}), 500
