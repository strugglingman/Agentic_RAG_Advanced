import asyncio
import json
import selectors
import sys
import pytest

from langgraph.checkpoint.memory import MemorySaver

from src.config.settings import Config
from src.services.langgraph_builder import build_langgraph_agent
from src.services.langgraph_state import create_initial_state, create_runtime_context
from src.services.query_supervisor import ExecutionRoute, QuerySupervisor
import src.services.langgraph_nodes_planning as planning_nodes
import src.services.langgraph_nodes_tools as tool_nodes


class _DummyFunction:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class _DummyToolCall:
    def __init__(self, name: str, arguments: str):
        self.function = _DummyFunction(name=name, arguments=arguments)


class _DummyMessage:
    def __init__(self, content: str | None = None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class _DummyChoice:
    def __init__(self, message: _DummyMessage):
        self.message = message


class _DummyResponse:
    def __init__(self, message: _DummyMessage):
        self.choices = [_DummyChoice(message)]


async def _fake_chat_completion_structured(*args, **kwargs):
    plan = {
        "steps": [
            {
                "tool": "send_email",
                "query": "Send a status email to qa@example.com",
            }
        ]
    }
    return _DummyResponse(_DummyMessage(content=json.dumps(plan)))


async def _fake_chat_completion_with_tools(*args, **kwargs):
    tool_args = {
        "to": "qa@example.com",
        "subject": "Status Update",
        "body": "All checks passed.",
    }
    tool_call = _DummyToolCall("send_email", json.dumps(tool_args))
    return _DummyResponse(_DummyMessage(tool_calls=[tool_call]))


async def _fake_execute_tool_call(tool_name, tool_args, context):
    assert tool_name == "send_email"
    return "Email sent successfully. Message-ID: mock-123"


def _patch_hitl_flow(monkeypatch):
    monkeypatch.setattr(
        planning_nodes, "semantic_route_query", lambda *args, **kwargs: (None, 0.0)
    )
    monkeypatch.setattr(
        planning_nodes, "chat_completion_structured", _fake_chat_completion_structured
    )
    monkeypatch.setattr(
        tool_nodes, "chat_completion_with_tools", _fake_chat_completion_with_tools
    )
    monkeypatch.setattr(tool_nodes, "execute_tool_call", _fake_execute_tool_call)
    monkeypatch.setattr(Config, "FORCE_INTERNAL_RETRIEVAL", False)
    monkeypatch.setattr(Config, "USE_LANGGRAPH", True)
    monkeypatch.setattr(Config, "CHECKPOINT_ENABLED", True)


def _run_async(coro):
    if sys.platform == "win32":
        with asyncio.Runner(
            loop_factory=lambda: asyncio.SelectorEventLoop(selectors.SelectSelector())
        ) as runner:
            return runner.run(coro)
    return asyncio.run(coro)


@pytest.mark.component
def test_hitl_interrupt_and_resume_with_memory_checkpoint(monkeypatch):
    _patch_hitl_flow(monkeypatch)

    runtime = create_runtime_context(
        openai_client=object(),
        dept_id="eng",
        user_id="user@example.com",
        conversation_id="hitl-memory-conv",
        conversation_history=[],
    )
    graph = build_langgraph_agent(runtime, checkpointer=MemorySaver())

    state = create_initial_state(query="Please send a status email.")
    state["plan"] = ["send_email: Send a status email to qa@example.com"]
    state["current_step"] = 0

    config = {"configurable": {"thread_id": "hitl-memory-thread"}}

    async def _run():
        async for _ in graph.astream(state, config=config, stream_mode="values"):
            pass

        snapshot = await graph.aget_state(config)
        assert snapshot.next
        assert snapshot.next[0] == "tool_send_email"

        final_state = None
        async for event in graph.astream(None, config=config, stream_mode="values"):
            final_state = event

        assert final_state is not None
        return final_state

    final = _run_async(_run())
    assert "Email sent successfully" in final.get("final_answer", "")


@pytest.mark.integration
def test_query_supervisor_hitl_resume_with_postgres_checkpoint(monkeypatch):
    _patch_hitl_flow(monkeypatch)

    async def _force_langgraph(self, query):
        return ExecutionRoute.LANGGRAPH

    monkeypatch.setattr(QuerySupervisor, "_classify_query", _force_langgraph)

    supervisor = QuerySupervisor(openai_client=object())
    if not supervisor._checkpoint_enabled:
        pytest.skip(
            "PostgreSQL checkpoint is unavailable for HITL/checkpoint integration test."
        )

    context = {
        "vector_db": None,
        "dept_id": "eng",
        "user_id": "user@example.com",
        "conversation_id": "hitl-pg-conv",
        "conversation_history": [],
        "request_data": {},
        "file_service": None,
        "available_files": [],
        "attachment_file_ids": [],
    }

    async def _run():
        first = await supervisor.process_query("Please send status email.", context)
        assert first.hitl_interrupt is not None
        assert first.hitl_interrupt.action == "send_email"

        resumed = await supervisor.resume_workflow(
            thread_id=first.hitl_interrupt.thread_id,
            context=context,
            confirmed=True,
        )
        return first, resumed

    first_result, resumed_result = _run_async(_run())
    assert first_result.hitl_interrupt is not None
    assert resumed_result.hitl_interrupt is None
    assert "Email sent successfully" in resumed_result.answer
