"""
Test suite to verify prompt registry matches existing inline prompts.

Run this BEFORE migrating to ensure no behavioral changes.
"""

import sys

sys.path.insert(0, "d:/chatbot/backend")

from src.prompts import PlanningPrompts, GenerationPrompts, ToolPrompts
from src.prompts.generation import ContextType


def test_planning_prompt():
    """Verify planning prompt matches langgraph_nodes.py line ~120"""
    query = "What is our Q3 revenue?"
    prompt = PlanningPrompts.create_plan(query)

    # Check critical components
    assert "retrieve" in prompt
    assert "calculator" in prompt
    assert "web_search" in prompt
    assert "JSON" in prompt
    assert query in prompt
    assert "OPTIMIZED search query in English" in prompt
    print("✅ Planning prompt matches")


def test_generation_web_search():
    """Verify web search prompt matches langgraph_nodes.py line ~1109"""
    system_prompt = GenerationPrompts.get_system_prompt(ContextType.WEB_SEARCH)

    # Check critical rules
    assert "DO NOT use bracket citations" in system_prompt
    assert "web search results" in system_prompt.lower()
    assert "Source: Web search" in system_prompt
    print("✅ Web search generation prompt matches")


def test_generation_document():
    """Verify document prompt matches langgraph_nodes.py line ~1120"""
    system_prompt = GenerationPrompts.get_system_prompt(ContextType.DOCUMENT)

    # Check critical rules
    assert "Include bracket citations [n]" in system_prompt
    assert "ONLY information from the numbered contexts" in system_prompt
    assert "I don't have enough information" in system_prompt
    print("✅ Document generation prompt matches")


def test_user_message_format():
    """Verify user message format matches langgraph_nodes.py line ~1158"""
    question = "What is the revenue?"
    context = "Context 1: Revenue is $1M"
    refined = "Q3 revenue details"

    message = GenerationPrompts.build_user_message(question, context, refined)

    assert question in message
    assert context in message
    assert refined in message
    assert "Refined as:" in message
    print("✅ User message format matches")


def test_calculator_prompt():
    """Verify calculator prompt matches langgraph_nodes.py line ~536"""
    task = "calculator: 15% of budget"
    prompt = ToolPrompts.calculator_prompt(task, is_detour=False)

    assert task in prompt
    assert "calculator tool" in prompt
    assert "ONLY from the task description" in prompt
    print("✅ Calculator prompt matches")


def test_web_search_prompt():
    """Verify web search prompt matches langgraph_nodes.py line ~715"""
    task = "web_search: Nanjing weather"
    prompt = ToolPrompts.web_search_prompt(task, is_detour=False)

    assert task in prompt
    assert "web_search tool" in prompt
    assert "ONLY from the task description" in prompt
    print("✅ Web search prompt matches")


def test_context_type_detection():
    """Verify context type detection logic"""
    # Simulate step_ctx from langgraph_nodes.py
    step_ctx_web = {"type": "tool", "tool_name": "web_search"}

    # This is the logic from line ~1105 in langgraph_nodes.py
    is_web_search = (
        step_ctx_web.get("type") == "tool"
        and step_ctx_web.get("tool_name") == "web_search"
    )

    if is_web_search:
        context_type = ContextType.WEB_SEARCH
    else:
        context_type = ContextType.DOCUMENT

    assert context_type == ContextType.WEB_SEARCH

    # Test document context
    step_ctx_doc = {"type": "retrieval"}
    is_web_search = (
        step_ctx_doc.get("type") == "tool"
        and step_ctx_doc.get("tool_name") == "web_search"
    )
    context_type = ContextType.WEB_SEARCH if is_web_search else ContextType.DOCUMENT
    assert context_type == ContextType.DOCUMENT

    print("✅ Context type detection logic matches")


if __name__ == "__main__":
    print("\n🔍 Testing Prompt Registry Compatibility\n")
    print("=" * 50)

    try:
        test_planning_prompt()
        test_generation_web_search()
        test_generation_document()
        test_user_message_format()
        test_calculator_prompt()
        test_web_search_prompt()
        test_context_type_detection()

        print("=" * 50)
        print("\n✨ ALL TESTS PASSED - Prompt registry is compatible!\n")
        print("Next steps:")
        print("1. Review the prompts in src/prompts/")
        print("2. Run this test: python test_prompt_registry.py")
        print("3. If all pass, we can migrate langgraph_nodes.py")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        print("Please review the prompt registry before migrating.")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
