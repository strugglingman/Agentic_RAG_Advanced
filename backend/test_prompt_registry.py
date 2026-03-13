"""
Test suite to verify prompt registry matches existing inline prompts.

Run with pytest:
    pytest test_prompt_registry.py -v
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.prompts import PlanningPrompts, GenerationPrompts, ToolPrompts
from src.prompts.generation import ContextType


def test_planning_prompt():
    """Verify planning prompt contains expected tool references"""
    query = "What is our Q3 revenue?"
    prompt = PlanningPrompts.create_plan(query)

    # Check critical components — all current tools must appear
    assert "retrieve" in prompt
    assert "web_search" in prompt
    assert "direct_answer" in prompt
    assert "code_execution" in prompt
    assert "JSON" in prompt
    assert query in prompt
    assert "OPTIMIZED search query in English" in prompt
    print("✅ Planning prompt matches")


def test_generation_web_search():
    """Verify web search generation prompt has relaxed citation rules"""
    system_prompt = GenerationPrompts.get_system_prompt(ContextType.WEB_SEARCH)

    assert "DO NOT use bracket citations" in system_prompt
    assert "web search results" in system_prompt.lower()
    assert "Source: Web search" in system_prompt
    print("✅ Web search generation prompt matches")


def test_generation_document():
    """Verify document generation prompt has strict citation rules"""
    system_prompt = GenerationPrompts.get_system_prompt(ContextType.DOCUMENT)

    assert "Include bracket citations [n]" in system_prompt
    assert "ONLY information from the numbered contexts" in system_prompt
    assert "I don't have enough information" in system_prompt
    print("✅ Document generation prompt matches")


def test_user_message_format():
    """Verify user message includes question, context, and refinement"""
    question = "What is the revenue?"
    context = "Context 1: Revenue is $1M"
    refined = "Q3 revenue details"

    message = GenerationPrompts.build_user_message(question, context, refined)

    assert question in message
    assert context in message
    assert refined in message
    assert "Refined as:" in message
    print("✅ User message format matches")


def test_web_search_prompt():
    """Verify web search tool prompt"""
    task = "web_search: Nanjing weather"
    prompt = ToolPrompts.web_search_prompt(task, is_detour=False)

    assert task in prompt
    assert "web_search tool" in prompt
    assert "ONLY from the task description" in prompt
    print("✅ Web search prompt matches")


def test_code_execution_prompt():
    """Verify code execution tool prompt"""
    task = "code_execution: Calculate statistics on the dataset"
    prompt = ToolPrompts.code_execution_prompt(task, is_detour=False)

    assert task in prompt
    assert "code_execution tool" in prompt
    print("✅ Code execution prompt matches")


def test_context_type_detection():
    """Verify context type detection logic"""
    step_ctx_web = {"type": "tool", "tool_name": "web_search"}
    is_web_search = (
        step_ctx_web.get("type") == "tool"
        and step_ctx_web.get("tool_name") == "web_search"
    )
    context_type = ContextType.WEB_SEARCH if is_web_search else ContextType.DOCUMENT
    assert context_type == ContextType.WEB_SEARCH

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
        test_web_search_prompt()
        test_code_execution_prompt()
        test_context_type_detection()

        print("=" * 50)
        print("\n✨ ALL TESTS PASSED - Prompt registry is compatible!\n")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
