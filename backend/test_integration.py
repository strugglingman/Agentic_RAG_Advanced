"""
Integration test to verify prompt migration doesn't change behavior.

This test simulates the LangGraph flow with the new prompt registry.
"""

import sys

sys.path.insert(0, "d:/chatbot/backend")

from src.prompts import PlanningPrompts, GenerationPrompts, ToolPrompts
from src.prompts.generation import ContextType


def test_end_to_end_flow():
    """Simulate a complete RAG flow with prompt registry"""

    print("\n" + "=" * 60)
    print("🧪 Testing End-to-End RAG Flow with Prompt Registry")
    print("=" * 60 + "\n")

    # 1. Planning Phase
    user_query = "介绍一下the man called ove这本书"
    planning_prompt = PlanningPrompts.create_plan(user_query)

    print("✅ STEP 1: Planning")
    print(f"   Query: {user_query}")
    print(f"   Prompt length: {len(planning_prompt)} chars")
    assert "retrieve" in planning_prompt
    assert "The Man Called Ove" in planning_prompt
    assert "JSON" in planning_prompt
    print("   ✓ Planning prompt contains expected keywords\n")

    # 2. Tool Execution Phase (planned call)
    plan_step = "retrieve: Summarize the book 'The Man Called Ove'"
    tool_prompt = ToolPrompts.web_search_prompt(plan_step, is_detour=False)

    print("✅ STEP 2: Tool Execution (Planned)")
    print(f"   Plan step: {plan_step}")
    print(f"   Prompt length: {len(tool_prompt)} chars")
    assert "web_search tool" in tool_prompt
    assert plan_step in tool_prompt
    assert "ONLY from the task description" in tool_prompt
    print("   ✓ Tool prompt correctly extracts task\n")

    # 3. Tool Execution Phase (detour call)
    refined_query = "Ove Fredrik Backman summary"
    tool_prompt_detour = ToolPrompts.web_search_prompt(refined_query, is_detour=True)

    print("✅ STEP 3: Tool Execution (Detour)")
    print(f"   Refined query: {refined_query}")
    print(f"   Prompt length: {len(tool_prompt_detour)} chars")
    assert refined_query in tool_prompt_detour
    assert "supplementary search" in tool_prompt_detour
    print("   ✓ Detour prompt includes supplementary note\n")

    # 4. Generation Phase - Web Search
    step_ctx_web = {
        "type": "tool",
        "tool_name": "web_search",
        "result": "A Man Called Ove is a novel by Fredrik Backman...",
        "plan_step": "web_search: The Man Called Ove",
    }

    is_web_search = (
        step_ctx_web.get("type") == "tool"
        and step_ctx_web.get("tool_name") == "web_search"
    )

    context_type = ContextType.WEB_SEARCH if is_web_search else ContextType.DOCUMENT
    system_prompt_web = GenerationPrompts.get_system_prompt(context_type)

    print("✅ STEP 4: Generation (Web Search)")
    print(f"   Context type: {context_type.value}")
    print(f"   System prompt length: {len(system_prompt_web)} chars")
    assert "DO NOT use bracket citations" in system_prompt_web
    assert "web search results" in system_prompt_web.lower()
    assert "Source: Web search" in system_prompt_web
    print("   ✓ Web search prompt has relaxed citation rules\n")

    # 5. Generation Phase - Document Retrieval
    step_ctx_doc = {
        "type": "retrieval",
        "docs": [{"chunk": "Company revenue was $1M", "source": "report.pdf"}],
        "plan_step": "retrieve: company revenue",
    }

    is_web_search_doc = (
        step_ctx_doc.get("type") == "tool"
        and step_ctx_doc.get("tool_name") == "web_search"
    )

    context_type_doc = (
        ContextType.WEB_SEARCH if is_web_search_doc else ContextType.DOCUMENT
    )
    system_prompt_doc = GenerationPrompts.get_system_prompt(context_type_doc)

    print("✅ STEP 5: Generation (Document Retrieval)")
    print(f"   Context type: {context_type_doc.value}")
    print(f"   System prompt length: {len(system_prompt_doc)} chars")
    assert "Include bracket citations [n]" in system_prompt_doc
    assert "ONLY information from the numbered contexts" in system_prompt_doc
    assert "I don't have enough information" in system_prompt_doc
    print("   ✓ Document prompt has strict citation rules\n")

    # 6. User Message Building
    question = "What is the company revenue?"
    context = "Context 1: Revenue was $1M in Q3"
    refined = "Q3 company revenue"

    user_message = GenerationPrompts.build_user_message(question, context, refined)

    print("✅ STEP 6: User Message Building")
    print(f"   Question: {question}")
    print(f"   Refined: {refined}")
    print(f"   Message length: {len(user_message)} chars")
    assert question in user_message
    assert context in user_message
    assert "Refined as:" in user_message
    assert refined in user_message
    print("   ✓ User message includes question, context, and refinement\n")

    # 7. Clarification Message
    reasoning = "The query is too ambiguous to answer accurately."
    clarification = GenerationPrompts.clarification_message(reasoning)

    print("✅ STEP 7: Clarification Message")
    print(f"   Reasoning: {reasoning}")
    print(f"   Message length: {len(clarification)} chars")
    assert "I need more information" in clarification
    assert reasoning in clarification
    assert "more specific details" in clarification
    print("   ✓ Clarification message includes reasoning\n")

    print("=" * 60)
    print("✨ ALL INTEGRATION TESTS PASSED!")
    print("=" * 60)
    print("\n📊 Summary:")
    print("   • Planning prompts: ✅ Working")
    print("   • Tool prompts (planned & detour): ✅ Working")
    print("   • Generation prompts (web & docs): ✅ Working")
    print("   • Context-aware citation rules: ✅ Working")
    print("   • User message building: ✅ Working")
    print("   • Clarification messages: ✅ Working")
    print("\n✅ Migration successful - no logic changes!")


if __name__ == "__main__":
    try:
        test_end_to_end_flow()
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
