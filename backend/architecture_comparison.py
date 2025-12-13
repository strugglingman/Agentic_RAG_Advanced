"""
Visual comparison: Before vs After prompt centralization
"""

BEFORE = """
❌ BEFORE: Scattered Prompts (Conflict Risk HIGH)

┌─────────────────────────────────────────────────────────────┐
│  langgraph_nodes.py (1421 lines)                            │
│                                                               │
│  • plan_node:                                                │
│    └── 50-line inline planning prompt                       │
│                                                               │
│  • tool_calculator_node:                                     │
│    ├── Planned call prompt (10 lines)                       │
│    └── Detour call prompt (8 lines)                         │
│                                                               │
│  • tool_web_search_node:                                     │
│    ├── Planned call prompt (10 lines)                       │
│    └── Detour call prompt (8 lines)                         │
│                                                               │
│  • generate_node:                                            │
│    ├── Web search system prompt (10 lines)  ← NO [n]        │
│    ├── Document system prompt (15 lines)    ← REQUIRES [n]  │
│    └── User message template (8 lines)                      │
│    └── Clarification message (f-string)                     │
│                                                               │
│  Problem: Easy to miss web vs doc citation rules!           │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  agent_service.py                                            │
│  • More tool calling prompts                                │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  query_refiner.py                                            │
│  • Refinement prompts                                       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  retrieval_evaluator.py                                      │
│  • Evaluation prompts (mode-based)                          │
└─────────────────────────────────────────────────────────────┘

⚠️  Issues:
   • Prompts scattered across 6+ files
   • Easy to create conflicts (web vs doc citation rules)
   • Hard to maintain consistency
   • No type safety for context detection
"""

AFTER = """
✅ AFTER: Centralized Prompts (Conflict Risk ZERO)

┌─────────────────────────────────────────────────────────────┐
│  src/prompts/  (SINGLE SOURCE OF TRUTH)                     │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  planning.py                                            ││
│  │  • PlanningPrompts.create_plan()                        ││
│  │    └── Multi-step decomposition with English queries   ││
│  └─────────────────────────────────────────────────────────┘│
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  tools.py                                               ││
│  │  • ToolPrompts.calculator_prompt(task, is_detour)       ││
│  │  • ToolPrompts.web_search_prompt(task, is_detour)       ││
│  │  • ToolPrompts.fallback_prompt(query, tool_type)        ││
│  └─────────────────────────────────────────────────────────┘│
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  generation.py  (CONTEXT-AWARE)                         ││
│  │                                                          ││
│  │  Enum: ContextType {WEB_SEARCH, DOCUMENT, TOOL}         ││
│  │                                                          ││
│  │  • GenerationPrompts.get_system_prompt(context_type)    ││
│  │    ├── WEB_SEARCH → Relaxed rules (no [n])              ││
│  │    ├── DOCUMENT   → Strict rules (requires [n])         ││
│  │    └── TOOL       → Tool-specific rules                 ││
│  │                                                          ││
│  │  • GenerationPrompts.build_user_message(q, ctx, ref)    ││
│  │  • GenerationPrompts.clarification_message(reasoning)   ││
│  └─────────────────────────────────────────────────────────┘│
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  evaluation.py                                          ││
│  │  • Coordination prompts for quality assessment          ││
│  └─────────────────────────────────────────────────────────┘│
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  refinement.py                                          ││
│  │  • Coordination prompts for query improvement           ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  langgraph_nodes.py (CLEAN, LOGIC-FOCUSED)                  │
│                                                               │
│  • plan_node:                                                │
│    └── prompt = PlanningPrompts.create_plan(query)          │
│                                                               │
│  • tool_calculator_node:                                     │
│    └── prompt = ToolPrompts.calculator_prompt(task, detour) │
│                                                               │
│  • tool_web_search_node:                                     │
│    └── prompt = ToolPrompts.web_search_prompt(task, detour) │
│                                                               │
│  • generate_node:                                            │
│    ├── if web_search: ctx = ContextType.WEB_SEARCH          │
│    │   else: ctx = ContextType.DOCUMENT                     │
│    ├── sys = GenerationPrompts.get_system_prompt(ctx)       │
│    └── usr = GenerationPrompts.build_user_message(...)      │
│                                                               │
│  No more inline prompts! Type-safe context detection!       │
└─────────────────────────────────────────────────────────────┘

✅ Benefits:
   • Single source of truth for all prompts
   • Impossible to miss context-aware rules (ContextType enum)
   • Easy to update (change once, affects all nodes)
   • Type-safe with IDE autocomplete
   • Testable independently
   • Clear documentation
"""

USAGE_EXAMPLE = """
📚 USAGE EXAMPLE: Context-Aware Generation

# OLD WAY (Error-Prone)
if is_web_search:
    system_prompt = '''You are a helpful assistant...
    RULES: DO NOT use bracket citations...'''
else:
    system_prompt = '''You are a helpful assistant...
    RULES: Include bracket citations [n]...'''

⚠️  Problem: Easy to forget which rule applies where!

---

# NEW WAY (Type-Safe)
from src.prompts import GenerationPrompts
from src.prompts.generation import ContextType

# Detect context type
is_web_search = (
    step_ctx.get("type") == "tool" 
    and step_ctx.get("tool_name") == "web_search"
)

# Automatic prompt selection based on type
context_type = ContextType.WEB_SEARCH if is_web_search else ContextType.DOCUMENT
system_prompt = GenerationPrompts.get_system_prompt(context_type)

✅ Benefit: IDE shows you the 3 options (WEB_SEARCH, DOCUMENT, TOOL)
✅ Benefit: Impossible to get wrong citation rules
✅ Benefit: Change rules in one place (generation.py)
"""


def print_comparison():
    print(BEFORE)
    print("\n" + "=" * 65 + "\n")
    print(AFTER)
    print("\n" + "=" * 65 + "\n")
    print(USAGE_EXAMPLE)


if __name__ == "__main__":
    print_comparison()
