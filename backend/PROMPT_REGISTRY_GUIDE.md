# Prompt Registry - Quick Reference Guide

## 🎯 When to Use Each Module

### 1. Planning Prompts (`planning.py`)
**Use when:** Creating execution plans from user queries

```python
from src.prompts import PlanningPrompts

prompt = PlanningPrompts.create_plan("What is our Q3 revenue?")
# Returns: JSON-structured prompt with tool options
```

**Features:**
- ✅ Multi-step decomposition
- ✅ English query optimization
- ✅ Tool name validation (retrieve, calculator, web_search)

---

### 2. Generation Prompts (`generation.py`)
**Use when:** Creating answers from contexts

```python
from src.prompts import GenerationPrompts
from src.prompts.generation import ContextType

# Step 1: Detect context type
if step_ctx.get("type") == "tool" and step_ctx.get("tool_name") == "web_search":
    context_type = ContextType.WEB_SEARCH
elif step_ctx.get("type") == "retrieval":
    context_type = ContextType.DOCUMENT
else:
    context_type = ContextType.TOOL

# Step 2: Get system prompt
system_prompt = GenerationPrompts.get_system_prompt(context_type)

# Step 3: Build user message
user_message = GenerationPrompts.build_user_message(
    question="What is the revenue?",
    context=formatted_context,
    refined_query="Q3 revenue"  # optional
)
```

**Citation Rules by Context:**

| Context Type | Citation Rule | Example Output |
|--------------|---------------|----------------|
| `WEB_SEARCH` | No [n] citations | "According to recent data... Source: Web search" |
| `DOCUMENT` | Required [n] | "Revenue is $1M [1]. Sources: report.pdf (page 5)" |
| `TOOL` | Tool mention | "According to calculator: $150K. Tool: calculator" |

**Clarification Messages:**
```python
clarification = GenerationPrompts.clarification_message(
    reasoning="The query is too ambiguous."
)
# Returns: "I need more information... Could you please provide more details?"
```

---

### 3. Tool Prompts (`tools.py`)
**Use when:** Calling calculator or web_search tools

```python
from src.prompts import ToolPrompts

# Planned call (from original plan)
prompt = ToolPrompts.calculator_prompt(
    task="calculator: 15% of budget",
    is_detour=False
)

# Detour call (ad-hoc from reflection)
prompt = ToolPrompts.web_search_prompt(
    task="Nanjing weather tomorrow",
    is_detour=True  # Adds "supplementary" note
)

# Fallback (no plan)
prompt = ToolPrompts.fallback_prompt(
    query="What is 15% of 1000?",
    tool_type="calculator"
)
```

**Detour vs Planned:**

| Call Type | When Used | Prompt Note |
|-----------|-----------|-------------|
| **Planned** | From original plan | "Extract from task description above" |
| **Detour** | From reflection/refinement | "This is a supplementary call..." |

---

### 4. Evaluation Prompts (`evaluation.py`)
**Use when:** Coordinating quality assessment

```python
from src.prompts import EvaluationPrompts

# Context note for logging
note = EvaluationPrompts.get_evaluation_context_note()
# Returns: "Evaluation uses step-specific query from plan..."

# Fallback reasoning
reasoning = EvaluationPrompts.fallback_reasoning()
# Returns: "Reflection failed due to error..."
```

**Note:** Most evaluation logic is in `retrieval_evaluator.py` (mode-based: FAST/BALANCED/THOROUGH)

---

### 5. Refinement Prompts (`refinement.py`)
**Use when:** Coordinating query improvement

```python
from src.prompts import RefinementPrompts

# Context note for logging
note = RefinementPrompts.get_refinement_context_note()
# Returns: "Refinement operates on step-specific query..."

# Extraction note
note = RefinementPrompts.extract_step_query_note()
# Returns: "Extract query from plan step format..."
```

**Note:** Most refinement logic is in `query_refiner.py`

---

## 🔍 Decision Tree: Which Module?

```
Are you creating an execution plan?
├─ YES → PlanningPrompts.create_plan()
└─ NO
   Are you calling a tool (calculator/web_search)?
   ├─ YES → ToolPrompts.calculator_prompt() or web_search_prompt()
   └─ NO
      Are you generating an answer?
      ├─ YES → GenerationPrompts (check context type!)
      │        ├─ Web search? → ContextType.WEB_SEARCH
      │        ├─ Documents? → ContextType.DOCUMENT
      │        └─ Tool result? → ContextType.TOOL
      └─ NO
         Are you assessing quality or refining?
         └─ Use EvaluationPrompts or RefinementPrompts for coordination
```

---

## ⚠️ Common Mistakes to Avoid

### ❌ DON'T: Hardcode prompts inline
```python
# BAD - Back to the old problem!
system_prompt = "You are a helpful assistant..."
```

### ✅ DO: Use the registry
```python
# GOOD - Centralized and maintainable
from src.prompts import GenerationPrompts
from src.prompts.generation import ContextType

system_prompt = GenerationPrompts.get_system_prompt(ContextType.DOCUMENT)
```

---

### ❌ DON'T: Forget context type
```python
# BAD - All contexts get same rules
system_prompt = GenerationPrompts.DOCUMENT_SYSTEM
```

### ✅ DO: Detect context type first
```python
# GOOD - Context-aware rules
is_web = (step_ctx.get("type") == "tool" and 
          step_ctx.get("tool_name") == "web_search")
context_type = ContextType.WEB_SEARCH if is_web else ContextType.DOCUMENT
system_prompt = GenerationPrompts.get_system_prompt(context_type)
```

---

### ❌ DON'T: Assume all tool calls are planned
```python
# BAD - Detour calls lose context
prompt = ToolPrompts.calculator_prompt(task, is_detour=False)
```

### ✅ DO: Check if it's a detour
```python
# GOOD - Proper detour detection
is_detour = state.get("evaluation_result") is not None
prompt = ToolPrompts.calculator_prompt(task, is_detour=is_detour)
```

---

## 📝 Checklist for Adding New Prompts

1. **Identify module:** Planning? Generation? Tools? Evaluation? Refinement?
2. **Add method:** Static method with clear docstring
3. **Consider context:** Does it need different rules for different contexts?
4. **Add tests:** Update `test_prompt_registry.py` and `test_integration.py`
5. **Document:** Add to this guide and `src/prompts/README.md`
6. **Update imports:** Export in `src/prompts/__init__.py` if needed

---

## 🧪 Testing Your Changes

### Unit Tests
```bash
python test_prompt_registry.py
```

### Integration Tests
```bash
python test_integration.py
```

### Manual Verification
```bash
# Check imports
python -c "from src.prompts import PlanningPrompts, GenerationPrompts, ToolPrompts; print('✅ OK')"

# Check context types
python -c "from src.prompts.generation import ContextType; print(list(ContextType)); print('✅ OK')"
```

---

## 🎯 Key Principles

1. **Single Source of Truth** - All prompts in `src/prompts/`
2. **Context-Aware** - Use `ContextType` for generation
3. **Type-Safe** - Use enums, not strings
4. **Documented** - Every method has docstring
5. **Tested** - Covered by unit + integration tests
6. **Maintainable** - Change once, affects everywhere

---

## 📞 Need Help?

- **Read:** [src/prompts/README.md](src/prompts/README.md)
- **Review:** [MIGRATION_REPORT.md](MIGRATION_REPORT.md)
- **Compare:** [architecture_comparison.py](architecture_comparison.py)
- **Test:** [test_prompt_registry.py](test_prompt_registry.py) and [test_integration.py](test_integration.py)

**Remember:** The goal is to make prompt conflicts IMPOSSIBLE, not just unlikely! 🎯
